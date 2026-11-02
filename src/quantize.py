"""Make the quantized checkpoint for a config that has a `quantize` section.

Usage (from the repository root, on a GPU):
    python src/quantize.py --config configs/qwen3-1.7b_int8.yaml   # in the BFCL environment
    python src/quantize.py --config configs/qwen3-1.7b_gptq.yaml   # in the quantization environment (GPTQModel)

Writes the checkpoint to checkpoints/<config name>/ (or $CHECKPOINT_DIR/<config name>/), from the model and
revision pinned in the config. quantize_info.json is written last, so a folder without it is unfinished.
src/run_eval.py then loads the weights from that folder.

Methods (the config's quantize.method):
    bnb-int8  LLM.int8() from bitsandbytes: 8-bit weights with one scale per row; input features above
              llm_int8_threshold are multiplied in FP16. vLLM can only load 8-bit bitsandbytes models that
              were quantized beforehand, so this is done here with transformers.
    gptq      GPTQ with GPTQModel, saved in the GPTQ format (vLLM runs it with its GPTQ kernels on a T4)
    awq       AWQ with GPTQModel, saved in the AWQ GEMM format (vLLM's AWQ kernels run on a T4)
GPTQ and AWQ use the same calibration set (CALIBRATION) for every model.
"""

import argparse
import gzip
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from common import checkpoint_path, package_versions, resolve_revision

# One calibration set for every GPTQ and AWQ checkpoint: the first `samples` English C4 documents (training
# shard 1) with at least `min_chars` characters, each cut to its first `tokens` tokens. The same documents are
# used for every model; only the tokenizer differs.
CALIBRATION = {
    "dataset": "allenai/c4",
    "revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
    "file": "en/c4-train.00001-of-01024.json.gz",
    "samples": 256,
    "min_chars": 3000,
    "tokens": 512,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="a file in configs/ with a quantize section")
    parser.add_argument("--overwrite", action="store_true", help="make the checkpoint again if it already exists")
    return parser.parse_args()


def bnb_int8(cfg, revision, out_dir):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    settings = cfg["quantize"]
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        revision=revision,
        torch_dtype=torch.float16,
        device_map={"": 0},  # bitsandbytes quantizes on the GPU while loading
        quantization_config=BitsAndBytesConfig(load_in_8bit=True,
                                               llm_int8_threshold=settings["llm_int8_threshold"]),
    )
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(cfg["model"], revision=revision).save_pretrained(out_dir)


def calibration_texts():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(CALIBRATION["dataset"], CALIBRATION["file"], repo_type="dataset",
                           revision=CALIBRATION["revision"])
    texts = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            if len(text) >= CALIBRATION["min_chars"]:
                texts.append(text)
                if len(texts) == CALIBRATION["samples"]:
                    return texts
    raise RuntimeError(f"{CALIBRATION['file']} has only {len(texts)} documents with "
                       f"{CALIBRATION['min_chars']}+ characters")


def fp16_copy(base_dir, copy_dir):
    """Save the base model with its weights in FP16.

    GPTQModel converts BF16 weights to FP16 in RAM, which is about 15 GiB for Qwen3-8B. Weights that are already in
    FP16 stay on disk until they are needed. The values are the same either way.
    """
    import torch
    from safetensors.torch import load_file, save_file

    size = sum(path.stat().st_size for path in base_dir.glob("*.safetensors"))
    if shutil.disk_usage(copy_dir.parent).free < 1.5 * size:  # room for this copy and for the 4-bit checkpoint
        raise RuntimeError(f"Not enough disk space for an FP16 copy of {size / 2**30:.1f} GiB in {copy_dir.parent}")
    shutil.rmtree(copy_dir, ignore_errors=True)
    shutil.copytree(base_dir, copy_dir, ignore=shutil.ignore_patterns("*.safetensors"))
    for path in sorted(base_dir.glob("*.safetensors")):
        weights = {name: tensor.to(torch.float16) for name, tensor in load_file(path).items()}
        save_file(weights, copy_dir / path.name, metadata={"format": "pt"})


def limit_awq_scales(x_mean_floor, max_scale):
    """Keep AWQ's channel scales in a range that FP16 can hold.

    A channel that is almost never active gets an extreme scale, and then all the others shift with it. AWQ divides
    the previous layer's weights by these scales, so its output can overflow FP16 and turn into NaN. Very quiet
    channels are treated as if they had x_mean_floor times the median activation, and every scale is kept between
    1 / max_scale and max_scale. Other channels are not changed.
    """
    from gptqmodel.looper.awq_processor import AWQProcessor

    search = AWQProcessor._compute_best_scale

    def limited(self, x, w_mean, x_mean, *args, **kwargs):
        x_mean = x_mean.clamp(min=float(x_mean.median()) * x_mean_floor)
        scales, loss = search(self, x, w_mean, x_mean, *args, **kwargs)
        return scales.clamp(1 / max_scale, max_scale), loss

    AWQProcessor._compute_best_scale = limited


def gptqmodel_4bit(cfg, revision, out_dir):
    import torch
    from gptqmodel import GPTQModel, QuantizeConfig
    from gptqmodel.quantization import FORMAT, METHOD
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    settings = cfg["quantize"]
    # GPTQModel reads the model config without a revision, so the pinned revision is downloaded first
    # (without Llama's original/ folder: the same weights in Meta's own format)
    base_dir = Path(snapshot_download(cfg["model"], revision=revision, ignore_patterns=["original/*"]))
    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    calibration = []
    for text in calibration_texts():
        encoded = tokenizer(text, truncation=True, max_length=CALIBRATION["tokens"])
        calibration.append({"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]})

    shared = dict(bits=settings["bits"], group_size=settings["group_size"],
                  failsafe=None,          # every layer must be calibrated; no plain rounding as a fallback
                  offload_to_disk=False)
    if settings["method"] == "gptq":
        quant_config = QuantizeConfig(**shared, quant_method=METHOD.GPTQ, format=FORMAT.GPTQ, sym=settings["sym"],
                                      desc_act=settings["desc_act"], act_group_aware=False)
    else:
        # GPTQModel packs AWQ weights with this class, and refuses to create it without its compiled CUDA kernels.
        # Only the class's forward() uses those kernels; packing is plain PyTorch and vLLM runs the model.
        from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMQuantLinear
        AwqGEMMQuantLinear.validate_once = classmethod(lambda cls: (True, None))
        if "max_scale" in settings:
            limit_awq_scales(settings["x_mean_floor"], settings["max_scale"])
        quant_config = QuantizeConfig(**shared, quant_method=METHOD.AWQ, format=FORMAT.GEMM,
                                      sym=not settings["zero_point"])

    load_dir = base_dir
    if settings.get("fp16_copy"):
        load_dir = out_dir.parent / f"{out_dir.name}_fp16"
        fp16_copy(base_dir, load_dir)
    model = GPTQModel.load(str(load_dir), quant_config, dtype=torch.float16)  # T4 has no BF16
    # batch_size also sets how many samples AWQ's scale search runs at once. With 1, GPTQModel runs all of them
    # in one forward pass, which does not fit on a T4.
    model.quantize(calibration, batch_size=settings.get("batch_size", 1))
    model.save(str(out_dir))
    if load_dir != base_dir:
        shutil.rmtree(load_dir, ignore_errors=True)

    # vLLM reads config.json with an older transformers than GPTQModel's. Keep the base model's config.json
    # (exactly what the FP16 runs used) and add only the quantization settings that GPTQModel wrote.
    with open(out_dir / "config.json", encoding="utf-8") as f:
        quantization_config = json.load(f)["quantization_config"]
    with open(base_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    config["quantization_config"] = quantization_config
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    # The base model's license files go with the checkpoint (needed if it is ever shared)
    for path in base_dir.iterdir():
        if path.name.upper().startswith(("LICENSE", "USE_POLICY", "NOTICE")):
            shutil.copy(path, out_dir / path.name)


METHODS = {"bnb-int8": bnb_int8, "gptq": gptqmodel_4bit, "awq": gptqmodel_4bit}


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "quantize" not in cfg:
        sys.exit(f"{args.config} has no quantize section: vLLM loads this model directly.")
    method = cfg["quantize"]["method"]
    if method not in METHODS:
        sys.exit(f"Unknown quantize.method {method!r} in {args.config}. Known: {', '.join(METHODS)}")

    out_dir = checkpoint_path(args.config)
    if (out_dir / "quantize_info.json").exists() and not args.overwrite:
        print(f"{out_dir} already exists. Use --overwrite to make it again.")
        return
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    revision = resolve_revision(cfg["model"], str(cfg.get("revision", "main")))
    print(f"Quantizing {cfg['model']} ({revision[:7]}) with {method} into {out_dir}")
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    METHODS[method](cfg, revision, out_dir)
    seconds = time.perf_counter() - t0

    size = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    info = {
        "config_file": args.config.as_posix(),
        "model": cfg["model"],
        "revision": revision,
        "method": method,
        "settings": cfg["quantize"],
        "calibration": CALIBRATION if method in ("gptq", "awq") else None,
        "started_utc": started.isoformat(timespec="seconds"),
        "seconds": round(seconds, 1),
        "size_gib": round(size / 2**30, 3),
        "versions": package_versions(),
    }
    with open(out_dir / "quantize_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"Done in {seconds:.0f} s: {info['size_gib']} GiB in {out_dir}")


if __name__ == "__main__":
    main()
