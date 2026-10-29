"""Run one model on BFCL questions with vLLM and save the raw outputs.

Usage (from the repository root, in the BFCL environment):
    python src/run_eval.py --config configs/qwen3-4b_fp16.yaml --mode free
    python src/run_eval.py --config configs/qwen3-4b_fp16.yaml --mode constrained --limit 3

Writes results/raw/<run_id>.jsonl (one line per question) and results/meta/<run_id>.json (model and
revision, software versions, GPU, memory, speed). Score the outputs afterwards with src/score.py.
The run id is <config name>_<mode>, with a suffix when only some of the questions are run.

A config with a `quantize` section loads its weights from the checkpoint that src/quantize.py made
(checkpoints/<config name>/, or $CHECKPOINT_DIR/<config name>/); the tokenizer still comes from the
pinned base model.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

os.environ.setdefault("VLLM_USE_V1", "0")  # T4 (compute capability 7.5) runs on vLLM's V0 engine

from common import REPO, checkpoint_path, package_versions, resolve_revision
from data import expand_categories, load_questions
from prompt_format import FORMAT_VERSION, build_messages
from schema import calls_schema

# Constrained mode uses xgrammar, which supports every schema we build (checked for all 1,390 questions).
# disable-any-whitespace prevents runaway whitespace (vLLM warns about this for Qwen models), and
# no-fallback makes vLLM stop with an error instead of silently switching to another backend.
GUIDED_BACKEND = "xgrammar:disable-any-whitespace,no-fallback"

# vLLM (V0 engine) log lines that report memory use
MEMORY_PATTERNS = {
    "model_weights_gib": r"Model loading took ([\d.]+) ?GiB",
    "activation_peak_gib": r"PyTorch activation peak memory takes ([\d.]+) ?GiB",
    "kv_cache_gib": r"reserved for KV Cache is ([\d.]+) ?GiB",
}


class LogCapture(logging.Handler):
    """Keeps vLLM's log messages, so the memory figures can be read from them."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="a file in configs/")
    parser.add_argument("--mode", required=True, choices=["free", "constrained"],
                        help="free: plain decoding; constrained: output forced to match the JSON schema")
    parser.add_argument("--categories", nargs="+", default=["non_live"],
                        help="non_live (default) or BFCL category names, e.g. simple_python parallel")
    parser.add_argument("--limit", type=int, help="only the first N questions of each category (quick tests)")
    parser.add_argument("--run-id", help="default: <config name>_<mode>[_<categories>][_limit<N>]")
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run with the same id")
    return parser.parse_args()


def code_version():
    """Git commit of this repository, and whether src/ or configs/ had uncommitted changes."""
    def git(*args):
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"),
                "uncommitted_changes": bool(git("status", "--porcelain", "--", "src", "configs"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes": None}


def gpu_memory_used():
    """Memory in use on each GPU (MiB), from nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        index, used, total = (int(x) for x in line.split(","))
        gpus.append({"index": index, "used_mib": used, "total_mib": total})
    return gpus


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    categories = expand_categories(args.categories)

    run_id = args.run_id or "_".join(
        [args.config.stem, args.mode]
        + ([] if args.categories == ["non_live"] else ["-".join(categories)])
        + ([f"limit{args.limit}"] if args.limit else [])
    )
    raw_path = args.results_dir / "raw" / f"{run_id}.jsonl"
    meta_path = args.results_dir / "meta" / f"{run_id}.json"
    if raw_path.exists() and not args.overwrite:
        sys.exit(f"{raw_path} already exists. Use --overwrite to replace it, or --run-id for a new name.")

    questions = []
    for category in categories:
        questions += load_questions(category)[: args.limit]
    print(f"Run {run_id}: {len(questions)} questions from {', '.join(categories)}")

    revision = resolve_revision(cfg["model"], str(cfg.get("revision", "main")))
    weights, weights_revision, quantize_info = cfg["model"], revision, None
    if "quantize" in cfg:
        checkpoint = checkpoint_path(args.config)
        info_path = checkpoint / "quantize_info.json"
        if not info_path.exists():
            sys.exit(f"No finished checkpoint in {checkpoint}. Make it first: "
                     f"python src/quantize.py --config {args.config}")
        with open(info_path, encoding="utf-8") as f:
            quantize_info = json.load(f)
        if quantize_info["revision"] != revision or quantize_info["settings"] != cfg["quantize"]:
            sys.exit(f"{checkpoint} was made from another revision or with other settings than {args.config}. "
                     f"Make it again: python src/quantize.py --config {args.config} --overwrite")
        weights, weights_revision = str(checkpoint), None
    started = datetime.now(timezone.utc)

    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import GuidedDecodingParams

    capture = LogCapture()
    logging.getLogger("vllm").addHandler(capture)

    t0 = time.perf_counter()
    llm = LLM(
        model=weights,
        revision=weights_revision,
        tokenizer=cfg["model"],
        tokenizer_revision=revision,
        dtype=cfg["dtype"],
        quantization=cfg.get("quantization"),
        tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
        max_model_len=cfg["max_model_len"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        seed=cfg["sampling"]["seed"],
        **cfg.get("llm_kwargs", {}),
    )
    load_seconds = time.perf_counter() - t0
    tokenizer = llm.get_tokenizer()

    # Every decoding parameter comes from the config, so nothing is taken from the model's
    # generation_config (unlike the BFCL run in Update 2)
    max_prompt_tokens = cfg["max_model_len"] - cfg["sampling"]["max_tokens"]
    prompts, prompt_texts, params = [], [], []
    for q in questions:
        text = tokenizer.apply_chat_template(build_messages(q), tokenize=False, add_generation_prompt=True,
                                             **cfg.get("chat_template_kwargs", {}))
        # The chat template already contains the special tokens (e.g. Llama's BOS); don't add them twice
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > max_prompt_tokens:
            sys.exit(f"{q['id']}: the prompt has {len(ids)} tokens, but max_model_len - max_tokens = "
                     f"{max_prompt_tokens}. Raise max_model_len in {args.config}.")
        prompts.append(TokensPrompt(prompt_token_ids=ids))
        prompt_texts.append(text)
        guided = None
        if args.mode == "constrained":
            guided = GuidedDecodingParams(json=calls_schema(q["functions"]), backend=GUIDED_BACKEND)
        params.append(SamplingParams(**cfg["sampling"], guided_decoding=guided))

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=True)
    generation_seconds = time.perf_counter() - t0

    rows = []
    for q, text, out in zip(questions, prompt_texts, outputs):
        completion = out.outputs[0]
        m = out.metrics
        # Time from first being scheduled to finishing (waiting in the queue is not counted).
        # Requests run in batches, so this depends on the batch as well as on the request itself.
        latency = None
        if m is not None and m.finished_time and m.first_scheduled_time:
            latency = round(m.finished_time - m.first_scheduled_time, 3)
        rows.append({
            "id": q["id"],
            "category": q["category"],
            "prompt": text,
            "output": completion.text,
            "finish_reason": completion.finish_reason,
            "n_prompt_tokens": len(out.prompt_token_ids),
            "n_output_tokens": len(completion.token_ids),
            "latency_s": latency,
        })

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    log = "\n".join(capture.messages)
    memory = {}
    for key, pattern in MEMORY_PATTERNS.items():
        found = re.search(pattern, log)
        memory[key] = float(found.group(1)) if found else None
    memory["gpus_after_run"] = gpu_memory_used()

    output_tokens = sum(r["n_output_tokens"] for r in rows)
    meta = {
        "run_id": run_id,
        "config_file": args.config.as_posix(),
        "config": cfg,
        "model": cfg["model"],
        "revision": revision,
        "precision": cfg["precision"],
        "quantized_checkpoint": quantize_info,
        "mode": args.mode,
        "guided_decoding_backend": GUIDED_BACKEND if args.mode == "constrained" else None,
        "prompt_format": FORMAT_VERSION,
        "categories": categories,
        "limit_per_category": args.limit,
        "n_questions": len(rows),
        "sampling_params": repr(SamplingParams(**cfg["sampling"])),
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_load_seconds": round(load_seconds, 1),
        "generation_seconds": round(generation_seconds, 1),
        "prompt_tokens": sum(r["n_prompt_tokens"] for r in rows),
        "output_tokens": output_tokens,
        "output_tokens_per_second": round(output_tokens / generation_seconds, 1),
        "finish_reasons": dict(Counter(r["finish_reason"] for r in rows)),
        "memory": memory,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "versions": package_versions(),
        "code": code_version(),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone: {len(rows)} answers in {generation_seconds:.0f} s "
          f"({meta['output_tokens_per_second']} output tokens/s); finish reasons {meta['finish_reasons']}")
    print(f"Model weights: {memory['model_weights_gib']} GiB")
    print(f"Saved {raw_path}\nSaved {meta_path}")


if __name__ == "__main__":
    main()
