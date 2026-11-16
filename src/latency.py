"""Latency at batch size 1: each question is sent on its own, after the previous answer has finished.

Usage (from the repository root, in the BFCL environment):
    python src/latency.py --config configs/qwen3-4b_awq.yaml
    python src/latency.py --config configs/qwen3-4b_awq.yaml --per-category 1 --warmup 1 --results-dir check

The model is loaded once. It answers a few warm-up questions (not timed), then the same sample of questions
(by default 5 per BFCL non-live category, 35 in all, chosen with seed 42) in free decoding and then in
constrained decoding. Prompt, decoding settings and checkpoint are exactly those of src/run_eval.py.

Writes results/latency/<config name>.jsonl (one line per question and mode) and results/latency/<config name>.json
(medians per mode, memory, software versions, GPUs).

The times per question:
  - first_token_ms: from the request's arrival to its first answer token, as vLLM records it. Reading the prompt
    happens in this time, and in constrained mode also building the grammar for the question's schema.
  - per_token_ms: the rest of the answer divided by the tokens after the first one (vLLM's records).
  - total_s: a stopwatch around llm.generate(), i.e. what a caller waits for the whole answer.
Answers can differ in length between models and precisions, so total_s mixes speed and answer length;
per_token_ms does not.
"""

import argparse
import hashlib
import json
import logging
import random
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from common import REPO, checkpoint_path, package_versions, resolve_revision
from data import NON_LIVE, load_questions
from prompt_format import FORMAT_VERSION, build_messages
from run_eval import GUIDED_BACKEND, MEMORY_PATTERNS, LogCapture, code_version, gpu_memory_used
from schema import calls_schema

# How much KV cache vLLM could set up (V0 engine log lines): an 8-bit KV cache should fit about twice the tokens
CACHE_PATTERNS = {
    "kv_cache_blocks": r"# \w+ blocks: (\d+)",
    "max_concurrency": r"Maximum concurrency for \d+ tokens per request: ([\d.]+)x",
}
WARMUP_CATEGORIES = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="a file in configs/")
    parser.add_argument("--modes", nargs="+", default=["free", "constrained"], choices=["free", "constrained"])
    parser.add_argument("--per-category", type=int, default=5, help="questions per non-live category (default 5)")
    parser.add_argument("--seed", type=int, default=42, help="seed of the question sample (default 42)")
    parser.add_argument("--warmup", type=int, default=3, help="untimed questions before each mode (default 3)")
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing result for this config")
    return parser.parse_args()


def sample_questions(per_category, seed):
    """The timed questions: the same for every config, `per_category` from each non-live category."""
    rng = random.Random(seed)
    return [q for category in NON_LIVE for q in rng.sample(load_questions(category), per_category)]


def warmup_questions(n, sample):
    """Untimed questions: the first questions (in BFCL's order) of a few categories that are not in the sample."""
    taken = {q["id"] for q in sample}
    questions = []
    for category in WARMUP_CATEGORIES:
        questions.append(next(q for q in load_questions(category) if q["id"] not in taken))
    return questions[:n]


def median(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 3) if values else None


def p90(values):
    values = sorted(v for v in values if v is not None)
    return round(values[min(len(values) - 1, int(0.9 * len(values)))], 3) if values else None


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    name = args.config.stem
    out_dir = args.results_dir / "latency"
    rows_path, meta_path = out_dir / f"{name}.jsonl", out_dir / f"{name}.json"
    if rows_path.exists() and not args.overwrite:
        sys.exit(f"{rows_path} already exists. Use --overwrite to replace it.")

    sample = sample_questions(args.per_category, args.seed)
    warmup = warmup_questions(args.warmup, sample)
    print(f"Latency {name}: {len(sample)} questions, {len(warmup)} warm-up, modes {', '.join(args.modes)}")

    # The same weights as run_eval.py: the pinned model, or the checkpoint that src/quantize.py made
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
            sys.exit(f"{checkpoint} was made from another revision or with other settings than {args.config}.")
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

    def prompt_of(q):
        """The prompt exactly as run_eval.py builds it."""
        text = tokenizer.apply_chat_template(build_messages(q), tokenize=False, add_generation_prompt=True,
                                             **cfg.get("chat_template_kwargs", {}))
        return text, TokensPrompt(prompt_token_ids=tokenizer.encode(text, add_special_tokens=False))

    def params_of(q, mode):
        guided = None
        if mode == "constrained":
            guided = GuidedDecodingParams(json=calls_schema(q["functions"]), backend=GUIDED_BACKEND)
        return SamplingParams(**cfg["sampling"], guided_decoding=guided)

    def timed(q, mode):
        """One question on its own; returns the answer and its times."""
        text, prompt = prompt_of(q)
        start = time.perf_counter()
        out = llm.generate([prompt], [params_of(q, mode)], use_tqdm=False)[0]
        total = time.perf_counter() - start
        completion, m = out.outputs[0], out.metrics
        n = len(completion.token_ids)
        first = rest = None
        if m is not None and m.first_token_time is not None:
            first = m.first_token_time - m.arrival_time
            if m.finished_time is not None and n > 1:
                rest = (m.finished_time - m.first_token_time) / (n - 1)
        return {
            "id": q["id"],
            "category": q["category"],
            "mode": mode,
            "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "n_prompt_tokens": len(out.prompt_token_ids),
            "n_output_tokens": n,
            "finish_reason": completion.finish_reason,
            "first_token_ms": None if first is None else round(first * 1000, 1),
            "per_token_ms": None if rest is None else round(rest * 1000, 2),
            "total_s": round(total, 3),
            "output": completion.text,
        }

    rows, summary = [], {}
    for mode in args.modes:
        for q in warmup:
            timed(q, mode)
        mode_rows = []
        for i, q in enumerate(sample, 1):
            row = timed(q, mode)
            mode_rows.append(row)
            print(f"  {mode:11s} {i:3d}/{len(sample)} {q['id']:28s} first {row['first_token_ms']} ms, "
                  f"{row['n_output_tokens']} tokens, {row['total_s']:.2f} s", flush=True)
        rows += mode_rows
        per_token = median(r["per_token_ms"] for r in mode_rows)
        summary[mode] = {
            "n_questions": len(mode_rows),
            "first_token_ms_median": median(r["first_token_ms"] for r in mode_rows),
            "first_token_ms_p90": p90(r["first_token_ms"] for r in mode_rows),
            "per_token_ms_median": per_token,
            "output_tokens_per_second": round(1000 / per_token, 1) if per_token else None,
            "total_s_median": median(r["total_s"] for r in mode_rows),
            "total_s_p90": p90(r["total_s"] for r in mode_rows),
            "total_s_sum": round(sum(r["total_s"] for r in mode_rows), 1),
            "output_tokens_mean": round(statistics.mean(r["n_output_tokens"] for r in mode_rows), 1),
            "finish_reasons": dict(Counter(r["finish_reason"] for r in mode_rows)),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(rows_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    log = "\n".join(capture.messages)
    memory = {}
    for key, pattern in {**MEMORY_PATTERNS, **CACHE_PATTERNS}.items():
        found = re.search(pattern, log)
        memory[key] = (int(found.group(1)) if key == "kv_cache_blocks" else float(found.group(1))) if found else None
    memory["gpus_after_run"] = gpu_memory_used()

    meta = {
        "config_file": args.config.as_posix(),
        "config": cfg,
        "model": cfg["model"],
        "revision": revision,
        "precision": cfg["precision"],
        "quantized_checkpoint": quantize_info,
        "kv_cache_dtype": cfg.get("llm_kwargs", {}).get("kv_cache_dtype", "auto"),
        "tensor_parallel_size": cfg.get("tensor_parallel_size", 1),
        "guided_decoding_backend": GUIDED_BACKEND,
        "prompt_format": FORMAT_VERSION,
        "batch_size": 1,
        "sample": {"per_category": args.per_category, "seed": args.seed, "ids": [q["id"] for q in sample]},
        "warmup_ids": [q["id"] for q in warmup],
        "sampling_params": repr(SamplingParams(**cfg["sampling"])),
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_load_seconds": round(load_seconds, 1),
        "summary": summary,
        "memory": memory,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "versions": package_versions(),
        "code": code_version(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    for mode, s in summary.items():
        print(f"\n{mode}: first token {s['first_token_ms_median']} ms, {s['output_tokens_per_second']} tokens/s, "
              f"whole answer {s['total_s_median']} s (medians); {s['output_tokens_mean']} tokens on average")
    print(f"Model weights: {memory['model_weights_gib']} GiB; KV cache blocks: {memory['kv_cache_blocks']}")
    print(f"Saved {rows_path}\nSaved {meta_path}")


if __name__ == "__main__":
    main()
