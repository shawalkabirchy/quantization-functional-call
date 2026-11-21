"""Build results/latency.md and results/latency.csv from the files that src/latency.py writes in results/latency/.

Usage (from the repository root; standard library only):
    python src/latency_table.py

One row per config and decoding mode in the CSV; the Markdown file shows the medians side by side, one table per
measure, with a column for the FP16 model with an 8-bit KV cache (Update 13).
"""

import csv
import json
from pathlib import Path

from main_table import MODELS, REPO

PRECISIONS = {"fp16": "FP16", "fp16-kv8": "FP16, 8-bit KV", "int8": "INT8", "nf4": "NF4", "gptq": "GPTQ",
              "awq": "AWQ"}
MODES = ["free", "constrained"]
BLOCK_TOKENS = 16   # vLLM's KV cache block size on CUDA (also: blocks x 16 / 8,192 = its "maximum concurrency")

NOTES = [
    "Each config was loaded on its own and answered the same 35 BFCL questions (5 per non-live category, seed 42) "
    "one at a time, after 3 untimed warm-up questions, first in free and then in constrained decoding "
    "(`src/latency.py`, notebook 08). Same prompt, greedy decoding, checkpoints and vLLM 0.8.5 as the main runs; "
    "one T4 per config, two for Qwen3-8B in FP16. Values are medians over the 35 questions.",
    "",
    "- **First token:** from the request's arrival to its first answer token (vLLM's timing). It includes reading "
    "the prompt, and in constrained decoding also building the grammar for the question's schema.",
    "- **Tokens/s:** answer tokens after the first one, divided by the time they took (1,000 / the median ms per "
    "token). This is the speed of writing the answer, independent of its length.",
    "- **Whole answer:** a stopwatch around the call, i.e. what a caller waits for. Models answer at different "
    "lengths, so this mixes speed and length; the mean answer length is in `results/latency.csv`.",
    "- **KV cache tokens:** how many tokens of keys and values vLLM could keep in the memory left after the weights "
    "(90% of the GPU, 81% for Qwen3-8B INT8), from its log: blocks × 16.",
    "- **Llama-3.2-3B in FP16 was timed twice.** The first try (16 Nov) took about 98 ms per answer token on "
    "every answer longer than two tokens: three times what its size and its own 8-bit-KV run (34 ms) suggest, "
    "while giving the same 70 answers as the second try. The table uses the second try (21 Nov, same notebook and "
    "code). In that try the GPU also slowed down during the run (28 to 64 ms per token in free decoding, 40 to "
    "74 ms in constrained decoding), so its constrained values are not reliable. The first try and a repeat of the "
    "8-bit-KV run (36 ms per token, against 34 ms) are in `results/latency/extra/`.",
    "- Where most answers are an empty list (Llama-3.2-1B NF4 and GPTQ in constrained decoding: 28 and 6 tokens "
    "on average), tokens/s rests on answers of one or two steps and is below the model's real writing speed.",
    "- In every other run the speed stayed steady from the first to the last question. Still, one run per config "
    "cannot rule out a GPU that was slower for a whole run (Kaggle's T4s can change speed during a session), so "
    "differences of about 20% or less are not meaningful.",
]


def main():
    rows = []
    for model in MODELS:
        for precision in PRECISIONS:
            path = REPO / "results" / "latency" / f"{model}_{precision}.json"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                meta = json.load(f)
            memory = meta["memory"]
            for mode in MODES:
                s = meta["summary"][mode]
                rows.append({
                    "model": MODELS[model], "precision": PRECISIONS[precision], "mode": mode,
                    "gpus": meta["tensor_parallel_size"], "n_questions": s["n_questions"],
                    "first_token_ms_median": s["first_token_ms_median"], "first_token_ms_p90": s["first_token_ms_p90"],
                    "per_token_ms_median": s["per_token_ms_median"],
                    "output_tokens_per_second": s["output_tokens_per_second"],
                    "total_s_median": s["total_s_median"], "total_s_p90": s["total_s_p90"],
                    "output_tokens_mean": s["output_tokens_mean"], "cut_off": s["finish_reasons"].get("length", 0),
                    "weights_gib_per_gpu": round(memory["model_weights_gib"], 2), "kv_cache_gib": memory["kv_cache_gib"],
                    "kv_cache_tokens": memory["kv_cache_blocks"] * BLOCK_TOKENS, "started_utc": meta["started_utc"],
                })

    out_csv = REPO / "results" / "latency.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def value(model, precision, mode, key):
        return next((r[key] for r in rows if r["model"] == MODELS[model] and r["precision"] == PRECISIONS[precision]
                     and r["mode"] == mode), None)

    def table(title, mode, key, fmt):
        header = ["Model"] + list(PRECISIONS.values())
        lines = [f"### {title}", "", "| " + " | ".join(header) + " |", "|---" + "|---:" * (len(header) - 1) + "|"]
        for model, name in MODELS.items():
            cells = [value(model, p, mode, key) for p in PRECISIONS]
            lines.append("| " + " | ".join([name] + ["" if v is None else fmt(v) for v in cells]) + " |")
        return lines + [""]

    lines = ["# Latency at Batch Size 1", "", *NOTES, "", "## Free decoding", ""]
    lines += table("Tokens per second while writing the answer", "free", "output_tokens_per_second", lambda v: f"{v:.1f}")
    lines += table("Time to the first token (ms)", "free", "first_token_ms_median", lambda v: f"{v:.0f}")
    lines += table("Time for the whole answer (s)", "free", "total_s_median", lambda v: f"{v:.2f}")
    lines += ["## Constrained decoding", ""]
    lines += table("Tokens per second while writing the answer", "constrained", "output_tokens_per_second",
                   lambda v: f"{v:.1f}")
    lines += table("Time to the first token (ms)", "constrained", "first_token_ms_median", lambda v: f"{v:.0f}")
    lines += table("Time for the whole answer (s)", "constrained", "total_s_median", lambda v: f"{v:.2f}")
    lines += ["## Memory (from vLLM's log)", ""]
    lines += table("Weights per GPU (GiB)", "free", "weights_gib_per_gpu", lambda v: f"{v:.2f}")
    lines += table("KV cache tokens", "free", "kv_cache_tokens", lambda v: f"{v:,}")
    lines += ["Built from `results/latency/` by `src/latency_table.py`; all values per config and mode, with the 90th "
              "percentiles and mean answer lengths, are in [`latency.csv`](latency.csv).", ""]
    out_md = REPO / "results" / "latency.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {out_csv} ({len(rows)} rows) and {out_md}")


if __name__ == "__main__":
    main()
