# FP16 Baselines (Update 4)

**Date:** 2026-10-12
**Goal:** function-calling accuracy of every core model in full precision (FP16), the baseline for every quantized run.

## Setup

- **Questions:** BFCL `non_live`, 1390 questions per model, bfcl-eval 2026.3.23
- **Prompt format:** `json-array-v1` (as in Update 3), free decoding; Qwen3 with thinking off
- **Decoding:** greedy (temperature 0), seed 42, at most 4096 answer tokens
- **Inference:** vLLM 0.8.5, FP16, on Kaggle's two Tesla T4s: one T4 per model, except Qwen3-8B, which is split across both
- **Software:** Python 3.11.17, torch 2.6.0, transformers 4.51.3
- **Code:** commit `fc7f9a7`; model revisions are in `results/meta/`
- **Overall** = BFCL's Non-Live Overall Acc (unweighted mean of simple, multiple, parallel and parallel-multiple; simple = mean of Python, Java, JavaScript). Irrelevance is reported on its own.

## Accuracy (%)

| Model | Python | Java | JavaScript | Multiple | Parallel | Parallel Multiple | Overall | Irrelevance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 86.25 | 65.00 | 64.00 | 87.50 | 71.50 | 59.50 | 72.56 | 77.92 |
| Qwen3-4B | 95.50 | 63.00 | 70.00 | 95.00 | 90.50 | 90.50 | 88.04 | 84.58 |
| Qwen3-8B | 95.50 | 65.00 | 68.00 | 93.00 | 92.50 | 91.50 | 88.29 | 87.08 |
| Llama-3.2-1B-Instruct | 39.75 | 37.00 | 48.00 | 38.50 | 41.00 | 24.50 | 36.40 | 82.92 |
| Llama-3.2-3B-Instruct | 22.00 | 32.00 | 24.00 | 30.00 | 42.50 | 27.00 | 31.38 | 91.25 |

## Format, length and speed

| Model | Parsed, AST questions (%) | Unparsed, AST questions | Cut off at max_tokens | GPUs | Weights per GPU (GiB) | Generation (s) | Output tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 89.48 | 121 | 0 | 1 | 3.22 | 167.80 | 422.20 |
| Qwen3-4B | 99.83 | 2 | 0 | 1 | 7.56 | 468.30 | 153.30 |
| Qwen3-8B | 99.91 | 1 | 0 | 2 | 7.64 | 401.50 | 178.90 |
| Llama-3.2-1B-Instruct | 90.52 | 109 | 10 | 1 | 2.32 | 210.60 | 612.70 |
| Llama-3.2-3B-Instruct | 52.00 | 552 | 0 | 1 | 6.02 | 322.60 | 230.40 |

## How the AST answers end

Every answer to the 1,150 AST questions (every category except irrelevance) falls into one of four
groups. These are plain counts over all 1,150 questions, read from `results/scores/` (fields
`parse_ok`, `n_calls` and `valid`).

| Model | Correct | Wrong call | No call (`[]`) | Unparsed |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | 879 | 149 | 1 | 121 |
| Qwen3-4B | 1,032 | 115 | 1 | 2 |
| Qwen3-8B | 1,035 | 109 | 5 | 1 |
| Llama-3.2-1B-Instruct | 428 | 325 | 288 | 109 |
| Llama-3.2-3B-Instruct | 331 | 256 | 11 | 552 |

## Notes

- **All five runs finished** with no errors. Every answer ended by itself, except 10 from Llama-3.2-1B
  that loop until the 4,096-token limit (repeating the same call, copying the function description into
  the arguments, or repeating words).
- **Re-scored locally:** scoring the raw answers again on a PC gives the same verdict on every question,
  for all five models.
- **Prompts checked:** every Qwen3 prompt ends with the empty `<think>` block (thinking off), and no
  answer contains `<think>`. Every Llama prompt has the fixed date (26 Jul 2024).
- **Qwen3 gains little from 4B to 8B.** Overall rises from 72.56 (1.7B) to 88.04 (4B), then only to
  88.29 (8B). Qwen3-4B is 1.25 points below Qwen3-4B-Instruct-2507 (89.29, Update 3).
- **Qwen3-1.7B loses many points on format.** 121 of its AST answers do not parse, 82 of them in
  parallel and parallel-multiple questions. 76 of the 121 have exactly one `}` too many, for example
  `[{"name": "math.factorial", "arguments": {"number": 5}}}]` (`simple_python_1`).
- **Llama-3.2-3B scores below Llama-3.2-1B (31.38 against 36.40) because of one format habit.** 552 of
  its 1,150 AST answers do not parse, and 503 of them are an array with the closing `]` missing: the
  answer ends with `"arguments": {"x": 4, "y": 5}}` and stops (`simple_python_2`). Our parser counts
  these as failures, as decided before the runs ([decisions](../notes/decisions.md), 2026-10-12). As
  a check only: adding the missing `]`, as BFCL's own prompt-mode parser does, would raise its Overall
  to 46.02. The reported score stays 31.38.
- **Llama-3.2-3B's irrelevance score (91.25) is too high for the same reason.** An answer that does not
  parse counts as "no call", which is the right answer for an irrelevance question (BFCL's rule). 76 of
  its 240 irrelevance answers do not parse. With the `]` added, 74 of them become calls, and
  irrelevance falls to 60.42.
- **Llama-3.2-1B often makes no call.** 288 of its 1,150 AST answers are an empty array `[]` (142 of
  them in simple Python), although every AST question needs a call. Wrong calls (325) are its other
  main failure.
- **No BFCL number to compare with.** BFCL's own entries for Qwen3-1.7B and Qwen3-8B run with thinking
  on, BFCL has no entry for plain Qwen3-4B, and it runs Llama-3.2 only in FC mode. None of these
  matches this setup.
- **What this means for the quantized runs.** For Qwen3-4B and Qwen3-8B almost every failure is a wrong
  call, so a drop after quantization will mostly be about content. For Qwen3-1.7B and especially
  Llama-3.2-3B, format failures are already large at FP16, so their quantized scores must be read
  together with the parse rate. The failure classifier (Update 7) and constrained decoding (Update 9)
  will separate the two.
- **Speed and memory:** Qwen3-8B generated faster than Qwen3-4B (401.5 s against 468.3 s) only because
  it is split across two T4s; its weights take 7.64 GiB on each. Llama-3.2-1B wrote the most tokens
  (129,061), 40,960 of them in its 10 cut-off answers.
