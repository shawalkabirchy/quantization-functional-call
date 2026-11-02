# INT8 and NF4 (Update 5)

**Date:** 2026-10-29
**Goal:** function-calling accuracy of every core model in INT8 and NF4 (bitsandbytes), compared with the FP16 baselines of Update 4.

## Setup

- **Questions:** BFCL `non_live`, 1390 questions per run, bfcl-eval 2026.3.23
- **Prompt format:** `json-array-v1` (as in Updates 3-4), free decoding; Qwen3 with thinking off
- **Decoding:** greedy (temperature 0), seed 42, at most 4096 answer tokens
- **INT8:** LLM.int8() from bitsandbytes 0.45.5: 8-bit weights with one scale per row; input features above 6.0 stay in FP16. The checkpoint is made with transformers 4.51.3 (`src/quantize.py`); vLLM runs it in eager mode.
- **NF4:** vLLM quantizes the FP16 weights while loading: 4-bit NormalFloat, blocks of 64, double quantization (bitsandbytes 0.45.5)
- **Inference:** vLLM 0.8.5 on one Tesla T4 per run (FP16 Qwen3-8B used two)
- **Software:** Python 3.11.17, torch 2.6.0, transformers 4.51.3
- **Code:** commit `1c349c7`, and `6b6903c` for Qwen3-8B INT8 and NF4 (the only change that touches these runs is Qwen3-8B INT8's memory setting); model revisions are in `results/meta/`
- **Overall** = BFCL's Non-Live Overall Acc (unweighted mean of simple, multiple, parallel and parallel-multiple; simple = mean of Python, Java, JavaScript). Irrelevance is reported on its own.

## Overall accuracy (%)

| Model | FP16 | INT8 | NF4 | INT8 - FP16 | NF4 - FP16 |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 72.56 | 69.42 | 69.06 | -3.14 | -3.50 |
| Qwen3-4B | 88.04 | 88.29 | 85.69 | 0.25 | -2.35 |
| Qwen3-8B | 88.29 | 87.27 | 88.00 | -1.02 | -0.29 |
| Llama-3.2-1B-Instruct | 36.40 | 34.94 | 16.02 | -1.46 | -20.38 |
| Llama-3.2-3B-Instruct | 31.38 | 34.02 | 52.75 | 2.64 | 21.37 |

## Accuracy per category (%)

| Model | Precision | Python | Java | JavaScript | Multiple | Parallel | Parallel Multiple | Overall | Irrelevance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | FP16 | 86.25 | 65.00 | 64.00 | 87.50 | 71.50 | 59.50 | 72.56 | 77.92 |
| Qwen3-1.7B | INT8 | 86.50 | 65.00 | 68.00 | 87.50 | 62.50 | 54.50 | 69.42 | 75.00 |
| Qwen3-1.7B | NF4 | 86.25 | 65.00 | 64.00 | 85.50 | 66.00 | 53.00 | 69.06 | 79.58 |
| Qwen3-4B | FP16 | 95.50 | 63.00 | 70.00 | 95.00 | 90.50 | 90.50 | 88.04 | 84.58 |
| Qwen3-4B | INT8 | 95.50 | 64.00 | 72.00 | 95.00 | 91.00 | 90.00 | 88.29 | 85.83 |
| Qwen3-4B | NF4 | 93.25 | 61.00 | 76.00 | 92.50 | 89.00 | 84.50 | 85.69 | 84.17 |
| Qwen3-8B | FP16 | 95.50 | 65.00 | 68.00 | 93.00 | 92.50 | 91.50 | 88.29 | 87.08 |
| Qwen3-8B | INT8 | 95.25 | 65.00 | 68.00 | 93.00 | 91.50 | 88.50 | 87.27 | 87.08 |
| Qwen3-8B | NF4 | 95.50 | 64.00 | 70.00 | 95.00 | 92.50 | 88.00 | 88.00 | 84.58 |
| Llama-3.2-1B-Instruct | FP16 | 39.75 | 37.00 | 48.00 | 38.50 | 41.00 | 24.50 | 36.40 | 82.92 |
| Llama-3.2-1B-Instruct | INT8 | 42.75 | 38.00 | 34.00 | 35.50 | 42.00 | 24.00 | 34.94 | 78.75 |
| Llama-3.2-1B-Instruct | NF4 | 36.75 | 11.00 | 8.00 | 17.50 | 20.50 | 7.50 | 16.02 | 73.33 |
| Llama-3.2-3B-Instruct | FP16 | 22.00 | 32.00 | 24.00 | 30.00 | 42.50 | 27.00 | 31.38 | 91.25 |
| Llama-3.2-3B-Instruct | INT8 | 22.75 | 21.00 | 18.00 | 30.00 | 52.50 | 33.00 | 34.02 | 87.92 |
| Llama-3.2-3B-Instruct | NF4 | 62.00 | 56.00 | 56.00 | 52.50 | 57.50 | 43.00 | 52.75 | 42.08 |

## Format, length and speed

| Model | Precision | Parsed, AST questions (%) | Unparsed answers | Cut off at max_tokens | GPUs | Weights per GPU (GiB) | Generation (s) | Output tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | FP16 | 89.48 | 121 | 0 | 1 | 3.22 | 167.80 | 422.20 |
| Qwen3-1.7B | INT8 | 86.87 | 151 | 0 | 1 | 1.90 | 485.10 | 146.50 |
| Qwen3-1.7B | NF4 | 90.43 | 110 | 1 | 1 | 1.27 | 1055.70 | 72.50 |
| Qwen3-4B | FP16 | 99.83 | 2 | 0 | 1 | 7.56 | 468.30 | 153.30 |
| Qwen3-4B | INT8 | 99.74 | 3 | 0 | 1 | 4.18 | 682.40 | 105.00 |
| Qwen3-4B | NF4 | 99.57 | 5 | 0 | 1 | 2.49 | 2541.80 | 28.60 |
| Qwen3-8B | FP16 | 99.91 | 1 | 0 | 2 | 7.64 | 401.50 | 178.90 |
| Qwen3-8B | INT8 | 99.83 | 2 | 0 | 1 | 8.80 | 1807.30 | 39.90 |
| Qwen3-8B | NF4 | 99.83 | 2 | 0 | 1 | 5.67 | 4895.00 | 14.70 |
| Llama-3.2-1B-Instruct | FP16 | 90.52 | 109 | 10 | 1 | 2.32 | 210.60 | 612.70 |
| Llama-3.2-1B-Instruct | INT8 | 86.09 | 160 | 12 | 1 | 1.41 | 481.30 | 350.20 |
| Llama-3.2-1B-Instruct | NF4 | 62.61 | 430 | 44 | 1 | 0.98 | 1477.80 | 252.30 |
| Llama-3.2-3B-Instruct | FP16 | 52.00 | 552 | 0 | 1 | 6.02 | 322.60 | 230.40 |
| Llama-3.2-3B-Instruct | INT8 | 55.65 | 510 | 0 | 1 | 3.42 | 384.80 | 196.20 |
| Llama-3.2-3B-Instruct | NF4 | 91.04 | 103 | 0 | 1 | 2.13 | 2035.30 | 42.60 |

## How the AST answers end

Every answer to the 1,150 AST questions (every category except irrelevance) falls into one of four
groups. These are plain counts over all 1,150 questions, read from `results/scores/` (fields
`parse_ok`, `n_calls` and `valid`).

| Model | Precision | Correct | Wrong call | No call (`[]`) | Unparsed |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | FP16 | 879 | 149 | 1 | 121 |
| Qwen3-1.7B | INT8 | 854 | 145 | 0 | 151 |
| Qwen3-1.7B | NF4 | 851 | 185 | 4 | 110 |
| Qwen3-4B | FP16 | 1,032 | 115 | 1 | 2 |
| Qwen3-4B | INT8 | 1,034 | 112 | 1 | 3 |
| Qwen3-4B | NF4 | 1,004 | 140 | 1 | 5 |
| Qwen3-8B | FP16 | 1,035 | 109 | 5 | 1 |
| Qwen3-8B | INT8 | 1,026 | 117 | 5 | 2 |
| Qwen3-8B | NF4 | 1,032 | 114 | 2 | 2 |
| Llama-3.2-1B-Instruct | FP16 | 428 | 325 | 288 | 109 |
| Llama-3.2-1B-Instruct | INT8 | 429 | 311 | 250 | 160 |
| Llama-3.2-1B-Instruct | NF4 | 253 | 181 | 286 | 430 |
| Llama-3.2-3B-Instruct | FP16 | 331 | 256 | 11 | 552 |
| Llama-3.2-3B-Instruct | INT8 | 352 | 283 | 5 | 510 |
| Llama-3.2-3B-Instruct | NF4 | 638 | 404 | 5 | 103 |

## Notes

- **All 10 runs finished,** with 1,390 answers each. Every prompt is the same as in that model's FP16
  run (checked on all 1,390 per run).
- **Re-scored locally:** scoring the raw answers again on a PC gives the same verdict on every question,
  for all 10 runs.
- **These are single runs.** No difference here has been tested yet, so small ones may mean nothing.
  Significance tests are planned for Update 8.
- **Qwen3-8B needed reruns.** INT8 first ran out of GPU memory at `gpu_memory_utilization: 0.90`, so its
  config uses 0.81 ([decisions](../notes/decisions.md)). Earlier NF4 tries did not finish: they were
  stopped by hand or cut off from outside the run, and none ended with an error. The NF4 result here
  comes from one full rerun.
- **INT8 stays close to FP16:** between -3.14 and +2.64 points on every model.
- **Qwen3-4B and Qwen3-8B barely change.** The largest change is -2.35 (Qwen3-4B NF4), and at least
  99.57% of their AST answers still parse. That drop is in content: 25 more wrong calls (115 to 140),
  and only 3 more unparsed answers.
- **Qwen3-1.7B loses about 3 points in both, in different ways.** INT8 adds format failures (151
  unparsed against 121), with about the same number of wrong calls (145 against 149). NF4 adds wrong
  calls (185 against 149), and unparsed answers fall to 110.
- **Llama-3.2-1B collapses at NF4 (-20.38), mostly on format.** Unparsed answers rise from 109 to 430.
  302 of the 430 contain `"description"` or `"properties"`: the model copies parts of the function
  definition instead of filling in a call. This is only a text search; the failure classifier (Update 7)
  will count it properly. 198 of its 1,390 answers are in a code fence (36 at FP16), and 44 reach the
  4,096-token limit (10 at FP16).
- **Llama-3.2-3B gains 21.37 points at NF4, but much of that is format too.** The missing closing `]`
  nearly disappears: 503 AST answers at FP16, 463 at INT8, 76 at NF4. As a check only, the bracket repair
  from Update 4 gives 46.02 (FP16), 48.58 (INT8) and 55.62 (NF4), so the NF4 gain shrinks from +21.37
  to +9.60. The reported scores stay strict.
- **Llama-3.2-3B's irrelevance falls from 91.25 to 42.08 at NF4.** Part of this is the same effect,
  because an unparsed answer counts as "no call". With the same repair the scores are 60.42 and 26.25, so
  NF4 does make more calls when no function fits: 177 of 240 answers against 95 at FP16.
- **We do not know yet why NF4 changes these two format habits while INT8 does not.** These notes only
  say what changed. The failure classifier (Update 7) and constrained decoding (Update 9) are planned to
  separate format from content.
- **Speed and memory.** Weights shrink as expected (Qwen3-4B: 7.56 GiB in FP16, 4.18 in INT8, 2.49 in
  NF4), but bitsandbytes is much slower than FP16 on a T4 (Qwen3-4B: 153.3 output tokens/s, then 105.0
  and 28.6). Two Qwen3-8B speeds should not be compared with the others: FP16 uses both T4s, and INT8's
  lower memory setting leaves room for only 1.15 full-length requests at a time. Memory numbers are
  vLLM's measurements at load time, not a live peak.
- **Notebooks.** The two runs are saved as they ran:
  [main session](../notebooks/04_int8_nf4_run_main.ipynb) and
  [Qwen3-8B NF4 rerun](../notebooks/04_int8_nf4_run_qwen3-8b_nf4.ipynb). Section 6 of the main session
  was run again for Qwen3-8B, so its output shows only those two runs; the settings and times of the
  other 8 are in `results/meta/`. To run everything again, use the clean
  [notebook 04](../notebooks/04_int8_nf4.ipynb).
