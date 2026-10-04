# BFCL Reproduction (Update 2)

**Date:** 2026-10-03
**Goal:** check that the BFCL harness runs on Kaggle's T4 GPU in FP16 and reproduces the published leaderboard.

## Setup

- **Models:** `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3-4B-Instruct-2507-FC`
- **Benchmark:** BFCL `non_live` categories, bfcl-eval 2026.3.23
- **Inference:** vLLM 0.8.5 on one Tesla T4 (tensor parallel size 1; the second T4 was idle); FP16 (T4 has no BF16)
- **Sampling:** BFCL requests temperature 0.001, but vLLM 0.8.5 raises anything below 0.01 to 0.01; top-k 20 and top-p 0.8 come from the model's generation config (see Notes)
- **Software:** Python 3.11.17, torch 2.6.0, transformers 4.51.3
- **Reference:** https://gorilla.cs.berkeley.edu/data_non_live.csv (downloaded 2026-10-03)

## Results (accuracy, %)

| Model | Category | Ours (FP16, T4) | Leaderboard | Difference |
|---|---|---:|---:|---:|
| Qwen3-4B-Instruct-2507 (Prompt) | Non-Live Overall Acc | 86.31 | 86.44 | -0.13 |
| Qwen3-4B-Instruct-2507 (Prompt) | Python Simple AST | 93.75 | 93.75 | 0.00 |
| Qwen3-4B-Instruct-2507 (Prompt) | Java Simple AST | 64.00 | 64.00 | 0.00 |
| Qwen3-4B-Instruct-2507 (Prompt) | JavaScript Simple AST | 74.00 | 74.00 | 0.00 |
| Qwen3-4B-Instruct-2507 (Prompt) | Multiple AST | 91.00 | 91.00 | 0.00 |
| Qwen3-4B-Instruct-2507 (Prompt) | Parallel AST | 88.00 | 88.00 | 0.00 |
| Qwen3-4B-Instruct-2507 (Prompt) | Parallel Multiple AST | 89.00 | 89.50 | -0.50 |
| Qwen3-4B-Instruct-2507 (Prompt) | Irrelevance Detection | 84.58 | 85.00 | -0.42 |
| Qwen3-4B-Instruct-2507 (FC) | Non-Live Overall Acc | 87.81 | 87.88 | -0.07 |
| Qwen3-4B-Instruct-2507 (FC) | Python Simple AST | 95.25 | 95.50 | -0.25 |
| Qwen3-4B-Instruct-2507 (FC) | Java Simple AST | 63.00 | 63.00 | 0.00 |
| Qwen3-4B-Instruct-2507 (FC) | JavaScript Simple AST | 66.00 | 68.00 | -2.00 |
| Qwen3-4B-Instruct-2507 (FC) | Multiple AST | 94.00 | 93.50 | 0.50 |
| Qwen3-4B-Instruct-2507 (FC) | Parallel AST | 92.50 | 92.50 | 0.00 |
| Qwen3-4B-Instruct-2507 (FC) | Parallel Multiple AST | 90.00 | 90.00 | 0.00 |
| Qwen3-4B-Instruct-2507 (FC) | Irrelevance Detection | 88.75 | 88.75 | 0.00 |

**Verdict:** Reproduced: every category is within 3 points of the leaderboard (largest difference: 2.00 points).

## Notes

- **Every gap is at most one question.** JavaScript Simple (FC) at -2.00 is 1 of 50 questions;
  Multiple (FC, +0.50) and Parallel Multiple (Prompt, -0.50) are 1 of 200; Irrelevance (Prompt, -0.42)
  is 1 of 240; Python Simple (FC, -0.25) is 1 of 400. The gaps go in both directions. This fits small
  numerical differences (FP16 instead of BF16, a different GPU, near-greedy sampling), but this run
  cannot say which of these caused them.
- **Complete run.** All 2,780 requests (1,390 per mode) were answered, with no failed requests.
- **Sampling is near-greedy, not greedy.** vLLM logged the temperature clamp on every request.
  Because BFCL sends only the temperature, vLLM fills in the model's recommended top-k 20 and top-p 0.8.
  BFCL's own vLLM launch does not override these defaults either, so this matches BFCL's standard
  local setup.
  Our own evaluation script (Update 3) should set every decoding parameter explicitly, so that precision
  is the only thing that changes between runs.
- **Runtime.** The server was ready about 2 minutes after launch. Generating both modes took about
  14 minutes on one T4 (08:25-08:39 UTC), well under the notebook's 40-minute estimate.
- **Scores kept.** Only `non_live` was run, so `results/bfcl_repro/score/` keeps `data_non_live.csv` and the
  per-category files. BFCL's other summary CSVs (overall, live, multi-turn, agentic, format sensitivity)
  were left out because they count the categories we did not run as 0%.
- **Server log.** vLLM cast the BF16 checkpoint to FP16, as intended. One request was preempted for KV-cache
  space and recomputed; recomputation does not change the output.
