# GPTQ and AWQ (Update 6)

**Date:** 2026-11-03 (runs on 29 October, 1 November and 2 November)
**Goal:** function-calling accuracy of every core model with 4-bit GPTQ and AWQ checkpoints made with one recipe, compared with the FP16 baselines of Update 4. All five precisions are in the [main table](main_table.md).

## Setup

- **Questions:** BFCL `non_live`, 1390 questions per run, bfcl-eval 2026.3.23
- **Prompt format:** `json-array-v1` (as in Updates 3-5), free decoding; Qwen3 with thinking off
- **Decoding:** greedy (temperature 0), seed 42, at most 4096 answer tokens
- **Inference:** vLLM 0.8.5 on one Tesla T4 per run (FP16 Qwen3-8B used two), with its GPTQ and AWQ GEMM kernels (the faster Marlin kernels need a newer GPU)
- **Quantization:** GPTQModel 5.7.0 (torch 2.9.1, transformers 4.57.6), `src/quantize.py`; the same pinned model revisions as FP16
- **Calibration (same for every model and both methods):** the first 256 documents with at least 3,000 characters from `allenai/c4` `en/c4-train.00001-of-01024.json.gz` (revision `1588ec4`), each cut to 512 tokens
- **GPTQ:** 4-bit weights, groups of 128, symmetric, desc_act=False; saved in the GPTQ format
- **AWQ:** 4-bit weights, groups of 128, with zero points, calibration in batches of 16; saved in the AWQ GEMM format. Qwen3-8B needed three more settings (see Notes)
- **Software:** Python 3.11.17, torch 2.6.0, transformers 4.51.3
- **Code:** commit `6b6903c` (GPTQ), `4f41d13` (AWQ) and `d7abb52` (Qwen3-8B AWQ). The commits in between change only the AWQ part of `src/quantize.py` and the AWQ configs; model revisions are in `results/meta/`
- **Overall** = BFCL's Non-Live Overall Acc (unweighted mean of simple, multiple, parallel and parallel-multiple; simple = mean of Python, Java, JavaScript). Irrelevance is reported on its own.

## Overall accuracy (%)

| Model | FP16 | GPTQ | AWQ | GPTQ - FP16 | AWQ - FP16 |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 72.56 | 39.25 | 75.94 | -33.31 | 3.38 |
| Qwen3-4B | 88.04 | 86.60 | 86.98 | -1.44 | -1.06 |
| Qwen3-8B | 88.29 | 85.98 | 88.42 | -2.31 | 0.13 |
| Llama-3.2-1B-Instruct | 36.40 | 11.40 | 17.81 | -25.00 | -18.59 |
| Llama-3.2-3B-Instruct | 31.38 | 59.52 | 39.73 | 28.14 | 8.35 |

## Accuracy per category (%)

| Model | Precision | Python | Java | JavaScript | Multiple | Parallel | Parallel Multiple | Overall | Irrelevance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | FP16 | 86.25 | 65.00 | 64.00 | 87.50 | 71.50 | 59.50 | 72.56 | 77.92 |
| Qwen3-1.7B | GPTQ | 77.50 | 60.00 | 68.00 | 63.50 | 6.00 | 19.00 | 39.25 | 62.08 |
| Qwen3-1.7B | AWQ | 91.25 | 64.00 | 66.00 | 86.50 | 77.50 | 66.00 | 75.94 | 69.58 |
| Qwen3-4B | FP16 | 95.50 | 63.00 | 70.00 | 95.00 | 90.50 | 90.50 | 88.04 | 84.58 |
| Qwen3-4B | GPTQ | 94.25 | 63.00 | 72.00 | 93.50 | 88.00 | 88.50 | 86.60 | 86.25 |
| Qwen3-4B | AWQ | 95.75 | 62.00 | 70.00 | 92.00 | 91.50 | 88.50 | 86.98 | 82.92 |
| Qwen3-8B | FP16 | 95.50 | 65.00 | 68.00 | 93.00 | 92.50 | 91.50 | 88.29 | 87.08 |
| Qwen3-8B | GPTQ | 95.25 | 62.00 | 66.00 | 91.50 | 90.50 | 87.50 | 85.98 | 89.58 |
| Qwen3-8B | AWQ | 95.00 | 66.00 | 72.00 | 93.00 | 90.50 | 92.50 | 88.42 | 87.50 |
| Llama-3.2-1B-Instruct | FP16 | 39.75 | 37.00 | 48.00 | 38.50 | 41.00 | 24.50 | 36.40 | 82.92 |
| Llama-3.2-1B-Instruct | GPTQ | 17.25 | 5.00 | 26.00 | 5.00 | 19.00 | 5.50 | 11.40 | 85.83 |
| Llama-3.2-1B-Instruct | AWQ | 41.25 | 8.00 | 16.00 | 6.00 | 39.50 | 4.00 | 17.81 | 72.92 |
| Llama-3.2-3B-Instruct | FP16 | 22.00 | 32.00 | 24.00 | 30.00 | 42.50 | 27.00 | 31.38 | 91.25 |
| Llama-3.2-3B-Instruct | GPTQ | 73.75 | 52.00 | 74.00 | 71.00 | 62.00 | 38.50 | 59.52 | 52.92 |
| Llama-3.2-3B-Instruct | AWQ | 41.75 | 35.00 | 22.00 | 36.00 | 56.00 | 34.00 | 39.73 | 82.08 |

## Format, length and speed

| Model | Precision | Parsed, AST questions (%) | Unparsed answers | Cut off at max_tokens | GPUs | Weights per GPU (GiB) | Generation (s) | Output tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | FP16 | 89.48 | 121 | 0 | 1 | 3.22 | 167.80 | 422.20 |
| Qwen3-1.7B | GPTQ | 65.65 | 395 | 0 | 1 | 1.27 | 193.90 | 409.30 |
| Qwen3-1.7B | AWQ | 95.22 | 55 | 0 | 1 | 1.27 | 218.00 | 329.10 |
| Qwen3-4B | FP16 | 99.83 | 2 | 0 | 1 | 7.56 | 468.30 | 153.30 |
| Qwen3-4B | GPTQ | 99.57 | 5 | 0 | 1 | 2.50 | 474.50 | 153.70 |
| Qwen3-4B | AWQ | 99.39 | 7 | 0 | 1 | 2.50 | 549.90 | 132.70 |
| Qwen3-8B | FP16 | 99.91 | 1 | 0 | 2 | 7.64 | 401.50 | 178.90 |
| Qwen3-8B | GPTQ | 99.74 | 3 | 0 | 1 | 5.71 | 844.80 | 84.50 |
| Qwen3-8B | AWQ | 99.74 | 3 | 1 | 1 | 5.71 | 939.30 | 80.50 |
| Llama-3.2-1B-Instruct | FP16 | 90.52 | 109 | 10 | 1 | 2.32 | 210.60 | 612.70 |
| Llama-3.2-1B-Instruct | GPTQ | 44.43 | 639 | 33 | 1 | 0.98 | 462.60 | 622.90 |
| Llama-3.2-1B-Instruct | AWQ | 41.65 | 671 | 102 | 1 | 0.98 | 1122.10 | 686.10 |
| Llama-3.2-3B-Instruct | FP16 | 52.00 | 552 | 0 | 1 | 6.02 | 322.60 | 230.40 |
| Llama-3.2-3B-Instruct | GPTQ | 90.61 | 108 | 1 | 1 | 2.14 | 403.60 | 237.20 |
| Llama-3.2-3B-Instruct | AWQ | 66.52 | 385 | 0 | 1 | 2.14 | 404.10 | 179.30 |

## How the AST answers end

Every answer to the 1,150 AST questions (every category except irrelevance) falls into one of four groups. These are plain counts over all 1,150 questions, read from `results/scores/` (fields `parse_ok`, `n_calls` and `valid`).

| Model | Precision | Correct | Wrong call | No call (`[]`) | Unparsed |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | FP16 | 879 | 149 | 1 | 121 |
| Qwen3-1.7B | GPTQ | 581 | 173 | 1 | 395 |
| Qwen3-1.7B | AWQ | 922 | 172 | 1 | 55 |
| Qwen3-4B | FP16 | 1,032 | 115 | 1 | 2 |
| Qwen3-4B | GPTQ | 1,016 | 125 | 4 | 5 |
| Qwen3-4B | AWQ | 1,024 | 118 | 1 | 7 |
| Qwen3-8B | FP16 | 1,035 | 109 | 5 | 1 |
| Qwen3-8B | GPTQ | 1,015 | 119 | 13 | 3 |
| Qwen3-8B | AWQ | 1,034 | 112 | 1 | 3 |
| Llama-3.2-1B-Instruct | FP16 | 428 | 325 | 288 | 109 |
| Llama-3.2-1B-Instruct | GPTQ | 146 | 205 | 160 | 639 |
| Llama-3.2-1B-Instruct | AWQ | 280 | 155 | 44 | 671 |
| Llama-3.2-3B-Instruct | FP16 | 331 | 256 | 11 | 552 |
| Llama-3.2-3B-Instruct | GPTQ | 727 | 306 | 9 | 108 |
| Llama-3.2-3B-Instruct | AWQ | 465 | 291 | 9 | 385 |

## Notes

- **All 10 runs finished,** with 1,390 answers each. Every prompt is the same as in that model's FP16
  run (checked on all 1,390 per run).
- **Re-scored locally:** scoring the raw answers again on a PC gives the same verdict on every question,
  for all 10 runs.
- **These are single runs.** No difference here has been tested yet, so small ones may mean nothing.
  Significance tests are planned for Update 8.
- **Qwen3-4B and Qwen3-8B barely change.** The changes run from -2.31 (Qwen3-8B GPTQ) to +0.13
  (Qwen3-8B AWQ), and at least 99.39% of their AST answers still parse.
- **Qwen3-1.7B falls 33.31 points with GPTQ, but gains 3.38 with AWQ.** The GPTQ fall is mostly format:
  unparsed answers rise from 121 to 395, and wrong calls only from 149 to 173. 316 of the 395 are
  parallel or parallel-multiple answers (Parallel falls from 71.50 to 6.00). 221 of the 395 become valid
  JSON when one `}` is removed: the model ends a call with one brace too many (`}}}]`). FP16 has the same
  habit, less often (75 of its 121 unparsed answers), and AWQ less still (23 of 55). We do not know yet
  why GPTQ makes this habit so much worse on this model. The constrained runs (Update 9) will show how
  much of the loss is format.
- **Llama-3.2-1B falls with both methods** (GPTQ -25.00, AWQ -18.59), mostly on format, as at NF4.
  Unparsed answers rise from 109 to 639 (GPTQ) and 671 (AWQ). 426 and 546 of them contain
  `"description"` or `"properties"`: the model copies parts of the function definition instead of
  filling in a call (a text search only; the failure classifier of Update 7 will count it properly).
  With AWQ, 323 of its 1,390 answers start with a code fence (36 at FP16), and 102 reach the
  4,096-token limit (10 at FP16).
- **Llama-3.2-3B gains with both methods** (GPTQ +28.14, AWQ +8.35), as at NF4, and again much of the
  gain is format. AST answers that leave out the closing `]` fall from 503 (FP16) to 83 (GPTQ) and
  346 (AWQ). As a check only, the bracket repair from Update 4 gives 46.02 (FP16), 66.17 (GPTQ) and
  52.42 (AWQ), so the gains shrink to +20.15 and +6.40. The reported scores stay strict.
- **Llama-3.2-3B's irrelevance falls from 91.25 to 52.92 with GPTQ** (82.08 with AWQ). With the same
  repair the scores are 60.42 (FP16), 52.08 (GPTQ) and 62.92 (AWQ). GPTQ makes a call on 113 of the
  240 irrelevance questions, against 21 at FP16.
- **Qwen3-8B AWQ needed three more settings, and only this checkpoint has them.** (1) The job stopped in
  its first layer, most likely out of system memory, so its weights are first copied to FP16 on disk
  (`fp16_copy`). (2) It then stopped at layer 3 because every value was NaN. Two read-only probe runs
  showed why: in layer 2, some channels had a mean activation of 0, AWQ gave them scales up to 2,450,
  and `up_proj`'s outputs then passed FP16's largest value (65,504). The fix raises very small mean
  activations and keeps the scales between 1/100 and 100 (`x_mean_floor: 0.001`, `max_scale: 100`);
  these two values are a judgment, not tuned.
  (3) It then ran out of GPU memory at layer 22 of 36, so AWQ's sums run in chunks of 128 MiB instead of
  1 GiB (`chunk_mib: 128`). The method settings and the calibration documents are the same as for the
  other models. On small test models, these changes give the same checkpoint whenever the scales are
  normal. The reasons are in [`notes/decisions.md`](../notes/decisions.md).
- **Some of AWQ's trial ratios give NaN.** AWQ tries 20 scale ratios per group of layers and keeps the
  best. In the Qwen3-8B probe (first layers only), the first 1-3 trials of every group gave NaN and were skipped, and the
  chosen ratio was always a later one. We think this is also an FP16 overflow, in GPTQModel's scale
  step, but it was not measured. The other four AWQ checkpoints were not probed, so they may have
  skipped such trials too.
- **Speed and memory.** Both methods shrink Qwen3-4B's weights from 7.56 to 2.50 GiB, and on the T4
  vLLM's GPTQ and AWQ kernels are much faster than bitsandbytes: Qwen3-4B gives 153.7 (GPTQ) and 132.7
  (AWQ) output tokens/s, against 153.3 at FP16 and 28.6 at NF4. Qwen3-8B fits on one T4 with both
  (5.71 GiB). Memory numbers are vLLM's measurements at load time, not a live peak.
- **Checkpoints.** The AWQ checkpoints are on Hugging Face (private): [shawalkabir/Qwen3-1.7B-AWQ-W4A16-g128](https://huggingface.co/shawalkabir/Qwen3-1.7B-AWQ-W4A16-g128) (`3cc45f1`), [shawalkabir/Qwen3-4B-AWQ-W4A16-g128](https://huggingface.co/shawalkabir/Qwen3-4B-AWQ-W4A16-g128) (`e8a7317`), [shawalkabir/Qwen3-8B-AWQ-W4A16-g128](https://huggingface.co/shawalkabir/Qwen3-8B-AWQ-W4A16-g128) (`6f1d2a8`), [shawalkabir/Llama-3.2-1B-Instruct-AWQ-W4A16-g128](https://huggingface.co/shawalkabir/Llama-3.2-1B-Instruct-AWQ-W4A16-g128) (`e8cd172`), [shawalkabir/Llama-3.2-3B-Instruct-AWQ-W4A16-g128](https://huggingface.co/shawalkabir/Llama-3.2-3B-Instruct-AWQ-W4A16-g128) (`6f9fd9e`). The GPTQ checkpoints
  were deleted after each run to save disk space and were not uploaded.
- **Notebooks.** The three sessions are saved as they ran:
  [GPTQ](../notebooks/05_gptq_awq_run_gptq.ipynb) (its quick check also tried AWQ, which ran out of GPU
  memory, so its AWQ runs were skipped), [AWQ](../notebooks/05_gptq_awq_run_awq.ipynb) (four models; its
  Qwen3-8B run stopped in the first layer) and [Qwen3-8B AWQ](../notebooks/05_gptq_awq_run_qwen3-8b_awq.ipynb).
  The failed tries and the probe runs are in [`notes/compute_log.md`](../notes/compute_log.md). To run
  everything again, use the clean [notebook 05](../notebooks/05_gptq_awq.ipynb).
