# Main Table: Accuracy by Precision (free decoding)

BFCL `non_live` (1,390 questions per run), our prompt format and scorer (`src/run_eval.py`, `src/score.py`), greedy decoding. Built from `results/summary.csv` by `src/main_table.py`; a missing run is shown as –.

## Non-Live Overall accuracy (%)

BFCL's Non-Live Overall Acc: the unweighted mean of simple (itself the mean of Python, Java and JavaScript), multiple, parallel and parallel-multiple.

| Model | FP16 | INT8 | NF4 | GPTQ | AWQ |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 72.56 | 69.42 | 69.06 | 39.25 | 75.94 |
| Qwen3-4B | 88.04 | 88.29 | 85.69 | 86.60 | 86.98 |
| Qwen3-8B | 88.29 | 87.27 | 88.00 | 85.98 | 88.42 |
| Llama-3.2-1B-Instruct | 36.40 | 34.94 | 16.02 | 11.40 | 17.81 |
| Llama-3.2-3B-Instruct | 31.38 | 34.02 | 52.75 | 59.52 | 39.73 |

## Change from FP16 (percentage points)

| Model | INT8 | NF4 | GPTQ | AWQ |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | -3.14 | -3.50 | -33.31 | +3.38 |
| Qwen3-4B | +0.25 | -2.35 | -1.44 | -1.06 |
| Qwen3-8B | -1.02 | -0.29 | -2.31 | +0.13 |
| Llama-3.2-1B-Instruct | -1.46 | -20.38 | -25.00 | -18.59 |
| Llama-3.2-3B-Instruct | +2.64 | +21.37 | +28.14 | +8.35 |

## Irrelevance accuracy (%)

Questions where no function fits; correct = no call. Reported apart from the overall score, as in BFCL.

| Model | FP16 | INT8 | NF4 | GPTQ | AWQ |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 77.92 | 75.00 | 79.58 | 62.08 | 69.58 |
| Qwen3-4B | 84.58 | 85.83 | 84.17 | 86.25 | 82.92 |
| Qwen3-8B | 87.08 | 87.08 | 84.58 | 89.58 | 87.50 |
| Llama-3.2-1B-Instruct | 82.92 | 78.75 | 73.33 | 85.83 | 72.92 |
| Llama-3.2-3B-Instruct | 91.25 | 87.92 | 42.08 | 52.92 | 82.08 |
