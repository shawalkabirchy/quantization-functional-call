# Main Table: Accuracy by Precision (constrained decoding)

BFCL `non_live` (1,390 questions per run), our prompt format and scorer (`src/run_eval.py`, `src/score.py`), greedy decoding. Built from `results/summary.csv` by `src/main_table.py`; a missing run is shown as –.

## Non-Live Overall accuracy (%)

BFCL's Non-Live Overall Acc: the unweighted mean of simple (itself the mean of Python, Java and JavaScript), multiple, parallel and parallel-multiple.

| Model | FP16 | INT8 | NF4 | GPTQ | AWQ |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 82.46 | 82.79 | 78.50 | 71.65 | 80.81 |
| Qwen3-4B | 88.25 | 89.04 | 86.12 | 86.85 | 87.48 |
| Qwen3-8B | 88.58 | 88.21 | 88.27 | 86.88 | 88.83 |
| Llama-3.2-1B-Instruct | 48.19 | 45.50 | 17.83 | 2.90 | 47.75 |
| Llama-3.2-3B-Instruct | 71.04 | 69.42 | 65.96 | 64.71 | 68.98 |

## Change from FP16 (percentage points)

| Model | INT8 | NF4 | GPTQ | AWQ |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | +0.33 | -3.96 | -10.81 | -1.65 |
| Qwen3-4B | +0.79 | -2.13 | -1.40 | -0.77 |
| Qwen3-8B | -0.37 | -0.31 | -1.70 | +0.25 |
| Llama-3.2-1B-Instruct | -2.69 | -30.36 | -45.29 | -0.44 |
| Llama-3.2-3B-Instruct | -1.62 | -5.08 | -6.33 | -2.06 |

## Irrelevance accuracy (%)

Questions where no function fits; correct = no call. Reported apart from the overall score, as in BFCL.

| Model | FP16 | INT8 | NF4 | GPTQ | AWQ |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 77.50 | 72.50 | 80.83 | 56.25 | 69.58 |
| Qwen3-4B | 84.58 | 86.67 | 83.75 | 86.25 | 83.33 |
| Qwen3-8B | 87.08 | 87.50 | 84.58 | 89.58 | 87.50 |
| Llama-3.2-1B-Instruct | 84.17 | 80.83 | 77.08 | 99.58 | 50.83 |
| Llama-3.2-3B-Instruct | 74.58 | 70.42 | 38.75 | 65.00 | 69.58 |
