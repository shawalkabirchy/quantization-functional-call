# Final Table: Every Score of the Project

All scores of the study in one place, after the experiments were frozen (Update 15). Built by [`src/final_table.py`](../src/final_table.py) from the files in `results/`; nothing here is typed by hand. The same master table, with p-values and the constrained-decoding speed, is in [`final_table.csv`](final_table.csv).

**Setup in short:** 5 models (Qwen3-1.7B, 4B, 8B; Llama-3.2-1B, 3B-Instruct) × 5 weight precisions (FP16, INT8, NF4, GPTQ, AWQ) × 2 decoding modes (free, JSON-schema-constrained) on the 1,390 BFCL `non_live` questions and on our 175-question schema-stress suite, plus the 5 FP16 models with an 8-bit KV cache (free decoding only): 110 runs. Greedy decoding, vLLM 0.8.5, Kaggle Tesla T4.

**How to read:**

- **BFCL score** = BFCL's Non-Live Overall accuracy (%): the unweighted mean of simple (itself the mean of Python, Java and JavaScript), multiple, parallel and parallel multiple. Irrelevance (240 questions where no function fits) is reported apart, as in BFCL.
- **Change** = points against FP16 in the same decoding mode. **Bold** = significant: McNemar's exact test, paired by question on the 1,150 AST questions, Holm-adjusted p < 0.05.
- **Constraint gain** = constrained score minus free score of the same run.
- **Stress** = correct answers of the 175 stress-suite questions (7 groups of 25).
- **Speed** = batch size 1, median over 35 questions. Tokens/s = speed of writing the answer. Differences of about 20% or less are within the noise of Kaggle's T4s.
- – = not run (the 8-bit KV cache was run in free decoding only).

## 1. Master table

| Model | Precision | Weights (GiB per GPU) | BFCL free | Change | BFCL constrained | Change | Constraint gain | Irrelevance free | Irrelevance constrained | Stress free | Stress constrained | Tokens/s | First token (ms) | Whole answer (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | FP16 | 3.22 | 72.56 | base | 82.46 | base | +9.90 | 77.92 | 77.50 | 154 | 171 | 52.9 | 77 | 0.80 |
|  | FP16, 8-bit KV | 3.22 | 71.83 | **-0.73** | – | – | – | 78.75 | – | 147 | – | 40.6 | 83 | 1.00 |
|  | INT8 | 1.90 | 69.42 | **-3.14** | 82.79 | +0.33 | +13.37 | 75.00 | 72.50 | 155 | 172 | 7.3 | 354 | 5.46 |
|  | NF4 | 1.27 | 69.06 | -3.50 | 78.50 | **-3.96** | +9.44 | 79.58 | 80.83 | 160 | 169 | 47.4 | 567 | 1.50 |
|  | GPTQ | 1.27 | 39.25 | **-33.31** | 71.65 | **-10.81** | +32.40 | 62.08 | 56.25 | 158 | 169 | 101.4 | 86 | 0.48 |
|  | AWQ | 1.27 | 75.94 | **+3.38** | 80.81 | -1.65 | +4.87 | 69.58 | 69.58 | 168 | 171 | 80.3 | 106 | 0.58 |
| Qwen3-4B | FP16 | 7.56 | 88.04 | base | 88.25 | base | +0.21 | 84.58 | 84.58 | 175 | 175 | 25.2 | 181 | 1.80 |
|  | FP16, 8-bit KV | 7.56 | 87.29 | -0.75 | – | – | – | 84.58 | – | 174 | – | 21.2 | 194 | 2.15 |
|  | INT8 | 4.18 | 88.29 | +0.25 | 89.04 | +0.79 | +0.75 | 85.83 | 86.67 | 173 | 174 | 6.1 | 394 | 7.03 |
|  | NF4 | 2.49 | 85.69 | **-2.35** | 86.12 | **-2.13** | +0.43 | 84.17 | 83.75 | 167 | 174 | 21.6 | 1512 | 3.46 |
|  | GPTQ | 2.50 | 86.60 | -1.44 | 86.85 | -1.40 | +0.25 | 86.25 | 86.25 | 168 | 175 | 58.9 | 220 | 0.90 |
|  | AWQ | 2.50 | 86.98 | -1.06 | 87.48 | -0.77 | +0.50 | 82.92 | 83.33 | 167 | 173 | 41.1 | 305 | 1.30 |
| Qwen3-8B | FP16 | 7.64 ×2 | 88.29 | base | 88.58 | base | +0.29 | 87.08 | 87.08 | 175 | 175 | 25.7 | 196 | 1.70 |
|  | FP16, 8-bit KV | 7.64 ×2 | 88.58 | +0.29 | – | – | – | 87.08 | – | 175 | – | 22.0 | 198 | 1.96 |
|  | INT8 | 8.80 | 87.27 | -1.02 | 88.21 | -0.37 | +0.94 | 87.08 | 87.50 | 175 | 175 | 6.1 | 532 | 6.72 |
|  | NF4 | 5.67 | 88.00 | -0.29 | 88.27 | -0.31 | +0.27 | 84.58 | 84.58 | 174 | 174 | 13.3 | 2907 | 5.86 |
|  | GPTQ | 5.71 | 85.98 | **-2.31** | 86.88 | -1.70 | +0.90 | 89.58 | 89.58 | 175 | 175 | 38.5 | 407 | 1.44 |
|  | AWQ | 5.71 | 88.42 | +0.13 | 88.83 | +0.25 | +0.41 | 87.50 | 87.50 | 175 | 175 | 24.2 | 554 | 2.20 |
| Llama-3.2-1B-Instruct | FP16 | 2.32 | 36.40 | base | 48.19 | base | +11.79 | 82.92 | 84.17 | 42 | 63 | 79.9 | 47 | 0.57 |
|  | FP16, 8-bit KV | 2.32 | 34.98 | -1.42 | – | – | – | 82.50 | – | 39 | – | 66.9 | 57 | 0.65 |
|  | INT8 | 1.41 | 34.94 | -1.46 | 45.50 | **-2.69** | +10.56 | 78.75 | 80.83 | 42 | 57 | 17.4 | 144 | 3.11 |
|  | NF4 | 0.98 | 16.02 | **-20.38** | 17.83 | **-30.36** | +1.81 | 73.33 | 77.08 | 32 | 30 | 67.0 | 430 | 2.01 |
|  | GPTQ | 0.98 | 11.40 | **-25.00** | 2.90 | **-45.29** | -8.50 | 85.83 | 99.58 | 16 | 13 | 148.6 | 61 | 0.56 |
|  | AWQ | 0.98 | 17.81 | **-18.59** | 47.75 | **-0.44** | +29.94 | 72.92 | 50.83 | 75 | 64 | 108.2 | 88 | 2.85 |
| Llama-3.2-3B-Instruct | FP16 | 6.02 | 31.38 | base | 71.04 | base | +39.66 | 91.25 | 74.58 | 79 | 142 | 34.2 | 158 | 1.36 |
|  | FP16, 8-bit KV | 6.02 | 31.40 | +0.02 | – | – | – | 90.83 | – | 78 | – | 28.8 | 154 | 1.48 |
|  | INT8 | 3.42 | 34.02 | +2.64 | 69.42 | -1.62 | +35.40 | 87.92 | 70.42 | 85 | 141 | 10.2 | 252 | 4.17 |
|  | NF4 | 2.13 | 52.75 | **+21.37** | 65.96 | -5.08 | +13.21 | 42.08 | 38.75 | 121 | 139 | 28.5 | 1358 | 2.97 |
|  | GPTQ | 2.14 | 59.52 | **+28.14** | 64.71 | **-6.33** | +5.19 | 52.92 | 65.00 | 129 | 119 | 76.6 | 191 | 0.85 |
|  | AWQ | 2.14 | 39.73 | **+8.35** | 68.98 | -2.06 | +29.25 | 82.08 | 69.58 | 83 | 139 | 43.4 | 291 | 1.23 |

Speed columns are free decoding; constrained decoding is in section 6. ×2 = the model is split over two T4s (Qwen3-8B in FP16); every other config runs on one.

In free decoding 11 of the 20 changes from FP16 are significant, in constrained decoding 8; the constraint raises 24 of the 25 runs. Of the 5 runs with an 8-bit KV cache, 1 changes significantly.

## 2. BFCL accuracy by category (%)

Python 400, Java 100, JavaScript 50, multiple 200, parallel 200, parallel multiple 200 and irrelevance 240 questions. Simple = mean of Python, Java and JavaScript.

### Free decoding

| Model | Precision | Python | Java | JavaScript | Simple | Multiple | Parallel | Parallel multiple | Overall | Irrelevance |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | FP16 | 86.25 | 65.00 | 64.00 | 71.75 | 87.50 | 71.50 | 59.50 | 72.56 | 77.92 |
|  | FP16, 8-bit KV | 79.00 | 66.00 | 66.00 | 70.33 | 83.00 | 75.00 | 59.00 | 71.83 | 78.75 |
|  | INT8 | 86.50 | 65.00 | 68.00 | 73.17 | 87.50 | 62.50 | 54.50 | 69.42 | 75.00 |
|  | NF4 | 86.25 | 65.00 | 64.00 | 71.75 | 85.50 | 66.00 | 53.00 | 69.06 | 79.58 |
|  | GPTQ | 77.50 | 60.00 | 68.00 | 68.50 | 63.50 | 6.00 | 19.00 | 39.25 | 62.08 |
|  | AWQ | 91.25 | 64.00 | 66.00 | 73.75 | 86.50 | 77.50 | 66.00 | 75.94 | 69.58 |
| Qwen3-4B | FP16 | 95.50 | 63.00 | 70.00 | 76.17 | 95.00 | 90.50 | 90.50 | 88.04 | 84.58 |
|  | FP16, 8-bit KV | 95.50 | 63.00 | 70.00 | 76.17 | 94.50 | 89.00 | 89.50 | 87.29 | 84.58 |
|  | INT8 | 95.50 | 64.00 | 72.00 | 77.17 | 95.00 | 91.00 | 90.00 | 88.29 | 85.83 |
|  | NF4 | 93.25 | 61.00 | 76.00 | 76.75 | 92.50 | 89.00 | 84.50 | 85.69 | 84.17 |
|  | GPTQ | 94.25 | 63.00 | 72.00 | 76.42 | 93.50 | 88.00 | 88.50 | 86.60 | 86.25 |
|  | AWQ | 95.75 | 62.00 | 70.00 | 75.92 | 92.00 | 91.50 | 88.50 | 86.98 | 82.92 |
| Qwen3-8B | FP16 | 95.50 | 65.00 | 68.00 | 76.17 | 93.00 | 92.50 | 91.50 | 88.29 | 87.08 |
|  | FP16, 8-bit KV | 95.50 | 64.00 | 68.00 | 75.83 | 93.00 | 93.50 | 92.00 | 88.58 | 87.08 |
|  | INT8 | 95.25 | 65.00 | 68.00 | 76.08 | 93.00 | 91.50 | 88.50 | 87.27 | 87.08 |
|  | NF4 | 95.50 | 64.00 | 70.00 | 76.50 | 95.00 | 92.50 | 88.00 | 88.00 | 84.58 |
|  | GPTQ | 95.25 | 62.00 | 66.00 | 74.42 | 91.50 | 90.50 | 87.50 | 85.98 | 89.58 |
|  | AWQ | 95.00 | 66.00 | 72.00 | 77.67 | 93.00 | 90.50 | 92.50 | 88.42 | 87.50 |
| Llama-3.2-1B-Instruct | FP16 | 39.75 | 37.00 | 48.00 | 41.58 | 38.50 | 41.00 | 24.50 | 36.40 | 82.92 |
|  | FP16, 8-bit KV | 39.75 | 35.00 | 42.00 | 38.92 | 40.00 | 39.00 | 22.00 | 34.98 | 82.50 |
|  | INT8 | 42.75 | 38.00 | 34.00 | 38.25 | 35.50 | 42.00 | 24.00 | 34.94 | 78.75 |
|  | NF4 | 36.75 | 11.00 | 8.00 | 18.58 | 17.50 | 20.50 | 7.50 | 16.02 | 73.33 |
|  | GPTQ | 17.25 | 5.00 | 26.00 | 16.08 | 5.00 | 19.00 | 5.50 | 11.40 | 85.83 |
|  | AWQ | 41.25 | 8.00 | 16.00 | 21.75 | 6.00 | 39.50 | 4.00 | 17.81 | 72.92 |
| Llama-3.2-3B-Instruct | FP16 | 22.00 | 32.00 | 24.00 | 26.00 | 30.00 | 42.50 | 27.00 | 31.38 | 91.25 |
|  | FP16, 8-bit KV | 26.25 | 32.00 | 20.00 | 26.08 | 34.00 | 44.00 | 21.50 | 31.40 | 90.83 |
|  | INT8 | 22.75 | 21.00 | 18.00 | 20.58 | 30.00 | 52.50 | 33.00 | 34.02 | 87.92 |
|  | NF4 | 62.00 | 56.00 | 56.00 | 58.00 | 52.50 | 57.50 | 43.00 | 52.75 | 42.08 |
|  | GPTQ | 73.75 | 52.00 | 74.00 | 66.58 | 71.00 | 62.00 | 38.50 | 59.52 | 52.92 |
|  | AWQ | 41.75 | 35.00 | 22.00 | 32.92 | 36.00 | 56.00 | 34.00 | 39.73 | 82.08 |

### Constrained decoding

| Model | Precision | Python | Java | JavaScript | Simple | Multiple | Parallel | Parallel multiple | Overall | Irrelevance |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | FP16 | 93.50 | 65.00 | 66.00 | 74.83 | 91.00 | 84.00 | 80.00 | 82.46 | 77.50 |
|  | INT8 | 92.50 | 65.00 | 68.00 | 75.17 | 90.00 | 84.50 | 81.50 | 82.79 | 72.50 |
|  | NF4 | 89.00 | 67.00 | 66.00 | 74.00 | 90.00 | 78.50 | 71.50 | 78.50 | 80.83 |
|  | GPTQ | 83.25 | 61.00 | 72.00 | 72.08 | 63.50 | 76.50 | 74.50 | 71.65 | 56.25 |
|  | AWQ | 92.25 | 65.00 | 70.00 | 75.75 | 87.50 | 81.50 | 78.50 | 80.81 | 69.58 |
| Qwen3-4B | FP16 | 95.50 | 64.00 | 70.00 | 76.50 | 95.50 | 90.50 | 90.50 | 88.25 | 84.58 |
|  | INT8 | 95.00 | 65.00 | 76.00 | 78.67 | 96.00 | 90.00 | 91.50 | 89.04 | 86.67 |
|  | NF4 | 93.50 | 63.00 | 76.00 | 77.50 | 93.50 | 89.00 | 84.50 | 86.12 | 83.75 |
|  | GPTQ | 94.75 | 64.00 | 72.00 | 76.92 | 94.50 | 88.50 | 87.50 | 86.85 | 86.25 |
|  | AWQ | 96.25 | 64.00 | 72.00 | 77.42 | 92.50 | 91.50 | 88.50 | 87.48 | 83.33 |
| Qwen3-8B | FP16 | 96.00 | 65.00 | 74.00 | 78.33 | 92.50 | 92.50 | 91.00 | 88.58 | 87.08 |
|  | INT8 | 96.00 | 65.00 | 74.00 | 78.33 | 93.00 | 92.00 | 89.50 | 88.21 | 87.50 |
|  | NF4 | 95.75 | 64.00 | 76.00 | 78.58 | 95.00 | 92.50 | 87.00 | 88.27 | 84.58 |
|  | GPTQ | 95.50 | 62.00 | 72.00 | 76.50 | 91.50 | 91.00 | 88.50 | 86.88 | 89.58 |
|  | AWQ | 95.50 | 66.00 | 78.00 | 79.83 | 93.00 | 90.50 | 92.00 | 88.83 | 87.50 |
| Llama-3.2-1B-Instruct | FP16 | 50.75 | 33.00 | 52.00 | 45.25 | 53.50 | 55.00 | 39.00 | 48.19 | 84.17 |
|  | INT8 | 45.00 | 27.00 | 36.00 | 36.00 | 55.00 | 52.00 | 39.00 | 45.50 | 80.83 |
|  | NF4 | 20.50 | 1.00 | 2.00 | 7.83 | 53.50 | 5.00 | 5.00 | 17.83 | 77.08 |
|  | GPTQ | 0.25 | 0.00 | 0.00 | 0.08 | 9.00 | 0.50 | 2.00 | 2.90 | 99.58 |
|  | AWQ | 68.00 | 51.00 | 64.00 | 61.00 | 49.00 | 47.50 | 33.50 | 47.75 | 50.83 |
| Llama-3.2-3B-Instruct | FP16 | 88.50 | 48.00 | 44.00 | 60.17 | 86.50 | 76.00 | 61.50 | 71.04 | 74.58 |
|  | INT8 | 88.00 | 44.00 | 38.00 | 56.67 | 86.00 | 75.00 | 60.00 | 69.42 | 70.42 |
|  | NF4 | 90.50 | 54.00 | 50.00 | 64.83 | 92.50 | 57.00 | 49.50 | 65.96 | 38.75 |
|  | GPTQ | 81.50 | 47.00 | 54.00 | 60.83 | 90.50 | 60.50 | 47.00 | 64.71 | 65.00 |
|  | AWQ | 88.75 | 58.00 | 48.00 | 64.92 | 84.50 | 64.50 | 62.00 | 68.98 | 69.58 |

## 3. Paired tests against FP16

Change in Overall with its bootstrap 95% interval; *FP16 only* / *Quantized only* (*8-bit KV only*) = AST questions right in only one of the two runs; p = McNemar's exact test, Holm-adjusted (over the 20 runs of a mode; over the 5 models for the 8-bit KV cache).

### Quantized weights

| Model | Precision | Free change | 95% interval | FP16 only | Quantized only | p (Holm) | Constrained change | 95% interval | FP16 only | Quantized only | p (Holm) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | INT8 | **-3.14** | -4.88 to -1.44 | 49 | 24 | 0.0463 | +0.33 | -0.73 to +1.40 | 14 | 13 | 1 |
|  | NF4 | -3.50 | -6.29 to -0.69 | 107 | 79 | 0.38 | **-3.96** | -6.02 to -1.83 | 74 | 28 | 9.45e-05 |
|  | GPTQ | **-33.31** | -36.65 to -30.04 | 342 | 44 | 4.97e-57 | **-10.81** | -13.33 to -8.27 | 150 | 27 | 1.29e-20 |
|  | AWQ | **+3.38** | +0.83 to +5.90 | 54 | 97 | 0.00702 | -1.65 | -3.56 to +0.21 | 49 | 31 | 0.567 |
| Qwen3-4B | INT8 | +0.25 | -0.60 to +1.12 | 7 | 9 | 1 | +0.79 | -0.08 to +1.73 | 6 | 10 | 1 |
|  | NF4 | **-2.35** | -3.92 to -0.81 | 43 | 15 | 0.00399 | **-2.13** | -3.65 to -0.62 | 40 | 15 | 0.0142 |
|  | GPTQ | -1.44 | -2.98 to +0.04 | 35 | 19 | 0.362 | -1.40 | -2.87 to +0.04 | 31 | 17 | 0.567 |
|  | AWQ | -1.06 | -2.48 to +0.29 | 24 | 16 | 1 | -0.77 | -2.21 to +0.62 | 21 | 17 | 1 |
| Qwen3-8B | INT8 | -1.02 | -1.94 to -0.17 | 14 | 5 | 0.445 | -0.37 | -1.15 to +0.35 | 8 | 5 | 1 |
|  | NF4 | -0.29 | -1.81 to +1.19 | 23 | 20 | 1 | -0.31 | -1.81 to +1.21 | 24 | 20 | 1 |
|  | GPTQ | **-2.31** | -3.71 to -0.96 | 32 | 12 | 0.0402 | -1.70 | -3.10 to -0.35 | 30 | 14 | 0.249 |
|  | AWQ | +0.13 | -1.10 to +1.40 | 19 | 18 | 1 | +0.25 | -0.96 to +1.50 | 18 | 18 | 1 |
| Llama-3.2-1B-Instruct | INT8 | -1.46 | -3.88 to +0.94 | 82 | 83 | 1 | **-2.69** | -4.98 to -0.38 | 100 | 60 | 0.0254 |
|  | NF4 | **-20.38** | -23.65 to -17.15 | 250 | 75 | 6.17e-22 | **-30.36** | -33.65 to -27.19 | 393 | 47 | 8.03e-68 |
|  | GPTQ | **-25.00** | -28.48 to -21.69 | 327 | 45 | 1.01e-52 | **-45.29** | -48.69 to -42.00 | 538 | 5 | 5.42e-151 |
|  | AWQ | **-18.59** | -21.92 to -15.25 | 257 | 109 | 9.25e-14 | **-0.44** | -3.75 to +2.92 | 120 | 178 | 0.0139 |
| Llama-3.2-3B-Instruct | INT8 | +2.64 | +0.48 to +4.92 | 60 | 81 | 0.551 | -1.62 | -3.54 to +0.31 | 43 | 28 | 0.767 |
|  | NF4 | **+21.37** | +18.04 to +24.81 | 39 | 346 | 2.47e-61 | -5.08 | -7.63 to -2.52 | 89 | 56 | 0.0919 |
|  | GPTQ | **+28.14** | +24.29 to +32.00 | 63 | 459 | 5.63e-74 | **-6.33** | -9.27 to -3.37 | 137 | 61 | 1.17e-06 |
|  | AWQ | **+8.35** | +5.17 to +11.52 | 66 | 200 | 1.12e-15 | -2.06 | -4.62 to +0.54 | 73 | 60 | 1 |

### 8-bit KV cache (FP16 weights, free decoding)

| Model | Change | 95% interval | FP16 only | 8-bit KV only | p (Holm) |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | **-0.73** | -2.48 to +1.02 | 58 | 28 | 0.00803 |
| Qwen3-4B | -0.75 | -1.75 to +0.21 | 10 | 4 | 0.432 |
| Qwen3-8B | +0.29 | -0.33 to +1.00 | 3 | 5 | 0.727 |
| Llama-3.2-1B-Instruct | -1.42 | -2.96 to +0.12 | 29 | 18 | 0.432 |
| Llama-3.2-3B-Instruct | +0.02 | -1.52 to +1.56 | 21 | 36 | 0.251 |

### How much of the free change the constraint removes

Difference = constrained change minus free change, with its bootstrap 95% interval. Loss removed is given for the 7 significant free losses only; 100% = the whole loss is gone, a negative value = the loss grows.

| Model | Precision | Free change | Constrained change | Difference | 95% interval | Loss removed |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | INT8 | **-3.14** | +0.33 | +3.47 | +1.71 to +5.23 | 111% |
|  | NF4 | -3.50 | -3.96 | -0.46 | -2.77 to +1.85 | – |
|  | GPTQ | **-33.31** | -10.81 | +22.50 | +19.69 to +25.37 | 68% |
|  | AWQ | **+3.38** | -1.65 | -5.03 | -6.92 to -3.15 | – |
| Qwen3-4B | INT8 | +0.25 | +0.79 | +0.54 | -0.48 to +1.58 | – |
|  | NF4 | **-2.35** | -2.13 | +0.22 | -0.35 to +0.83 | 9% |
|  | GPTQ | -1.44 | -1.40 | +0.04 | -0.52 to +0.58 | – |
|  | AWQ | -1.06 | -0.77 | +0.29 | -0.17 to +0.83 | – |
| Qwen3-8B | INT8 | -1.02 | -0.37 | +0.65 | -0.21 to +1.52 | – |
|  | NF4 | -0.29 | -0.31 | -0.02 | -0.77 to +0.73 | – |
|  | GPTQ | **-2.31** | -1.70 | +0.61 | -0.08 to +1.35 | 26% |
|  | AWQ | +0.13 | +0.25 | +0.12 | -0.04 to +0.42 | – |
| Llama-3.2-1B-Instruct | INT8 | -1.46 | -2.69 | -1.23 | -4.08 to +1.69 | – |
|  | NF4 | **-20.38** | -30.36 | -9.98 | -13.46 to -6.48 | -49% |
|  | GPTQ | **-25.00** | -45.29 | -20.29 | -23.65 to -17.04 | -81% |
|  | AWQ | **-18.59** | -0.44 | +18.15 | +14.10 to +22.17 | 98% |
| Llama-3.2-3B-Instruct | INT8 | +2.64 | -1.62 | -4.26 | -7.15 to -1.48 | – |
|  | NF4 | **+21.37** | -5.08 | -26.45 | -30.46 to -22.54 | – |
|  | GPTQ | **+28.14** | -6.33 | -34.47 | -38.96 to -29.81 | – |
|  | AWQ | **+8.35** | -2.06 | -10.41 | -14.37 to -6.52 | – |

## 4. Schema-stress suite (correct of 25 per group)

### Free decoding

| Model | Precision | Plain (control) | Nested | Enum | Number | Optional | Unicode | Menu | Total (of 175) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | FP16 | 25 | 24 | 22 | 9 | 25 | 24 | 25 | 154 |
|  | FP16, 8-bit KV | 22 | 22 | 23 | 8 | 24 | 23 | 25 | 147 |
|  | INT8 | 25 | 20 | 23 | 13 | 25 | 24 | 25 | 155 |
|  | NF4 | 25 | 25 | 23 | 19 | 23 | 23 | 22 | 160 |
|  | GPTQ | 25 | 25 | 21 | 15 | 25 | 23 | 24 | 158 |
|  | AWQ | 25 | 24 | 21 | 25 | 25 | 23 | 25 | 168 |
| Qwen3-4B | FP16 | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
|  | FP16, 8-bit KV | 25 | 24 | 25 | 25 | 25 | 25 | 25 | 174 |
|  | INT8 | 25 | 24 | 24 | 25 | 25 | 25 | 25 | 173 |
|  | NF4 | 25 | 19 | 25 | 25 | 25 | 24 | 24 | 167 |
|  | GPTQ | 25 | 19 | 25 | 25 | 25 | 25 | 24 | 168 |
|  | AWQ | 25 | 19 | 24 | 25 | 25 | 24 | 25 | 167 |
| Qwen3-8B | FP16 | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
|  | FP16, 8-bit KV | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
|  | INT8 | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
|  | NF4 | 25 | 25 | 24 | 25 | 25 | 25 | 25 | 174 |
|  | GPTQ | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
|  | AWQ | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
| Llama-3.2-1B-Instruct | FP16 | 2 | 11 | 5 | 3 | 14 | 4 | 3 | 42 |
|  | FP16, 8-bit KV | 2 | 11 | 5 | 3 | 12 | 4 | 2 | 39 |
|  | INT8 | 4 | 15 | 2 | 1 | 15 | 2 | 3 | 42 |
|  | NF4 | 2 | 4 | 2 | 2 | 19 | 1 | 2 | 32 |
|  | GPTQ | 6 | 0 | 0 | 0 | 8 | 2 | 0 | 16 |
|  | AWQ | 16 | 18 | 3 | 11 | 20 | 7 | 0 | 75 |
| Llama-3.2-3B-Instruct | FP16 | 16 | 12 | 11 | 9 | 7 | 5 | 19 | 79 |
|  | FP16, 8-bit KV | 16 | 11 | 11 | 9 | 8 | 4 | 19 | 78 |
|  | INT8 | 19 | 17 | 8 | 9 | 9 | 2 | 21 | 85 |
|  | NF4 | 20 | 22 | 22 | 11 | 20 | 1 | 25 | 121 |
|  | GPTQ | 25 | 24 | 23 | 5 | 23 | 7 | 22 | 129 |
|  | AWQ | 16 | 14 | 15 | 5 | 11 | 4 | 18 | 83 |

### Constrained decoding

| Model | Precision | Plain (control) | Nested | Enum | Number | Optional | Unicode | Menu | Total (of 175) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | FP16 | 25 | 25 | 22 | 25 | 25 | 24 | 25 | 171 |
|  | INT8 | 25 | 25 | 23 | 25 | 25 | 24 | 25 | 172 |
|  | NF4 | 25 | 25 | 23 | 25 | 25 | 23 | 23 | 169 |
|  | GPTQ | 25 | 25 | 21 | 25 | 25 | 24 | 24 | 169 |
|  | AWQ | 25 | 25 | 23 | 25 | 25 | 23 | 25 | 171 |
| Qwen3-4B | FP16 | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
|  | INT8 | 25 | 25 | 24 | 25 | 25 | 25 | 25 | 174 |
|  | NF4 | 25 | 25 | 25 | 25 | 25 | 24 | 25 | 174 |
|  | GPTQ | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
|  | AWQ | 25 | 25 | 24 | 25 | 25 | 24 | 25 | 173 |
| Qwen3-8B | FP16 | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
|  | INT8 | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
|  | NF4 | 25 | 25 | 24 | 25 | 25 | 25 | 25 | 174 |
|  | GPTQ | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
|  | AWQ | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
| Llama-3.2-1B-Instruct | FP16 | 2 | 20 | 8 | 9 | 12 | 6 | 6 | 63 |
|  | INT8 | 4 | 23 | 9 | 3 | 8 | 7 | 3 | 57 |
|  | NF4 | 0 | 0 | 2 | 0 | 3 | 0 | 25 | 30 |
|  | GPTQ | 0 | 0 | 0 | 0 | 1 | 0 | 12 | 13 |
|  | AWQ | 2 | 13 | 6 | 7 | 9 | 7 | 20 | 64 |
| Llama-3.2-3B-Instruct | FP16 | 25 | 25 | 24 | 13 | 25 | 8 | 22 | 142 |
|  | INT8 | 25 | 25 | 24 | 13 | 25 | 5 | 24 | 141 |
|  | NF4 | 25 | 21 | 25 | 15 | 25 | 8 | 20 | 139 |
|  | GPTQ | 21 | 24 | 20 | 12 | 25 | 7 | 10 | 119 |
|  | AWQ | 25 | 25 | 24 | 12 | 25 | 8 | 20 | 139 |

### The 7 significant changes of the 280 stress-suite tests

McNemar's exact test on the 25 questions of a group, Holm's correction over the 140 tests of each mode. For the 8-bit KV cache, 0 of its 35 group tests are significant.

| Model | Precision | Mode | Group | FP16 | Quantized | Change | p (Holm) |
|---|---|---|---|---:|---:|---:|---:|
| Qwen3-1.7B | AWQ | free | Number | 9 | 25 | +16 | 0.00427 |
| Llama-3.2-1B-Instruct | AWQ | free | Plain (control) | 2 | 16 | +14 | 0.017 |
| Llama-3.2-3B-Instruct | GPTQ | free | Optional | 7 | 23 | +16 | 0.02 |
| Llama-3.2-1B-Instruct | NF4 | constrained | Nested | 20 | 0 | -20 | 0.000267 |
| Llama-3.2-1B-Instruct | NF4 | constrained | Menu | 6 | 25 | +19 | 0.000526 |
| Llama-3.2-1B-Instruct | GPTQ | constrained | Nested | 20 | 0 | -20 | 0.000267 |
| Llama-3.2-1B-Instruct | AWQ | constrained | Menu | 6 | 20 | +14 | 0.0167 |

## 5. Failure classes (BFCL, all 1,390 questions)

S1 not valid JSON, S2 valid JSON that does not match the schema (syntactic); M1 wrong function, M2 wrong argument value, M3 wrong number of calls, M4 a call on an irrelevance question, M5 no call when one was needed (semantic).

### Free decoding

| Model | Precision | Failures | S1 | S2 | M1 | M2 | M3 | M4 | M5 | Syntactic (S) | Semantic (M) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | FP16 | 324 | 121 | 24 | 1 | 95 | 32 | 50 | 1 | 145 | 179 |
|  | INT8 | 356 | 151 | 20 | 2 | 95 | 32 | 56 | 0 | 171 | 185 |
|  | NF4 | 348 | 106 | 18 | 3 | 120 | 50 | 47 | 4 | 124 | 224 |
|  | GPTQ | 660 | 393 | 25 | 1 | 109 | 44 | 87 | 1 | 418 | 242 |
|  | AWQ | 301 | 55 | 18 | 5 | 114 | 39 | 69 | 1 | 73 | 228 |
| Qwen3-4B | FP16 | 155 | 2 | 8 | 0 | 91 | 18 | 35 | 1 | 10 | 145 |
|  | INT8 | 150 | 3 | 8 | 0 | 88 | 18 | 32 | 1 | 11 | 139 |
|  | NF4 | 184 | 5 | 8 | 1 | 117 | 15 | 37 | 1 | 13 | 171 |
|  | GPTQ | 167 | 5 | 9 | 1 | 102 | 15 | 31 | 4 | 14 | 153 |
|  | AWQ | 167 | 7 | 9 | 0 | 92 | 19 | 39 | 1 | 16 | 151 |
| Qwen3-8B | FP16 | 146 | 0 | 11 | 0 | 83 | 18 | 29 | 5 | 11 | 135 |
|  | INT8 | 155 | 2 | 12 | 1 | 89 | 17 | 29 | 5 | 14 | 141 |
|  | NF4 | 155 | 2 | 11 | 1 | 88 | 16 | 35 | 2 | 13 | 142 |
|  | GPTQ | 160 | 3 | 14 | 0 | 88 | 19 | 23 | 13 | 17 | 143 |
|  | AWQ | 146 | 3 | 11 | 1 | 82 | 20 | 28 | 1 | 14 | 132 |
| Llama-3.2-1B-Instruct | FP16 | 763 | 69 | 266 | 2 | 79 | 39 | 20 | 288 | 335 | 428 |
|  | INT8 | 772 | 81 | 290 | 3 | 91 | 32 | 25 | 250 | 371 | 401 |
|  | NF4 | 961 | 225 | 361 | 1 | 46 | 19 | 23 | 286 | 586 | 375 |
|  | GPTQ | 1,038 | 470 | 346 | 0 | 28 | 28 | 6 | 160 | 816 | 222 |
|  | AWQ | 935 | 353 | 441 | 2 | 46 | 22 | 27 | 44 | 794 | 141 |
| Llama-3.2-3B-Instruct | FP16 | 840 | 519 | 238 | 1 | 46 | 15 | 10 | 11 | 757 | 83 |
|  | INT8 | 827 | 475 | 261 | 2 | 53 | 17 | 14 | 5 | 736 | 91 |
|  | NF4 | 651 | 99 | 352 | 2 | 91 | 25 | 77 | 5 | 451 | 200 |
|  | GPTQ | 536 | 107 | 209 | 1 | 110 | 25 | 75 | 9 | 316 | 220 |
|  | AWQ | 728 | 364 | 254 | 3 | 55 | 20 | 23 | 9 | 618 | 110 |
| **All 25 runs** |  | **11,625** | **3,620** | **3,224** | **34** | **2,098** | **614** | **927** | **1,108** | **6,844** | **4,781** |

Syntactic: 6,844 of 11,625 failures (58.9%).

### Constrained decoding

| Model | Precision | Failures | S1 | S2 | M1 | M2 | M3 | M4 | M5 | Syntactic (S) | Semantic (M) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | FP16 | 222 | 4 | 0 | 2 | 117 | 44 | 54 | 1 | 4 | 218 |
|  | INT8 | 235 | 2 | 0 | 2 | 120 | 44 | 66 | 1 | 2 | 233 |
|  | NF4 | 260 | 1 | 0 | 5 | 144 | 57 | 46 | 7 | 1 | 259 |
|  | GPTQ | 396 | 1 | 0 | 2 | 161 | 125 | 105 | 2 | 1 | 395 |
|  | AWQ | 259 | 1 | 0 | 5 | 128 | 51 | 73 | 1 | 1 | 258 |
| Qwen3-4B | FP16 | 153 | 2 | 0 | 0 | 94 | 19 | 37 | 1 | 2 | 151 |
|  | INT8 | 144 | 1 | 0 | 0 | 94 | 16 | 32 | 1 | 1 | 143 |
|  | NF4 | 180 | 1 | 0 | 1 | 121 | 17 | 39 | 1 | 1 | 179 |
|  | GPTQ | 163 | 1 | 0 | 1 | 109 | 15 | 33 | 4 | 1 | 162 |
|  | AWQ | 160 | 4 | 0 | 0 | 96 | 19 | 40 | 1 | 4 | 156 |
| Qwen3-8B | FP16 | 143 | 0 | 0 | 0 | 89 | 18 | 31 | 5 | 0 | 143 |
|  | INT8 | 145 | 0 | 0 | 1 | 90 | 19 | 30 | 5 | 0 | 145 |
|  | NF4 | 153 | 0 | 0 | 1 | 96 | 17 | 37 | 2 | 0 | 153 |
|  | GPTQ | 153 | 0 | 0 | 0 | 96 | 19 | 25 | 13 | 0 | 153 |
|  | AWQ | 142 | 1 | 0 | 1 | 89 | 20 | 30 | 1 | 1 | 141 |
| Llama-3.2-1B-Instruct | FP16 | 631 | 8 | 0 | 9 | 146 | 86 | 38 | 344 | 8 | 623 |
|  | INT8 | 679 | 11 | 0 | 10 | 142 | 78 | 46 | 392 | 11 | 668 |
|  | NF4 | 994 | 2 | 0 | 6 | 72 | 41 | 55 | 818 | 2 | 992 |
|  | GPTQ | 1,127 | 0 | 0 | 1 | 6 | 7 | 1 | 1,112 | 0 | 1,127 |
|  | AWQ | 653 | 10 | 0 | 7 | 184 | 117 | 118 | 217 | 10 | 643 |
| Llama-3.2-3B-Instruct | FP16 | 339 | 1 | 0 | 2 | 115 | 97 | 61 | 63 | 1 | 338 |
|  | INT8 | 364 | 0 | 0 | 2 | 110 | 102 | 71 | 79 | 0 | 364 |
|  | NF4 | 458 | 0 | 0 | 3 | 105 | 169 | 147 | 34 | 0 | 458 |
|  | GPTQ | 438 | 1 | 0 | 1 | 115 | 161 | 84 | 76 | 1 | 437 |
|  | AWQ | 364 | 1 | 0 | 3 | 134 | 106 | 73 | 47 | 1 | 363 |
| **All 25 runs** |  | **8,955** | **53** | **0** | **65** | **2,773** | **1,464** | **1,372** | **3,228** | **53** | **8,902** |

Syntactic: 53 of 8,955 failures (0.6%).

### 8-bit KV cache (free decoding): failed answers

| Model | FP16 | 8-bit KV |
|---|---:|---:|
| Qwen3-1.7B | 324 | 352 |
| Qwen3-4B | 155 | 161 |
| Qwen3-8B | 146 | 144 |
| Llama-3.2-1B-Instruct | 763 | 775 |
| Llama-3.2-3B-Instruct | 840 | 826 |

## 6. Speed and memory at batch size 1

Weights per GPU and KV cache tokens (keys and values that fit in the memory left after the weights) from vLLM's log. Llama-3.2-3B in FP16: constrained values not reliable (the GPU slowed down during that run). Llama-3.2-1B NF4 and GPTQ in constrained decoding: most answers are an empty list, so tokens/s is below the model's real writing speed.

| Model | Precision | GPUs | Weights (GiB per GPU) | KV cache tokens | Free: tokens/s | Free: first token (ms) | Free: whole answer (s) | Constrained: tokens/s | Constrained: first token (ms) | Constrained: whole answer (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | FP16 | 1 | 3.22 | 78,896 | 52.9 | 77 | 0.80 | 49.6 | 583 | 1.41 |
|  | FP16, 8-bit KV | 1 | 3.22 | 157,792 | 40.6 | 83 | 1.00 | 41.3 | 600 | 1.53 |
|  | INT8 | 1 | 1.90 | 91,040 | 7.3 | 354 | 5.46 | 7.2 | 873 | 6.25 |
|  | NF4 | 1 | 1.27 | 97,120 | 47.4 | 567 | 1.50 | 46.3 | 839 | 1.69 |
|  | GPTQ | 1 | 1.27 | 97,056 | 101.4 | 86 | 0.48 | 100.4 | 604 | 1.03 |
|  | AWQ | 1 | 1.27 | 97,088 | 80.3 | 106 | 0.58 | 79.5 | 604 | 1.13 |
| Qwen3-4B | FP16 | 1 | 7.56 | 29,696 | 25.2 | 181 | 1.80 | 24.9 | 614 | 2.21 |
|  | FP16, 8-bit KV | 1 | 7.56 | 59,408 | 21.2 | 194 | 2.15 | 21.4 | 610 | 2.48 |
|  | INT8 | 1 | 4.18 | 54,144 | 6.1 | 394 | 7.03 | 6.1 | 903 | 7.49 |
|  | NF4 | 1 | 2.49 | 66,560 | 21.6 | 1512 | 3.46 | 21.6 | 1754 | 3.68 |
|  | GPTQ | 1 | 2.50 | 66,464 | 58.9 | 220 | 0.90 | 59.5 | 597 | 1.28 |
|  | AWQ | 1 | 2.50 | 66,480 | 41.1 | 305 | 1.30 | 41.9 | 600 | 1.67 |
| Qwen3-8B | FP16 | 2 | 7.64 | 57,008 | 25.7 | 196 | 1.70 | 25.6 | 656 | 2.22 |
|  | FP16, 8-bit KV | 2 | 7.64 | 114,032 | 22.0 | 198 | 1.96 | 22.0 | 642 | 2.44 |
|  | INT8 | 1 | 8.80 | 9,440 | 6.1 | 532 | 6.72 | 6.1 | 993 | 7.42 |
|  | NF4 | 1 | 5.67 | 43,264 | 13.3 | 2907 | 5.86 | 13.3 | 3150 | 6.09 |
|  | GPTQ | 1 | 5.71 | 42,976 | 38.5 | 407 | 1.44 | 38.8 | 648 | 1.73 |
|  | AWQ | 1 | 5.71 | 42,976 | 24.2 | 554 | 2.20 | 24.1 | 791 | 2.42 |
| Llama-3.2-1B-Instruct | FP16 | 1 | 2.32 | 312,560 | 79.9 | 47 | 0.57 | 79.6 | 519 | 0.95 |
|  | FP16, 8-bit KV | 1 | 2.32 | 625,136 | 66.9 | 57 | 0.65 | 69.5 | 492 | 1.00 |
|  | INT8 | 1 | 1.41 | 341,840 | 17.4 | 144 | 3.11 | 16.5 | 597 | 2.54 |
|  | NF4 | 1 | 0.98 | 356,480 | 67.0 | 430 | 2.01 | 32.6 | 653 | 0.70 |
|  | GPTQ | 1 | 0.98 | 356,304 | 148.6 | 61 | 0.56 | 46.1 | 503 | 0.53 |
|  | AWQ | 1 | 0.98 | 356,368 | 108.2 | 88 | 2.85 | 113.1 | 515 | 0.82 |
| Llama-3.2-3B-Instruct | FP16 | 1 | 6.02 | 54,528 | 34.2 | 158 | 1.36 | 22.5 | 512 | 2.64 |
|  | FP16, 8-bit KV | 1 | 6.02 | 109,072 | 28.8 | 154 | 1.48 | 28.7 | 519 | 1.89 |
|  | INT8 | 1 | 3.42 | 78,720 | 10.2 | 252 | 4.17 | 10.1 | 701 | 4.51 |
|  | NF4 | 1 | 2.13 | 90,944 | 28.5 | 1358 | 2.97 | 28.4 | 1567 | 2.94 |
|  | GPTQ | 1 | 2.14 | 90,832 | 76.6 | 191 | 0.85 | 77.6 | 527 | 0.97 |
|  | AWQ | 1 | 2.14 | 90,848 | 43.4 | 291 | 1.23 | 46.8 | 534 | 1.33 |

## 7. Setup and method checks

### BFCL reproduced on Kaggle (Qwen3-4B-Instruct-2507, BFCL's own harness, FP16)

| Mode | Category | Ours (FP16, T4) | Leaderboard | Difference |
|---|---|---:|---:|---:|
| Prompt | Non-Live Overall Acc | 86.31 | 86.44 | -0.13 |
| Prompt | Python Simple AST | 93.75 | 93.75 | +0.00 |
| Prompt | Java Simple AST | 64.00 | 64.00 | +0.00 |
| Prompt | JavaScript Simple AST | 74.00 | 74.00 | +0.00 |
| Prompt | Multiple AST | 91.00 | 91.00 | +0.00 |
| Prompt | Parallel AST | 88.00 | 88.00 | +0.00 |
| Prompt | Parallel Multiple AST | 89.00 | 89.50 | -0.50 |
| Prompt | Irrelevance Detection | 84.58 | 85.00 | -0.42 |
| FC | Non-Live Overall Acc | 87.81 | 87.88 | -0.07 |
| FC | Python Simple AST | 95.25 | 95.50 | -0.25 |
| FC | Java Simple AST | 63.00 | 63.00 | +0.00 |
| FC | JavaScript Simple AST | 66.00 | 68.00 | -2.00 |
| FC | Multiple AST | 94.00 | 93.50 | +0.50 |
| FC | Parallel AST | 92.50 | 92.50 | +0.00 |
| FC | Parallel Multiple AST | 90.00 | 90.00 | +0.00 |
| FC | Irrelevance Detection | 88.75 | 88.75 | +0.00 |

### First run of our own script (Update 3, Qwen3-4B-Instruct-2507, FP16, free decoding)

| Model | Python | Java | JavaScript | Simple | Multiple | Parallel | Parallel multiple | Overall | Irrelevance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 | 96.00 | 65.00 | 72.00 | 77.67 | 94.00 | 94.00 | 91.50 | 89.29 | 85.83 |

### Other checks

- Our scorer against BFCL's: re-scoring the 1,390 reproduction answers gives BFCL's verdict on every question ([`src/check_scorer.py`](../src/check_scorer.py)).
- Failure classifier: checked by hand on 80 random failures, 80 agree ([`notes/taxonomy_check.md`](../notes/taxonomy_check.md)).
- Schema against BFCL's checker: 14 of the 34,750 free answers are correct by BFCL but do not match the schema ([`taxonomy_free.md`](taxonomy_free.md)).
- Parser: Llama-3.2-3B FP16 scores 31.38 with our strict parser and would score 46.02 with BFCL's bracket repair (diagnostic only; the reported score stays 31.38) ([`baseline_fp16.md`](baseline_fp16.md)).
- Every answer of every run was scored again on a PC with the same result.
