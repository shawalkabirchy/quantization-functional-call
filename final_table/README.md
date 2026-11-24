# Final Table

Every score of the project in one place. The experiments were frozen in Update 15; every value in this folder is read from the files in [`results/`](../results/) by [`src/final_table.py`](../src/final_table.py), and none is typed by hand.

**What was run:** 5 models (Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct) at 5 weight precisions (FP16, INT8, NF4, GPTQ, AWQ), in free and in JSON-schema-constrained decoding, on the 1,390 BFCL `non_live` questions and on our 175-question schema-stress suite; and the 5 FP16 models with an 8-bit KV cache, in free decoding. That is 110 runs, plus a speed test of all 30 configs. Greedy decoding, vLLM 0.8.5, Kaggle Tesla T4 GPUs.

## How to read the tables

- **BFCL** = BFCL's Non-Live Overall accuracy (%): the unweighted mean of simple (itself the mean of Python, Java and JavaScript), multiple, parallel and parallel multiple.
- **Change** = points against FP16 in the same decoding mode. **Bold** = significant: McNemar's exact test, paired by question on the 1,150 AST questions, with Holm's correction (p < 0.05). A small change can be significant when the two runs get many different questions right: Llama-3.2-1B AWQ in constrained decoding changes by -0.44, with 178 AST questions right only with AWQ and 120 only with FP16.
- **Constraint gain** = constrained score minus free score of the same run.
- **Irrelevance** = accuracy (%) on the 240 questions where no function fits (correct = no call). As in BFCL, it is not part of the Overall score.
- **Stress** = correct answers out of the 175 stress-suite questions (7 groups of 25).
- – = not run: the 8-bit KV cache was tested in free decoding only.

## BFCL accuracy (%)

| Model / precision | Free | Change | Constrained | Change | Constraint<br>gain | Irrelevance<br>free | Irrelevance<br>constrained |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3-1.7B** |  |  |  |  |  |  |  |
| FP16 | 72.56 | base | 82.46 | base | +9.90 | 77.92 | 77.50 |
| FP16, 8-bit KV | 71.83 | **-0.73** | – | – | – | 78.75 | – |
| INT8 | 69.42 | **-3.14** | 82.79 | +0.33 | +13.37 | 75.00 | 72.50 |
| NF4 | 69.06 | -3.50 | 78.50 | **-3.96** | +9.44 | 79.58 | 80.83 |
| GPTQ | 39.25 | **-33.31** | 71.65 | **-10.81** | +32.40 | 62.08 | 56.25 |
| AWQ | 75.94 | **+3.38** | 80.81 | -1.65 | +4.87 | 69.58 | 69.58 |
| **Qwen3-4B** |  |  |  |  |  |  |  |
| FP16 | 88.04 | base | 88.25 | base | +0.21 | 84.58 | 84.58 |
| FP16, 8-bit KV | 87.29 | -0.75 | – | – | – | 84.58 | – |
| INT8 | 88.29 | +0.25 | 89.04 | +0.79 | +0.75 | 85.83 | 86.67 |
| NF4 | 85.69 | **-2.35** | 86.12 | **-2.13** | +0.43 | 84.17 | 83.75 |
| GPTQ | 86.60 | -1.44 | 86.85 | -1.40 | +0.25 | 86.25 | 86.25 |
| AWQ | 86.98 | -1.06 | 87.48 | -0.77 | +0.50 | 82.92 | 83.33 |
| **Qwen3-8B** |  |  |  |  |  |  |  |
| FP16 | 88.29 | base | 88.58 | base | +0.29 | 87.08 | 87.08 |
| FP16, 8-bit KV | 88.58 | +0.29 | – | – | – | 87.08 | – |
| INT8 | 87.27 | -1.02 | 88.21 | -0.37 | +0.94 | 87.08 | 87.50 |
| NF4 | 88.00 | -0.29 | 88.27 | -0.31 | +0.27 | 84.58 | 84.58 |
| GPTQ | 85.98 | **-2.31** | 86.88 | -1.70 | +0.90 | 89.58 | 89.58 |
| AWQ | 88.42 | +0.13 | 88.83 | +0.25 | +0.41 | 87.50 | 87.50 |
| **Llama-3.2-1B-Instruct** |  |  |  |  |  |  |  |
| FP16 | 36.40 | base | 48.19 | base | +11.79 | 82.92 | 84.17 |
| FP16, 8-bit KV | 34.98 | -1.42 | – | – | – | 82.50 | – |
| INT8 | 34.94 | -1.46 | 45.50 | **-2.69** | +10.56 | 78.75 | 80.83 |
| NF4 | 16.02 | **-20.38** | 17.83 | **-30.36** | +1.81 | 73.33 | 77.08 |
| GPTQ | 11.40 | **-25.00** | 2.90 | **-45.29** | -8.50 | 85.83 | 99.58 |
| AWQ | 17.81 | **-18.59** | 47.75 | **-0.44** | +29.94 | 72.92 | 50.83 |
| **Llama-3.2-3B-Instruct** |  |  |  |  |  |  |  |
| FP16 | 31.38 | base | 71.04 | base | +39.66 | 91.25 | 74.58 |
| FP16, 8-bit KV | 31.40 | +0.02 | – | – | – | 90.83 | – |
| INT8 | 34.02 | +2.64 | 69.42 | -1.62 | +35.40 | 87.92 | 70.42 |
| NF4 | 52.75 | **+21.37** | 65.96 | -5.08 | +13.21 | 42.08 | 38.75 |
| GPTQ | 59.52 | **+28.14** | 64.71 | **-6.33** | +5.19 | 52.92 | 65.00 |
| AWQ | 39.73 | **+8.35** | 68.98 | -2.06 | +29.25 | 82.08 | 69.58 |

## Stress suite, memory and speed

| Model / precision | Stress<br>free | Stress<br>constrained | Weights<br>(GiB per GPU) | KV cache<br>tokens | Tokens/s | First token<br>(ms) | Whole answer<br>(s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3-1.7B** |  |  |  |  |  |  |  |
| FP16 | 154 | 171 | 3.22 | 78,896 | 52.9 | 77 | 0.80 |
| FP16, 8-bit KV | 147 | – | 3.22 | 157,792 | 40.6 | 83 | 1.00 |
| INT8 | 155 | 172 | 1.90 | 91,040 | 7.3 | 354 | 5.46 |
| NF4 | 160 | 169 | 1.27 | 97,120 | 47.4 | 567 | 1.50 |
| GPTQ | 158 | 169 | 1.27 | 97,056 | 101.4 | 86 | 0.48 |
| AWQ | 168 | 171 | 1.27 | 97,088 | 80.3 | 106 | 0.58 |
| **Qwen3-4B** |  |  |  |  |  |  |  |
| FP16 | 175 | 175 | 7.56 | 29,696 | 25.2 | 181 | 1.80 |
| FP16, 8-bit KV | 174 | – | 7.56 | 59,408 | 21.2 | 194 | 2.15 |
| INT8 | 173 | 174 | 4.18 | 54,144 | 6.1 | 394 | 7.03 |
| NF4 | 167 | 174 | 2.49 | 66,560 | 21.6 | 1,512 | 3.46 |
| GPTQ | 168 | 175 | 2.50 | 66,464 | 58.9 | 220 | 0.90 |
| AWQ | 167 | 173 | 2.50 | 66,480 | 41.1 | 305 | 1.30 |
| **Qwen3-8B** |  |  |  |  |  |  |  |
| FP16 | 175 | 175 | 7.64 ×2 | 57,008 | 25.7 | 196 | 1.70 |
| FP16, 8-bit KV | 175 | – | 7.64 ×2 | 114,032 | 22.0 | 198 | 1.96 |
| INT8 | 175 | 175 | 8.80 | 9,440 | 6.1 | 532 | 6.72 |
| NF4 | 174 | 174 | 5.67 | 43,264 | 13.3 | 2,907 | 5.86 |
| GPTQ | 175 | 175 | 5.71 | 42,976 | 38.5 | 407 | 1.44 |
| AWQ | 175 | 175 | 5.71 | 42,976 | 24.2 | 554 | 2.20 |
| **Llama-3.2-1B-Instruct** |  |  |  |  |  |  |  |
| FP16 | 42 | 63 | 2.32 | 312,560 | 79.9 | 47 | 0.57 |
| FP16, 8-bit KV | 39 | – | 2.32 | 625,136 | 66.9 | 57 | 0.65 |
| INT8 | 42 | 57 | 1.41 | 341,840 | 17.4 | 144 | 3.11 |
| NF4 | 32 | 30 | 0.98 | 356,480 | 67.0 | 430 | 2.01 |
| GPTQ | 16 | 13 | 0.98 | 356,304 | 148.6 | 61 | 0.56 |
| AWQ | 75 | 64 | 0.98 | 356,368 | 108.2 | 88 | 2.85 |
| **Llama-3.2-3B-Instruct** |  |  |  |  |  |  |  |
| FP16 | 79 | 142 | 6.02 | 54,528 | 34.2 | 158 | 1.36 |
| FP16, 8-bit KV | 78 | – | 6.02 | 109,072 | 28.8 | 154 | 1.48 |
| INT8 | 85 | 141 | 3.42 | 78,720 | 10.2 | 252 | 4.17 |
| NF4 | 121 | 139 | 2.13 | 90,944 | 28.5 | 1,358 | 2.97 |
| GPTQ | 129 | 119 | 2.14 | 90,832 | 76.6 | 191 | 0.85 |
| AWQ | 83 | 139 | 2.14 | 90,848 | 43.4 | 291 | 1.23 |

Memory is read from vLLM's log: the weights per GPU, and how many tokens of keys and values fit in the memory left after them. Speed is for free decoding at batch size 1, as the median over 35 questions; tokens/s is the speed of writing the answer. ×2 = the model is split over two T4s (Qwen3-8B with FP16 weights); every other config runs on one. Speed differences of about 20% or less are within the noise of Kaggle's T4s. Speed in constrained decoding: [Speed and memory](speed_memory.md).

## In numbers

- **Significant changes from FP16:** 11 of 20 in free decoding, 8 of 20 in constrained decoding, and 1 of 5 with the 8-bit KV cache.
- **The constraint raises 24 of the 25 runs** (all but Llama-3.2-1B-Instruct GPTQ).
- **Format errors** (syntactic failures): 6,844 of 11,625 failures (58.9%) in free decoding, 53 of 8,955 (0.6%) in constrained decoding.
- **Stress suite:** 7 of the 280 group tests against FP16 are significant; with the 8-bit KV cache, 0 of 35.
- **BFCL reproduced on Kaggle** within 0.13 points (prompt mode) and 0.07 points (FC mode) of the leaderboard.

## More tables

| Page | What it has |
|---|---|
| [BFCL by category](bfcl_by_category.md) | accuracy in each BFCL category, for every run |
| [Significance tests](significance.md) | every change from FP16 with its 95% interval and p-value, and how much of each change the constraint removes |
| [Stress suite](stress_suite.md) | correct answers in each of the 7 stress groups, for every run, and the 7 significant changes |
| [Failure classes](failure_classes.md) | every failed answer, sorted into 2 syntactic and 5 semantic classes |
| [Speed and memory](speed_memory.md) | memory, KV cache size and speed of all 30 configs, in both decoding modes |
| [Setup checks](setup_checks.md) | BFCL reproduced against its leaderboard, and the other checks of the method |
| [`final_table.csv`](final_table.csv) | the two tables above as data, with p-values and the speed in constrained decoding |

What the numbers mean: [`findings/`](../findings/README.md). Answers to the four research questions: [`research_questions/`](../research_questions/README.md). To build this folder again: `python src/final_table.py`.
