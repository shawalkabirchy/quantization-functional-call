# Score Changes

How the scores change with precision and decoding mode, in one place. **State:** after Update 12 (Wed 11 Nov 2026).

Score = BFCL's Non-Live Overall accuracy (%) on the 1,390 `non_live` questions. Change = quantized minus FP16 of the
same model in the same decoding mode, in points. **Bold** = significant (McNemar's exact test on the 1,150 AST
questions, Holm-adjusted p < 0.05 over the 20 tests of a mode). Sources: [`main_table.md`](../results/main_table.md),
[`main_table_constrained.md`](../results/main_table_constrained.md),
[`significance_free.md`](../results/significance_free.md),
[`constrained_vs_free.md`](../results/constrained_vs_free.md).

## 1. Free decoding

| Model | FP16 score | INT8 | NF4 | GPTQ | AWQ |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 72.56 | **-3.14** | -3.50 | **-33.31** | **+3.38** |
| Qwen3-4B | 88.04 | +0.25 | **-2.35** | -1.44 | -1.06 |
| Qwen3-8B | 88.29 | -1.02 | -0.29 | **-2.31** | +0.13 |
| Llama-3.2-1B-Instruct | 36.40 | -1.46 | **-20.38** | **-25.00** | **-18.59** |
| Llama-3.2-3B-Instruct | 31.38 | +2.64 | **+21.37** | **+28.14** | **+8.35** |

11 of the 20 changes are significant.

## 2. Constrained decoding

| Model | FP16 score | INT8 | NF4 | GPTQ | AWQ |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 82.46 | +0.33 | **-3.96** | **-10.81** | -1.65 |
| Qwen3-4B | 88.25 | +0.79 | **-2.13** | -1.40 | -0.77 |
| Qwen3-8B | 88.58 | -0.37 | -0.31 | -1.70 | +0.25 |
| Llama-3.2-1B-Instruct | 48.19 | **-2.69** | **-30.36** | **-45.29** | **-0.44** |
| Llama-3.2-3B-Instruct | 71.04 | -1.62 | -5.08 | **-6.33** | -2.06 |

8 of the 20 changes are significant. Llama-3.2-1B AWQ is significant although its change is small: it is right more
often than FP16 in the simple categories and less often in the others, and Overall weights the categories while the
test counts every question equally.

## 3. What the constraint does to the FP16 score

| Model | Free | Constrained | Change |
|---|---:|---:|---:|
| Qwen3-1.7B | 72.56 | 82.46 | +9.90 |
| Qwen3-4B | 88.04 | 88.25 | +0.21 |
| Qwen3-8B | 88.29 | 88.58 | +0.29 |
| Llama-3.2-1B-Instruct | 36.40 | 48.19 | +11.79 |
| Llama-3.2-3B-Instruct | 31.38 | 71.04 | +39.66 |

24 of the 25 runs score higher with the constraint. The exception is Llama-3.2-1B GPTQ (11.40 to 2.90), which then
answers with no call on 1,112 of the 1,150 AST questions.

## 4. The 7 significant free losses, with and without the constraint

| Model | Precision | Free change | Constrained change | Loss removed |
|---|---|---:|---:|---:|
| Qwen3-1.7B | INT8 | -3.14 | +0.33 | 111% |
| Qwen3-1.7B | GPTQ | -33.31 | -10.81 | 68% |
| Llama-3.2-1B-Instruct | AWQ | -18.59 | -0.44 | 98% |
| Qwen3-4B | NF4 | -2.35 | -2.13 | 9% (not different from 0) |
| Qwen3-8B | GPTQ | -2.31 | -1.70 | 26% (not different from 0) |
| Llama-3.2-1B-Instruct | NF4 | -20.38 | -30.36 | -49% (larger) |
| Llama-3.2-1B-Instruct | GPTQ | -25.00 | -45.29 | -81% (larger) |

Loss removed = (constrained change - free change) / size of the free loss. 100% = the loss is gone; below 0% = the
loss is larger. Intervals for every run: [`gap.csv`](../results/gap.csv).

## 5. Failures in all 25 runs, by class

| Mode | S1 | S2 | M1 | M2 | M3 | M4 | M5 | Total | Syntactic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Free | 3,620 | 3,224 | 34 | 2,098 | 614 | 927 | 1,108 | 11,625 | 58.9% |
| Constrained | 53 | 0 | 65 | 2,773 | 1,464 | 1,372 | 3,228 | 8,955 | 0.6% |

S1 not valid JSON, S2 does not match the schema (syntactic); M1 wrong function, M2 wrong argument value, M3 wrong
number of calls, M4 a call on an irrelevance question, M5 no call (semantic). Per run:
[`taxonomy_free.csv`](../results/taxonomy_free.csv), [`taxonomy_constrained.csv`](../results/taxonomy_constrained.csv).

## 6. Schema-stress suite: the changes that pass Holm's correction (7 of 280)

Correct answers out of 25 in the group. Holm's correction is over the 140 tests of each mode.

| Model | Precision | Mode | Group | FP16 | Quantized |
|---|---|---|---|---:|---:|
| Qwen3-1.7B | AWQ | free | Number | 9 | 25 |
| Llama-3.2-1B-Instruct | AWQ | free | Plain (control) | 2 | 16 |
| Llama-3.2-3B-Instruct | GPTQ | free | Optional | 7 | 23 |
| Llama-3.2-1B-Instruct | NF4 | constrained | Nested | 20 | 0 |
| Llama-3.2-1B-Instruct | GPTQ | constrained | Nested | 20 | 0 |
| Llama-3.2-1B-Instruct | NF4 | constrained | Menu | 6 | 25 |
| Llama-3.2-1B-Instruct | AWQ | constrained | Menu | 6 | 20 |

All counts per group and level: [`stress.md`](../results/stress.md), [`stress_accuracy.csv`](../results/stress_accuracy.csv).

## Irrelevance accuracy

Irrelevance questions (no function fits; correct = no call) are scored apart from Overall, as in BFCL. The largest
change is Llama-3.2-3B with free decoding: 91.25 at FP16, 42.08 at NF4. Part of this is format: an answer that does
not parse makes no call, so it counts as correct, and 76 of its 240 FP16 irrelevance answers do not parse. With the
missing `]` added, the two scores are 60.42 and 26.25, so NF4 does make more calls when no function fits (177 of 240
answers, against 95 at FP16) ([report](../results/int8_nf4.md)). Tables:
[`main_table.md`](../results/main_table.md), [`main_table_constrained.md`](../results/main_table_constrained.md).
