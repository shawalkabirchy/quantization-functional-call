# Schema-Stress Suite

[Back to the final table](README.md)

Correct answers of every run in each group of our schema-stress suite: 7 groups of 25 questions. Six groups each stress one feature of a function's schema; the plain group is a control without any of them. Full description: [`data/stress_suite/`](../data/stress_suite/README.md).

| Group | What it stresses |
|---|---|
| Plain (control) | nothing: a flat function with 2-3 required strings and integers |
| Nested | objects as arguments: one object, a list of objects, 2 or 3 levels deep |
| Enum | a code from a fixed list of 4, 12 or 30 codes |
| Number | integers (with thousands separators, in words, negative) and floats |
| Optional | 2, 4 or 8 optional arguments with defaults |
| Unicode | strings to copy exactly: accented Latin, non-Latin scripts, emoji |
| Menu | choosing 1 of 50 functions with near-duplicate names |

## Free decoding

| Model / precision | Plain (control) | Nested | Enum | Number | Optional | Unicode | Menu | Total<br>(of 175) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3-1.7B** |  |  |  |  |  |  |  |  |
| FP16 | 25 | 24 | 22 | 9 | 25 | 24 | 25 | 154 |
| FP16, 8-bit KV | 22 | 22 | 23 | 8 | 24 | 23 | 25 | 147 |
| INT8 | 25 | 20 | 23 | 13 | 25 | 24 | 25 | 155 |
| NF4 | 25 | 25 | 23 | 19 | 23 | 23 | 22 | 160 |
| GPTQ | 25 | 25 | 21 | 15 | 25 | 23 | 24 | 158 |
| AWQ | 25 | 24 | 21 | 25 | 25 | 23 | 25 | 168 |
| **Qwen3-4B** |  |  |  |  |  |  |  |  |
| FP16 | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
| FP16, 8-bit KV | 25 | 24 | 25 | 25 | 25 | 25 | 25 | 174 |
| INT8 | 25 | 24 | 24 | 25 | 25 | 25 | 25 | 173 |
| NF4 | 25 | 19 | 25 | 25 | 25 | 24 | 24 | 167 |
| GPTQ | 25 | 19 | 25 | 25 | 25 | 25 | 24 | 168 |
| AWQ | 25 | 19 | 24 | 25 | 25 | 24 | 25 | 167 |
| **Qwen3-8B** |  |  |  |  |  |  |  |  |
| FP16 | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
| FP16, 8-bit KV | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
| INT8 | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
| NF4 | 25 | 25 | 24 | 25 | 25 | 25 | 25 | 174 |
| GPTQ | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
| AWQ | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
| **Llama-3.2-1B-Instruct** |  |  |  |  |  |  |  |  |
| FP16 | 2 | 11 | 5 | 3 | 14 | 4 | 3 | 42 |
| FP16, 8-bit KV | 2 | 11 | 5 | 3 | 12 | 4 | 2 | 39 |
| INT8 | 4 | 15 | 2 | 1 | 15 | 2 | 3 | 42 |
| NF4 | 2 | 4 | 2 | 2 | 19 | 1 | 2 | 32 |
| GPTQ | 6 | 0 | 0 | 0 | 8 | 2 | 0 | 16 |
| AWQ | 16 | 18 | 3 | 11 | 20 | 7 | 0 | 75 |
| **Llama-3.2-3B-Instruct** |  |  |  |  |  |  |  |  |
| FP16 | 16 | 12 | 11 | 9 | 7 | 5 | 19 | 79 |
| FP16, 8-bit KV | 16 | 11 | 11 | 9 | 8 | 4 | 19 | 78 |
| INT8 | 19 | 17 | 8 | 9 | 9 | 2 | 21 | 85 |
| NF4 | 20 | 22 | 22 | 11 | 20 | 1 | 25 | 121 |
| GPTQ | 25 | 24 | 23 | 5 | 23 | 7 | 22 | 129 |
| AWQ | 16 | 14 | 15 | 5 | 11 | 4 | 18 | 83 |

## Constrained decoding

| Model / precision | Plain (control) | Nested | Enum | Number | Optional | Unicode | Menu | Total<br>(of 175) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3-1.7B** |  |  |  |  |  |  |  |  |
| FP16 | 25 | 25 | 22 | 25 | 25 | 24 | 25 | 171 |
| INT8 | 25 | 25 | 23 | 25 | 25 | 24 | 25 | 172 |
| NF4 | 25 | 25 | 23 | 25 | 25 | 23 | 23 | 169 |
| GPTQ | 25 | 25 | 21 | 25 | 25 | 24 | 24 | 169 |
| AWQ | 25 | 25 | 23 | 25 | 25 | 23 | 25 | 171 |
| **Qwen3-4B** |  |  |  |  |  |  |  |  |
| FP16 | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
| INT8 | 25 | 25 | 24 | 25 | 25 | 25 | 25 | 174 |
| NF4 | 25 | 25 | 25 | 25 | 25 | 24 | 25 | 174 |
| GPTQ | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
| AWQ | 25 | 25 | 24 | 25 | 25 | 24 | 25 | 173 |
| **Qwen3-8B** |  |  |  |  |  |  |  |  |
| FP16 | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
| INT8 | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
| NF4 | 25 | 25 | 24 | 25 | 25 | 25 | 25 | 174 |
| GPTQ | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
| AWQ | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 175 |
| **Llama-3.2-1B-Instruct** |  |  |  |  |  |  |  |  |
| FP16 | 2 | 20 | 8 | 9 | 12 | 6 | 6 | 63 |
| INT8 | 4 | 23 | 9 | 3 | 8 | 7 | 3 | 57 |
| NF4 | 0 | 0 | 2 | 0 | 3 | 0 | 25 | 30 |
| GPTQ | 0 | 0 | 0 | 0 | 1 | 0 | 12 | 13 |
| AWQ | 2 | 13 | 6 | 7 | 9 | 7 | 20 | 64 |
| **Llama-3.2-3B-Instruct** |  |  |  |  |  |  |  |  |
| FP16 | 25 | 25 | 24 | 13 | 25 | 8 | 22 | 142 |
| INT8 | 25 | 25 | 24 | 13 | 25 | 5 | 24 | 141 |
| NF4 | 25 | 21 | 25 | 15 | 25 | 8 | 20 | 139 |
| GPTQ | 21 | 24 | 20 | 12 | 25 | 7 | 10 | 119 |
| AWQ | 25 | 25 | 24 | 12 | 25 | 8 | 20 | 139 |

## Significant changes

McNemar's exact test on the 25 questions of a group, against FP16 in the same decoding mode, with Holm's correction over the 140 tests of each mode (280 in all). Of these, 7 are significant:

| Model | Precision | Mode | Group | FP16<br>(of 25) | Quantized<br>(of 25) | Change | p (Holm) |
|---|---|---|---|---:|---:|---:|---:|
| Qwen3-1.7B | AWQ | free | Number | 9 | 25 | +16 | 0.004 |
| Llama-3.2-1B-Instruct | AWQ | free | Plain (control) | 2 | 16 | +14 | 0.017 |
| Llama-3.2-3B-Instruct | GPTQ | free | Optional | 7 | 23 | +16 | 0.020 |
| Llama-3.2-1B-Instruct | NF4 | constrained | Nested | 20 | 0 | -20 | 2.67 × 10<sup>-4</sup> |
| Llama-3.2-1B-Instruct | NF4 | constrained | Menu | 6 | 25 | +19 | 5.26 × 10<sup>-4</sup> |
| Llama-3.2-1B-Instruct | GPTQ | constrained | Nested | 20 | 0 | -20 | 2.67 × 10<sup>-4</sup> |
| Llama-3.2-1B-Instruct | AWQ | constrained | Menu | 6 | 20 | +14 | 0.017 |

With the 8-bit KV cache, none of the 35 group tests (Holm's correction over the 35) is significant.

---

Built by [`src/final_table.py`](../src/final_table.py) from [`summary.csv`](../results/summary.csv), [`stress_significance.csv`](../results/stress_significance.csv), [`kv_cache_stress.csv`](../results/kv_cache_stress.csv).
