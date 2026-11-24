# Failure Classes

[Back to the final table](README.md)

Every failed answer of the 50 main BFCL runs (25 free, 25 constrained), sorted into one class by [`src/taxonomy.py`](../src/taxonomy.py) (checked by hand on 80 failures: 80 agree). The checks run in this order, so an answer with a format error is syntactic even if its content is also wrong.

- **Syntactic** (the format is wrong): **S1** not valid JSON; **S2** valid JSON that does not match the schema (wrong keys, a function that is not in the list, an extra or missing argument, a wrong type, or a value outside an enum)
- **Semantic** (the format is right, the content is wrong): **M1** wrong function; **M2** wrong argument value; **M3** wrong number of calls; **M4** a call on an irrelevance question; **M5** no call when one was needed

An irrelevance answer that does not parse makes no call, so BFCL counts it as correct; these answers (576 in the free runs, 40 in the constrained runs) are not failures.

## Free decoding

| Model / precision | Failures | S1 | S2 | M1 | M2 | M3 | M4 | M5 | Syntactic<br>(S1+S2) | Semantic<br>(M1-M5) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3-1.7B** |  |  |  |  |  |  |  |  |  |  |
| FP16 | 324 | 121 | 24 | 1 | 95 | 32 | 50 | 1 | 145 | 179 |
| INT8 | 356 | 151 | 20 | 2 | 95 | 32 | 56 | 0 | 171 | 185 |
| NF4 | 348 | 106 | 18 | 3 | 120 | 50 | 47 | 4 | 124 | 224 |
| GPTQ | 660 | 393 | 25 | 1 | 109 | 44 | 87 | 1 | 418 | 242 |
| AWQ | 301 | 55 | 18 | 5 | 114 | 39 | 69 | 1 | 73 | 228 |
| **Qwen3-4B** |  |  |  |  |  |  |  |  |  |  |
| FP16 | 155 | 2 | 8 | 0 | 91 | 18 | 35 | 1 | 10 | 145 |
| INT8 | 150 | 3 | 8 | 0 | 88 | 18 | 32 | 1 | 11 | 139 |
| NF4 | 184 | 5 | 8 | 1 | 117 | 15 | 37 | 1 | 13 | 171 |
| GPTQ | 167 | 5 | 9 | 1 | 102 | 15 | 31 | 4 | 14 | 153 |
| AWQ | 167 | 7 | 9 | 0 | 92 | 19 | 39 | 1 | 16 | 151 |
| **Qwen3-8B** |  |  |  |  |  |  |  |  |  |  |
| FP16 | 146 | 0 | 11 | 0 | 83 | 18 | 29 | 5 | 11 | 135 |
| INT8 | 155 | 2 | 12 | 1 | 89 | 17 | 29 | 5 | 14 | 141 |
| NF4 | 155 | 2 | 11 | 1 | 88 | 16 | 35 | 2 | 13 | 142 |
| GPTQ | 160 | 3 | 14 | 0 | 88 | 19 | 23 | 13 | 17 | 143 |
| AWQ | 146 | 3 | 11 | 1 | 82 | 20 | 28 | 1 | 14 | 132 |
| **Llama-3.2-1B-Instruct** |  |  |  |  |  |  |  |  |  |  |
| FP16 | 763 | 69 | 266 | 2 | 79 | 39 | 20 | 288 | 335 | 428 |
| INT8 | 772 | 81 | 290 | 3 | 91 | 32 | 25 | 250 | 371 | 401 |
| NF4 | 961 | 225 | 361 | 1 | 46 | 19 | 23 | 286 | 586 | 375 |
| GPTQ | 1,038 | 470 | 346 | 0 | 28 | 28 | 6 | 160 | 816 | 222 |
| AWQ | 935 | 353 | 441 | 2 | 46 | 22 | 27 | 44 | 794 | 141 |
| **Llama-3.2-3B-Instruct** |  |  |  |  |  |  |  |  |  |  |
| FP16 | 840 | 519 | 238 | 1 | 46 | 15 | 10 | 11 | 757 | 83 |
| INT8 | 827 | 475 | 261 | 2 | 53 | 17 | 14 | 5 | 736 | 91 |
| NF4 | 651 | 99 | 352 | 2 | 91 | 25 | 77 | 5 | 451 | 200 |
| GPTQ | 536 | 107 | 209 | 1 | 110 | 25 | 75 | 9 | 316 | 220 |
| AWQ | 728 | 364 | 254 | 3 | 55 | 20 | 23 | 9 | 618 | 110 |
| **All 25 runs** | 11,625 | 3,620 | 3,224 | 34 | 2,098 | 614 | 927 | 1,108 | 6,844 | 4,781 |

Syntactic: 6,844 of 11,625 failures (58.9%).

## Constrained decoding

| Model / precision | Failures | S1 | S2 | M1 | M2 | M3 | M4 | M5 | Syntactic<br>(S1+S2) | Semantic<br>(M1-M5) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3-1.7B** |  |  |  |  |  |  |  |  |  |  |
| FP16 | 222 | 4 | 0 | 2 | 117 | 44 | 54 | 1 | 4 | 218 |
| INT8 | 235 | 2 | 0 | 2 | 120 | 44 | 66 | 1 | 2 | 233 |
| NF4 | 260 | 1 | 0 | 5 | 144 | 57 | 46 | 7 | 1 | 259 |
| GPTQ | 396 | 1 | 0 | 2 | 161 | 125 | 105 | 2 | 1 | 395 |
| AWQ | 259 | 1 | 0 | 5 | 128 | 51 | 73 | 1 | 1 | 258 |
| **Qwen3-4B** |  |  |  |  |  |  |  |  |  |  |
| FP16 | 153 | 2 | 0 | 0 | 94 | 19 | 37 | 1 | 2 | 151 |
| INT8 | 144 | 1 | 0 | 0 | 94 | 16 | 32 | 1 | 1 | 143 |
| NF4 | 180 | 1 | 0 | 1 | 121 | 17 | 39 | 1 | 1 | 179 |
| GPTQ | 163 | 1 | 0 | 1 | 109 | 15 | 33 | 4 | 1 | 162 |
| AWQ | 160 | 4 | 0 | 0 | 96 | 19 | 40 | 1 | 4 | 156 |
| **Qwen3-8B** |  |  |  |  |  |  |  |  |  |  |
| FP16 | 143 | 0 | 0 | 0 | 89 | 18 | 31 | 5 | 0 | 143 |
| INT8 | 145 | 0 | 0 | 1 | 90 | 19 | 30 | 5 | 0 | 145 |
| NF4 | 153 | 0 | 0 | 1 | 96 | 17 | 37 | 2 | 0 | 153 |
| GPTQ | 153 | 0 | 0 | 0 | 96 | 19 | 25 | 13 | 0 | 153 |
| AWQ | 142 | 1 | 0 | 1 | 89 | 20 | 30 | 1 | 1 | 141 |
| **Llama-3.2-1B-Instruct** |  |  |  |  |  |  |  |  |  |  |
| FP16 | 631 | 8 | 0 | 9 | 146 | 86 | 38 | 344 | 8 | 623 |
| INT8 | 679 | 11 | 0 | 10 | 142 | 78 | 46 | 392 | 11 | 668 |
| NF4 | 994 | 2 | 0 | 6 | 72 | 41 | 55 | 818 | 2 | 992 |
| GPTQ | 1,127 | 0 | 0 | 1 | 6 | 7 | 1 | 1,112 | 0 | 1,127 |
| AWQ | 653 | 10 | 0 | 7 | 184 | 117 | 118 | 217 | 10 | 643 |
| **Llama-3.2-3B-Instruct** |  |  |  |  |  |  |  |  |  |  |
| FP16 | 339 | 1 | 0 | 2 | 115 | 97 | 61 | 63 | 1 | 338 |
| INT8 | 364 | 0 | 0 | 2 | 110 | 102 | 71 | 79 | 0 | 364 |
| NF4 | 458 | 0 | 0 | 3 | 105 | 169 | 147 | 34 | 0 | 458 |
| GPTQ | 438 | 1 | 0 | 1 | 115 | 161 | 84 | 76 | 1 | 437 |
| AWQ | 364 | 1 | 0 | 3 | 134 | 106 | 73 | 47 | 1 | 363 |
| **All 25 runs** | 8,955 | 53 | 0 | 65 | 2,773 | 1,464 | 1,372 | 3,228 | 53 | 8,902 |

Syntactic: 53 of 8,955 failures (0.6%).

## 8-bit KV cache (FP16 weights, free decoding)

Failed answers at FP16 and with the 8-bit KV cache, and the classes of the answers that were right at FP16 and are wrong with the 8-bit KV cache.

| Model | Failures<br>at FP16 | Failures with<br>8-bit KV | Newly wrong:<br>S1 | Newly wrong:<br>S2 | Newly wrong:<br>M1-M5 |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 324 | 352 | 51 | 1 | 6 |
| Qwen3-4B | 155 | 161 | 2 | 1 | 7 |
| Qwen3-8B | 146 | 144 | 0 | 0 | 3 |
| Llama-3.2-1B-Instruct | 763 | 775 | 3 | 9 | 18 |
| Llama-3.2-3B-Instruct | 840 | 826 | 13 | 7 | 5 |

---

Built by [`src/final_table.py`](../src/final_table.py) from [`taxonomy_free.csv`](../results/taxonomy_free.csv), [`taxonomy_constrained.csv`](../results/taxonomy_constrained.csv), [`kv_cache.csv`](../results/kv_cache.csv).
