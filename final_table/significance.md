# Significance Tests

[Back to the final table](README.md)

Every change from FP16 is tested question by question: McNemar's exact test on the 1,150 AST questions (every category except irrelevance), with Holm's correction over the 20 runs of a decoding mode (over the 5 models for the 8-bit KV cache). *Right at FP16 only* and *right quantized only* count the AST questions that only one of the two runs gets right. The 95% interval of the change in Overall is a bootstrap: 10,000 resamples of the questions, drawn inside each category. **Bold** = Holm-adjusted p < 0.05.

The test and the interval can disagree. The test counts the questions whose answer changes; the interval is for the change in Overall, an average over the categories, and it is not adjusted for the number of tests. So Qwen3-8B INT8 in free decoding has an interval without 0 but no significant test, and Qwen3-1.7B with the 8-bit KV cache has a significant test but an interval with 0.

## Quantized weights, free decoding

| Model / precision | Change | 95% interval | Right at<br>FP16 only | Right<br>quantized only | p (Holm) |
|---|---:|---:|---:|---:|---:|
| **Qwen3-1.7B** |  |  |  |  |  |
| INT8 | **-3.14** | -4.88 to -1.44 | 49 | 24 | 0.046 |
| NF4 | -3.50 | -6.29 to -0.69 | 107 | 79 | 0.380 |
| GPTQ | **-33.31** | -36.65 to -30.04 | 342 | 44 | 4.97 × 10<sup>-57</sup> |
| AWQ | **+3.38** | +0.83 to +5.90 | 54 | 97 | 0.007 |
| **Qwen3-4B** |  |  |  |  |  |
| INT8 | +0.25 | -0.60 to +1.12 | 7 | 9 | 1.000 |
| NF4 | **-2.35** | -3.92 to -0.81 | 43 | 15 | 0.004 |
| GPTQ | -1.44 | -2.98 to +0.04 | 35 | 19 | 0.362 |
| AWQ | -1.06 | -2.48 to +0.29 | 24 | 16 | 1.000 |
| **Qwen3-8B** |  |  |  |  |  |
| INT8 | -1.02 | -1.94 to -0.17 | 14 | 5 | 0.445 |
| NF4 | -0.29 | -1.81 to +1.19 | 23 | 20 | 1.000 |
| GPTQ | **-2.31** | -3.71 to -0.96 | 32 | 12 | 0.040 |
| AWQ | +0.13 | -1.10 to +1.40 | 19 | 18 | 1.000 |
| **Llama-3.2-1B-Instruct** |  |  |  |  |  |
| INT8 | -1.46 | -3.88 to +0.94 | 82 | 83 | 1.000 |
| NF4 | **-20.38** | -23.65 to -17.15 | 250 | 75 | 6.17 × 10<sup>-22</sup> |
| GPTQ | **-25.00** | -28.48 to -21.69 | 327 | 45 | 1.01 × 10<sup>-52</sup> |
| AWQ | **-18.59** | -21.92 to -15.25 | 257 | 109 | 9.25 × 10<sup>-14</sup> |
| **Llama-3.2-3B-Instruct** |  |  |  |  |  |
| INT8 | +2.64 | +0.48 to +4.92 | 60 | 81 | 0.551 |
| NF4 | **+21.37** | +18.04 to +24.81 | 39 | 346 | 2.47 × 10<sup>-61</sup> |
| GPTQ | **+28.14** | +24.29 to +32.00 | 63 | 459 | 5.63 × 10<sup>-74</sup> |
| AWQ | **+8.35** | +5.17 to +11.52 | 66 | 200 | 1.12 × 10<sup>-15</sup> |

## Quantized weights, constrained decoding

| Model / precision | Change | 95% interval | Right at<br>FP16 only | Right<br>quantized only | p (Holm) |
|---|---:|---:|---:|---:|---:|
| **Qwen3-1.7B** |  |  |  |  |  |
| INT8 | +0.33 | -0.73 to +1.40 | 14 | 13 | 1.000 |
| NF4 | **-3.96** | -6.02 to -1.83 | 74 | 28 | 9.45 × 10<sup>-5</sup> |
| GPTQ | **-10.81** | -13.33 to -8.27 | 150 | 27 | 1.29 × 10<sup>-20</sup> |
| AWQ | -1.65 | -3.56 to +0.21 | 49 | 31 | 0.567 |
| **Qwen3-4B** |  |  |  |  |  |
| INT8 | +0.79 | -0.08 to +1.73 | 6 | 10 | 1.000 |
| NF4 | **-2.13** | -3.65 to -0.62 | 40 | 15 | 0.014 |
| GPTQ | -1.40 | -2.87 to +0.04 | 31 | 17 | 0.567 |
| AWQ | -0.77 | -2.21 to +0.62 | 21 | 17 | 1.000 |
| **Qwen3-8B** |  |  |  |  |  |
| INT8 | -0.37 | -1.15 to +0.35 | 8 | 5 | 1.000 |
| NF4 | -0.31 | -1.81 to +1.21 | 24 | 20 | 1.000 |
| GPTQ | -1.70 | -3.10 to -0.35 | 30 | 14 | 0.249 |
| AWQ | +0.25 | -0.96 to +1.50 | 18 | 18 | 1.000 |
| **Llama-3.2-1B-Instruct** |  |  |  |  |  |
| INT8 | **-2.69** | -4.98 to -0.38 | 100 | 60 | 0.025 |
| NF4 | **-30.36** | -33.65 to -27.19 | 393 | 47 | 8.03 × 10<sup>-68</sup> |
| GPTQ | **-45.29** | -48.69 to -42.00 | 538 | 5 | 5.42 × 10<sup>-151</sup> |
| AWQ | **-0.44** | -3.75 to +2.92 | 120 | 178 | 0.014 |
| **Llama-3.2-3B-Instruct** |  |  |  |  |  |
| INT8 | -1.62 | -3.54 to +0.31 | 43 | 28 | 0.767 |
| NF4 | -5.08 | -7.63 to -2.52 | 89 | 56 | 0.092 |
| GPTQ | **-6.33** | -9.27 to -3.37 | 137 | 61 | 1.17 × 10<sup>-6</sup> |
| AWQ | -2.06 | -4.62 to +0.54 | 73 | 60 | 1.000 |

## 8-bit KV cache (FP16 weights, free decoding)

| Model | Change | 95% interval | Right at<br>FP16 only | Right with<br>8-bit KV only | p (Holm) |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | **-0.73** | -2.48 to +1.02 | 58 | 28 | 0.008 |
| Qwen3-4B | -0.75 | -1.75 to +0.21 | 10 | 4 | 0.432 |
| Qwen3-8B | +0.29 | -0.33 to +1.00 | 3 | 5 | 0.727 |
| Llama-3.2-1B-Instruct | -1.42 | -2.96 to +0.12 | 29 | 18 | 0.432 |
| Llama-3.2-3B-Instruct | +0.02 | -1.52 to +1.56 | 21 | 36 | 0.251 |

## How much of the free change the constraint removes

*Difference* = constrained change minus free change, with its bootstrap 95% interval. *Loss removed* = the part of a significant free loss that is gone under the constraint (100% = all of it; below 0% = the loss grows). It is given for the 7 significant free losses only.

| Model / precision | Free change | Constrained<br>change | Difference | 95% interval | Loss removed |
|---|---:|---:|---:|---:|---:|
| **Qwen3-1.7B** |  |  |  |  |  |
| INT8 | **-3.14** | +0.33 | +3.47 | +1.71 to +5.23 | 111% |
| NF4 | -3.50 | -3.96 | -0.46 | -2.77 to +1.85 | – |
| GPTQ | **-33.31** | -10.81 | +22.50 | +19.69 to +25.37 | 68% |
| AWQ | **+3.38** | -1.65 | -5.03 | -6.92 to -3.15 | – |
| **Qwen3-4B** |  |  |  |  |  |
| INT8 | +0.25 | +0.79 | +0.54 | -0.48 to +1.58 | – |
| NF4 | **-2.35** | -2.13 | +0.22 | -0.35 to +0.83 | 9% |
| GPTQ | -1.44 | -1.40 | +0.04 | -0.52 to +0.58 | – |
| AWQ | -1.06 | -0.77 | +0.29 | -0.17 to +0.83 | – |
| **Qwen3-8B** |  |  |  |  |  |
| INT8 | -1.02 | -0.37 | +0.65 | -0.21 to +1.52 | – |
| NF4 | -0.29 | -0.31 | -0.02 | -0.77 to +0.73 | – |
| GPTQ | **-2.31** | -1.70 | +0.61 | -0.08 to +1.35 | 26% |
| AWQ | +0.13 | +0.25 | +0.12 | -0.04 to +0.42 | – |
| **Llama-3.2-1B-Instruct** |  |  |  |  |  |
| INT8 | -1.46 | -2.69 | -1.23 | -4.08 to +1.69 | – |
| NF4 | **-20.38** | -30.36 | -9.98 | -13.46 to -6.48 | -49% |
| GPTQ | **-25.00** | -45.29 | -20.29 | -23.65 to -17.04 | -81% |
| AWQ | **-18.59** | -0.44 | +18.15 | +14.10 to +22.17 | 98% |
| **Llama-3.2-3B-Instruct** |  |  |  |  |  |
| INT8 | +2.64 | -1.62 | -4.26 | -7.15 to -1.48 | – |
| NF4 | **+21.37** | -5.08 | -26.45 | -30.46 to -22.54 | – |
| GPTQ | **+28.14** | -6.33 | -34.47 | -38.96 to -29.81 | – |
| AWQ | **+8.35** | -2.06 | -10.41 | -14.37 to -6.52 | – |

---

Built by [`src/final_table.py`](../src/final_table.py) from [`significance_free.csv`](../results/significance_free.csv), [`significance_constrained.csv`](../results/significance_constrained.csv), [`kv_cache.csv`](../results/kv_cache.csv), [`gap.csv`](../results/gap.csv).
