# Latency at Batch Size 1

Each config was loaded on its own and answered the same 35 BFCL questions (5 per non-live category, seed 42) one at a time, after 3 untimed warm-up questions, first in free and then in constrained decoding (`src/latency.py`, notebook 08). Same prompt, greedy decoding, checkpoints and vLLM 0.8.5 as the main runs; one T4 per config, two for Qwen3-8B in FP16. Values are medians over the 35 questions.

- **First token:** from the request's arrival to its first answer token (vLLM's timing). It includes reading the prompt, and in constrained decoding also building the grammar for the question's schema.
- **Tokens/s:** answer tokens after the first one, divided by the time they took (1,000 / the median ms per token). This is the speed of writing the answer, independent of its length.
- **Whole answer:** a stopwatch around the call, i.e. what a caller waits for. Models answer at different lengths, so this mixes speed and length; the mean answer length is in `results/latency.csv`.
- **KV cache tokens:** how many tokens of keys and values vLLM could keep in the memory left after the weights (90% of the GPU, 81% for Qwen3-8B INT8), from its log: blocks × 16.
- **Llama-3.2-3B in FP16 was timed twice.** The first try (16 Nov) took about 98 ms per answer token on every answer longer than two tokens: three times what its size and its own 8-bit-KV run (34 ms) suggest, while giving the same 70 answers as the second try. The table uses the second try (21 Nov, same notebook and code). In that try the GPU also slowed down during the run (28 to 64 ms per token in free decoding, 40 to 74 ms in constrained decoding), so its constrained values are not reliable. The first try and a repeat of the 8-bit-KV run (36 ms per token, against 34 ms) are in `results/latency/extra/`.
- Where most answers are an empty list (Llama-3.2-1B NF4 and GPTQ in constrained decoding: 28 and 6 tokens on average), tokens/s rests on answers of one or two steps and is below the model's real writing speed.
- In every other run the speed stayed steady from the first to the last question. Still, one run per config cannot rule out a GPU that was slower for a whole run (Kaggle's T4s can change speed during a session), so differences of about 20% or less are not meaningful.

## Free decoding

### Tokens per second while writing the answer

| Model | FP16 | FP16, 8-bit KV | INT8 | NF4 | GPTQ | AWQ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 52.9 | 40.6 | 7.3 | 47.4 | 101.4 | 80.3 |
| Qwen3-4B | 25.2 | 21.2 | 6.1 | 21.6 | 58.9 | 41.1 |
| Qwen3-8B | 25.7 | 22.0 | 6.1 | 13.3 | 38.5 | 24.2 |
| Llama-3.2-1B-Instruct | 79.9 | 66.9 | 17.4 | 67.0 | 148.6 | 108.2 |
| Llama-3.2-3B-Instruct | 34.2 | 28.8 | 10.2 | 28.5 | 76.6 | 43.4 |

### Time to the first token (ms)

| Model | FP16 | FP16, 8-bit KV | INT8 | NF4 | GPTQ | AWQ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 77 | 83 | 354 | 567 | 86 | 106 |
| Qwen3-4B | 181 | 194 | 394 | 1512 | 220 | 305 |
| Qwen3-8B | 196 | 198 | 532 | 2907 | 407 | 554 |
| Llama-3.2-1B-Instruct | 47 | 57 | 144 | 430 | 61 | 88 |
| Llama-3.2-3B-Instruct | 158 | 154 | 252 | 1358 | 191 | 291 |

### Time for the whole answer (s)

| Model | FP16 | FP16, 8-bit KV | INT8 | NF4 | GPTQ | AWQ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 0.80 | 1.00 | 5.46 | 1.50 | 0.48 | 0.58 |
| Qwen3-4B | 1.80 | 2.15 | 7.03 | 3.46 | 0.90 | 1.30 |
| Qwen3-8B | 1.70 | 1.96 | 6.72 | 5.86 | 1.44 | 2.20 |
| Llama-3.2-1B-Instruct | 0.57 | 0.65 | 3.11 | 2.01 | 0.56 | 2.85 |
| Llama-3.2-3B-Instruct | 1.36 | 1.48 | 4.17 | 2.97 | 0.85 | 1.23 |

## Constrained decoding

### Tokens per second while writing the answer

| Model | FP16 | FP16, 8-bit KV | INT8 | NF4 | GPTQ | AWQ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 49.6 | 41.3 | 7.2 | 46.3 | 100.4 | 79.5 |
| Qwen3-4B | 24.9 | 21.4 | 6.1 | 21.6 | 59.5 | 41.9 |
| Qwen3-8B | 25.6 | 22.0 | 6.1 | 13.3 | 38.8 | 24.1 |
| Llama-3.2-1B-Instruct | 79.6 | 69.5 | 16.5 | 32.6 | 46.1 | 113.1 |
| Llama-3.2-3B-Instruct | 22.5 | 28.7 | 10.1 | 28.4 | 77.6 | 46.8 |

### Time to the first token (ms)

| Model | FP16 | FP16, 8-bit KV | INT8 | NF4 | GPTQ | AWQ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 583 | 600 | 873 | 839 | 604 | 604 |
| Qwen3-4B | 614 | 610 | 903 | 1754 | 597 | 600 |
| Qwen3-8B | 656 | 642 | 993 | 3150 | 648 | 791 |
| Llama-3.2-1B-Instruct | 519 | 492 | 597 | 653 | 503 | 515 |
| Llama-3.2-3B-Instruct | 512 | 519 | 701 | 1567 | 527 | 534 |

### Time for the whole answer (s)

| Model | FP16 | FP16, 8-bit KV | INT8 | NF4 | GPTQ | AWQ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 1.41 | 1.53 | 6.25 | 1.69 | 1.03 | 1.13 |
| Qwen3-4B | 2.21 | 2.48 | 7.49 | 3.68 | 1.28 | 1.67 |
| Qwen3-8B | 2.22 | 2.44 | 7.42 | 6.09 | 1.73 | 2.42 |
| Llama-3.2-1B-Instruct | 0.95 | 1.00 | 2.54 | 0.70 | 0.53 | 0.82 |
| Llama-3.2-3B-Instruct | 2.64 | 1.89 | 4.51 | 2.94 | 0.97 | 1.33 |

## Memory (from vLLM's log)

### Weights per GPU (GiB)

| Model | FP16 | FP16, 8-bit KV | INT8 | NF4 | GPTQ | AWQ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 3.22 | 3.22 | 1.90 | 1.27 | 1.27 | 1.27 |
| Qwen3-4B | 7.56 | 7.56 | 4.18 | 2.49 | 2.50 | 2.50 |
| Qwen3-8B | 7.64 | 7.64 | 8.80 | 5.67 | 5.71 | 5.71 |
| Llama-3.2-1B-Instruct | 2.32 | 2.32 | 1.41 | 0.98 | 0.98 | 0.98 |
| Llama-3.2-3B-Instruct | 6.02 | 6.02 | 3.42 | 2.13 | 2.14 | 2.14 |

### KV cache tokens

| Model | FP16 | FP16, 8-bit KV | INT8 | NF4 | GPTQ | AWQ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 78,896 | 157,792 | 91,040 | 97,120 | 97,056 | 97,088 |
| Qwen3-4B | 29,696 | 59,408 | 54,144 | 66,560 | 66,464 | 66,480 |
| Qwen3-8B | 57,008 | 114,032 | 9,440 | 43,264 | 42,976 | 42,976 |
| Llama-3.2-1B-Instruct | 312,560 | 625,136 | 341,840 | 356,480 | 356,304 | 356,368 |
| Llama-3.2-3B-Instruct | 54,528 | 109,072 | 78,720 | 90,944 | 90,832 | 90,848 |

Built from `results/latency/` by `src/latency_table.py`; all values per config and mode, with the 90th percentiles and mean answer lengths, are in [`latency.csv`](latency.csv).
