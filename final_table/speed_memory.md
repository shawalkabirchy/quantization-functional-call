# Speed and Memory

[Back to the final table](README.md)

Each config was loaded on its own and answered the same 35 BFCL questions (5 per category, seed 42) one at a time (batch size 1), after 3 untimed warm-up questions, first in free and then in constrained decoding ([`src/latency.py`](../src/latency.py); notebooks [`08_kv_latency_run_A`](../notebooks/08_kv_latency_run_A.ipynb) and [`_B`](../notebooks/08_kv_latency_run_B.ipynb)). Prompts, checkpoints and vLLM 0.8.5 are those of the accuracy runs. One T4 per config, two for Qwen3-8B with FP16 weights. Values are medians over the 35 questions.

- **Weights per GPU** and **KV cache tokens** are read from vLLM's log. KV cache tokens = how many tokens of keys and values fit in the memory left after the weights (90% of the GPU; 81% for Qwen3-8B INT8).
- **Tokens/s** = the speed of writing the answer: answer tokens after the first one, per second.
- **First token** = from the request's arrival to the first answer token. It includes reading the prompt, and in constrained decoding also building the grammar for the question's schema.
- **Whole answer** = what a caller waits for. It mixes speed and answer length.

## Memory

| Model / precision | GPUs | Weights<br>(GiB per GPU) | KV cache<br>tokens |
|---|---:|---:|---:|
| **Qwen3-1.7B** |  |  |  |
| FP16 | 1 | 3.22 | 78,896 |
| FP16, 8-bit KV | 1 | 3.22 | 157,792 |
| INT8 | 1 | 1.90 | 91,040 |
| NF4 | 1 | 1.27 | 97,120 |
| GPTQ | 1 | 1.27 | 97,056 |
| AWQ | 1 | 1.27 | 97,088 |
| **Qwen3-4B** |  |  |  |
| FP16 | 1 | 7.56 | 29,696 |
| FP16, 8-bit KV | 1 | 7.56 | 59,408 |
| INT8 | 1 | 4.18 | 54,144 |
| NF4 | 1 | 2.49 | 66,560 |
| GPTQ | 1 | 2.50 | 66,464 |
| AWQ | 1 | 2.50 | 66,480 |
| **Qwen3-8B** |  |  |  |
| FP16 | 2 | 7.64 | 57,008 |
| FP16, 8-bit KV | 2 | 7.64 | 114,032 |
| INT8 | 1 | 8.80 | 9,440 |
| NF4 | 1 | 5.67 | 43,264 |
| GPTQ | 1 | 5.71 | 42,976 |
| AWQ | 1 | 5.71 | 42,976 |
| **Llama-3.2-1B-Instruct** |  |  |  |
| FP16 | 1 | 2.32 | 312,560 |
| FP16, 8-bit KV | 1 | 2.32 | 625,136 |
| INT8 | 1 | 1.41 | 341,840 |
| NF4 | 1 | 0.98 | 356,480 |
| GPTQ | 1 | 0.98 | 356,304 |
| AWQ | 1 | 0.98 | 356,368 |
| **Llama-3.2-3B-Instruct** |  |  |  |
| FP16 | 1 | 6.02 | 54,528 |
| FP16, 8-bit KV | 1 | 6.02 | 109,072 |
| INT8 | 1 | 3.42 | 78,720 |
| NF4 | 1 | 2.13 | 90,944 |
| GPTQ | 1 | 2.14 | 90,832 |
| AWQ | 1 | 2.14 | 90,848 |

## Speed at batch size 1

| Model / precision | Free<br>tokens/s | Free<br>first token (ms) | Free<br>whole answer (s) | Constrained<br>tokens/s | Constrained<br>first token (ms) | Constrained<br>whole answer (s) |
|---|---:|---:|---:|---:|---:|---:|
| **Qwen3-1.7B** |  |  |  |  |  |  |
| FP16 | 52.9 | 77 | 0.80 | 49.6 | 583 | 1.41 |
| FP16, 8-bit KV | 40.6 | 83 | 1.00 | 41.3 | 600 | 1.53 |
| INT8 | 7.3 | 354 | 5.46 | 7.2 | 873 | 6.25 |
| NF4 | 47.4 | 567 | 1.50 | 46.3 | 839 | 1.69 |
| GPTQ | 101.4 | 86 | 0.48 | 100.4 | 604 | 1.03 |
| AWQ | 80.3 | 106 | 0.58 | 79.5 | 604 | 1.13 |
| **Qwen3-4B** |  |  |  |  |  |  |
| FP16 | 25.2 | 181 | 1.80 | 24.9 | 614 | 2.21 |
| FP16, 8-bit KV | 21.2 | 194 | 2.15 | 21.4 | 610 | 2.48 |
| INT8 | 6.1 | 394 | 7.03 | 6.1 | 903 | 7.49 |
| NF4 | 21.6 | 1,512 | 3.46 | 21.6 | 1,754 | 3.68 |
| GPTQ | 58.9 | 220 | 0.90 | 59.5 | 597 | 1.28 |
| AWQ | 41.1 | 305 | 1.30 | 41.9 | 600 | 1.67 |
| **Qwen3-8B** |  |  |  |  |  |  |
| FP16 | 25.7 | 196 | 1.70 | 25.6 | 656 | 2.22 |
| FP16, 8-bit KV | 22.0 | 198 | 1.96 | 22.0 | 642 | 2.44 |
| INT8 | 6.1 | 532 | 6.72 | 6.1 | 993 | 7.42 |
| NF4 | 13.3 | 2,907 | 5.86 | 13.3 | 3,150 | 6.09 |
| GPTQ | 38.5 | 407 | 1.44 | 38.8 | 648 | 1.73 |
| AWQ | 24.2 | 554 | 2.20 | 24.1 | 791 | 2.42 |
| **Llama-3.2-1B-Instruct** |  |  |  |  |  |  |
| FP16 | 79.9 | 47 | 0.57 | 79.6 | 519 | 0.95 |
| FP16, 8-bit KV | 66.9 | 57 | 0.65 | 69.5 | 492 | 1.00 |
| INT8 | 17.4 | 144 | 3.11 | 16.5 | 597 | 2.54 |
| NF4 | 67.0 | 430 | 2.01 | 32.6 | 653 | 0.70 |
| GPTQ | 148.6 | 61 | 0.56 | 46.1 | 503 | 0.53 |
| AWQ | 108.2 | 88 | 2.85 | 113.1 | 515 | 0.82 |
| **Llama-3.2-3B-Instruct** |  |  |  |  |  |  |
| FP16 | 34.2 | 158 | 1.36 | 22.5 | 512 | 2.64 |
| FP16, 8-bit KV | 28.8 | 154 | 1.48 | 28.7 | 519 | 1.89 |
| INT8 | 10.2 | 252 | 4.17 | 10.1 | 701 | 4.51 |
| NF4 | 28.5 | 1,358 | 2.97 | 28.4 | 1,567 | 2.94 |
| GPTQ | 76.6 | 191 | 0.85 | 77.6 | 527 | 0.97 |
| AWQ | 43.4 | 291 | 1.23 | 46.8 | 534 | 1.33 |

## Notes

- Llama-3.2-3B with FP16 weights was timed twice; the table uses the second try (21 Nov 2026). The GPU slowed down during that run, so its constrained-decoding values are not reliable. Both tries are in [`results/latency/extra/`](../results/latency/extra/) and [`results/latency.md`](../results/latency.md).
- Llama-3.2-1B NF4 and GPTQ in constrained decoding: most answers are an empty list, so tokens/s rests on very short answers and is below the model's real writing speed.
- One run per config: speed differences of about 20% or less are not meaningful.

---

Built by [`src/final_table.py`](../src/final_table.py) from [`latency.csv`](../results/latency.csv).
