# 8-Bit KV Cache (Stretch Item, Update 13)

**Question:** the main study quantizes the weights. What happens to function calling if the KV cache (the keys
and values of the earlier tokens, which the model keeps in GPU memory while it writes) is stored in 8 bits instead
of 16?

**Setup:** the five FP16 models answered the 1,390 BFCL `non_live` questions and the 175-question stress suite
again, in free decoding, with one change: `kv_cache_dtype: fp8` in vLLM 0.8.5 (FP8 E4M3, scale 1.0, because the
checkpoints have no KV scales). Weights, prompts, decoding and scorer are those of the FP16 runs, so every
difference comes from the KV cache. The T4 has no FP8 hardware; vLLM converts between FP16 and FP8 in its own CUDA
code. Configs: `configs/<model>_fp16-kv8.yaml`; runs: notebook 08 on Kaggle in two parts (16 Nov 2026), all
finished. Checks: all 7,825 prompts are the same as in the FP16 runs, and scoring every answer again on a PC gives
the same result. Tests as for the quantized weights: McNemar's exact test on the 1,150 AST questions with Holm's
correction over the five models, and a bootstrap 95% interval for the change in Overall
([`src/kv_cache.py`](../src/kv_cache.py)).

## BFCL, free decoding

| Model | FP16 | 8-bit KV | Change | 95% interval | Right at FP16 only | Right with 8-bit KV only | p (Holm) | Irrelevance (of 240) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 72.56 | 71.83 | -0.73 | -2.48 to +1.02 | 58 | 28 | **0.008** | 187 → 189 |
| Qwen3-4B | 88.04 | 87.29 | -0.75 | -1.75 to +0.21 | 10 | 4 | 0.43 | 203 → 203 |
| Qwen3-8B | 88.29 | 88.58 | +0.29 | -0.33 to +1.00 | 3 | 5 | 0.73 | 209 → 209 |
| Llama-3.2-1B-Instruct | 36.40 | 34.98 | -1.42 | -2.96 to +0.12 | 29 | 18 | 0.43 | 199 → 198 |
| Llama-3.2-3B-Instruct | 31.38 | 31.40 | +0.02 | -1.52 to +1.56 | 21 | 36 | 0.25 | 219 → 218 |

Per category: the rows `<model>_fp16-kv8_free` in [`summary.csv`](summary.csv); tests and failure classes:
[`kv_cache.csv`](kv_cache.csv).

**Four models do not change.** Their Overall moves by -1.42 to +0.29 points, every interval includes 0, no test is
significant, and irrelevance moves by at most one answer.

**Qwen3-1.7B makes more format errors.** It is the only significant change: 58 AST answers are lost and 28 gained.
Of the 58 lost, 51 are not valid JSON, and 44 of these have exactly one `}` too many, e.g.
`[{"name": "solve_quadratic", "arguments": {"a": 2, "b": 5, "c": 3}}}]`; 42 are right once that brace is removed.
Over the whole run, answers with one extra `}` rise from 75 at FP16 to 102. This is the same weak spot that GPTQ hit
hardest (221 such answers, Update 6). The change in Overall is small (-0.73, interval includes 0) because the losses
fall mostly in simple Python (31 lost, 2 gained) and multiple (12 lost, 3 gained), while parallel gains (4 lost,
11 gained): McNemar's test counts questions, while Overall averages the categories, and simple Python is only part
of one of its four parts.

Failure classes of the answers that were right at FP16 and are wrong with the 8-bit KV cache (all 1,390
questions, so irrelevance is included here):

| Model | Not valid JSON (S1) | Off-schema (S2) | Wrong values or calls (M1-M5) |
|---|---:|---:|---:|
| Qwen3-1.7B | 51 | 1 | 6 |
| Qwen3-4B | 2 | 1 | 7 |
| Qwen3-8B | 0 | 0 | 3 |
| Llama-3.2-1B-Instruct | 3 | 9 | 18 |
| Llama-3.2-3B-Instruct | 13 | 7 | 5 |

## Schema-stress suite, free decoding

No group changes significantly (Holm's correction over 35 tests). Correct answers out of 175:

| Model | FP16 | 8-bit KV |
|---|---:|---:|
| Qwen3-1.7B | 154 | 147 |
| Qwen3-4B | 175 | 174 |
| Qwen3-8B | 175 | 175 |
| Llama-3.2-1B-Instruct | 42 | 39 |
| Llama-3.2-3B-Instruct | 79 | 78 |

Qwen3-1.7B's largest drops are in the plain control (25 to 22) and nesting (24 to 22). All 9 answers it loses are
invalid JSON, and 6 of them are right once one extra `}` is removed: the same habit as on BFCL, not a stressed
feature. All tests: [`kv_cache_stress.csv`](kv_cache_stress.csv).

## Memory

The weights take the same memory, and the same memory now holds twice as many tokens of keys and values (from
vLLM's log, in the 90% of the GPU it may use):

| Model | KV cache tokens, FP16 | KV cache tokens, 8-bit |
|---|---:|---:|
| Qwen3-1.7B | 78,896 | 157,792 |
| Qwen3-4B | 29,696 | 59,408 |
| Qwen3-8B (two T4s) | 57,008 | 114,032 |
| Llama-3.2-1B-Instruct | 312,560 | 625,136 |
| Llama-3.2-3B-Instruct | 54,528 | 109,072 |

Our longest prompt is about 3,700 tokens plus a 4,096-token answer budget, so every model already fits a full
request at FP16; the extra room matters for longer contexts or many requests at once.

## Speed

Writing the answer is slower with the 8-bit KV cache in all five models, most likely because the T4 has no FP8
hardware, so vLLM converts the cache back to FP16 in software (batch size 1, free decoding, [`latency.md`](latency.md)):

| Model | Tokens/s, FP16 | Tokens/s, 8-bit KV | Change |
|---|---:|---:|---:|
| Qwen3-1.7B | 52.9 | 40.6 | -23% |
| Qwen3-4B | 25.2 | 21.2 | -16% |
| Qwen3-8B (two T4s) | 25.7 | 22.0 | -14% |
| Llama-3.2-1B-Instruct | 79.9 | 66.9 | -16% |
| Llama-3.2-3B-Instruct | 34.2 | 28.8 | -16% |

Each value is one run, and single runs on Kaggle's T4s can differ by about 20% (see [`latency.md`](latency.md)), so
the size of the slowdown is uncertain; that it appears in all five models is the firmer result. The time to the
first token barely changes. A GPU with FP8 support could behave differently; that was not tested.

## In short

An 8-bit KV cache leaves function calling unchanged in four of the five models. In Qwen3-1.7B it adds format
errors of the kind the model already makes (one extra `}`), and the content underneath is mostly right. It doubles
the room for keys and values, and on a T4 it makes writing the answer somewhat slower. The constrained decoding of
Update 10 removes this kind of error when it comes from the weights; it was not run with the 8-bit KV cache here.
