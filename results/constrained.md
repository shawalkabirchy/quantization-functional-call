# Constrained Decoding (Update 9)

**Date:** 2026-11-03 (runs on 2 and 3 November)
**Goal:** function-calling accuracy of every core model at every precision when every answer is forced to match the JSON schema of the question's functions. The comparison with free decoding is Update 10.

## Setup

Everything is the same as in the free runs of Updates 4-6 (questions, prompt, greedy decoding, scorer, model revisions and quantization settings), except the constraint:

- **Constraint:** vLLM 0.8.5 guided decoding with XGrammar 0.1.18 (`xgrammar:disable-any-whitespace,no-fallback`). The schema is built by [`src/schema.py`](../src/schema.py): a JSON array of calls, each naming one of the question's functions, with its arguments typed as declared and no extra arguments. An empty array is allowed.
- **Same prompts:** every prompt is the same as in the free run of the same model and precision (checked on all 34,750 answers).
- **Code:** commit `d7abb52` for part A (Qwen3-1.7B, Llama-3.2-1B, Llama-3.2-3B) and `0cdf81b` for part B (Qwen3-4B, Qwen3-8B); `src/` and `configs/` are the same in both. Notebooks: [part A](../notebooks/06_constrained_run_A.ipynb), [part B](../notebooks/06_constrained_run_B.ipynb), from the template [`06_constrained.ipynb`](../notebooks/06_constrained.ipynb) with only `PART` changed. The runs were split into two parts so that one Kaggle session could not lose all of them.
- **INT8 and GPTQ checkpoints** were made again with the same code and settings, because the checkpoints of Updates 5 and 6 were not kept. For every GPTQ checkpoint, the GPTQ loss of every quantized layer in the log is identical to Update 6 (1,008 layers in the five models), and every INT8 and GPTQ checkpoint loads with the same weight size as in the free run. The main packages are the same as in Updates 5 and 6; Python went from 3.11.17 to 3.11.18, and a few other packages, such as google-auth and hf-xet, have newer versions (see the freeze files). The GPTQ checkpoints are now on Hugging Face (private): [shawalkabir/Qwen3-1.7B-GPTQ-W4A16-g128](https://huggingface.co/shawalkabir/Qwen3-1.7B-GPTQ-W4A16-g128) (`7803089`), [shawalkabir/Qwen3-4B-GPTQ-W4A16-g128](https://huggingface.co/shawalkabir/Qwen3-4B-GPTQ-W4A16-g128) (`a82f180`), [shawalkabir/Qwen3-8B-GPTQ-W4A16-g128](https://huggingface.co/shawalkabir/Qwen3-8B-GPTQ-W4A16-g128) (`bec7853`), [shawalkabir/Llama-3.2-1B-Instruct-GPTQ-W4A16-g128](https://huggingface.co/shawalkabir/Llama-3.2-1B-Instruct-GPTQ-W4A16-g128) (`b14742c`), [shawalkabir/Llama-3.2-3B-Instruct-GPTQ-W4A16-g128](https://huggingface.co/shawalkabir/Llama-3.2-3B-Instruct-GPTQ-W4A16-g128) (`59872d0`).
- **AWQ checkpoints** were downloaded from the private Hugging Face repositories made in Update 6: [shawalkabir/Qwen3-1.7B-AWQ-W4A16-g128](https://huggingface.co/shawalkabir/Qwen3-1.7B-AWQ-W4A16-g128) (`3cc45f1`), [shawalkabir/Qwen3-4B-AWQ-W4A16-g128](https://huggingface.co/shawalkabir/Qwen3-4B-AWQ-W4A16-g128) (`e8a7317`), [shawalkabir/Qwen3-8B-AWQ-W4A16-g128](https://huggingface.co/shawalkabir/Qwen3-8B-AWQ-W4A16-g128) (`6f1d2a8`), [shawalkabir/Llama-3.2-1B-Instruct-AWQ-W4A16-g128](https://huggingface.co/shawalkabir/Llama-3.2-1B-Instruct-AWQ-W4A16-g128) (`e8cd172`), [shawalkabir/Llama-3.2-3B-Instruct-AWQ-W4A16-g128](https://huggingface.co/shawalkabir/Llama-3.2-3B-Instruct-AWQ-W4A16-g128) (`6f9fd9e`).

## Overall accuracy (%)

Overall = BFCL's Non-Live Overall Acc. All five precisions are also in [`main_table_constrained.md`](main_table_constrained.md).

| Model | FP16 | INT8 | NF4 | GPTQ | AWQ |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 82.46 | 82.79 | 78.50 | 71.65 | 80.81 |
| Qwen3-4B | 88.25 | 89.04 | 86.12 | 86.85 | 87.48 |
| Qwen3-8B | 88.58 | 88.21 | 88.27 | 86.88 | 88.83 |
| Llama-3.2-1B-Instruct | 48.19 | 45.50 | 17.83 | 2.90 | 47.75 |
| Llama-3.2-3B-Instruct | 71.04 | 69.42 | 65.96 | 64.71 | 68.98 |

## Accuracy per category (%)

| Model | Precision | Python | Java | JavaScript | Multiple | Parallel | Parallel Multiple | Overall | Irrelevance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | FP16 | 93.50 | 65.00 | 66.00 | 91.00 | 84.00 | 80.00 | 82.46 | 77.50 |
| Qwen3-1.7B | INT8 | 92.50 | 65.00 | 68.00 | 90.00 | 84.50 | 81.50 | 82.79 | 72.50 |
| Qwen3-1.7B | NF4 | 89.00 | 67.00 | 66.00 | 90.00 | 78.50 | 71.50 | 78.50 | 80.83 |
| Qwen3-1.7B | GPTQ | 83.25 | 61.00 | 72.00 | 63.50 | 76.50 | 74.50 | 71.65 | 56.25 |
| Qwen3-1.7B | AWQ | 92.25 | 65.00 | 70.00 | 87.50 | 81.50 | 78.50 | 80.81 | 69.58 |
| Qwen3-4B | FP16 | 95.50 | 64.00 | 70.00 | 95.50 | 90.50 | 90.50 | 88.25 | 84.58 |
| Qwen3-4B | INT8 | 95.00 | 65.00 | 76.00 | 96.00 | 90.00 | 91.50 | 89.04 | 86.67 |
| Qwen3-4B | NF4 | 93.50 | 63.00 | 76.00 | 93.50 | 89.00 | 84.50 | 86.12 | 83.75 |
| Qwen3-4B | GPTQ | 94.75 | 64.00 | 72.00 | 94.50 | 88.50 | 87.50 | 86.85 | 86.25 |
| Qwen3-4B | AWQ | 96.25 | 64.00 | 72.00 | 92.50 | 91.50 | 88.50 | 87.48 | 83.33 |
| Qwen3-8B | FP16 | 96.00 | 65.00 | 74.00 | 92.50 | 92.50 | 91.00 | 88.58 | 87.08 |
| Qwen3-8B | INT8 | 96.00 | 65.00 | 74.00 | 93.00 | 92.00 | 89.50 | 88.21 | 87.50 |
| Qwen3-8B | NF4 | 95.75 | 64.00 | 76.00 | 95.00 | 92.50 | 87.00 | 88.27 | 84.58 |
| Qwen3-8B | GPTQ | 95.50 | 62.00 | 72.00 | 91.50 | 91.00 | 88.50 | 86.88 | 89.58 |
| Qwen3-8B | AWQ | 95.50 | 66.00 | 78.00 | 93.00 | 90.50 | 92.00 | 88.83 | 87.50 |
| Llama-3.2-1B-Instruct | FP16 | 50.75 | 33.00 | 52.00 | 53.50 | 55.00 | 39.00 | 48.19 | 84.17 |
| Llama-3.2-1B-Instruct | INT8 | 45.00 | 27.00 | 36.00 | 55.00 | 52.00 | 39.00 | 45.50 | 80.83 |
| Llama-3.2-1B-Instruct | NF4 | 20.50 | 1.00 | 2.00 | 53.50 | 5.00 | 5.00 | 17.83 | 77.08 |
| Llama-3.2-1B-Instruct | GPTQ | 0.25 | 0.00 | 0.00 | 9.00 | 0.50 | 2.00 | 2.90 | 99.58 |
| Llama-3.2-1B-Instruct | AWQ | 68.00 | 51.00 | 64.00 | 49.00 | 47.50 | 33.50 | 47.75 | 50.83 |
| Llama-3.2-3B-Instruct | FP16 | 88.50 | 48.00 | 44.00 | 86.50 | 76.00 | 61.50 | 71.04 | 74.58 |
| Llama-3.2-3B-Instruct | INT8 | 88.00 | 44.00 | 38.00 | 86.00 | 75.00 | 60.00 | 69.42 | 70.42 |
| Llama-3.2-3B-Instruct | NF4 | 90.50 | 54.00 | 50.00 | 92.50 | 57.00 | 49.50 | 65.96 | 38.75 |
| Llama-3.2-3B-Instruct | GPTQ | 81.50 | 47.00 | 54.00 | 90.50 | 60.50 | 47.00 | 64.71 | 65.00 |
| Llama-3.2-3B-Instruct | AWQ | 88.75 | 58.00 | 48.00 | 84.50 | 64.50 | 62.00 | 68.98 | 69.58 |

## Format, length and speed

"Time vs. free" is the generation time divided by that of the free run of the same model and precision.

| Model | Precision | Parsed, AST questions (%) | Unparsed answers | Cut off at max_tokens | GPUs | Weights per GPU (GiB) | Generation (s) | Time vs. free | Output tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | FP16 | 99.65 | 4 | 4 | 1 | 3.22 | 974.1 | 5.81 | 88.9 |
| Qwen3-1.7B | INT8 | 99.83 | 2 | 2 | 1 | 1.90 | 1876.9 | 3.87 | 41.9 |
| Qwen3-1.7B | NF4 | 99.91 | 1 | 2 | 1 | 1.27 | 1964.8 | 1.86 | 39.1 |
| Qwen3-1.7B | GPTQ | 99.91 | 1 | 1 | 1 | 1.27 | 919.6 | 4.74 | 86.7 |
| Qwen3-1.7B | AWQ | 99.91 | 1 | 1 | 1 | 1.27 | 946.5 | 4.34 | 79.8 |
| Qwen3-4B | FP16 | 99.83 | 2 | 0 | 1 | 7.56 | 1029.0 | 2.20 | 69.7 |
| Qwen3-4B | INT8 | 99.91 | 1 | 0 | 1 | 4.18 | 1538.7 | 2.25 | 46.6 |
| Qwen3-4B | NF4 | 99.91 | 1 | 0 | 1 | 2.49 | 3208.2 | 1.26 | 22.6 |
| Qwen3-4B | GPTQ | 99.91 | 1 | 0 | 1 | 2.50 | 1031.5 | 2.17 | 70.5 |
| Qwen3-4B | AWQ | 99.65 | 4 | 0 | 1 | 2.50 | 1083.1 | 1.97 | 67.3 |
| Qwen3-8B | FP16 | 100.00 | 0 | 0 | 2 | 7.64 | 1015.3 | 2.53 | 70.7 |
| Qwen3-8B | INT8 | 100.00 | 0 | 0 | 1 | 8.80 | 2621.7 | 1.45 | 27.4 |
| Qwen3-8B | NF4 | 100.00 | 0 | 0 | 1 | 5.67 | 5782.3 | 1.18 | 12.4 |
| Qwen3-8B | GPTQ | 100.00 | 0 | 0 | 1 | 5.71 | 1235.7 | 1.46 | 57.7 |
| Qwen3-8B | AWQ | 99.91 | 1 | 1 | 1 | 5.71 | 1542.4 | 1.64 | 49.0 |
| Llama-3.2-1B-Instruct | FP16 | 99.30 | 8 | 16 | 1 | 2.32 | 834.2 | 3.96 | 143.3 |
| Llama-3.2-1B-Instruct | INT8 | 99.04 | 11 | 16 | 1 | 1.41 | 1165.6 | 2.42 | 100.2 |
| Llama-3.2-1B-Instruct | NF4 | 99.83 | 2 | 8 | 1 | 0.98 | 1381.0 | 0.93 | 39.8 |
| Llama-3.2-1B-Instruct | GPTQ | 100.00 | 0 | 0 | 1 | 0.98 | 712.1 | 1.54 | 8.7 |
| Llama-3.2-1B-Instruct | AWQ | 99.13 | 10 | 26 | 1 | 0.98 | 949.0 | 0.85 | 175.8 |
| Llama-3.2-3B-Instruct | FP16 | 99.91 | 1 | 2 | 1 | 6.02 | 963.3 | 2.99 | 74.8 |
| Llama-3.2-3B-Instruct | INT8 | 100.00 | 0 | 0 | 1 | 3.42 | 1075.7 | 2.80 | 59.4 |
| Llama-3.2-3B-Instruct | NF4 | 100.00 | 0 | 0 | 1 | 2.13 | 2447.2 | 1.20 | 26.1 |
| Llama-3.2-3B-Instruct | GPTQ | 99.91 | 1 | 1 | 1 | 2.14 | 874.9 | 2.17 | 73.7 |
| Llama-3.2-3B-Instruct | AWQ | 99.91 | 1 | 1 | 1 | 2.14 | 934.6 | 2.31 | 73.3 |

## Notes

- **All 25 runs finished.** Part A ran on 2-3 November (20:12-02:09 UTC), part B on 3 November (06:47-13:30 UTC). Re-scoring the saved answers on a PC gives the same scores for all 25 runs.
- **The constraint works:** all 34,657 answers that parse (of 34,750) match their schema (checked with `jsonschema`). So no constrained answer has a schema error (class S2 of Update 7).
- **93 answers do not parse** (0.27%). 81 were cut off at 4,096 tokens: the constraint fixes the format, but it cannot make a model stop. 67 of them repeat calls without end (23 to 187 calls), 10 make one call with a list that never ends, 3 write one string that never ends, and 1 writes a number with endless zeros (Qwen3-8B AWQ, `parallel_95`). 66 of the 81 are Llama-3.2-1B's. 40 of the 81 are irrelevance answers; they contain no complete call, so BFCL counts them as correct, as in the free runs (38 of the 40 are Llama-3.2-1B's).
- **12 answers stopped early, inside an unclosed string** (finish reason `stop`): 9 from Qwen3-4B on Java and JavaScript questions, whose arguments are strings of code, and 3 from Llama-3.2-1B (`multiple_119` at FP16 and INT8, `simple_javascript_23` at AWQ). In 10 of them the model wrote the closing `}}]` inside the string, then a chat-template token (`</think>` for Qwen3-4B, `<|start_header_id|>` for Llama), and the answer ended there. Why the decoder let these answers end before the JSON was complete was not investigated. They count as failures.
- **Empty answers:** the schema allows an empty array (no call), so a model can always answer `[]`. Llama-3.2-1B does this much more under the constraint: on the 1,150 AST questions, FP16 344 (free run: 288), NF4 818 (286), GPTQ 1,112 (160). Llama-3.2-1B GPTQ also answers `[]` on 239 of the 240 irrelevance questions, so its irrelevance score is 99.58 while its Overall is 2.90.
- **First look at free vs. constrained** (the tests are Update 10): 24 of the 25 runs score higher with the constraint. FP16 rises for every model: Qwen3-1.7B +9.90, Qwen3-4B +0.21, Qwen3-8B +0.29, Llama-3.2-1B +11.79, Llama-3.2-3B +39.66 points. The one run that drops is Llama-3.2-1B GPTQ (11.40 to 2.90), because of the empty answers above.
- **Speed:** 23 of the 25 runs took longer than the free run of the same model and precision (0.85 to 5.81 times as long; median 2.17). These are single runs on Kaggle, not a speed test; the speed study is Update 13.
