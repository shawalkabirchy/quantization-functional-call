# Evaluation Script Sanity Check (Update 3)

**Date:** 2026-10-12
**Goal:** check that our own evaluation script gives scores close to BFCL's own harness, on the same model and GPU.

## Setup

- **Model:** `Qwen/Qwen3-4B-Instruct-2507` (revision `cdbee75f17c0`), FP16
- **Questions:** BFCL `non_live`, 1390 questions, bfcl-eval 2026.3.23
- **Prompt format:** `json-array-v1`: one system prompt asking for a JSON array of calls
- **Decoding:** `SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0, seed=42, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=4096, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None)`
- **Inference:** vLLM 0.8.5 on Tesla T4, Tesla T4; tensor parallel size 1
- **Software:** Python 3.11.17, torch 2.6.0, transformers 4.51.3, xgrammar 0.1.18
- **Code:** commit `e5b2d67`

## Results: free mode (accuracy, %)

| Category | Our script | BFCL prompt | BFCL FC | vs prompt | vs FC | Parsed (%) |
|---|---:|---:|---:|---:|---:|---:|
| Non-Live Overall Acc | 89.29 | 86.31 | 87.81 | 2.98 | 1.48 | n/a |
| Python Simple AST | 96.00 | 93.75 | 95.25 | 2.25 | 0.75 | 100.00 |
| Java Simple AST | 65.00 | 64.00 | 63.00 | 1.00 | 2.00 | 96.00 |
| JavaScript Simple AST | 72.00 | 74.00 | 66.00 | -2.00 | 6.00 | 90.00 |
| Multiple AST | 94.00 | 91.00 | 94.00 | 3.00 | 0.00 | 100.00 |
| Parallel AST | 94.00 | 88.00 | 92.50 | 6.00 | 1.50 | 100.00 |
| Parallel Multiple AST | 91.50 | 89.00 | 90.00 | 2.50 | 1.50 | 100.00 |
| Irrelevance Detection | 85.83 | 84.58 | 88.75 | 1.25 | -2.92 | 100.00 |

**Verdict:** Pass: the overall score (89.29%) is 1.48 points from the closer BFCL run (tolerance 5 points).

## Run information

- Generation: 492 s for 1390 questions, 163.7 output tokens/s; model loading 100 s
- Finish reasons: {'stop': 1390}
- Memory (vLLM): weights 7.6065 GiB, activation peak 1.42 GiB, KV cache 4.03 GiB
- Constrained mode, 21 questions: every answer parsed; 20 s

## Notes

- **Passed.** Overall 89.29%: 1.48 points above the BFCL FC run (87.81%) and 2.98 points above the BFCL
  prompt run (86.31%) on the same model and GPU. Our script scores higher than BFCL's prompt mode in 6
  of 7 categories. Two things differ from the BFCL runs: the answer format (a JSON array instead of
  Python call syntax or Qwen's tool-call tags) and the decoding (strictly greedy instead of near-greedy).
  This run cannot say which of the two causes the difference.
- **Re-scored locally:** scoring the raw answers again on a PC gives the same numbers in every category.
- **9 of 1,390 answers did not parse** (0.6%): 4 in Java and 5 in JavaScript, none elsewhere. In all 9,
  an argument has to be a string that itself contains a JSON object, and the model got the inner
  quotes or escaping wrong (for example `simple_java_1`). The parser reads them correctly: these
  answers are not valid JSON. They are counted as wrong, as planned: syntactic failures that already
  occur at FP16.
- **No answer was cut off.** All 1,390 ended by themselves (finish reason `stop`); the longest answer
  was 322 tokens, far below the 4,096-token limit.
- **Constrained decoding works on the T4.** vLLM 0.8.5 (V0 engine) with xgrammar answered all 21
  test questions with valid JSON. Accuracy on 3 questions per category means nothing yet.
- **Speed and memory:** 492 s of generation for 1,390 questions on one T4 (163.7 output tokens/s);
  the FP16 weights take 7.61 GiB. The second T4 was idle.
