# Related Work

Last literature check: 5 Oct 2026 (arXiv, Google Scholar, GitHub).

## Directly related

### Quantization and tool calling

| # | Paper | Year | What it does | What it misses (our gap) |
|---|---|---|---|---|
| 1 | Dong et al., *Can Compressed LLMs Truly Act?* (ACBench), ICML 2025, [arXiv:2505.19433](https://arxiv.org/abs/2505.19433) | 2025 | GPTQ/AWQ (2-8-bit weights, 4/8-bit KV cache) and pruning on 12 agent tasks (tool use via T-Eval, ToolBench, ToolAlpaca), 15 models from Gemma-2B to Qwen2.5-32B; tool use drops only 1-3% at 4 bits; JSON output degrades more than plain-string output | Aggregate scores only; no breakdown by error type (malformed output, wrong tool, wrong argument); constrained decoding not studied |
| 2 | Paramanayakam et al., *Less is More*, DATE 2025, [arXiv:2411.15399](https://arxiv.org/abs/2411.15399) | 2025 | Tool-filtering method for edge devices; its motivating table shows Llama-3.1-8B on 230 BFCL queries falling from 63.0% (full precision) to 20.4-44.4% (Ollama q4_0 to q8_0) | Full precision ran on Hugging Face and the quantized builds on Ollama, so the drop mixes quantization with the serving stack; no error analysis |
| 3 | Erdogan et al., *TinyAgent*, EMNLP 2024 Demo, [arXiv:2409.00608](https://arxiv.org/abs/2409.00608) | 2024 | Fine-tuned 1.1B/7B function-calling agents; 4-bit llama.cpp builds keep their success rate (80.06 → 80.35%, 84.95 → 85.14%) | Uses quantization-aware fine-tuning on one custom task; says nothing about off-the-shelf post-training quantization |
| 4 | Jang et al., *Flat Score, Amplified Failures*, [arXiv:2607.27275](https://arxiv.org/abs/2607.27275) | 2026 | τ²-bench with 9-35B models (Gemma-4, Qwen3.5/3.6, dense and MoE) at BF16, FP8 and AWQ INT4 on vLLM: task score stays flat, but tool-name hallucination grows up to 2.5× and the benchmark's error budget hides it; counts failures per channel (hallucinated tool, wrong entity/argument, malformed arguments), and malformed arguments stay near zero at every precision | Large models in multi-turn episodes only; suggests constrained decoding but tests a re-prompting repair loop instead; no single-call scoring like BFCL's |
| 5 | Wu et al., *Which Decisions Low-Bit Quantization Breaks, and How to Predict Them*, [arXiv:2608.06564](https://arxiv.org/abs/2608.06564) | 2026 | 16 models (0.6-32B, including Qwen3) at 4/3/2 bits (RTN, activation-aware scaling, GPTQ, GGUF): decision margins shrink predictably; *whether* to call a tool flips before *which* tool, and argument-filling decisions barely flip; at 3-bit RTN, three of five models lose more completed calls than correct tool choices on 400 BFCL tasks | Studies decisions, mostly at 3 and 2 bits; no JSON-validity analysis, no error taxonomy, no constrained decoding |

### Quantization beyond perplexity (analogues)

| # | Paper | Year | What it does | What it misses (our gap) |
|---|---|---|---|---|
| 6 | Afrin et al., *Quantize with Confidence? An Empirical Study of Quantization for Code Generation*, [arXiv:2607.14181](https://arxiv.org/abs/2607.14181) | 2026 | Six methods (GPTQ, AWQ, QuIP#, AQLM, bitsandbytes, GGUF) on code models; methods differ (AQLM matches the baseline, QuIP# degrades most) | Code is graded by execution; tool calls need argument-level correctness |
| 7 | On-device LLM quantization evaluation, [arXiv:2505.15030](https://arxiv.org/abs/2505.15030) | 2025 | Accuracy, latency and memory of quantized on-device LLMs | Does not examine structured-output validity |

### Function-calling evaluation

| # | Paper | Year | What it does | What it misses (our gap) |
|---|---|---|---|---|
| 8 | Berkeley Function Calling Leaderboard (BFCL), ICML 2025, PMLR 267:48371-48392 | 2025 | Standard evaluation harness for function calling; AST and executable checks across many models | Compares models, not numerical precisions of the same model |
| 9 | ToolScan, [arXiv:2411.13547](https://arxiv.org/abs/2411.13547) | 2024 | Error taxonomy for tool-using LLMs | Studied at full precision only |
| 10 | Uncertainty quantification for function calling, [arXiv:2604.22985](https://arxiv.org/abs/2604.22985) | 2026 | Full-precision baseline verifier for function calls | No quantization; useful comparison for our verifier |

### Structured output and constrained decoding (full precision)

| # | Paper | Year | What it does | What it misses (our gap) |
|---|---|---|---|---|
| 11 | Tam et al., *Let Me Speak Freely?*, [arXiv:2408.02442](https://arxiv.org/abs/2408.02442) | 2024 | Format restrictions (JSON, XML) lower reasoning accuracy; stricter formats hurt more | Reasoning tasks, not tool calls; no quantization |
| 12 | Geng et al., *JSONSchemaBench*, [arXiv:2501.10868](https://arxiv.org/abs/2501.10868) | 2025 | 10K real-world JSON schemas; compares six constrained-decoding engines on coverage, speed and output quality | Compares engines, not model precisions; useful source of schema features for RQ4 |
| 13 | Ray, *The Constraint Tax*, [arXiv:2605.26128](https://arxiv.org/abs/2605.26128) | 2026 | Qwen2.5-0.5B/1.5B and SmolLM2-1.7B: hard schema decoding raises validity from 61.5% to 100% but lowers accuracy from 19.7% to 11.0%; on a calendar tool-call task, Qwen2.5-1.5B's executable accuracy falls from 91.5% to 48.0% | Full precision only |
| 14 | Li et al., *Constraint Tax in Open-Weight LLMs*, [arXiv:2606.25605](https://arxiv.org/abs/2606.25605) | 2026 | With JSON-schema constraints and tool calling on together, some models stop calling tools; the authors attribute this to the grammar masking the tool-call tokens, and propose two-pass execution | A serving-configuration failure at full precision; no quantization |
| 15 | Lee, *Repair, Not Improvement*, [arXiv:2608.13959](https://arxiv.org/abs/2608.13959) | 2026 | 0.6-4B tool routers: constraining output to valid tool names mostly repairs unreadable output rather than improving decisions; combined with a stop at line breaks, it lowers abstention accuracy by up to 29.5 points | Full precision; tool choice and abstention only, no arguments |
| 16 | Chavan, *Constrained Decoding Eliminates Structural Failures in Small LLMs but Reveals a Scale-Dependent Semantic Gap*, [arXiv:2609.23742](https://arxiv.org/abs/2609.23742) | 2026 | Five 0.6-4B models, 14 structured-output tasks, Outlines vs. XGrammar: constrained decoding removes every structural failure, but a semantic gap remains | Full precision only |

### Prior work that is not a paper

- **QuantCliff** ([github.com/YeeDEA/quantcliff](https://github.com/YeeDEA/quantcliff), August 2026, no
  preprint). Tests llama.cpp GGUF builds in 11 configurations: Qwen3-1.7B and Qwen3-4B-Instruct at
  BF16, Q8_0, Q4_K_M and Q3_K_M, and Llama-3.1-8B-Instruct at Q8_0, Q4_K_M and Q3_K_M. Each runs with
  and without a JSON-schema grammar on two test sets: 100 BFCL v4 items (50 simple, 50 multiple) and
  20 multi-step mock-tool scenarios. Single-call BFCL accuracy moves only 1-3 points down the
  precision ladder (Qwen3-1.7B: 89% to 86%). On the scenarios, the grammar removed all 104 malformed
  outputs, yet task success rose by 0.0 points in 9 of 11 configurations. The errors came back as
  wrong tools and arguments: for Llama-3.1-8B at Q3_K_M, invalid output fell from 24 to 0 while
  wrong-tool calls rose from 2 to 16. This is the result RQ3 expects, at small scale.
- **KV-cache quantization and tool calling:** ACBench includes 4- and 8-bit KV-cache settings but
  reports aggregate scores only; otherwise only practitioner write-ups.

## Gap (revised 5 Oct 2026)

The Update 1 claim, that no peer-reviewed study measures quantization against function calling, no
longer holds. ACBench (ICML 2025) and Less is More (DATE 2025) both report tool-calling accuracy
under quantization. Two 2026 preprints (4, 5) and one GitHub study go further. What is still
missing:

1. **An error breakdown for small models on one controlled serving stack (RQ1, RQ2).** No study
   scores the full BFCL AST categories across FP16, INT8, NF4, GPTQ and AWQ on one serving stack, or
   splits the failures into syntactic (S1-S2) and semantic (M1-M5) classes. ACBench reports
   aggregate scores. Less is More also reports aggregates, and its comparison mixes quantization with
   the serving stack. Wu et al. measure decisions, not error types. Jang et al. count failure
   channels, but for 9-35B models in multi-turn episodes, where malformed calls almost never occur.
2. **Constrained decoding combined with quantization (RQ3).** Only QuantCliff has tested this: GGUF
   builds of three models, a substitution result from 20 multi-step scenarios, no paper and no
   significance tests. Jang et al. suggest constrained decoding as a fix but do not test it.
   QuantCliff's finding, that constrained decoding substitutes errors rather than removing them,
   matches this project's hypothesis. RQ3 therefore becomes a test of that claim at scale: five
   precisions, the full non-live set, and paired tests. Papers 13-16 show the same effect at full
   precision only.
3. **Schema features under quantization (RQ4).** JSONSchemaBench catalogues schema features, but
   only to compare decoding engines. No study asks which features break first as precision drops.
4. **KV-cache quantization (stretch):** ACBench includes KV4/KV8 settings, but with aggregate scores
   only; no study breaks down tool-call errors under a quantized KV cache.

## Implications for the study (for discussion)

- Report the decision of whether to call a tool (irrelevance, M4 and M5) separately; Wu et al. find
  it breaks first.
- Report error counts per failure class, not only accuracy; Jang et al. show accuracy can stay flat
  while failures grow.
- Compare precisions with paired tests on the same items, and state the minimum detectable effect
  before running ([arXiv:2605.28873](https://arxiv.org/abs/2605.28873)).
- Cite QuantCliff, and present RQ3 as a larger, controlled test of its finding.

## Background (methods used in this project)

| Topic | Paper |
|---|---|
| GPTQ | Frantar et al., [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) |
| AWQ | Lin et al., [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) |
| LLM.int8() (INT8) | Dettmers et al., [arXiv:2208.07339](https://arxiv.org/abs/2208.07339) |
| QLoRA (NF4 data type) | Dettmers et al., [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) |
| Low-bit quantized LLaMA3 | Huang et al., [arXiv:2404.14047](https://arxiv.org/abs/2404.14047) |
| vLLM / PagedAttention | Kwon et al., [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) |
| XGrammar (constrained decoding) | Dong et al., [arXiv:2411.15100](https://arxiv.org/abs/2411.15100) |
| Guided generation (Outlines) | Willard & Louf, [arXiv:2307.09702](https://arxiv.org/abs/2307.09702) |
| API-Bank | Li et al., [arXiv:2304.08244](https://arxiv.org/abs/2304.08244) |
| ToolACE | Liu et al., [arXiv:2409.00920](https://arxiv.org/abs/2409.00920) |
| Sample size for 4-bit quantization benchmarks (paired minimum detectable effect) | Zhuang et al., [arXiv:2605.28873](https://arxiv.org/abs/2605.28873) |

## To do

- [x] Fresh arXiv / Google Scholar search: "quantization function calling", "quantized LLM tool use",
      "quantization structured output JSON", "low-bit LLM agents" (5 Oct 2026)
- [x] Reach 12+ directly related papers (16, plus QuantCliff)
- [x] Update the existing-work and novelty claims in `docs/proposal.tex` to match (5 Oct 2026)
