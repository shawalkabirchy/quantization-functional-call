# Related Work

## Directly related

| # | Paper | Year | What it does | What it misses (our gap) |
|---|---|---|---|---|
| 1 | Berkeley Function Calling Leaderboard (BFCL), ICML 2025, PMLR 267:48371-48392 | 2025 | Standard evaluation harness for function calling; AST and executable checks across many models | Compares models, not numerical precisions of the same model |
| 2 | ToolScan, [arXiv:2411.13547](https://arxiv.org/abs/2411.13547) | 2024 | Error taxonomy for tool-using LLMs | Studied at full precision only |
| 3 | Quantization for code generation, [arXiv:2607.14181](https://arxiv.org/abs/2607.14181) | 2026 | Finds that quantization methods differ in their effect on code correctness; nearest analogue to this project | Code is graded by execution; tool calls need argument-level correctness |
| 4 | On-device LLM quantization evaluation, [arXiv:2505.15030](https://arxiv.org/abs/2505.15030) | 2025 | Accuracy, latency and memory of quantized on-device LLMs | Does not examine structured-output validity |
| 5 | Uncertainty quantification for function calling, [arXiv:2604.22985](https://arxiv.org/abs/2604.22985) | 2026 | Full-precision baseline verifier for function calls | No quantization; useful comparison for our verifier |

**Gap:** no peer-reviewed study measures quantization against function calling. Reports of silent
tool-call failures under INT4 KV-cache and W4A16 builds exist only as practitioner write-ups.

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

## To do (Update 2)

- [ ] Fresh arXiv / Google Scholar search: "quantization function calling", "quantized LLM tool use",
      "quantization structured output JSON", "low-bit LLM agents"
- [ ] Reach 12+ directly related papers
