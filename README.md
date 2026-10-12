# Silent Tool-Call Failures

**How Quantization Degrades Function Calling and Structured Output in Small LLMs**

CSE498R Directed Research · Supervisor: Dr. Mohammad Shifat-E-Rabbi
**Status:** Week 2 of 9: evaluation script and FP16 baselines · [Progress log](UPDATES.md) · [Roadmap](ROADMAP.md)

---

## Problem

Local and on-device agents almost always run quantized models, usually at 4-bit. Practitioners
report a typical failure pattern: chat quality and standard benchmark scores look unchanged, but a
JSON field is silently given the wrong type, an enum value drifts, or a tool is called with a
plausible but wrong argument. A wrong tool argument is not a badly worded sentence. It is a wrong
database query, a wrong transaction, or a wrong command sent to a device.

This project measures how much of that degradation is caused by quantization. It separates
**syntactic** failures (invalid JSON, schema violations) from **semantic** ones (valid output, but
the wrong tool or the wrong argument value). It then tests whether the standard fix,
grammar-constrained decoding, removes the failures that matter.

## Research questions

| | Question |
|---|---|
| **RQ1** | How much does quantization hurt function-calling accuracy, and how does that depend on model size and quantization method? |
| **RQ2** | How much of that damage is syntactic, and how much is semantic? |
| **RQ3** | How much of the quantization penalty does grammar-constrained decoding remove, and which errors remain? |
| **RQ4** | Which schema features (nesting, enums, integer vs. float, optional fields, unicode, large tool menus) break first? |

## Planned setup

| | Core | Stretch (if time allows) |
|---|---|---|
| **Benchmarks** | BFCL v4 AST categories (simple, multiple, parallel, parallel-multiple, irrelevance); schema-stress suite (~150 cases) | API-Bank, ToolACE |
| **Models** | Qwen3-1.7B/4B/8B (thinking mode off), Llama-3.2-1B/3B-Instruct | Qwen3-0.6B, Phi-4-mini, xLAM-2-1b |
| **Precisions** | FP16 (baseline), INT8, NF4, GPTQ-4bit, AWQ-4bit | KV-cache quantization |
| **Decoding** | Free generation vs. JSON-schema-constrained decoding | |
| **Extras** | Accuracy vs. memory and latency | Pre-execution verifier |

The core study is 5 models × 5 precisions × 2 decoding modes = **50 evaluation runs**.

**Hardware:** Kaggle (2× NVIDIA T4). T4 GPUs do not support BF16, so FP16 is the full-precision
baseline.

## Failure taxonomy (planned)

| Type | Class | Example |
|---|---|---|
| Syntactic | S1 Unparseable output | Truncated or malformed JSON |
| Syntactic | S2 Schema violation | `"count": "5"` where an integer is required; value outside an enum |
| Semantic | M1 Wrong function | Calls `get_weather` instead of `get_forecast` |
| Semantic | M2 Wrong argument value | Valid schema, but `unit: "celsius"` where the query asked for Fahrenheit |
| Semantic | M3 Wrong number of calls | Two calls expected, one produced |
| Semantic | M4 Unneeded tool call | A tool called for a query that needs none |
| Semantic | M5 Missed tool call | No call when one was required |

Rule: output that fails JSON parsing or schema validation is **syntactic**. Output that passes the
schema but fails the BFCL checker is **semantic**.

## Repository layout

```
configs/            one config per model x precision
src/                evaluation, scoring and taxonomy code
data/stress_suite/  schema-stress test cases
results/            summary tables and figures
notebooks/          Kaggle notebooks and analysis
notes/              related work, design decisions
paper/              paper source (LaTeX)
docs/               project proposal and roadmap
```

## Running an evaluation

GPU runs happen in Kaggle notebooks (see [`notebooks/`](notebooks/)), in a Python 3.11 environment
with `bfcl-eval==2026.3.23` and vLLM 0.8.5. From the repository root:

```
python src/run_eval.py --config configs/qwen3-4b_fp16.yaml --mode free   # raw answers -> results/raw/
python src/score.py results/raw/qwen3-4b_fp16_free.jsonl                # scores -> results/summary.csv
```

`--mode constrained` forces every answer to match the functions' JSON schema. Every run also saves
its model revision, software versions, GPU, memory and speed in `results/meta/`.

## Progress

Progress is presented to the supervisor twice a week (Monday and Wednesday), and each update is
pushed here before the meeting. See [UPDATES.md](UPDATES.md) for the dated log and
[ROADMAP.md](ROADMAP.md) for the full plan.

## License

[MIT](LICENSE)
