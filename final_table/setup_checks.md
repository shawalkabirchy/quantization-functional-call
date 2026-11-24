# Setup Checks

[Back to the final table](README.md)

Checks that the benchmark, the scorer and the classifier can be trusted, made before and during the study.

## BFCL reproduced on Kaggle

BFCL's own harness, run on Qwen3-4B-Instruct-2507 in FP16 on one Kaggle T4, in BFCL's prompt mode and in its function-calling (FC) mode, against BFCL's leaderboard (downloaded 3 Oct 2026). Accuracy (%); report: [`notes/bfcl_repro.md`](../notes/bfcl_repro.md).

| Category | Prompt mode<br>ours | Prompt mode<br>leaderboard | Prompt mode<br>difference | FC mode<br>ours | FC mode<br>leaderboard | FC mode<br>difference |
|---|---:|---:|---:|---:|---:|---:|
| Non-Live Overall Acc | 86.31 | 86.44 | -0.13 | 87.81 | 87.88 | -0.07 |
| Python Simple AST | 93.75 | 93.75 | 0.00 | 95.25 | 95.50 | -0.25 |
| Java Simple AST | 64.00 | 64.00 | 0.00 | 63.00 | 63.00 | 0.00 |
| JavaScript Simple AST | 74.00 | 74.00 | 0.00 | 66.00 | 68.00 | -2.00 |
| Multiple AST | 91.00 | 91.00 | 0.00 | 94.00 | 93.50 | +0.50 |
| Parallel AST | 88.00 | 88.00 | 0.00 | 92.50 | 92.50 | 0.00 |
| Parallel Multiple AST | 89.00 | 89.50 | -0.50 | 90.00 | 90.00 | 0.00 |
| Irrelevance Detection | 84.58 | 85.00 | -0.42 | 88.75 | 88.75 | 0.00 |

## First run of our own script (Update 3)

Our prompt format and scorer on Qwen3-4B-Instruct-2507 (not one of the five study models) in FP16 with free decoding. Accuracy (%); report: [`notes/eval_sanity_check.md`](../notes/eval_sanity_check.md).

| Model | Python | Java | JavaScript | Simple | Multiple | Parallel | Parallel<br>multiple | Overall | Irrelevance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 | 96.00 | 65.00 | 72.00 | 77.67 | 94.00 | 94.00 | 91.50 | 89.29 | 85.83 |

## Other checks

- **Scorer:** re-scoring the 1,390 answers of the BFCL reproduction with our scorer gives BFCL's verdict on every question ([`src/check_scorer.py`](../src/check_scorer.py)).
- **Re-scoring:** every answer of every run was scored again on a PC, with the same result.
- **Same prompts:** all 7,825 prompts of the 8-bit KV cache runs are the same as in the FP16 runs ([`results/kv_cache.md`](../results/kv_cache.md)).
- **Failure classifier:** checked by hand on 80 random failures; all 80 agree ([`notes/taxonomy_check.md`](../notes/taxonomy_check.md)).
- **Schema against BFCL's checker:** 14 of the 34,750 free answers are correct by BFCL but do not match the schema ([`results/taxonomy_free.md`](../results/taxonomy_free.md)).
- **Parser:** Llama-3.2-3B FP16 scores 31.38 with our strict parser and would score 46.02 with BFCL's bracket repair. This was a diagnostic only; the reported score stays 31.38 ([`results/baseline_fp16.md`](../results/baseline_fp16.md)).

---

Built by [`src/final_table.py`](../src/final_table.py) from [`bfcl_repro/comparison.csv`](../results/bfcl_repro/comparison.csv), [`summary.csv`](../results/summary.csv).
