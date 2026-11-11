# Findings

The main results of the project so far, in one place.

**State:** after Update 12 (Wed 11 Nov 2026). Every number comes from the files in [`results/`](../results/),
linked under each finding. Full score tables: [`score_changes.md`](score_changes.md). Answers to the four research
questions: [`research_questions/`](../research_questions/README.md).

**Setup in short:** 5 models (Qwen3-1.7B, 4B, 8B; Llama-3.2-1B, 3B-Instruct) × 5 precisions (FP16, INT8, NF4,
GPTQ, AWQ) × 2 decoding modes (free, JSON-schema-constrained) = 50 runs on the 1,390 BFCL `non_live` questions,
and 50 more runs on our 175-question schema-stress suite. Score = BFCL's Non-Live Overall accuracy (%).
"Significant" = McNemar's exact test, paired by question, Holm-adjusted p < 0.05.

---

## Major findings

### 1. The larger Qwen3 models barely change at any precision

Qwen3-4B and Qwen3-8B stay within 2.35 points of FP16 at every precision with free decoding, and within 2.13 points
with constrained decoding. Only two of their eight free changes are significant: Qwen3-4B NF4 (-2.35) and Qwen3-8B
GPTQ (-2.31).

Evidence: [main table](../results/main_table.md), [tests](../results/significance_free.md)

### 2. The smaller models change a lot, in both directions

| Model | Change from FP16 with free decoding (points) |
|---|---|
| Qwen3-1.7B | GPTQ -33.31 |
| Llama-3.2-1B-Instruct | NF4 -20.38, GPTQ -25.00, AWQ -18.59 |
| Llama-3.2-3B-Instruct | NF4 +21.37, GPTQ +28.14, AWQ +8.35 |

All seven are significant (11 of the 20 free changes are). Within each family the smallest model is hurt most, but
size alone does not predict the change: Llama-3.2-3B gains with every 4-bit method, while Qwen3-4B, of similar size,
barely moves.

Evidence: [main table](../results/main_table.md), [tests](../results/significance_free.md),
[figure](../results/figures/penalty_by_size_free.png)

### 3. INT8 is the gentlest precision; GPTQ gives the largest drop most often

INT8 has a significant change for only one model (Qwen3-1.7B, -3.14). GPTQ gives the largest drop of the four
precisions for three of the five models with free decoding (Qwen3-1.7B, Qwen3-8B, Llama-3.2-1B) and for four of the
five with constrained decoding (all but Qwen3-4B). All GPTQ and AWQ checkpoints were made with one recipe and one
calibration set, so this is a result for that recipe.

Evidence: [score changes](score_changes.md), [GPTQ and AWQ report](../results/gptq_awq.md)

### 4. In the small models, most failures are format errors

Over all 25 free runs, 6,844 of the 11,625 failures (58.9%) are syntactic: the answer is not valid JSON, or it does
not match the function's schema. In Qwen3-4B and Qwen3-8B only 6.5% to 10.6% are syntactic; they fail mostly on
argument values. In the three smaller models, quantization mostly moves the syntactic counts. Each small model has
its own format habit:

- Qwen3-1.7B GPTQ: 221 of its 395 unparsed AST answers have one `}` too many
- Llama-3.2-1B: copies the function definition into its answer (413 answers in all five free runs)
- Llama-3.2-3B FP16: leaves out the closing `]`; 503 of its 519 invalid-JSON answers would parse with it

Evidence: [failure classes](../results/taxonomy_free.md), [figure](../results/figures/taxonomy_free.png),
[hand check of the classifier](../notes/taxonomy_check.md) (80 of 80 agree)

### 5. Some "gains" after quantization are only better format

Llama-3.2-3B gains 8.35 to 28.14 points with NF4, GPTQ and AWQ with free decoding, mostly because its habit of
leaving out the closing `]` fades. Under the constraint, which fixes FP16's format too, all three are below FP16
(-2.06 to -6.33). So a higher score after quantization does not mean better calls.

Evidence: [free vs. constrained](../results/constrained_vs_free.md)

### 6. Headline: constrained decoding removes a loss only when it is a format error over the right content

Of the 7 significant free losses:

| Outcome | Runs | Loss removed |
|---|---|---|
| Removed | Qwen3-1.7B INT8, Qwen3-1.7B GPTQ, Llama-3.2-1B AWQ | 111%, 68%, 98% |
| Unchanged (content losses) | Qwen3-4B NF4, Qwen3-8B GPTQ | 9%, 26% (not different from 0) |
| Larger | Llama-3.2-1B NF4, Llama-3.2-1B GPTQ | -49%, -81% |

Example: Qwen3-1.7B GPTQ failed on format on 414 AST questions with free decoding; under the constraint, 279 of them
are right. A schema cannot fix a wrong argument value, so the small content losses of Qwen3-4B and Qwen3-8B stay.

Evidence: [free vs. constrained](../results/constrained_vs_free.md), [table](../results/gap.csv),
[figure](../results/figures/constrained_gap.png)

### 7. Forcing valid JSON can make a model stop calling

Under the constraint, Llama-3.2-1B NF4 and GPTQ escape to an empty list (no call). Llama-3.2-1B GPTQ does this on
1,112 of the 1,150 AST questions, and its score falls from 11.40 (free) to 2.90. It is the only one of the 25 runs
that scores lower with the constraint. The schema has to allow an empty list, because irrelevance questions need it.

Evidence: [constrained runs](../results/constrained.md), [free vs. constrained](../results/constrained_vs_free.md)

### 8. Under the constraint, format errors almost vanish and content errors show

The constraint raises FP16 for every model (Llama-3.2-3B by 39.66 points) and raises 24 of the 25 runs. Syntactic
failures fall from 6,844 to 53, but semantic ones rise from 4,781 to 8,902: many answers that broke the format also
had wrong content, which the constraint now shows. Under the constraint, 8 of the 20 changes from FP16 are
significant (free: 11).

Evidence: [free vs. constrained](../results/constrained_vs_free.md),
[constrained main table](../results/main_table_constrained.md)

### 9. No schema feature breaks across the board as precision drops (stress suite)

The clearest case is objects nested three levels deep in Qwen3-4B:

| Qwen3-4B, nested 3 levels deep (6 questions) | FP16 | INT8 | NF4 | GPTQ | AWQ |
|---|---:|---:|---:|---:|---:|
| Free decoding | 6 | 5 | 0 | 0 | 0 |
| Constrained decoding | 6 | 6 | 6 | 6 | 6 |

All 19 failures have the right values and wrong closing braces, so the constraint removes them (the same pattern as
finding 6). The drop does not pass Holm's correction: with 25 questions per group, a group must lose at least 13 to
pass. Qwen3-8B misses one question in each of two of its ten stress runs and nothing else.

Evidence: [stress-suite report](../results/stress.md), [accuracy](../results/stress_accuracy.csv),
[failure classes](../results/stress_failures.csv)

### 10. The weak spots of the small models are already there at FP16

- Qwen3-1.7B adds an extra closing brace after a number (44 of its 125 number answers with free decoding; FP16 is
  the worst, 9 of 25 right)
- Llama-3.2-3B writes non-ASCII letters as wrong `\u` escapes (5 to 8 of 25 right with the constraint, at every
  precision)
- Llama-3.2-1B answers 23 of the 25 plain control questions with no call at FP16
- No well-formed answer has a wrong number: all 104 wrong values in the number group are in a product ID argument

Only 7 of the 280 stress-suite changes pass Holm's correction.

Evidence: [stress-suite report](../results/stress.md), [tests](../results/stress_significance.csv)

---

## Other findings (setup and method)

- **BFCL reproduced on Kaggle:** Qwen3-4B-Instruct-2507 scores 86.31% (prompt mode) and 87.81% (FC mode), against
  86.44% and 87.88% on the leaderboard ([report](../notes/bfcl_repro.md))
- **Our scorer matches BFCL:** re-scoring the 1,390 reproduction answers gives BFCL's verdict on every question
  ([`src/check_scorer.py`](../src/check_scorer.py))
- **The parser decides a lot for small models:** Llama-3.2-3B FP16 scores 31.38 with our strict parser and would
  score 46.02 with BFCL's bracket repair (diagnostic only; the strict parser was fixed before the runs)
  ([FP16 report](../results/baseline_fp16.md))
- **The schema is very slightly stricter than BFCL's checker:** 14 of the 34,750 free answers are correct by BFCL
  but do not match the schema ([failure classes](../results/taxonomy_free.md))
- **Qwen3-8B AWQ needed three extra settings** that no other checkpoint uses (an FP16 overflow made every value NaN
  without limits on AWQ's scales); the quantization settings and calibration are unchanged
  ([GPTQ and AWQ report](../results/gptq_awq.md), [`notes/decisions.md`](../notes/decisions.md))
