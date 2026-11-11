# Research Questions and Answers

The four research questions of the project and the answers we have so far.

**State:** after Update 12 (Wed 11 Nov 2026). RQ1 to RQ4 all have answers from finished runs. The answers may still
be refined before the final paper (Update 18). Main findings in one list: [`findings/`](../findings/README.md).

**Setup in short:** 5 models (Qwen3-1.7B, 4B, 8B; Llama-3.2-1B, 3B-Instruct) × 5 precisions (FP16, INT8, NF4,
GPTQ, AWQ) × 2 decoding modes (free, JSON-schema-constrained), on the 1,390 BFCL `non_live` questions and on our
175-question schema-stress suite. Score = BFCL's Non-Live Overall accuracy (%). "Significant" = McNemar's exact test,
paired by question, Holm-adjusted p < 0.05.

| | Question | Short answer |
|---|---|---|
| **RQ1** | How much does quantization hurt function-calling accuracy, and how does that depend on model size and quantization method? | Little for Qwen3-4B and 8B (at most 2.35 points); a lot, in both directions, for the three smaller models (up to 33 points). INT8 is the gentlest. |
| **RQ2** | How much of that damage is syntactic, and how much is semantic? | In the small models, mostly syntactic (format); in Qwen3-4B and 8B, mostly semantic (wrong argument values). |
| **RQ3** | How much of the quantization penalty does grammar-constrained decoding remove, and which errors remain? | All of it when the loss is a format error over the right content; none when it is a content error; and it can make a loss larger when the model then answers with no call. |
| **RQ4** | Which schema features break first? | None breaks across the board. The clearest case is three-level nesting in Qwen3-4B at 4 bits, and it is a format error only. |

---

## RQ1. How much does quantization hurt function-calling accuracy, and how does that depend on model size and quantization method?

**Answer:** it depends mostly on the model.

- **Larger Qwen3 models: very little.** Qwen3-4B and Qwen3-8B stay within 2.35 points of FP16 at every precision.
  Only two of their eight changes are significant: Qwen3-4B NF4 (-2.35) and Qwen3-8B GPTQ (-2.31).
- **Smaller models: a lot, in both directions.** Qwen3-1.7B GPTQ falls 33.31 points. Llama-3.2-1B falls 18.59 to
  25.00 points with NF4, GPTQ and AWQ. Llama-3.2-3B rises 8.35 to 28.14 points with the same three methods.
- **Size:** within each family the smallest model is hurt most. Across families, size alone does not predict the
  change: Llama-3.2-3B gains, while Qwen3-4B, of similar size, barely moves.
- **Method:** INT8 is the gentlest (a significant change only for Qwen3-1.7B, -3.14). GPTQ gives the largest drop
  for three of the five models with free decoding, and for four of the five with constrained decoding.
- **In total,** 11 of the 20 changes from FP16 are significant.

| Change from FP16, free decoding (points) | INT8 | NF4 | GPTQ | AWQ |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | **-3.14** | -3.50 | **-33.31** | **+3.38** |
| Qwen3-4B | +0.25 | **-2.35** | -1.44 | -1.06 |
| Qwen3-8B | -1.02 | -0.29 | **-2.31** | +0.13 |
| Llama-3.2-1B-Instruct | -1.46 | **-20.38** | **-25.00** | **-18.59** |
| Llama-3.2-3B-Instruct | +2.64 | **+21.37** | **+28.14** | **+8.35** |

**Bold** = significant.

**Evidence:** [main table](../results/main_table.md), [significance tests](../results/significance_free.md),
[figure](../results/figures/penalty_by_size_free.png) (Updates 4-8)

---

## RQ2. How much of that damage is syntactic, and how much is semantic?

**Answer:** in the small models, mostly syntactic; in the larger Qwen3 models, mostly semantic.

- Over all 25 free runs, 6,844 of the 11,625 failures (58.9%) are syntactic (not valid JSON, or not matching the
  function's schema).
- Qwen3-4B and Qwen3-8B: only 6.5% to 10.6% of their failures are syntactic. They fail mostly on argument values,
  at every precision, and quantization changes their failure counts by at most 29 answers.
- In the three smaller models, quantization mostly moves the syntactic counts: Qwen3-1.7B GPTQ has 273 more
  syntactic failures than FP16 and 63 more semantic ones; Llama-3.2-1B GPTQ has 481 more syntactic ones;
  Llama-3.2-3B GPTQ has 441 fewer.
- Each small model has its own format habit: one `}` too many (Qwen3-1.7B), a copied function definition
  (Llama-3.2-1B), a missing closing `]` (Llama-3.2-3B at FP16).

| Model | Syntactic share of failures (five precisions) |
|---|---|
| Qwen3-1.7B | 24.3% to 63.3% |
| Qwen3-4B | 6.5% to 9.6% |
| Qwen3-8B | 7.5% to 10.6% |
| Llama-3.2-1B-Instruct | 43.9% to 84.9% |
| Llama-3.2-3B-Instruct | 59.0% to 90.1% |

**Caveat:** an answer with a format error is never checked for content, so a format error can hide a content error.
The constrained runs (RQ3) show what is behind the format errors.

**Evidence:** [failure classes](../results/taxonomy_free.md), [figure](../results/figures/taxonomy_free.png),
[hand check](../notes/taxonomy_check.md) (the classifier agrees on all 80 failures checked by hand) (Update 7)

---

## RQ3. How much of the quantization penalty does grammar-constrained decoding remove, and which errors remain?

**Answer:** the constraint removes a quantization loss only when the loss is a format error over the right content.

| Significant free loss | Free change | Constrained change | Loss removed |
|---|---:|---:|---:|
| Qwen3-1.7B INT8 | -3.14 | +0.33 | 111% |
| Qwen3-1.7B GPTQ | -33.31 | -10.81 | 68% |
| Llama-3.2-1B-Instruct AWQ | -18.59 | -0.44 | 98% |
| Qwen3-4B NF4 | -2.35 | -2.13 | 9% (not different from 0) |
| Qwen3-8B GPTQ | -2.31 | -1.70 | 26% (not different from 0) |
| Llama-3.2-1B-Instruct NF4 | -20.38 | -30.36 | -49% (larger) |
| Llama-3.2-1B-Instruct GPTQ | -25.00 | -45.29 | -81% (larger) |

- **Removed (3 losses):** format errors with the right content underneath. Qwen3-1.7B GPTQ failed on format on 414
  AST questions with free decoding; under the constraint, 279 of them are right.
- **Unchanged (2 losses):** the small losses of Qwen3-4B and Qwen3-8B were already about content (wrong argument
  values), which a schema cannot fix.
- **Larger (2 losses):** forced to write valid JSON, Llama-3.2-1B NF4 and GPTQ answer with an empty list (no call).
- **Free-mode gains disappear:** Llama-3.2-3B's gains of 8.35 to 28.14 points become changes below FP16 (-2.06 to
  -6.33), because the constraint also fixes FP16's format (FP16 rises by 39.66 points).

**Which errors remain:** syntactic failures fall from 6,844 to 53 (none breaks the schema; the 53 were cut off or
stopped early). Semantic failures rise from 4,781 to 8,902, because the constraint fixes the format and shows the
content errors behind it. What is left depends on the model: wrong argument values in Qwen3 (49% to 64% of their
failures), no call in Llama-3.2-1B (71%), and a mix in Llama-3.2-3B. Under the constraint, 8 of the 20 changes from
FP16 are significant (free: 11).

**Evidence:** [free vs. constrained](../results/constrained_vs_free.md), [table](../results/gap.csv),
[figure](../results/figures/constrained_gap.png), [constrained runs](../results/constrained.md) (Updates 9-10)

---

## RQ4. Which schema features (nesting, enums, integer vs. float, optional fields, unicode, large tool menus) break first?

**Answer:** no schema feature breaks across the board as precision drops. The clearest case is objects nested three
levels deep in Qwen3-4B, and it is a format error only.

| Qwen3-4B, nested 3 levels deep (6 questions) | FP16 | INT8 | NF4 | GPTQ | AWQ |
|---|---:|---:|---:|---:|---:|
| Free decoding | 6 | 5 | 0 | 0 | 0 |
| Constrained decoding | 6 | 6 | 6 | 6 | 6 |

- All 19 of these failures have the right values and wrong closing braces; each is right once only its closing
  brackets are fixed, so the constraint removes them (the pattern of RQ3). The drop is the same for all three 4-bit
  methods, but it does not pass Holm's correction: a group of 25 must lose at least 13 questions to pass.
- Qwen3-4B and Qwen3-8B handle every feature at every precision; Qwen3-4B misses at most 2 questions in any
  constrained run.
- The other weak spots are already there at FP16 and change little with precision: Qwen3-1.7B adds an extra brace
  after a number, Llama-3.2-3B writes non-ASCII letters as wrong `\u` escapes, and Llama-3.2-1B answers 23 of the 25
  plain control questions with no call.
- No well-formed answer has a wrong number: the 104 wrong values in the number group are all in a product ID.
- With 30 enum codes, Qwen3-1.7B GPTQ picks a wrong code on 4 of 8 questions, the largest content loss on a stressed
  feature in a Qwen3 model (it rests on 3 questions).
- Only 7 of the 280 changes pass Holm's correction.

**Caveat:** each group has 25 questions, so only large changes can be told from chance.

**Evidence:** [stress-suite report](../results/stress.md), [accuracy](../results/stress_accuracy.csv),
[failure classes](../results/stress_failures.csv), [tests](../results/stress_significance.csv),
[the suite](../data/stress_suite/README.md) (Updates 11-12)
