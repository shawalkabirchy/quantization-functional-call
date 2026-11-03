# Progress Updates

Newest first. One entry per supervisor meeting (Monday and Wednesday, 4:00 pm), pushed before the
meeting.

---

## Update 8: Accuracy drop by precision and model size (midpoint)
Planned: Wed 28 Oct · Pushed: Tue 3 Nov (delayed: campus closure and illness)

**Done:**
- Key figure: change in accuracy from FP16 for each precision, models smallest first
  ([figure](results/figures/penalty_by_size_free.png))
- Significance tests for all 20 quantized runs, paired by question: McNemar's exact test with Holm's
  correction, and bootstrap 95% intervals ([`src/significance.py`](src/significance.py),
  [report](results/significance_free.md), [table](results/significance_free.csv))
- Paper outline: every section in bullet points, IEEE conference template ([`paper/main.tex`](paper/main.tex))
- Choices recorded in [`notes/decisions.md`](notes/decisions.md)

**Result:** 11 of the 20 changes are significant (Holm-adjusted p < 0.05).

| Model | Significant changes (points) | Not significant |
|---|---|---|
| Qwen3-1.7B | GPTQ -33.31, INT8 -3.14, AWQ +3.38 | NF4 |
| Qwen3-4B | NF4 -2.35 | INT8, GPTQ, AWQ |
| Qwen3-8B | GPTQ -2.31 | INT8, NF4, AWQ |
| Llama-3.2-1B-Instruct | GPTQ -25.00, NF4 -20.38, AWQ -18.59 | INT8 |
| Llama-3.2-3B-Instruct | GPTQ +28.14, NF4 +21.37, AWQ +8.35 | INT8 |

The larger Qwen3 models barely move, with only small changes (at most 2.35 points) significant.
Within each family the smallest model is hurt most, but size alone does not predict the change:
Llama-3.2-3B gains with every 4-bit method. INT8 is the gentlest precision: a significant change only
for Qwen3-1.7B. These are single greedy runs, so the tests cover the choice of questions, not
run-to-run noise.

**For the meeting:** choose one stretch item for Week 7 (pre-execution verifier, API-Bank or KV-cache
quantization), or drop it to catch up (the plan's first fallback rule); confirm the paper template.

**Next (Update 9, planned Mon 2 Nov):** constrained decoding for all 25 combinations; Setup section.

---

## Update 7: Syntactic vs. semantic failure classifier
Planned: Mon 26 Oct · Pushed: Tue 3 Nov (delayed: campus closure and illness)

**Done:**
- [`src/taxonomy.py`](src/taxonomy.py) sorts every failed answer into a class: S1 not valid JSON, S2 does
  not match the schema (syntactic); M1 wrong function, M2 wrong argument value, M3 wrong number of calls,
  M4 a call on an irrelevance question, M5 no call (semantic). The schema is the one constrained
  decoding enforces ([`src/schema.py`](src/schema.py), from Update 3)
- All 25 free runs classified: [report](results/taxonomy_free.md), [counts](results/taxonomy_free.csv),
  [figure](results/figures/taxonomy_free.png)
- Checked by hand on two random samples of 40 failures: the final version agrees on all 80
  ([`notes/taxonomy_check.md`](notes/taxonomy_check.md)). The counts match the scores on every run
- Rules recorded in [`notes/decisions.md`](notes/decisions.md): an invented argument or function name is
  syntactic, because the schema does not allow it

**Result:** 6,844 of the 11,625 failures in the 25 runs (58.9%) are syntactic.

| Model | Syntactic share of failures (5 precisions) |
|---|---|
| Qwen3-1.7B | 24.3% to 63.3% |
| Qwen3-4B | 6.5% to 9.6% |
| Qwen3-8B | 7.5% to 10.6% |
| Llama-3.2-1B-Instruct | 43.9% to 84.9% |
| Llama-3.2-3B-Instruct | 59.0% to 90.1% |

Qwen3-4B and Qwen3-8B fail mostly on argument values, at every precision. In the three smaller models,
quantization mostly moves the syntactic counts: Qwen3-1.7B GPTQ has 273 more syntactic failures than
FP16 and 63 more semantic ones; Llama-3.2-1B GPTQ has 481 more syntactic ones; Llama-3.2-3B GPTQ has
441 fewer. A format error hides any content error in the same answer, so the two counts are not
independent.

**Next (Update 8, planned Wed 28 Oct):** accuracy drop by precision and model size, with significance
tests; paper outline.

---

## Update 6: GPTQ and AWQ results; first main table
Planned: Wed 21 Oct · Pushed: Tue 3 Nov (delayed: campus closure and illness)

**Done:**
- GPTQ and AWQ runs for all five core models (free mode, 1,390 BFCL `non_live` questions each):
  [report](results/gptq_awq.md), [scores](results/summary.csv), notebooks:
  [GPTQ](notebooks/05_gptq_awq_run_gptq.ipynb), [AWQ](notebooks/05_gptq_awq_run_awq.ipynb),
  [Qwen3-8B AWQ](notebooks/05_gptq_awq_run_qwen3-8b_awq.ipynb)
- First main table: all 25 model and precision combinations ([main table](results/main_table.md))
- AWQ checkpoints saved on Hugging Face (private)
- All 10 runs re-scored on a PC: the same verdict on every question. Every prompt is the same as in
  the FP16 run

**Result** (Overall = BFCL's Non-Live Overall Acc, %):

| Model | FP16 | INT8 | NF4 | GPTQ | AWQ |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 72.56 | 69.42 | 69.06 | 39.25 | 75.94 |
| Qwen3-4B | 88.04 | 88.29 | 85.69 | 86.60 | 86.98 |
| Qwen3-8B | 88.29 | 87.27 | 88.00 | 85.98 | 88.42 |
| Llama-3.2-1B-Instruct | 36.40 | 34.94 | 16.02 | 11.40 | 17.81 |
| Llama-3.2-3B-Instruct | 31.38 | 34.02 | 52.75 | 59.52 | 39.73 |

Qwen3-4B and Qwen3-8B stay within 2.35 points of FP16 at every precision. The three smaller models move
a lot, and mostly through format. Qwen3-1.7B falls 33.31 points with GPTQ: 395 of its AST answers do
not parse (121 at FP16), 221 of them with one `}` too many. Llama-3.2-1B falls with NF4, GPTQ and AWQ,
often copying the function definition. Llama-3.2-3B rises with all three, partly because its habit of leaving
out the closing `]` fades. These are single runs, not yet tested for significance (Update 8).
Constrained decoding (Update 9) will show how much of each change is format.

**Problem:** Qwen3-8B AWQ needed three more settings that no other checkpoint uses: an FP16 copy of the
weights on disk (system memory), limits on AWQ's scales (an FP16 overflow made every value NaN), and
smaller working chunks (GPU memory). The quantization settings and calibration are unchanged; the
[report](results/gptq_awq.md) and [`notes/decisions.md`](notes/decisions.md) have the details.

**Next (Update 7, planned Mon 26 Oct):** syntactic vs. semantic failure classifier.

---

## Update 5: INT8 and NF4 results
Planned: Mon 19 Oct · Pushed: Mon 2 Nov (delayed: campus closure and illness)

**Done:**
- INT8 and NF4 runs for all five core models (free mode, 1,390 BFCL `non_live` questions each):
  [report](results/int8_nf4.md), [scores](results/summary.csv),
  [main notebook](notebooks/04_int8_nf4_run_main.ipynb),
  [Qwen3-8B NF4 rerun](notebooks/04_int8_nf4_run_qwen3-8b_nf4.ipynb)
- INT8 checkpoints made with [`src/quantize.py`](src/quantize.py), because vLLM only loads 8-bit
  bitsandbytes models that are already quantized; NF4 is quantized by vLLM while loading
- 4-bit checkpoints made with one recipe: GPTQ for all five models, AWQ for four; the recipe and the
  reasons for it are in [`notes/decisions.md`](notes/decisions.md)
- All 10 runs re-scored on a PC: the same verdict on every question. Every prompt is the same as in
  the FP16 run

**Result** (Overall = BFCL's Non-Live Overall Acc, %):

| Model | FP16 | INT8 | NF4 |
|---|---:|---:|---:|
| Qwen3-1.7B | 72.56 | 69.42 | 69.06 |
| Qwen3-4B | 88.04 | 88.29 | 85.69 |
| Qwen3-8B | 88.29 | 87.27 | 88.00 |
| Llama-3.2-1B-Instruct | 36.40 | 34.94 | 16.02 |
| Llama-3.2-3B-Instruct | 31.38 | 34.02 | 52.75 |

INT8 stays within about 3 points of FP16 on every model, and Qwen3-4B and 8B barely change in either
format. The two big moves are both Llama at NF4, and format plays a large part in both. Llama-3.2-1B
falls 20.38 points: 430 of its AST answers do not parse (109 at FP16), many of them copying the
function definition. Llama-3.2-3B rises 21.37 points, because its habit of leaving out the closing `]`
nearly disappears (503 answers at FP16, 76 at NF4). With the bracket repair from Update 4 applied to
both runs as a check, its gain is 9.60 points, and its irrelevance score falls from 60.42 to 26.25.
These are single runs, not yet tested for significance (Update 8).

**Problem:** The Qwen3-8B AWQ checkpoint is not ready. That job stopped in its first layer with no
error, most likely out of system memory. A fix is tested on small models and has not yet run on the
full model. AWQ failed on the T4 twice before that (GPU memory, then a compiled module we do not
build); both are fixed, and the other four checkpoints are made. Qwen3-8B also needed reruns for INT8
and NF4; the [report](results/int8_nf4.md) has the details.

**Next (Update 6, planned Wed 21 Oct):** GPTQ and AWQ runs; first main results table.

---

## 2026-10-12 (Update 4)

**Done:**
- FP16 baselines for all five core models with the Update 3 script (free mode, 1,390 BFCL `non_live`
  questions each): Qwen3-1.7B, Qwen3-4B and Qwen3-8B (thinking off), Llama-3.2-1B-Instruct and
  Llama-3.2-3B-Instruct ([notebook](notebooks/03_fp16_baselines.ipynb),
  [report](results/baseline_fp16.md), [scores](results/summary.csv))
- Qwen3-8B split across Kaggle's two T4s; the other models use one T4
- All five runs re-scored on a PC: the same verdict on every question

**Result** (Overall = BFCL's Non-Live Overall Acc, %):

| Model | Overall | Irrelevance | Parsed, AST questions |
|---|---:|---:|---:|
| Qwen3-1.7B | 72.56 | 77.92 | 89.48 |
| Qwen3-4B | 88.04 | 84.58 | 99.83 |
| Qwen3-8B | 88.29 | 87.08 | 99.91 |
| Llama-3.2-1B-Instruct | 36.40 | 82.92 | 90.52 |
| Llama-3.2-3B-Instruct | 31.38 | 91.25 | 52.00 |

Qwen3 gains a lot from 1.7B to 4B and very little from 4B to 8B. Llama-3.2-3B scores below
Llama-3.2-1B because 503 of its answers leave out the closing `]` of the JSON array, which our strict
parser counts as a failure (with the `]` added it would score 46.02). Llama-3.2-1B often makes no call
at all (288 of 1,150 AST questions).

**Problem:** For Qwen3-1.7B and Llama-3.2-3B, format failures are already large at FP16, so for these
two models a drop after quantization will mix format and content. The failure classifier (Update 7)
and constrained decoding (Update 9) are planned to separate the two.

**Next (Update 5, Mon 19 Oct):** INT8 and NF4 runs; GPTQ and AWQ checkpoints prepared.

---

## 2026-10-12 (Update 3)

**Done:**
- Our own evaluation script: [`src/run_eval.py`](src/run_eval.py) runs a model with vLLM in free or
  JSON-schema-constrained mode and saves every raw answer, plus the model revision, software versions,
  GPU, memory and speed; [`src/score.py`](src/score.py) scores the answers with BFCL's own AST checker.
  Design choices (prompt format, greedy decoding, strict parser) are in
  [`notes/decisions.md`](notes/decisions.md)
- Scorer checked against BFCL without a GPU: re-scoring the 1,390 Update 2 answers gives BFCL's verdict
  on every question ([`src/check_scorer.py`](src/check_scorer.py))
- Sanity check on Kaggle ([notebook](notebooks/02_eval_sanity_check.ipynb),
  [report](notes/eval_sanity_check.md), [scores](results/summary.csv))
- One config per model in [`configs/`](configs/), each pinned to a Hugging Face commit;
  [compute log](notes/compute_log.md) started

**Result:** The script works. On Qwen3-4B-Instruct-2507 (FP16, one T4) it scores 89.29% overall
(BFCL's Non-Live Overall Acc), against 87.81% (FC mode) and 86.31% (prompt mode) for BFCL's own
harness: 1.48 points from the closer run. Only 9 of 1,390 answers were invalid JSON, all in Java and
JavaScript questions where an argument is a string that contains JSON. Constrained decoding returned
valid JSON for all 21 test questions.

**Problem:** None. Access to Llama 3.2 has been approved, so the second model family stays as planned.

**Next (Update 4, Wed 14 Oct):** FP16 baselines for all 5 core models (Qwen3-1.7B, 4B and 8B,
Llama-3.2-1B and 3B).

---

## 2026-10-05 (Update 2)

**Done:**
- BFCL reproduced on Kaggle (one T4, FP16) with `Qwen/Qwen3-4B-Instruct-2507` in prompt and FC modes,
  on all 1,390 `non_live` questions ([notebook](notebooks/01_bfcl_reproduction.ipynb),
  [report](notes/bfcl_repro.md), [scores](results/bfcl_repro/))
- Related work grown from 5 to 16 directly related papers after a fresh novelty check on arXiv,
  Google Scholar and GitHub ([`notes/related_work.md`](notes/related_work.md))

**Result:** Reproduced. Overall non-live accuracy is 86.31% (prompt) and 87.81% (FC), against 86.44% and
87.88% on the leaderboard. Every category is within 2 points, and no category differs by more than
one question.

**Problem:** vLLM 0.8.5 raises BFCL's temperature of 0.001 to 0.01 and adds the model's default top-k
and top-p, so decoding is near-greedy rather than strictly greedy. Our own evaluation script will set
every decoding parameter explicitly.

**Next (Update 3, Mon 12 Oct):** our own evaluation script (raw outputs saved, scored with BFCL's AST
checker); configs for all models.

---

## 2026-10-03 (Update 1)

**Done:**
- Repository set up: README with the problem statement and four research questions, a 9-week
  [roadmap](ROADMAP.md), and the folder structure
- Project proposal and roadmap added to [`docs/`](docs/)
- Related-work notes started: 5 directly related papers and 10 background papers
  ([`notes/related_work.md`](notes/related_work.md))
- Design decisions log started ([`notes/decisions.md`](notes/decisions.md))

**Result:** Scope fixed: 5 models × 5 precisions × 2 decoding modes = 50 core evaluation runs.

**Problem:** No lab GPU is available, so all experiments will run on Kaggle (2× T4), with Google
Colab as backup. T4 GPUs do not support BF16, so FP16 will be the full-precision baseline.
Access to Llama 3.2 has been requested on Hugging Face and is awaiting approval.

**Next (Update 2, Wed 7 Oct):** reproduce BFCL on Qwen3-4B-Instruct-2507 (FP16, prompt and FC
modes) and compare against the leaderboard; grow related work to 12+ papers with a fresh check for any recent study on
quantization and function calling.
