# Progress Updates

Newest first. One entry per supervisor meeting (Monday and Wednesday, 4:00 pm), pushed before the
meeting.

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
