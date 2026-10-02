# Progress Updates

Newest first. One entry per supervisor meeting (Monday and Wednesday, 4:00 pm), pushed before the
meeting.

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

**Next (Update 2, Wed 7 Oct):** reproduce BFCL on Qwen2.5-1.5B-Instruct (FP16) and compare against
the leaderboard; grow related work to 12+ papers with a fresh check for any recent study on
quantization and function calling.
