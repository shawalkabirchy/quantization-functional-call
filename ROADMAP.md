# Roadmap

**Timeline:** 4 October to 2 December 2026 (9 weeks)
**Updates:** presented to the supervisor every Monday (A) and Wednesday (B) at 4:00 pm (Bangladesh time), 18 in total; each update is pushed here before the meeting
**Experiment freeze:** 22 November
Full plan: [`docs/roadmap.tex`](docs/roadmap.tex)

## Week 1 (4-10 Oct): Setup and reproduction
- [x] **Update 1 (Mon 5 Oct):** repository setup: README, roadmap, progress log, related-work notes
- [x] **Update 2 (Wed 7 Oct):** BFCL reproduced on Qwen3-4B-Instruct-2507 (FP16, prompt and FC modes) and compared with the leaderboard; related work grown to 12+ papers

## Week 2 (11-17 Oct): Evaluation script and baselines
- [ ] **Update 3 (Mon 12 Oct):** own evaluation script (raw outputs saved, scored with BFCL's AST checker); configs for all models
- [ ] **Update 4 (Wed 14 Oct):** FP16 baselines for all 5 core models

## Week 3 (18-24 Oct): Quantization sweep
- [ ] **Update 5 (Mon 19 Oct):** INT8 and NF4 runs; GPTQ/AWQ checkpoints prepared
- [ ] **Update 6 (Wed 21 Oct):** GPTQ and AWQ runs; **first main results table**

## Week 4 (25-31 Oct): Failure taxonomy and midpoint
- [ ] **Update 7 (Mon 26 Oct):** syntactic vs. semantic failure classifier
- [ ] **Update 8 (Wed 28 Oct):** **midpoint**: accuracy drop by precision and model size; paper outline

## Week 5 (1-7 Nov): Constrained decoding
- [ ] **Update 9 (Mon 2 Nov):** JSON-schema-constrained runs for all model x precision combinations; Setup section
- [ ] **Update 10 (Wed 4 Nov):** **headline result**: how much of the quantization penalty constrained decoding removes

## Week 6 (8-14 Nov): Schema-stress suite
- [ ] **Update 11 (Mon 9 Nov):** stress suite v1 (~150 cases, 6 failure classes); Introduction drafted
- [ ] **Update 12 (Wed 11 Nov):** stress-suite results by failure class; Related Work section

## Week 7 (15-21 Nov): Stretch item and accuracy per gigabyte
- [ ] **Update 13 (Mon 16 Nov):** one stretch item; latency benchmark
- [ ] **Update 14 (Wed 18 Nov):** accuracy vs. memory and latency curve; Method section

## Week 8 (22-28 Nov): Freeze and full draft
- [ ] **Update 15 (Mon 23 Nov):** experiments frozen (22 Nov); all figures final; Results, Discussion, Limitations
- [ ] **Update 16 (Wed 25 Nov):** full paper draft to supervisor

## Week 9 (29 Nov - 2 Dec): Revisions and release
- [ ] **Update 17 (Mon 30 Nov):** revisions from supervisor feedback; release preparation
- [ ] **Update 18 (Wed 2 Dec):** final paper submitted; public release v1.0 (code, results, stress suite, reproduction steps)
