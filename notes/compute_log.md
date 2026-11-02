# Compute Log

Kaggle GPU time, to keep track of the weekly GPU quota. Kaggle counts the whole session while a GPU
is attached, not only the time the GPU is busy.

| Date | Run | GPU | Time (h) | Notes |
|---|---|---|---:|---|
| 2026-10-03 | Update 2: BFCL reproduction, Qwen3-4B-Instruct-2507, prompt + FC mode (2 × 1,390 questions) | T4 ×2 (one used) | 0.3 | Notebook ran 18 minutes (08:21-08:40 UTC) |
| 2026-10-12 | Update 3: sanity check of our script, Qwen3-4B-Instruct-2507, free mode (1,390 questions) + constrained check (21 questions) | T4 ×2 (one used) | ~1.4 | Session from about 07:30 to 08:51 UTC, including a first try that stopped at the environment check (07:34) and pauses between steps; the GPU was busy for about 13 minutes |
| 2026-10-12 | Update 4: FP16 baselines, 5 models, free mode (5 × 1,390 questions) | T4 ×2 (both for Qwen3-8B, one for the others) | ≥0.6 | Notebook ran 36 minutes (15:02-15:38 UTC); loading and answering took about 34 of them. The session was open longer: the packages came from the cache of an earlier start in the same session, whose time was not recorded |
| 2026-10-29 | Update 5: INT8 and NF4 main session, 9 of 10 runs finished, plus the quick check and INT8 checkpoints | T4 ×2 (one used per run) | ~8.0 | Interactive session, 10:07-18:08 UTC. Includes the Qwen3-8B INT8 run that ran out of memory, the unfinished Qwen3-8B NF4 tries and idle time |
| 2026-10-29 | Update 5: Qwen3-8B NF4 rerun (1,390 questions) | T4 ×2 (one used) | 1.5 | Save & Run All, 18:21-19:49 UTC |
