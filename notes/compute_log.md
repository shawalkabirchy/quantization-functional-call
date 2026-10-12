# Compute Log

Kaggle GPU time, to keep track of the weekly GPU quota. Kaggle counts the whole session while a GPU
is attached, not only the time the GPU is busy.

| Date | Run | GPU | Time (h) | Notes |
|---|---|---|---:|---|
| 2026-10-03 | Update 2: BFCL reproduction, Qwen3-4B-Instruct-2507, prompt + FC mode (2 × 1,390 questions) | T4 ×2 (one used) | 0.3 | Notebook ran 18 minutes (08:21-08:40 UTC) |
| 2026-10-12 | Update 3: sanity check of our script, Qwen3-4B-Instruct-2507, free mode (1,390 questions) + constrained check (21 questions) | T4 ×2 (one used) | ~1.4 | Session from about 07:30 to 08:51 UTC, including a first try that stopped at the environment check (07:34) and pauses between steps; the GPU was busy for about 13 minutes |
