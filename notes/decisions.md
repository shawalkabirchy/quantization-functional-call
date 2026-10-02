# Design Decisions

Every design choice that affects results, with the reason. Newest at the bottom.

| Date | Decision | Reason |
|---|---|---|
| 2026-10-03 | Compute: Kaggle (2× T4) as the main platform, Google Colab as backup | No lab GPU available |
| 2026-10-03 | Full-precision baseline is FP16, not BF16 | T4 GPUs do not support BF16 |
| 2026-10-03 | Qwen2.5-7B in FP16 runs with tensor parallelism across both T4s | About 15 GB of weights does not fit on one 16 GB T4 with room for the KV cache |
| 2026-10-03 | Core scope: 5 models × 5 precisions × 2 decoding modes; everything else is stretch | Fits the 9-week timeline |
| 2026-10-03 | Second model family: Llama-3.2-1B/3B; Phi-4-mini as fallback if access is denied | Two sizes in a second family; designed for on-device use; xLAM-2-1b and Hammer2.1 are Qwen-based |
| 2026-10-03 | License: MIT | Permissive and standard for research code |
