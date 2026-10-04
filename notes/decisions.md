# Design Decisions

Every design choice that affects results, with the reason. Newest at the bottom.

| Date | Decision | Reason |
|---|---|---|
| 2026-10-03 | Compute: Kaggle (2× T4) as the main platform, Google Colab as backup | No lab GPU available |
| 2026-10-03 | Full-precision baseline is FP16, not BF16 | T4 GPUs do not support BF16 |
| 2026-10-03 | Qwen2.5-7B in FP16 runs with tensor parallelism across both T4s (superseded below: Qwen2.5 replaced by Qwen3) | About 15 GB of weights does not fit on one 16 GB T4 with room for the KV cache |
| 2026-10-03 | Core scope: 5 models × 5 precisions × 2 decoding modes; everything else is stretch | Fits the 9-week timeline |
| 2026-10-03 | Second model family: Llama-3.2-1B/3B; Phi-4-mini as fallback if access is denied | Two sizes in a second family; designed for on-device use; xLAM-2-1b and Hammer2.1 are Qwen-based |
| 2026-10-03 | License: MIT | Permissive and standard for research code |
| 2026-10-03 | BFCL pinned to `bfcl-eval==2026.3.23` with vLLM 0.8.5, in a separate Python 3.11 environment | Latest BFCL release; vLLM 0.8.5 is the version BFCL pins for local models and still supports T4 GPUs |
| 2026-10-03 | The vLLM server is started manually in FP16, and BFCL connects to it with `--skip-server-setup` | BFCL always launches vLLM with `--dtype bfloat16`, which a T4 cannot run |
| 2026-10-03 | BFCL reproduction model: `Qwen/Qwen3-4B-Instruct-2507`, prompt and FC modes | BFCL v4 dropped Qwen2.5 (July 2025), so Qwen2.5 has no current leaderboard score to compare with; this model is on the leaderboard in both modes, has no thinking mode, and fits on one T4 |
| 2026-10-03 | Qwen family switched from Qwen2.5 to Qwen3: Qwen3-1.7B, Qwen3-4B, Qwen3-8B (Qwen3-0.6B as stretch) | BFCL v4 dropped Qwen2.5; Qwen3 is the current generation and has BFCL v4 leaderboard scores; the three sizes replace Qwen2.5-1.5B/3B/7B one for one |
| 2026-10-03 | Qwen3 runs with thinking mode off (`enable_thinking=False`) | All three sizes come from the same release; short outputs keep T4 runs fast; comparable with Llama-3.2, which has no thinking mode |
| 2026-10-03 | Qwen3-8B in FP16 runs with tensor parallelism across both T4s | About 16 GB of weights does not fit on one 16 GB T4 |
| 2026-10-03 | GPTQ and AWQ checkpoints are made with one recipe for every model, not downloaded | Qwen publishes official AWQ only for Qwen3-4B/8B and no 4-bit GPTQ at all; one recipe for all models keeps the comparison fair |
| 2026-10-05 | The study's 4B model is `Qwen/Qwen3-4B` (thinking off), not `Qwen/Qwen3-4B-Instruct-2507` | Same release and training as Qwen3-1.7B and Qwen3-8B, so size is the only difference between the three; Instruct-2507 has no 1.7B or 8B version. BFCL has no `Qwen/Qwen3-4B` entry and scores Qwen3-1.7B/8B with thinking on, so the core Qwen models have no comparable leaderboard score; our own FP16 runs are the baseline |
| 2026-10-05 | `Qwen/Qwen3-4B-Instruct-2507` is kept only as the sanity check for our own evaluation script (Update 3), compared with the BFCL run (86.31% prompt, 87.81% FC) | It is the only model with a BFCL run on our hardware. Our script uses its own prompt format, so a few points apart is fine; a big gap means a parsing bug |
