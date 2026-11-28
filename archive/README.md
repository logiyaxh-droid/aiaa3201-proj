# Archived Explorations

This directory preserves three completed research explorations that were **terminated after controlled negative or No-Go results**. Retaining their code, self-checks, and numeric outputs ensures complete scientific provenance without cluttering the primary pipeline.

| Directory | Method Explored | Domain | Outcome & Root Cause | Detailed Report |
|---|---|---|---|---|
| [`utrd/`](utrd/) | **U-TRD**: Frozen EDVR + 4-step pixel-space residual diffusion (deterministic / uniform noise / uncertainty-scaled noise) | REDS clean bicubic | **Stopped.** Under small-budget pixel-space diffusion, both diffusion variants lost to the capacity-matched deterministic residual control (PSNR −4.38 to −5.69 dB) and uncertainty-scaled noise did not beat uniform noise. | [`utrd/RESULTS_PROTO1.md`](utrd/RESULTS_PROTO1.md) |
| [`stablesr_fusion/`](stablesr_fusion/) | **StableSR Output Fusion**: Frozen single-frame StableSR fused with EDVR | REDS clean bicubic | **Stopped.** Best fixed blend gained only `+0.00045 dB`; oracle per-pixel upper bound was `+0.020 dB`. | [`stablesr_fusion/PROBE_CONCLUSION.md`](stablesr_fusion/PROBE_CONCLUSION.md) |
| [`m2_stageA_latent_lora/`](m2_stageA_latent_lora/) | **M2 Stage A**: Zero-initialized SD 2.1 latent residual LoRA + uncertainty gate (5 pilot arms) | REDS-Real (2nd-order degradation) | **No-Go (all 5 arms).** Even after auditing and fixing 7 implementation bugs, SD 2.1's 8× lossy VAE round-trip error (`~11.5/255`) matched RealBasicVSR's residual error floor and introduced 8×8 cyan block artifacts via `neg_raw` channel bias amplification. Directly motivated pivoting to pixel-space fine-tuning (`experiments/06_realworld_pixel_ft/`). | [`m2_stageA_latent_lora/RESULTS_M2.md`](m2_stageA_latent_lora/RESULTS_M2.md) |

For an integrated academic discussion of all negative results and ablations across the project, see [`../docs/NEGATIVE_RESULTS_AND_ABLATIONS.md`](../docs/NEGATIVE_RESULTS_AND_ABLATIONS.md).
