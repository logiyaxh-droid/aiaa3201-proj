# Evaluation Protocol & Metric Audits

This document specifies the formal metric definitions, implementation conventions, audit verifications, and methodological safeguards enforced across all experiments in this repository (`evalkit/`).

---

## 1. Metric Definitions

### 1.1 Spatial Fidelity Metrics (Full-Reference)

| Metric | Implementation | Convention |
|---|---|---|
| **PSNR** (↑) | `evalkit.metrics.psnr` | Computed on RGB `[0, 255]` (`float64` reference; streaming `float32` verified within `1.8e-6 dB` max discrepancy). Note: Real-world VSR evaluations (`score_reds_real.py`, `eval_cross.py`) use luma PSNR (`cv2.COLOR_RGB2GRAY`, `float64`) consistently across all rows within each table. |
| **SSIM** (↑) | `evalkit.metrics.ssim_gpu` | `7×7` Gaussian window (`float64`), parity-checked against legacy CPU implementation in `results/02_sota_benchmarks/audit/selfcheck_metrics.json`. |

> **Performance Safeguard:** `torch.backends.cudnn.benchmark` must remain `False`. Enabling it causes cuDNN autotuning on the first `float64` `7×7` convolution to stall for **126 s** per clip; with it disabled, the identical bit-exact computation completes in **0.4 s**.

### 1.2 Perceptual & Distribution Metrics

| Metric | Implementation | Convention |
|---|---|---|
| **LPIPS** (↓) | `evalkit.metrics.LPIPS` | Official `lpips==0.1.4` AlexNet trunk, inputs normalized to `[-1, 1]`. All compared methods share the same LPIPS instance within each run. |
| **DISTS** (↓) | `experiments/04_uncertainty_and_ft/quality_metrics/common.py` | Deep Image Structure and Texture Similarity (`DISTS-pytorch`). When DISTS is used in training (e.g., `FT-EDVR`, `B1_uniform`), **it is never cited as sole independent evidence**; independent LPIPS and temporal metrics are always reported alongside it. |
| **FID** (↓) | `experiments/02_sota_benchmarks/fid_full.py` | `pytorch-fid==0.3.0`, `pt_inception-2015-12-05`, 2048-D `pool3`, RGB/255, bilinear resize to `299×299`, `2x-1` normalization, evaluated over **all 3,000 frames** (30 clips × 100 frames) with GT statistics computed once. Covariance square root uses symmetric eigendecomposition (`eigh`, ~1 s) instead of `scipy.linalg.sqrtm` (>25 min), verified to agree within `1.4e-14`. |

> **FID Caveat on Video Frames:** 3,000 frames drawn from 30 contiguous 100-frame clips are temporally correlated, so the effective sample size is smaller than 3,000 independent images. We do not duplicate frames to inflate sample size, and FID numbers should only be compared within this exact protocol.

### 1.3 Temporal Consistency Metrics

A central finding of this repository is that **different temporal metrics measure fundamentally different properties and frequently rank methods in opposite orders**. We therefore compute and report three distinct temporal metrics explicitly:

1. **Formal Reference-Based tLP (↓)** (`evalkit.temporal.tlp`, following TecoGAN):
   $$\text{tLP} = \frac{1}{T-1} \sum_{t=1}^{T-1} \Big| \text{LPIPS}(Y_{t-1}, Y_t) - \text{LPIPS}(G_{t-1}, G_t) \Big|$$
   - Protocol: `cutfr=0`, `crop32=False`, `x100=False`, covering all 99 consecutive frame pairs per clip on REDS-val.
   - Measures whether the **frame-to-frame perceptual dynamics of the SR output ($d_{\text{SR}}$) match those of the ground truth ($d_{\text{GT}}$)**.
   - Undefined (`null`) whenever contiguous GT frames do not exist (e.g., Vimeo-90K center-frame-only GT, VideoLQ).

2. **Flow-warp LPIPS (↓)** (no-reference temporal self-consistency; formerly mislabeled `tLPIPS` in early drafts):
   $$\text{Flow-warp LPIPS} = \frac{1}{|\mathcal{P}|} \sum_{(t-2, t) \in \mathcal{P}} \text{LPIPS}\big(\mathcal{W}(Y_{t-2}, \mathbf{f}_{t \to t-2}),\; Y_t\big)$$
   - Protocol: `stride=2`, Farneback optical flow estimated at `1/4` resolution (`scale=4`).
   - Measures **output smoothness along optical flow trajectories** without requiring GT.

3. **Occlusion-Aware Warping Error $E_{\text{warp}}$ (↓)** (real-world VSR benchmark):
   $$E_{\text{warp}} = \frac{1}{T-1} \sum_{t=1}^{T-1} \frac{\sum_{p} O_t(p)\, \big| Y_t(p) - \mathcal{W}(Y_{t-1}, \mathbf{f}_{t \to t-1})(p) \big|}{\sum_{p} O_t(p)}$$
   - Computed on `[0, 255]` luminance with forward-backward occlusion mask $O_t(p) = \mathbb{1}\!\left[\|\mathbf{f}_{\text{fwd}}(p) + \mathbf{f}_{\text{bwd}}(p + \mathbf{f}_{\text{fwd}}(p))\|_1 \le 3.0\right]$ and an 8-pixel border margin.
   - Always compared against **GT self-warping** ($E_{\text{warp}}(\text{GT}) = 5.757$ on REDS-Real c9): negative $E_{\text{warp}} - E_{\text{warp}}(\text{GT})$ indicates over-smoothing relative to GT, while positive values indicate temporal jitter/flicker absent from GT.

### 1.4 Why Formal tLP and Flow-warp LPIPS Rank Methods Oppositely

On REDS-val (30 clips × 99 pairs/clip, $d_{\text{GT}} = 0.23043$):

| Method | Formal tLP ↓ | $d_{\text{SR}}$ (mean SR change) | $d_{\text{GT}}$ (mean GT change) | Flow-warp LPIPS ↓ |
|---|---:|---:|---:|---:|
| **EDSR ×4** | **0.00723** | 0.22657 | 0.23043 | 0.0811 |
| TecoGAN | 0.01062 | 0.22084 | 0.23043 | 0.0800 |
| EDVR-L | 0.01226 | 0.21864 | 0.23043 | 0.0647 |
| Bicubic + USM | 0.01595 | 0.21481 | 0.23043 | 0.0642 |
| Bicubic | 0.02699 | 0.20346 | 0.23043 | 0.0519 |
| Lanczos | 0.02807 | 0.20236 | 0.23043 | 0.0507 |
| 5-Frame Flow + USM | 0.03485 | 0.19559 | 0.23043 | 0.0276 |
| 3-Frame Flow | 0.03874 | 0.19169 | 0.23043 | 0.0281 |
| 5-Frame Flow | 0.04374 | 0.18669 | 0.23043 | **0.0236** |
| 3-Frame Naive Avg | 0.08946 | 0.14102 | 0.23043 | 0.1017 |
| 5-Frame Naive Avg + USM | 0.12737 | 0.10308 | 0.23043 | 0.1006 |
| 5-Frame Naive Avg | 0.13507 | 0.09536 | 0.23043 | 0.0893 |

**Takeaway:** Temporal low-pass smoothing (`flowtavg5`, `tavg5`) suppresses frame-to-frame detail ($d_{\text{SR}}$ collapses to `0.0954` vs $d_{\text{GT}} = 0.2304$), which rewards Flow-warp LPIPS (`0.0236`) while severely penalizing formal tLP (`0.0437`–`0.1351`). **"Smoother output" does not equal "faithful temporal dynamics."** Both metrics must be reported jointly.

### 1.5 No-Reference Quality Metrics (No-GT Real-World Data)

For VideoLQ and real mobile footage without paired GT, we report **NIQE (↓)**, **MUSIQ (↑)**, and **DOVER (↑, technical and aesthetic raw scores on contiguous 97-frame sequences)**. Three strict boundaries apply:

1. **No-reference scores measure perceived sharpness/naturalness, not faithfulness to GT.** On RealVSR (where GT exists), OSEDiff dominates NIQE/MUSIQ on VideoLQ yet drops **2.1 dB below bicubic** in PSNR.
2. **`selfPSNR` is not a quality metric.** Bicubic achieves higher `selfPSNR` (`23.74`) than EDVR (`20.49`) simply because doing nothing stays closest to the degraded input.
3. **Never divide HR $E_{\text{warp}}$ by LR $E_{\text{warp}}$.** On VideoLQ, `e_warp` is measured at `4×` HR resolution while `e_warp_in` is measured at `1×` LR resolution; Farneback pixel scale and occlusion tolerance (`tol=3.0`) are not invariant across 4× resolution changes. Only **same-resolution horizontal comparisons** across methods are valid (e.g., `B1` `32.882` vs `RealBasicVSR` `33.962`, `−3.18%`).

---

## 2. Verification & Provenance Audits

All audit records are frozen in `results/02_sota_benchmarks/audit/`:

| Audit | File | Verified Property |
|---|---|---|
| **Metric parity** | `selfcheck_metrics.json`, `regression_part1.json` | Exact numerical match between `evalkit` and legacy Part 1/2 measurements (e.g., Bicubic PSNR `26.403205`, Flow-warp LPIPS `0.051851`). |
| **tLP cross-check** | `tlp_check.json` | Verified clip `000` formal tLP (`0.026102`) vs Flow-warp LPIPS (`0.056855`) on the same 100 frames. |
| **Optical-flow direction calibration** | `experiments/01_baselines/flow_gt.py` | Known synthetic sub-pixel/integer translation verifies sign and coordinate convention of `cv2.remap` / `grid_sample` without relying on documentation. |
| **Forward-backward consistency bug check** | `models/warp.py`, `results/02_sota_benchmarks/diagnostics/align/align.json` | Verified consistency test must be a **sum** `|f_fwd(p) + f_bwd(p + f_fwd(p))| <= tol`; writing a difference yields ~14 px median error on identical flow and silently disables the occlusion mask. |
| **TecoGAN grid convention** | `grid_convention.json`, `bf_shift_scan.json`, `teco_ab_000.json` | Measured TensorFlow `bicubic_four` sub-pixel grid shift (`--shift 0.375` LR pixels), recovering **+2.664 dB** on TecoGAN and ruling out grid misalignment as a remaining confound. |
| **REDS split integrity** | `reds_audit_report.json`, `course_val_sharp_hashes.json` | SHA256 verification across 12 frames (`clips 000/011/015/020`, frames `0/49/98`) confirms `val_sharp` matches upstream `snah/REDS` `val_sharp.zip` pixel-for-pixel and has **zero overlap** with `train_sharp.zip` (REDS4). |
| **Strict frame pairing** | `evalkit/data.py` | All evaluations pair LR, SR, and GT strictly by parsed `(clip_id, frame_id)` rather than positional directory sort order. |
