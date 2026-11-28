# Datasets, Checkpoints, and Reproducibility Guide

Datasets and pretrained weights are **not tracked in git**. This document provides the complete dataset manifest, SHA256 checkpoint verification table, environment configuration, and exact commands to reproduce every benchmark, fine-tuning run, and demo video.

---

## 1. Datasets Manifest

### 1.1 Clean Synthetic Benchmarks

| Dataset | Split & Structure | Resolution & Scale | Provenance & Overlap Audit |
|---|---|---|---|
| **REDS Validation** | `val_sharp/` (GT) & `val_sharp_bicubic/` (×4 LR): 30 clips `000`–`029` × 100 frames | `1280×720` GT / `320×180` LR (MATLAB anti-aliased bicubic `a = −0.5`) | Original REDS official val (BasicSR regroup `240–269`). SHA256 audit (`results/02_sota_benchmarks/audit/reds_audit_report.json`) proves pixel-exact match to upstream `val_sharp.zip` and **zero overlap** with `train_sharp.zip` / REDS4. |
| **REDS Train Sample** | `train_sample_sharp/` & `train_sample_bicubic/`: 20 clips `000`–`019` × 100 frames | `1280×720` GT / `320×180` LR | Used strictly for fine-tuning (`FT-EDVR` and `UA-RealBasicVSR` synthetic 2nd-order training LR); never used for evaluation. |
| **Vimeo-90K Septuplet Test** | `vimeo90k_test/`: 7,824 sequences (full) / 94 subset; GT only on center frame `im4.png` | `448×256` GT / `112×64` LR | Center-frame metrics only; formal tLP is `null` (no contiguous GT). Split overlap with upstream EDVR-Vimeo / TecoGAN training lists is unverified (marked accordingly). |

### 1.2 Real-World & Wild Video Benchmarks

| Dataset | Source | Scale & Protocol | Paired GT |
|---|---|---|---|
| **Wild Video (Ours)** | 5 original handheld recordings in `demos/wild_source/` (`A`–`E`) | 20 clips × 100 frames; calibrated $\sigma=1.5$ + bicubic ×4 LR | Pseudo-GT (real content, synthetic LR) |
| **RealVSR** (ICCV 2021) | Official `LQ_test.zip` / `GT_test.zip` | 50 clips × 5 frames (`512×1024`); measured effective bandwidth ~`2×`, downsampled `×4` for unified ×4 evaluation | Yes |
| **MVSR4x** (CVPRW 2023) | HuggingFace `RuixuanJiang/Video_Deblurring_Datasets` → `MVSR/` (40.18 GB) | 15 clips × 5 frames (`1024×1024`), verified true `×4` wide-vs-telephoto dual-camera | Yes |
| **REDS-Real (Ours)** | Built from clean REDS `val_sharp/` via `experiments/05_generative_priors/build_reds_real.py` | 30 clips × 9 contiguous frames (`c9`, frames `20–28`, 270 frames) & `s10` (300 frames); two-pass RealBasicVSR 2nd-order degradation, manifest SHA256 `8ab95add…` | Yes |
| **VideoLQ** (CVPR 2022) | OpenDataLab `OpenDataLab/VideoLQ` (`openxlab` CLI) | 50 clips (`000`–`049`, `320×240` to `720×576`) | No |

---

## 2. Pretrained Checkpoints Manifest (SHA256 Verified)

Place downloaded checkpoints under `$VSR_CKPT_ROOT`:

| Model | Filename | Size (Bytes) | SHA256 | Training Data & Architecture Notes |
|---|---|---:|---|---|
| **EDSR ×4** | `EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth` | 6,084,475 | `0c287733e70d1b8b8fc5885613ecbe451e5f3010bcae0307612ef5e4aa08dd5f` | **DIV2K** (800 images), single-frame |
| **EDVR-L ×4 (REDS)** | `EDVR_L_x4_SR_REDS_official-9f5f5039.pth` | 82,587,950 (78.8 MiB) | `9f5f50393965194cc180e1313fd162bc85003bb4c2a58c23a434a33fdb3786a2` | REDS `train_sharp` (`val_partition=REDS4`), **`num_frame=5`** (`fusion` in=640) |
| **EDVR-L ×4 (Vimeo)** | `EDVR_L_x4_SR_Vimeo90K_official-162b54e4.pth` | 82,850,094 | `162b54e4935bf7e66c4a2a0cf0d3620bc4ec287eb4d4dfe1587b9ecdcc6768cb` | Vimeo-90K septuplet train, **`num_frame=7`** (`fusion` in=896) |
| **TecoGAN ×4** | `TecoGAN.data-00000-of-00001` | 12,128,916 | `e009bc08ec84c17bd8525252933ceafb164a743727285360259160e81a8e6961` | Vimeo-90K + 28 online videos, **Gaussian ($\sigma=1.5$)** synthetic LR; TF1 checkpoint (requires `--shift 0.375` grid compensation) |
| **TecoGAN ×4** | `TecoGAN.index` | 4,814 | `c100584c2ec416c2daa4175ce43c5d0ebb87f13db5c340441b8e75cee0da70d3` | Companion index file |
| **TecoGAN ×4** | `TecoGAN.meta` | 496,308 | `b5e126d40aced8c952b84c63a3cbe9258848a7f9a866e7e225c44ac29c979259` | Companion meta graph |
| **RealBasicVSR ×4** | `RealBasicVSR_x4.pth` | — | Official `ckkelvinchan/RealBasicVSR` release | Loaded standalone by `models/realbasicvsr.py` (no `mmcv`/`mmediting` dependency); `6.29M` parameters |
| **External Priors** | OSEDiff (`SD 2.1` + `osediff.pkl`), SUPIR (`v0F` + `SDXL`), DOVE (`CogVideoX-5B`) | — | Official HuggingFace releases | Always verify downloaded `.safetensors` shards via `safetensors.safe_open` (file size match alone does not guarantee valid tensor headers). |

---

## 3. Environment Setup & Engineering Notes

### 3.1 Installation

```bash
conda create -n vsr python=3.10 -y
conda activate vsr
pip install -e .
# Optional: BasicSR (for EDSR / EDVR / BasicVSR++ reference models)
pip install basicsr==1.4.2
# Optional: Perceptual IQA (DISTS, NIQE, MUSIQ) and diffusion priors
pip install -e ".[iqa,priors]"
```

### 3.2 Critical Engineering Safeguards

1. **PyTorch / Torchvision Compatibility:** `basicsr==1.4.2` imports `torchvision.transforms.functional_tensor`, which was removed in `torchvision>=0.17`. Our evaluation scripts inject a clean runtime compatibility shim automatically.
2. **FP32 Requirement for RealBasicVSR SPyNet:** `models/realbasicvsr.py::_warp` constructs `grid_sample` sampling coordinates in `float32`. Keep `UARBVSR` in `fp32` (`5.76 GB` peak VRAM for 5-frame `96×96` training crops, `~1.5 GB` for 9-frame `320×180 → 1280×720` whole-clip inference).
3. **cuDNN Autotuning:** Keep `torch.backends.cudnn.benchmark = False` when running `evalkit.metrics.ssim_gpu` to prevent a 126 s stall on `float64` `7×7` convolutions.
4. **Large Diffusion Model Memory Tips:** For SUPIR on 24–32 GB RAM hosts, build the SDXL skeleton under `torch.set_default_dtype(torch.float16)` (`experiments/05_generative_priors/supir_lean.py`) and use `--min_size 768` under GPU memory pressure. For DOVE on 24 GB GPUs, set `--tile_size_hw 256` to prevent VAE OOM.

---

## 4. End-to-End Reproduction Commands

First export paths once:

```bash
export VSR_DATA_ROOT=/path/to/project1_video_sr
export VSR_CKPT_ROOT=/path/to/checkpoints
export RBVSR_CKPT=$VSR_CKPT_ROOT/RealBasicVSR_x4.pth
export RBVSR_DATA=/path/to/reds_real_c9
export VSR_REAL_DATA_ROOT=/path/to/realworld_datasets   # contains mvsr4x/, vlq/, realvsr/, reds_real_c9/
export TORCH_HOME=~/.cache/torch
```

### 4.1 Clean-Domain 12-Method Benchmark & Diagnostics

```bash
# 1. Optical-flow direction self-calibration
python experiments/01_baselines/flow_gt.py

# 2. Full 12-method REDS-val evaluation (resumable; smoke-test clip 000 first)
python experiments/02_sota_benchmarks/full_eval.py --out outputs/full --clips 000 --methods bicubic --verify-clip 000
python experiments/02_sota_benchmarks/full_eval.py --out outputs/full
python experiments/02_sota_benchmarks/fid_full.py outputs/full
python experiments/02_sota_benchmarks/aggregate_full.py outputs/full
python experiments/02_sota_benchmarks/figures_full.py outputs/full

# 3. Diagnostic suite (motion groups, alignment, degradation refutation, neighbor ablation, Vimeo-90K, speed)
python experiments/02_sota_benchmarks/motion_groups.py --out outputs/motion --results outputs/full/results
python experiments/02_sota_benchmarks/align_diag.py --out outputs/align --clips 004 011 019 --frames 12
python experiments/02_sota_benchmarks/teco_degradation.py --out outputs/degradation --clips 000 004 011 019 022 --frames 25 --with-tecogan
python experiments/02_sota_benchmarks/edvr_neighbor.py --out outputs/edvr_neighbor
python experiments/02_sota_benchmarks/vimeo_part2.py --out outputs/vimeo --n 0
python experiments/02_sota_benchmarks/bench_speed.py --out outputs/bench/speed.json --frames 100 --reps 5 --gap 3
```

### 4.2 Wild Video Study

```bash
python experiments/03_wild_video/prep_auto.py demos/wild_source/VID_20261116_113830.mp4 A 4
python experiments/03_wild_video/p1_wild.py gt lr part1
python experiments/03_wild_video/p2_edvr_wild.py gt lr part2
bash   experiments/03_wild_video/p2_teco_wild.sh gt lr part2
python experiments/03_wild_video/p2_teco_metrics_wild.py gt part2
python experiments/03_wild_video/table_all.py
python experiments/03_wild_video/demo_2x2.py A02 B01 C03 D00 E00
```

### 4.3 Clean-Domain EDVR Perceptual Fine-Tuning

```bash
python experiments/04_uncertainty_and_ft/edvr_finetune/smoke.py
python experiments/04_uncertainty_and_ft/edvr_finetune/train.py --lr 3e-5 --w-dists 0.03 --w-adv 0.001 --steps 900
python experiments/04_uncertainty_and_ft/edvr_finetune/validate_30.py
```

### 4.4 Real-World Pixel-Space Fine-Tuning (`UA-RealBasicVSR`) & Soft Blending

```bash
# 1. Build REDS-Real evaluation set & 3-tier training LR
python experiments/05_generative_priors/build_reds_real.py

# 2. Run 8-point pre-training self-check
python experiments/06_realworld_pixel_ft/selfcheck_rbv.py

# 3. Train the three arms (1,500 steps each; ~5.8 GB peak VRAM)
python experiments/06_realworld_pixel_ft/train_rbv_ft.py --name B1_uniform --uniform
python experiments/06_realworld_pixel_ft/train_rbv_ft.py --name B2_uncertain
python experiments/06_realworld_pixel_ft/train_rbv_ft.py --name B2b_perc_boost --w-pix 0.35 --w-dists 0.25 --w-adv 0.005 --w-temp 0.20

# 4. Whole-clip inference on all 30 REDS-Real clips (~1 min/arm) + scoring
python experiments/06_realworld_pixel_ft/eval_rbv_ft.py --run runs/realworld_pixel_ft/B1_uniform/last.pth --name full_b1
python experiments/05_generative_priors/score_reds_real.py --name full_b1 --dir outputs/realworld_pixel_ft/full_b1 --out results/06_realworld_pixel_ft/metrics/full_b1/metrics.json

# 5. Soft blending & constant-weight controls + cross-domain evaluation
python experiments/06_realworld_pixel_ft/blend_sweep.py
python experiments/06_realworld_pixel_ft/eval_cross.py --run runs/realworld_pixel_ft/B1_uniform/last.pth --name cross_mvsr_b1 --dataset mvsr4x
python experiments/06_realworld_pixel_ft/eval_cross.py --run runs/realworld_pixel_ft/B1_uniform/last.pth --name cross_vlq_b1 --dataset vlq
python experiments/06_realworld_pixel_ft/eval_mvsr_blend.py --out outputs/cross_mvsr_blend_sqrt/metrics.json

# 6. Generate final 100-frame 2×2 real-world demo videos (normal + 2× loupe inset)
python experiments/demo_generation/gen_realworld_100f.py
python experiments/demo_generation/make_realworld_blend_100f.py
python experiments/demo_generation/build_realworld_demo.py
```
