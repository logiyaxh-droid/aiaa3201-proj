# 全量评测：正式 tLP + FID + 统一空间指标

REDS 官方 validation 划分（课程 `val_sharp`），**30 clip × 100 帧 × 12 个方法**，
全部在本仓库的共用评测包 `p0/evalkit/` 下重跑。执行 2026-11-17，脚本见
[`p0/scripts/full_eval.py`](../scripts/full_eval.py)、[`fid_full.py`](../scripts/fid_full.py)、
[`aggregate_full.py`](../scripts/aggregate_full.py)、[`figures_full.py`](../scripts/figures_full.py)。

## 主表

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 正式 tLP ↓ | Flow-warp LPIPS ↓ | FID ↓ |
|---|---|---|---|---|---|---|
| bicubic | 26.403 | 0.7384 | 0.4835 | 0.0270 | 0.0519 | 41.41 |
| lanczos | 26.546 | 0.7441 | 0.4861 | 0.0281 | 0.0507 | 36.33 |
| bicubic_unsharp | 26.642 | 0.7517 | 0.4398 | 0.0159 | 0.0642 | 35.62 |
| **edsr** | 28.284 | 0.8067 | 0.3087 | **0.0072** | 0.0811 | 16.56 |
| tavg3 | 23.203 | 0.6374 | 0.5595 | 0.0895 | 0.1017 | 65.26 |
| tavg5 | 21.852 | 0.5903 | 0.6052 | 0.1351 | 0.0893 | 97.15 |
| tavg5_unsharp | 21.960 | 0.5983 | 0.5590 | 0.1274 | 0.1006 | 92.15 |
| flowtavg3 | 26.132 | 0.7286 | 0.4993 | 0.0387 | 0.0281 | 46.12 |
| flowtavg5 | 25.834 | 0.7195 | 0.5096 | 0.0437 | **0.0236** | 48.67 |
| flowtavg5_unsharp | 26.164 | 0.7364 | 0.4692 | 0.0349 | 0.0276 | 43.62 |
| **edvr** | **30.904** | **0.8789** | **0.2039** | 0.0123 | 0.0647 | **6.31** |
| **tecogan** | 25.843 | 0.7187 | 0.3337 | 0.0106 | 0.0800 | 40.65 |

PSNR / SSIM / LPIPS / Flow-warp LPIPS 与 2026-11-11/14 的旧结果**逐位一致**
（例：bicubic PSNR 26.403205366330504、flow-warp 0.05185096894639233），
本表只是把旧表补全了 tLP 与 FID 两列。

## tLP 的分解，以及为什么排序反直觉

tLP 衡量的是"输出序列的帧间变化"与"真值序列的帧间变化"的**差距**（99 对/clip）。

| 方法 | tLP | d_SR（输出抖动均值） | d_GT（真值抖动均值） |
|---|---|---|---|
| edsr | **0.00723** | 0.22657 | 0.23043 |
| tecogan | 0.01062 | 0.22084 | 0.23043 |
| edvr | 0.01226 | 0.21864 | 0.23043 |
| bicubic_unsharp | 0.01595 | 0.21481 | 0.23043 |
| bicubic | 0.02699 | 0.20346 | 0.23043 |
| lanczos | 0.02807 | 0.20236 | 0.23043 |
| flowtavg5_unsharp | 0.03485 | 0.19559 | 0.23043 |
| flowtavg3 | 0.03874 | 0.19169 | 0.23043 |
| flowtavg5 | 0.04374 | 0.18669 | 0.23043 |
| tavg3 | 0.08946 | 0.14102 | 0.23043 |
| tavg5_unsharp | 0.12737 | 0.10308 | 0.23043 |
| tavg5 | 0.13507 | 0.09536 | 0.23043 |

EDSR 的 d_SR 最接近 d_GT，所以 tLP 最低；tavg5 把帧抖没了一大半（d_SR 0.0954），
离真值的行为最远，tLP 反而最高。**"输出不抖" ≠ "时域一致"** —— 纯平滑会压低 d_SR，
从而拉大与 d_GT 的差距。这正是无 GT 的 Flow-warp LPIPS 会把若干平滑方法排在前面的原因，
也是两个指标排序几乎相反的原因。所以本仓库并列报告两者，不单独用任一个下"最稳"的结论。

## 配对对照（清单任务 5）

| 对照 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔtLP |
|---|---|---|---|---|
| EDSR − Bicubic | +1.881 | +0.0683 | −0.1747 | −0.0198 |
| 3-frame avg − Bicubic | −3.200 | −0.1010 | +0.0760 | +0.0625 |
| flowtavg3 − Bicubic | −0.271 | −0.0098 | +0.0159 | +0.0118 |
| 5-frame avg − Bicubic | −4.552 | −0.1481 | +0.1217 | +0.1081 |
| tavg3 − tavg5 | +1.352 | — | — | −0.0456 |
| **flowtavg5 − tavg5** | **+3.983** | **+0.1292** | **−0.0955** | **−0.0913** |
| Bicubic + USM − Bicubic | +0.239 | +0.0134 | −0.0437 | −0.0110 |
| flowtavg5 + USM − flowtavg5 | +0.330 | +0.0169 | −0.0404 | −0.0089 |
| flowtavg5 − flowtavg3 | −0.298 | — | — | +0.005 |

"更多邻帧是否加重重影"：**朴素平均会**（3→5 帧再掉 1.35 dB、tLP 再差 0.046），
**光流对齐后基本不会**（3→5 帧只掉 0.30 dB、tLP 差 0.005）。
USM 在两个位置都是"空间变好、tLP 变好、Flow-warp LPIPS 变差"的分裂效果。

## 逐 clip 胜负（脚本统计，不是只看均值）

| 对照 | PSNR | LPIPS | tLP |
|---|---|---|---|
| EDSR vs Bicubic | 30/30 胜（+0.88 ~ +4.37 dB，中位 +1.82） | 30/30 | **30/30** |
| flowtavg5 vs tavg5 | 30/30 胜（+0.74 ~ +6.95 dB，中位 +4.49） | 30/30 | **30/30** |
| EDVR vs EDSR | 30/30 胜（+1.56 ~ +4.39 dB，中位 +2.50） | 30/30 | **4/30**（均值 +0.005 更差） |
| TecoGAN vs Bicubic | **0/30**（−0.40 ~ −0.98 dB） | 30/30 | 30/30 |

EDVR 在 tLP 上不如 EDSR 是最值得注意的一条：它在空间与感知指标上全胜，
但按参考式 tLP 比 EDSR 抖得更多。合理解读是它的输出更锐、帧间变化更大，
而 d_GT 固定；这不等于"时域不一致"，也不能据此反过来说 EDSR 更好，
只能说单一数值不足以判定时域质量。

## FID

配置固定在 `metrics/run_config.json`：pytorch-fid 0.3.0、`pt_inception-2015-12-05` 权重、
2048 维 pool3、RGB/255、bilinear 到 299、`2x-1` 归一化、**全部 3000 帧**、
所有方法同一帧集合、GT 特征只算一次。

EDVR 6.31 ≪ EDSR 16.56 ≪ bicubic_unsharp 35.62 ≈ lanczos 36.33 ≈ TecoGAN 40.65
≈ bicubic 41.41 < flowtavg5_unsharp 43.62 < flowtavg3 46.12 < flowtavg5 48.67
< tavg3 65.26 < tavg5_unsharp 92.15 < tavg5 97.15。

FID 与 PSNR 同向。TecoGAN 的 FID（40.65）只比 bicubic（41.41）好一点，
远不及其 LPIPS 优势（0.3337 vs 0.4835）—— 它在"看起来像"上得分，
但分布层面仍明显偏离 GT。

**限制**：3000 帧来自 30 个连续片段的 100 帧，视频帧高度相关，
名义样本量远大于独立场景数；不通过重复帧凑样本，也不与其它实现的 FID 混用。

## 复算

```bash
cd ~/p0 && export TORCH_HOME=/tmp/torch_home
/tmp/srenv/bin/python scripts/full_eval.py --out ~/p0/out/full     # 可断点续跑
/tmp/srenv/bin/python scripts/fid_full.py  ~/p0/out/full           # 复用缓存特征
/tmp/srenv/bin/python scripts/aggregate_full.py ~/p0/out/full
/tmp/srenv/bin/python scripts/figures_full.py  ~/p0/out/full
```

口径核对：`--clips 000 --methods bicubic --verify-clip 000` 输出
PSNR 25.61704565628519、tLP 0.02610237、flow-warp 0.056854682564735415，与旧 JSON 及
`p0/audit/tlp_check.json` 一致。float32 流式 PSNR 与 evalkit float64 的最大差 1.8e-6 dB
（记录在每行的 `psnr_abs_diff`）。
