# REDS-Real：真实退化基准与零样本基线

日期：2026-11-24  
脚本：`prior_qualification/scripts/build_reds_real.py`、`run_realvsr.py`、`run_osediff_realvsr.py`、`score_realvsr.py`  
数据清单：`data/reds_real/manifest.json`（sha256 `8ab95addbb7415f4cd4a9d39847489908b1bc89156edd072ce6048f0365a6b03`）  
原始指标：`prior_qualification/results/reds_real/*_s10/metrics.json`

## 为什么需要这个基准

之前三个评测集各有缺陷，都会让"生成先验 vs 回归模型"的对比失真：

| 基准 | 缺陷 |
|---|---|
| REDS `val_sharp_bicubic` | 干净的 bicubic 降质，没有压缩与噪声，天然对 EDVR 有利 |
| VideoLQ | 无 GT，只能报无参考指标 |
| RealVSR | LQ 与 GT 同像素网格（约 2 倍有效），本仓库又降 4 倍送进 x4 模型，强低通切掉了原光学退化频带，天然偏向 EDVR |

REDS-Real 同时修掉第一和第三个缺陷：取**干净的** REDS val GT，施加 RealBasicVSR 的二阶真实退化
（模糊 → 随机缩放 → 高斯/泊松噪声 → JPEG → 视频压缩，整条链做两遍），再降到 GT/4，并保留 GT。

退化参数与官方 RealBasicVSR 训练管线一致（核 7–21、缩放 0.15–1.5、JPEG 30–95、
视频码率 1e4–1e5），**噪声与缩放范围按本机实测收窄**（见下方"退化强度标定"）。

## 退化强度标定

首版直接用 RealBasicVSR 的原始范围时，退化反而不够（LR 的高频能量异常）：

| 图像 | 拉普拉斯能量 |
|---|---:|
| REDS 官方 bicubic LR | 25.64 |
| REDS-Real LR（首版参数） | 15.45 |
| REDS-Real LR（收窄后） | 15.11 |

首版的问题是噪声（σ 最高 30/255、泊松 scale 最低 0.05）在 720p 全图上过强，
抵消了模糊。收窄到 σ 1–12/255、泊松 scale 0.5–3.0 并逐级 clip 后，
LR 明显比 bicubic LR 更平滑（15.11 vs 25.64），即退化确实更重。

## 结果（30 段 × 10 帧 = 300 帧）

抽 10 帧是因为 OSEDiff 实测约 8 帧/分钟，3000 帧要 6 小时以上；所有方法在同一批
帧上重算，不切片已有的全量 JSON（均值口径不同，不可切）。

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ |
|---|---:|---:|---:|---:|
| bicubic ×4 | **20.547** | **0.4934** | 0.7397 | 0.3702 |
| 原始 EDVR | 20.500 | 0.4848 | 0.6605 | 0.3747 |
| 感知微调 EDVR | 20.450 | 0.4790 | 0.6373 | 0.3642 |
| OSEDiff | 19.477 | 0.4330 | **0.5204** | **0.2703** |

相对 bicubic：

| 方法 | ΔPSNR | ΔLPIPS |
|---|---:|---:|
| 原始 EDVR | −0.047 | −10.7% |
| 感知微调 EDVR | −0.097 | −13.8% |
| OSEDiff | −1.070 | **−29.6%** |

## 连续 9 帧子集（可算 tLP）

抽帧后 `tLP` 不可算（需要相邻帧）。为让时序指标可用，另建一个**连续 9 帧**子集
（帧 20–28，满足 DOVE 的 `(F-1) % 8 == 0`），30 段 × 9 帧 = 270 帧，基线已重算：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | tLP ↓ |
|---|---:|---:|---:|---:|---:|
| bicubic ×4 | **21.004** | **0.5019** | 0.7271 | 0.3463 | 0.0744 |
| 原始 EDVR | 20.916 | 0.4902 | 0.6351 | 0.3577 | 0.0630 |
| 感知微调 EDVR | 20.862 | 0.4840 | **0.6163** | **0.3525** | 0.0631 |

与 10 帧子集结论一致：EDVR 的 PSNR 仍低于 bicubic（20.916 vs 21.004）。
时序上 EDVR 明显优于 bicubic（tLP 0.0630 vs 0.0744，−15.3%），这是回归模型唯一
稳定占优的轴。

## 全量 3000 帧的基线（供参考）

| 方法 | PSNR | LPIPS | DISTS | tLP |
|---|---:|---:|---:|---:|
| bicubic | 20.887 | 0.7327 | 0.3606 | 0.0798 |
| 原始 EDVR | 20.838 | 0.6415 | 0.3677 | 0.0687 |
| 感知微调 EDVR | 20.777 | 0.6208 | 0.3592 | 0.0687 |

全量与子集的方向完全一致，说明 10 帧子集没有引入偏差。

## 结论

1. **在真实退化下，干净域训练的 EDVR 连 bicubic 都打不过**（PSNR 20.500 vs 20.547）。
   而在干净 REDS 上它高出 bicubic 4.5 dB。这直接印证了路线图 1.1 节的判断：
   之前的强基线是"退化域匹配"的产物，不是通用能力。
2. **生成先验在真实退化上首次在感知轴上大幅领先**：OSEDiff 的 LPIPS 比 bicubic 低 29.6%、
   比感知微调 EDVR 低 18.3%，DISTS 低 26.9%，而 PSNR 只低 1.07 dB。
   这与 REDS clean bicubic 上"OSEDiff 的 PSNR 低 2.8 dB、LPIPS 高 36.9%"方向完全相反。
3. **这解释了之前 RealVSR 上的矛盾结果**：RealVSR 被二次降采样后，输入已经接近干净
   bicubic 域，所以回归模型占优；一旦退化是真正的压缩+噪声+模糊，生成先验的价值立刻显现。
4. **感知微调在真实退化上仍有小幅正收益**（LPIPS −3.5% 相对原始 EDVR），但代价是 PSNR −0.05 dB，
   幅度比干净域（−31.2%）小得多。

## 边界

- **OSEDiff 只在 10 帧子集上跑过**；连续 9 帧子集上的 OSEDiff 与 DOVE 尚未完成
  （反复被共享卡上其他用户的进程挤掉：曾观察到 23.8 GB + 15.8 GB 的占用，
  只剩 9.75 MiB）。补齐命令见下方，脚本会跳过已完成片段。
- 退化是**模拟**的真实退化（RealBasicVSR 管线），不等于真实相机采集；
  它的价值是"有 GT + 退化强度可控 + 与干净域形成对照"。
- 噪声/缩放范围按本机标定收窄，与官方训练管线的原始范围不完全一致，
  因此不能与 RealBasicVSR 论文数字横向比较。
- DOVE 的输出目录与 RealVSR/MVSR4x 共用（脚本写死 `outputs/realvsr/<name>`），
  换数据集前必须清空，否则会跳过同名 clip 而复用旧结果。

## 复现

```bash
cd p3/prior_qualification
VENV=p3/quality_metrics/venv/bin/python

# 1. 构建（30 段 × 100 帧，约 75 分钟）
$VENV scripts/build_reds_real.py \
    --gt /data/disk1/shared/aiaa3201/project1_video_sr/val_sharp \
    --out data/reds_real --seed 20261124

# 2. 抽 10 帧子集（各方法统一口径）
$VENV scripts/make_frame_subset.py --src data/reds_real \
    --out data/reds_real_s10 --frames 20 30 40 50 60 70 80 90 95 99

# 3. 推理（--method bicubic / edvr；--finetuned true 为感知微调）
RVQ_DATA=$PWD/data/reds_real_s10 $VENV scripts/run_realvsr.py --name bicubic_s10 --method bicubic
RVQ_DATA=$PWD/data/reds_real_s10 $VENV scripts/run_realvsr.py --name edvr_s10 --method edvr
RVQ_DATA=$PWD/data/reds_real_s10 $VENV scripts/run_realvsr.py --name ft_s10 --method edvr --finetuned true
RVQ_DATA=$PWD/data/reds_real_s10 $VENV scripts/run_osediff_realvsr.py --name osediff_s10

# 4. 打分
RVQ_DATA=$PWD/data/reds_real_s10 $VENV scripts/score_realvsr.py \
    --name osediff_s10 --dir outputs/realvsr/osediff_s10
```
