# RealVSR 配对评测：真实退化上第一次有 GT 的生成先验检验

日期：2026-11-24  
数据：RealVSR 测试集（ICCV 2021），50 段 × 5 帧 = 250 帧  
脚本：`prior_qualification/scripts/{run_realvsr.py, run_osediff_realvsr.py, run_supir_realvsr.py, run_dove_realvsr.py, score_realvsr.py}`  
原始指标：`prior_qualification/results/realvsr/*/metrics.json`

## 结论

**在配对真值下，三个生成模型的 PSNR 全部落后于回归模型（差 0.79–2.82 dB），但感知轴上 DOVE 已经追到回归模型附近。** DOVE 的 LPIPS 只比感知微调 EDVR 高 11.9%，并且优于原始 EDVR（0.2215 vs 0.2404）；SUPIR 和 OSEDiff 在感知轴上也落后。

这与 VideoLQ 无参考指标下"生成模型全面领先"的结论部分相反：领先只在"看起来清晰"这个意义上成立。

## 协议

RealVSR 的 LQ 与 GT 同为 512×1024，不是 4 倍配对。官方测试脚本用 `EDVR_NoUp`（`upscale=1`）直接吃 LQ。

实测 LQ 的有效分辨率约为 GT 的 1/2（高频能量 LQ 0.316% vs GT 降 2 倍 0.298% vs GT 降 4 倍 0.065%）。

为让所有 x4 模型共用一条协议，本实验：

1. LQ 再降 4 倍（PIL BICUBIC，缩小时带抗锯齿前置滤波）→ 128×256；
2. 各 x4 模型重建到 512×1024；
3. 与 GT 原生对齐算 PSNR/SSIM/LPIPS/DISTS/tLP。

两项核对保证这不是协议自身的问题：

- **上限**：GT 自身降 4 倍再上采样，只比本路径高 0.5–2.7 dB（016/108/364 为 26.49/21.27/23.93 dB），输入路径没有额外丢信息；
- **对齐**：±6 px 平移搜索只带来 ≤0.29 dB 收益，LQ–GT 已对齐。

## 结果

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | tLP ↓ |
|---|---:|---:|---:|---:|---:|
| 原始 EDVR | **23.036** | **0.7590** | 0.2404 | 0.1315 | 0.0128 |
| 感知微调 EDVR | 22.871 | 0.7452 | **0.1979** | **0.0944** | **0.0094** |
| bicubic ×4 | 22.147 | 0.6480 | 0.4997 | 0.2118 | 0.0287 |
| SUPIR v0F | 22.082 | 0.6470 | 0.3347 | 0.1774 | 0.0111 |
| DOVE 5B | 21.577 | 0.6666 | 0.2215 | 0.1118 | 0.0177 |
| OSEDiff | 20.049 | 0.5875 | 0.2710 | 0.1397 | 0.0143 |

相对感知微调 EDVR：

| 方法 | ΔPSNR | ΔLPIPS |
|---|---:|---:|
| 原始 EDVR | +0.165 | +21.5% |
| bicubic | −0.724 | +152.5% |
| SUPIR | −0.789 | +69.1% |
| DOVE | −1.294 | **+11.9%** |
| OSEDiff | −2.822 | +36.9% |

## 解读

1. **"生成先验在真实退化上更差"这个概括不成立，需要分轴说。** PSNR 轴上三个生成模型全部落后（−0.79 到 −2.82 dB）；感知轴上 DOVE 已经和回归模型接近，SUPIR/OSEDiff 没有。把两者混成一个"好/坏"判断会丢掉这条差别。
2. **无参考指标不能替代真值对照。** VideoLQ 上 OSEDiff 的 NIQE 4.109/MUSIQ 66.85 优于 EDVR，但在 RealVSR 上它的 PSNR 比 bicubic 还低 2.1 dB，且 50/50 段全部更差。高 NIQE/MUSIQ 反映"看起来清晰"，不保证"更像真值"。
3. **感知微调的权衡在真实退化上稳定成立。** 0.165 dB 的 PSNR 换 LPIPS −17.7%、DISTS −28.2%，与 REDS 上的方向一致（REDS 为 −0.381 dB 换 LPIPS −31.2%）。
4. **DOVE 存在约 +4.5/255 的整体亮度偏高。** clip 016 第 20、22 帧的 RGB 三通道一致偏高 4.2–4.8，这解释了一部分 PSNR 损失，但不改变排名。
5. **三个生成模型的失败方式不同。** SUPIR 保真接近 bicubic 但感知最差；OSEDiff 感知优于 bicubic 但像素偏离最远；DOVE 在两条轴上都是三者中最好的，但仍未超过回归模型。

## 复现

```bash
# 构建子集（本机，从 LQ_test.zip / GT_test.zip）
python p3/prior_qualification/scripts/build_realvsr_subset.py \
  --lq LQ_test.zip --gt GT_test.zip --out p3/prior_qualification/data/realvsr

cd p3/prior_qualification
VENV=p3/quality_metrics/venv/bin/python
$VENV scripts/run_realvsr.py --name bicubic --method bicubic
$VENV scripts/run_realvsr.py --name edvr    --method edvr
$VENV scripts/run_realvsr.py --name ft      --method edvr --finetuned true
$VENV scripts/run_osediff_realvsr.py --name osediff
$VENV scripts/run_supir_realvsr.py
$VENV scripts/run_dove_realvsr.py
$VENV scripts/score_realvsr.py --name osediff --dir outputs/realvsr/osediff
$VENV scripts/score_realvsr.py --name supir   --dir outputs/realvsr/supir
$VENV scripts/score_realvsr.py --name dove    --dir outputs/realvsr/dove
```

## 复现注意

- **SUPIR**：四份权重（SUPIR-v0F fp16、SDXL juggernaut、CLIP1、CLIP2）就位后，需改 `vendor/SUPIR/{CKPT_PTH.py, options/SUPIR_v0.yaml}` 指向本地路径。CLIP1 必须是**目录**（含 tokenizer），CLIP2 需转成 open_clip 的 `.bin` 容器。加载时用 `scripts/supir_lean.py`（fp16 骨架 + 逐张量流式加载），否则在 31 GB 内存机器上会被 OOM killer 杀掉。
- **DOVE**：输入须是视频文件（模型只 glob 视频扩展名），所以帧序列先经 imageio-ffmpeg 的 ffmpeg 编成 mp4。DOVE 输出写成 `<out>/<clip>/<clip>/NNN.png` 并从 000 编号，脚本会重映射回 RealVSR 帧号。首轮 50 段中 17 段因 OOM（`rc=-9`）失败，需分批补跑。
- **可复现性**：本次三个生成模型都在共享主机的资源竞争下完成，DOVE 的 PSNR 会受其他用户占用影响（OOM 导致部分片段重跑）。重跑时建议确认 GPU 至少 40 GB 空闲、内存至少 20 GB 空闲。
