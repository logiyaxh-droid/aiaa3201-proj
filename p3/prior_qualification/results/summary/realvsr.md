# RealVSR 配对评测：真实退化上第一次有 GT 的生成先验检验

日期：2026-11-23  
数据：RealVSR 测试集（ICCV 2021），50 段 × 5 帧 = 250 帧  
脚本：`prior_qualification/scripts/{run_realvsr.py, run_osediff_realvsr.py, score_realvsr.py}`  
原始指标：`prior_qualification/results/realvsr/*/metrics.json`

## 结论

**在有配对真值的真实退化上，OSEDiff 比 bicubic 还差 2.1 dB，且在 50/50 段上全部更差。** 这与 VideoLQ 无参考指标下"生成模型全面领先"的结论方向相反。

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
| OSEDiff | 20.049 | 0.5875 | 0.2710 | 0.1397 | 0.0143 |

相对感知微调 EDVR：

| 方法 | ΔPSNR | ΔLPIPS |
|---|---:|---:|
| 原始 EDVR | +0.165 | +21.5% |
| bicubic | −0.724 | +152.5% |
| OSEDiff | −2.822 | +36.9% |

## 解读

1. **无参考指标不能替代真值对照。** VideoLQ 上 OSEDiff 的 NIQE 4.109/MUSIQ 66.85 优于 EDVR，但在 RealVSR 上它的 PSNR 比 bicubic 还低。高 NIQE/MUSIQ 反映"看起来清晰"，不保证"更像真值"。
2. **感知微调的权衡在真实退化上稳定成立。** 0.165 dB 的 PSNR 换 LPIPS −17.7%、DISTS −28.2%，与 REDS 上的方向一致（REDS 为 −0.381 dB 换 LPIPS −31.2%）。
3. **生成先验的偏差是可测量的。** OSEDiff 并非无意义输出，它的 LPIPS 0.271 好于 bicubic 0.500，只是仍不如回归模型；说明它确实在改善感知质量，但把像素推离了真值。

## 未完成部分

SUPIR 与 DOVE 没有跑成，原因都是权重下载：

| 模型 | 缺失 | 大小 | 状态 |
|---|---|---:|---|
| SUPIR v0F | CLIP2（`CLIP-ViT-bigG-14`） | 10.16 GB | HF 大文件单连接 184 B/s–1.6 KB/s，多线程高并发触发 429 |
| DOVE 5B | CogVideoX transformer + text_encoder 分片 | ~20 GB | 同类限制，实测 8 B/s |

因此本表**只有一个生成模型样本**，不能外推为"所有生成先验在真实退化上都劣于回归模型"。这是本轮最需要补的缺口。

## 复现

```bash
# 构建子集（本地，从 LQ_test.zip / GT_test.zip）
python scripts/build_realvsr_subset.py          # LQ 降 4 倍 + GT 原样，50 段 × 帧 20-24

# 推理
cd p3/prior_qualification
venv/bin/python scripts/run_realvsr.py --name bicubic --method bicubic
venv/bin/python scripts/run_realvsr.py --name edvr    --method edvr
venv/bin/python scripts/run_realvsr.py --name ft      --method edvr --finetuned true
venv/bin/python scripts/run_osediff_realvsr.py --name osediff

# 打分（前三者自带打分；OSEDiff 用同一个 score_realvsr.py）
venv/bin/python scripts/score_realvsr.py --name osediff --dir outputs/realvsr/osediff
```
