# RealVSR 配对评测：真实退化上第一次有 GT 的生成先验检验

日期：2026-11-23  
数据：RealVSR 测试集（ICCV 2021），50 段 × 5 帧 = 250 帧  
脚本：`prior_qualification/scripts/{run_realvsr.py, run_osediff_realvsr.py, run_supir_realvsr.py, score_realvsr.py}`  
原始指标：`prior_qualification/results/realvsr/*/metrics.json`

## 结论

**在有配对真值的真实退化上，两个生成模型都不占优**：SUPIR 的 PSNR 比感知微调 EDVR 低 0.79 dB、LPIPS 高 69%；OSEDiff 的 PSNR 比 bicubic 还低 2.1 dB，且 50/50 段全部更差。这与 VideoLQ 无参考指标下"生成模型全面领先"的结论方向相反。

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
| OSEDiff | 20.049 | 0.5875 | 0.2710 | 0.1397 | 0.0143 |

相对感知微调 EDVR：

| 方法 | ΔPSNR | ΔLPIPS |
|---|---:|---:|
| 原始 EDVR | +0.165 | +21.5% |
| bicubic | −0.724 | +152.5% |
| SUPIR | −0.789 | +69.1% |
| OSEDiff | −2.822 | +36.9% |

## 解读

1. **无参考指标不能替代真值对照。** VideoLQ 上 OSEDiff 的 NIQE 4.109/MUSIQ 66.85 优于 EDVR，但在 RealVSR 上它的 PSNR 比 bicubic 还低。高 NIQE/MUSIQ 反映"看起来清晰"，不保证"更像真值"。
2. **感知微调的权衡在真实退化上稳定成立。** 0.165 dB 的 PSNR 换 LPIPS −17.7%、DISTS −28.2%，与 REDS 上的方向一致（REDS 为 −0.381 dB 换 LPIPS −31.2%）。
3. **两个生成模型的失败方式不同。** SUPIR 保真更接近基准（PSNR 22.08，与 bicubic 相当），但感知指标明显差于回归模型（LPIPS 0.335 vs EDVR 0.240）；OSEDiff 感知指标比 bicubic 好（0.271 vs 0.500）却把像素推离真值更远（PSNR 低 2.1 dB）。二者都没有找到"既保真又感知好"的解。

## 未完成部分

DOVE 5B 没有跑成。它需要约 47 GB 显存（bfloat16），而运行期间共享卡上：

| 时刻 | 其他用户占用显存 | 剩余 | 结果 |
|---|---:|---:|---|
| 第一次 | 29.8 + 0.5 GB | 18 GB | `pipe.to("cuda")` 被 OOM killer 杀掉，rc=-9 |
| 第二次 | 20.0 + 17.3 GB | 12 GB | 同上；`enable_model_cpu_offload` 仍因 CPU 内存不足被 SIGKILL |

`enable_sequential_cpu_offload` 在 bfloat16 下触发 `Cannot copy out of meta tensor`，diffusers 0.32.2 的已知限制。

因此结论基于 **2 个生成模型样本**，不能外推为"所有生成先验在真实退化上都劣于回归模型"。

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
$VENV scripts/score_realvsr.py --name osediff --dir outputs/realvsr/osediff
$VENV scripts/score_realvsr.py --name supir  --dir outputs/realvsr/supir
```

SUPIR 需要先按其官方流程准备四份权重（SUPIR-v0F fp16、SDXL juggernaut、CLIP1、CLIP2），并修改 `vendor/SUPIR/{CKPT_PTH.py,options/SUPIR_v0.yaml}` 指向本地路径；CLIP1 必须是**目录**（含 tokenizer），CLIP2 需转成 open_clip 的 `.bin` 容器。
