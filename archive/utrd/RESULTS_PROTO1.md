# U-TRD 阶段 2 结果：三原型对照

日期：2026-11-21  
状态：完成，门槛判定为**不通过**，扩散主线停止  
对应计划：`P3_direction_c_residual_diffusion_roadmap.md` 阶段 0～2

## 结论

在统一预算（1500 步、batch 8、DISTS 0.01、无对抗项、同一 U-Net）下：

- **确定性残差网络**与冻结 EDVR 基本持平，LPIPS 仅改善 1.2%，PSNR −0.02 dB。
- **均匀噪声扩散**严重劣于基线：PSNR −4.38 dB，LPIPS 恶化 74.7%。
- **不确定性噪声扩散**更差：PSNR −5.69 dB，LPIPS 恶化 45.8%，tLP 也明显变差。

方向 C 的核心假设——空间不确定性噪声优于同容量均匀扩散——在本预算内**得到相反结果**。按预设门槛，不扩展 30 片段，不继续加大训练步数，不进入正式训练阶段。

## 数字

开发片段 000 / 002 / 009，固定帧 40–69（各 30 帧），全帧、无缩放、uint8 量化。

| 模型 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | tLP ↓ | 相对基线 ΔPSNR | ΔLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| 冻结 EDVR（基线） | 32.4095 | 0.91391 | 0.15138 | 0.00934 | — | — |
| 确定性残差 | 32.3892 | 0.91344 | 0.14957 | 0.00890 | −0.020 | −1.2% |
| 均匀扩散 | 28.0344 | 0.72738 | 0.26446 | 0.00711 | −4.375 | +74.7% |
| 不确定性扩散 | 26.7161 | 0.76990 | 0.22065 | 0.02821 | −5.693 | +45.8% |

逐片段结果见 `results/*/metrics.json`。

## 失败原因已定位，不是采样公式错误

`scripts/diag_diffusion.py` 在真实训练状态上分别测了 t=1..4 的单步恢复：

| 模型 | 只喂训练时的噪声 | 喂另一种噪声 |
|---|---|---|
| 不确定性扩散 | t=1 单步 PSNR 27.14 | 18.66 |
| 均匀扩散 | t=1 单步 PSNR 29.62 | 17.59 |

两个模型都只在自己训练的噪声分布上表现正常，互换噪声后崩溃。这说明噪声调度、DDIM 更新与训练边际是一致的；失败来自模型学到的生成结果本身，不是推理实现错配。

`scripts/diag_temp.py` 进一步扫描起始噪声：

| 起始噪声倍数 | 均匀扩散 PSNR | 不确定性扩散 PSNR |
|---|---:|---:|
| ×1.00 | 24.31 | 23.76 |
| ×0.50 | 27.91 | 28.24 |
| ×0.25 | 27.85 | 28.83 |
| ×0.10 | 27.76 | 28.94 |
| ×0.00 | 27.74 | 28.96 |

即使把起始噪声降到零，两者仍低于基线 30.79 dB 约 1.8～3.1 dB。所以“起始噪声太强”不是主因，1500 步预算下扩散没有超过同容量确定性网络。

## 这个结果否定和没有否定什么

**已被否定：**

- 在这个小预算、像素空间、4 步、64×64 训练裁剪的设置下，残差扩散不能超过原始 EDVR。
- 空间不确定性噪声没有优于均匀噪声，反而在 PSNR 和 LPIPS 上都更差。
- 该规模下方向 C 的“少加噪保留平坦区”没有转化为整体指标收益。

**未被否定：**

- 更大模型、更多步数、更大裁剪或感知更强的目标函数下，扩散可能有效。
- 图像级不确定性扩散（如 UPSR）在完整预算下的结论不能由本轮直接推翻。
- 本结论只覆盖 REDS bicubic 和上述三条模型。

## 自检记录

`selfcheck/selfcheck.json` 15/15 通过：帧配对、LR/GT 尺寸、EDVR 特征形状、不确定性范围、均匀/不确定噪声 RMS 相等、固定种子复现、三臂前反向与权重更新、10 步过拟合损失下降。

训练耗时：三组并行各约 36 分钟，显存约 6.1 GB/组。

## 复算

```bash
cd archive/utrd
../quality_metrics/venv/bin/python scripts/selfcheck.py
../quality_metrics/venv/bin/python scripts/train.py --mode det --steps 1500 --batch 8 --tag proto1
../quality_metrics/venv/bin/python scripts/train.py --mode uniform --steps 1500 --batch 8 --tag proto1
../quality_metrics/venv/bin/python scripts/train.py --mode uncertain --steps 1500 --batch 8 --tag proto1
../quality_metrics/venv/bin/python scripts/evaluate.py --mode baseline --name baseline_dev
../quality_metrics/venv/bin/python scripts/evaluate.py --mode det --name det_proto1 --checkpoint runs/det_proto1/step_001500.pth
../quality_metrics/venv/bin/python scripts/diag_diffusion.py
../quality_metrics/venv/bin/python scripts/diag_temp.py
```

## 下一步建议

扩散主线按规则停止。若仍要继续方向 C，应先解决“新模型为什么可能比确定性网络更好”这个前提，例如换用预训练生成先验或多步采样预算，而不是在现有小模型上增加步数。确定性残差网络也只带来 1.2% LPIPS 改善，不足以作为新模型成果。
