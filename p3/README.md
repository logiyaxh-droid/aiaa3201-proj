# Part 3 — 探索：像素级不确定性 + 区域路由

> Direction C（Uncertainty-Aware）。PDF 要求"计算像素级不确定性，
> 确定区域交给回归模型、不确定纹理交给生成模型"。

## 为什么选这个方向

Part 2 暴露的矛盾：**EDVR 赢失真指标，TecoGAN 赢感知与时域指标**。
Part 3 要正视这个矛盾——先回答"模型在哪里拿不准"，再做区域路由。

PDF 给了三个方向，A（生成式 VSR）和 B（时域一致增强）都要 HuggingFace 权重，
而服务器 `huggingface.co` 不通；C 只用 BasicSR 系模型 + 自算不确定性，**确定做得出来**。

## 内容

```
scripts/
  make_figures.py        5 张报告图的生成脚本
  make_region_profile.py 区域（纹理/平坦）不确定性剖面
  compat_shim.py         让老脚本能跑在新依赖上的兼容层
metrics/
  uncertainty.json       四个不确定性估计量的 Spearman / 分位误差 / 选择性 PSNR
  attn_diag.json         注意力图诊断
  route_lean.json        精简路由实验结果
  route_lpips.json       以 LPIPS 为目标的路由
  route_probe.json       路由探针
  soft_fuse.json         软融合（soft mask）
  tta.json               TTA 集成 vs 单次推理
figures/
  fig1_uncertainty_compare.png  四种不确定性图 vs 真实误差图
  fig2_calibration.png          误差-不确定性分位曲线
  fig3_error_capture.png        丢弃最不确定 top-k% 能挽回多少平方误差
  fig4_tta_gain.png             TTA 集成 vs 单次，逐 clip PSNR / LPIPS
  fig5_region_profile.png       区域剖面
docs/
  P3_report_draft.md            报告草稿
frames/
  实验用帧（3 clip × 100 帧，各分支输出 + GT + 注意力图）
```

## 不确定性估计量

| 代号 | 含义 |
|---|---|
| `U2_disagree` | 两分支（fidelity / generative）的逐像素输出分歧，**免费**——两个分支本来都要算 |
| `U3_attn` | EDVR 注意力图导出的不确定性 |
| `U4_lap` | 拉普拉斯高频能量（盲基线，用来证明前面两个不是巧合） |
| U1 TTA | 翻转推理的 spread，需先跑 `p3_tta` |

**结论**：`U2_disagree` 与 `U4_lap` 的 Spearman 在 0.40–0.51，注意力导出的 `U3_attn`
接近 0（-0.04）——**注意力图不能当不确定性用**，这是个负面结果，报告里照实写。

## 数据

`frames/` 已包含复算所需的全部帧（约 1.6 GB，3 clip × 100 帧），
明细见 [`frames/README.md`](frames/README.md)。无需再准备数据即可重跑 `scripts/` 出图。
