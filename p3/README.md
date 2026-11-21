# Part 3 — 探索：像素级不确定性 + 区域路由

> Direction C（Uncertainty-Aware）。PDF 要求"计算像素级不确定性，
> 确定区域交给回归模型、不确定纹理交给生成模型"。

## 后续扩展：新模型尝试与最终结论

本目录的早期内容是"不确定性诊断 + 路由"的第一次探索。后续又完成了三组新模型实验，
**均未通过预设门槛**，最终结论与停止理由见
[`docs/P3_direction_c_final_conclusions.md`](docs/P3_direction_c_final_conclusions.md)。

| 实验 | 内容 | 结论 |
|---|---|---|
| `utrd/` | 冻结 EDVR + 少步残差扩散三原型（确定性 / 均匀噪声 / 不确定性噪声） | 均未超过同容量确定性对照，扩散主线停止 |
| `prior_qualification/` | OSEDiff、SUPIR、DOVE 外部强生成先验的区域资格测试 | 纹理块 LPIPS 全部劣于现有感知微调 EDVR 1.5～2 倍，全部不通过 |
| `stablesr_fusion/` | 冻结 StableSR 单帧生成结果与 EDVR 做输出级融合 | 最优固定混合仅 +0.00045 dB，oracle 上限 +0.020 dB，停止该路线（见 `stablesr_fusion/PROBE_CONCLUSION.md`） |

当前在 REDS bicubic 上真正有效的改进仍是 EDVR 感知微调
（LPIPS −31.2%，PSNR −0.381 dB），见
[`quality_metrics/fine_tune/RESULTS.md`](quality_metrics/fine_tune/RESULTS.md)。

## 为什么选这个方向

Part 2 暴露的是**取舍**而非矛盾：EDVR 在 PSNR/SSIM/LPIPS 上全面领先
（LPIPS 0.2039 对 TecoGAN 0.3337，即 TecoGAN 在感知指标上也并没有赢），
而时域稳定性在旧指标下无法判定。Part 3 要正视这个取舍——先回答
"模型在哪里拿不准"，再做区域路由。

> 早期草稿写的是「EDVR 赢失真指标、TecoGAN 赢感知与时域指标，这是矛盾」，
> 与实测不符，已在 P0 更正；数值未变，只改了结论表述。

PDF 给了三个方向，A（生成式 VSR）和 B（时域一致增强）都要 HuggingFace 权重，
而服务器 `huggingface.co` 不通；C 只用 BasicSR 系模型 + 自算不确定性，**确定做得出来**。

## 内容

```
scripts/
  make_figures.py        5 张报告图的生成脚本
  make_region_profile.py 区域（纹理/平坦）不确定性剖面
  compat_shim.py         让老脚本能跑在新依赖上的兼容层
p3_attn_dump.py          导出 EDVR 时间/空间注意力图，供离线分析
p3_bvsrpp_frames.py      跑 BasicVSR++ 逐帧输出（bvsrpp_frame/）
p3_route_lean.py         精简路由实验
p3_route_lpips.py        以 LPIPS 为目标的路由实验
p3_route_probe.py        路由探针（分歧是否可当误差代理）
p3_soft_fuse.py          软融合（soft mask）
p3_tta.py                翻转 TTA 集成与 U1 spread
p3_uncertainty.py        四个不确定性估计量的指标汇总
diag_attn.py             注意力图诊断
probe_realesrgan.py      Real-ESRGAN 探针
verify2.py               早期输出自检
verify_bvsrpp.py         校验 BasicVSR++ 帧输出
ms_files.py              共享盘文件清单小工具
ms_probe.py / ms_search.py / ms_search2.py  共享盘检索小工具
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
frames/                         实验用帧（不入库，见「数据」一节；按需生成）
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

`frames/`（3 clip × 100 帧、5 个分支、约 1.6 GB）**不入库**：这些是 720p 原始帧，
属于可再生的推理产物，放进 git 会让仓库膨胀到 1.6 GB 以上且无法回退。
需要复算时按下面重新生成即可。

```bash
# 目录约定：p3/frames/{val_sharp,bvsrpp_frame,edvr_frame,gen,attn}/<clip>/<frame>.png
# val_sharp/  从课程共享盘复制（只读）
cp -r /data/disk1/shared/aiaa3201/project1_video_sr/val_sharp p3/frames/val_sharp

# 三个分支的输出各自用对应推理脚本生成
p3/p3_bvsrpp_frames.py             # bvsrpp_frame/
p3/p3_attn_dump.py                 # attn/（含 000_attn.npz + 000_heat.png）
# edvr_frame/ 与 gen/ 的推理脚本在 Part 2 / Part 3 的服务器 scratch 里
```

脚本里的路径是相对的：`make_figures.py` / `make_region_profile.py` 默认取
脚本上一级的 `frames/`，找不到时回退到同级的 `data/`，所以放哪都不用改代码。
仓库里已有的 `figures/` 5 张图与 `metrics/*.json` 就是在这些帧上跑出来的结果，
可直接核对；`frames/` 缺失时脚本会报错退出，不会静默生成空图。
