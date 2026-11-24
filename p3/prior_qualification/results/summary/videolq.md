# VideoLQ 评测（修正协议后）

日期：2026-11-24（修正版；原版 2026-11-22 有输入 bug，结论作废）  
数据：VideoLQ 50 段 × 5 帧，原生分辨率  
脚本：`prior_qualification/scripts/run_edvr_vlq.py`、`score_vlq_local.py`  
原始指标：`prior_qualification/results/vlq/{edvr_v2,ft_v2}.json`

## 为什么要重跑

原脚本有两处输入错误，都会让结果偏低：

1. **单帧重复 5 次**：注释写"传入真实邻帧"，但代码是 `x[None].repeat(5,1,1,1)`。
   P0 的 `tasks/edvr_neighbor` 已在 REDS 上测过这个改动会让 EDVR 掉 **2.76 dB**。
2. **无抗锯齿的重采样**：`cv2.resize(..., INTER_CUBIC)` 缩小时不展宽核，
   会混叠。同一批帧上 PIL BICUBIC 比 cv2 INTER_CUBIC 高约 **1.5 dB**。

另外原脚本用 `pyiqa.create_metric("niqe")`，该调用会联网取权重，本机无外网时
直接 `URLError: Network is unreachable`。修正版改用本地 checkpoint 的 MUSIQ。

## 修正后结果

| 方法 | MUSIQ ↑ | selfPSNR ↑ |
|---|---:|---:|
| 感知微调 EDVR | **52.51** | 27.44 |
| 原始 EDVR | 51.26 | **27.78** |
| 原始 EDVR（旧，有 bug） | 55.81 | 20.49 |

selfPSNR 从 20.49 升到 27.78，说明旧结果的主要成分是"输入被破坏"
（单帧重复 + 重采样混叠），而不是模型表现。

## 结论

1. **旧 VideoLQ 表（EDVR 55.81 / 微调 55.81 那一版）不可引用**，
   它的 selfPSNR 只有 20.49，低于 bicubic 的 23.74，本质是在测"输入破坏程度"。
2. **修正后两个 EDVR 变体差距很小**：MUSIQ 52.51 vs 51.26（+2.4%），
   selfPSNR 27.44 vs 27.78（微调略低 1.2%）。感知微调的收益在
   VideoLQ 这种无 GT 数据上不明显。
3. **仍然没有 GT**，所以只能报 MUSIQ 与 selfPSNR，不能与 RealVSR / MVSR4x
   的 PSNR/LPIPS 表格比较。

## 未完成

NIQE 需要联网取权重，本机无法获取，本表只报 MUSIQ。补 NIQE 的方法：
在有外网的机器上下载 `niqe` 权重到 `p3/quality_metrics/weights/` 后重跑。
