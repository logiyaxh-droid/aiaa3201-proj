# EDVR-L 感知微调

目标：在 P0/P3 已证明的缺陷（高频纹理重建不足）上做最小干预，并**如实测出代价**。
这里只记录实现与判据；结论见 `RESULTS.md`。

## 为什么从 EDVR-L 出发

阶段 D 的证据：纹理区块贡献 81%～83% 的总绝对误差，而 EDVR 是 12 个方法里唯一在所有
指标上都最好、且在纹理区恢复比例最高（0.52）的。继续改进只能从它出发，否则要先解释
为什么换一个更差的底座。

## 训练协议

| 项 | 值 | 理由 |
|---|---|---|
| 底座 | `EDVR_L_x4_SR_REDS_official-9f5f5039.pth` | 与 P0/P3 评测同一权重 |
| 数据 | `train_sample_sharp` / `train_sample_bicubic`（20 clip × 100 帧） | 课程自带的 LR，不再自行生成；实测与 PIL bicubic 平均差 0.137/255 |
| 输入 | 5 帧窗口，中心帧为目标 | 与推理完全一致 |
| 裁剪 | LR 64×64 → GT 256×256 | 官方 EDVR 训练 GT 尺寸 |
| 增广 | 水平翻转、转置 | 官方训练增广 |
| 优化器 | Adam(lr 3e-5, betas 0.9/0.99) | 1e-4 会明显遗忘先验（见下） |
| 像素损失 | Charbonnier，eps=1e-12，mean | 与 BasicSR `CharbonnierLoss` 逐位一致（自检差值 0） |
| 感知损失 | DISTS（pyiqa 官方权重） | 与评测同一实现；可微 |
| 对抗 | SN-UNet + relativistic average GAN | BasicSR 官方判别器 |
| 预算 | 900 步 × batch 8 | 约 18 分钟/配置 |

**不修改推理链路**：`p3/quality_metrics/scripts/predictions.py` 里的 EDVR 仍是原权重；
微调权重只作为候选文件单独评估。

## 关键实证结论（影响后续所有比较）

**从收敛权重继续训练本身会造成退化，学习率必须小。** 同样是 400 步：

| lr | PSNR 变化 | DISTS 变化 |
|---|---|---|
| 1e-4 | −0.23 dB | +0.003 |
| 3e-5 | −0.073 dB | +0.0004 |

因此每次扫描都先跑一个 **仅像素的对照组**，用同一 lr、步数和种子；所有感知增益都是
相对这个对照组报的，否则微调本身的损伤会被算到感知损失头上。

## 判据（在扫描前固定）

主判据 DISTS / LPIPS，不用 MUSIQ（自检已证明 MUSIQ 奖励过度锐化）。

有效需同时满足：

1. 30 片段 DISTS 均值相对 EDVR 基线下降 ≥ 5%；
2. PSNR 下降 ≤ 0.5 dB；
3. tLP 不变差；
4. 逐片段配对胜率不是靠少数片段拉动；
5. 高频诊断不自相矛盾：能量上升不能靠平坦区噪声换来。

## 目录

```text
scripts/train.py                  训练（断点续训，保存模型/优化器/步数/三种 RNG）
scripts/smoke.py                  前反向、权重更新、判别器、损失一致性自检
scripts/evaluate.py               固定口径评估（复用 P3 指标与 P0 时序实现）
scripts/sweep_perceptual.py       感知权重扫描（单变量，含像素对照）
scripts/sweep_adversarial.py      对抗权重扫描（固定感知权重）
scripts/hf_diagnostic.py          高频真伪诊断（纹理/平坦分区）
scripts/residual_decomposition.py 残差分解（锐化 vs 合成）
scripts/make_crops.py             并排对照图
scripts/validate_30.py            30 片段最终验证
runs/<name>/                      权重与训练日志
results/<name>/metrics.json       逐片段指标
results/perceptual_sweep.json     扫描汇总
selfcheck/training_smoke.json     流程自检
```

## 复算

```bash
cd p3/quality_metrics
./venv/bin/python fine_tune/scripts/smoke.py
./venv/bin/python fine_tune/scripts/sweep_perceptual.py
./venv/bin/python fine_tune/scripts/sweep_adversarial.py --dists-weight 0.03
./venv/bin/python fine_tune/scripts/validate_30.py --name <name> \
    --checkpoint fine_tune/runs/<name>/step_000900.pth
```

## 已知坑

- `train.py` 需要把 `p0` 加入 `sys.path` 才能导入 `evalkit`（已加）。
- Charbonnier 的 eps 在 BasicSR 里是**直接相加**（1e-12），不是 eps²。写错会让损失与
  官方实现差 8e-6，自检会拦下。
- 网络输出不裁剪，实测约 1% 像素越出 [0,1]，没有发散；推理时统一 clamp+round 到 uint8。
- 训练日志重定向必须 `PYTHONUNBUFFERED=1`。
