# 阶段 A 指标自检（clip 000）

对应路线图 §3.3 的六项检查。**先做自检，再做全量**；任何一项失败只修该项，不扩到全量。

判定门（gate）为下表前四项，全部通过才继续。

| 检查 | 结果 | 判据 |
|---|---|---|
| 1 恒等 | PASS | DISTS(x, x) 最大绝对值 1.19e-07 |
| 2 退化单调性 | PASS | 5 种失真下 DISTS 全部上升 |
| 3 参考实现 | PASS | pyiqa vs 官方 DISTS_pytorch 最大差 0.0（容差 1e-5） |
| 4 重复性 | PASS | DISTS / MUSIQ 差 0.0；DOVER 同种子逐位相同 |
| 5 旧指标回归 | PASS | PSNR 差 7e-07，LPIPS 差 0.0 |
| 6 合理性 | PASS | edvr DISTS < bicubic，bicubic MUSIQ < GT |

## 1. 恒等检查

预测 = GT 时 DISTS 应为 0。实测最大绝对值 **1.19e-07**（float32 精度量级），
uint8 往返后同为 1.19e-07。通过。

## 2. 退化单调性（固定 10 帧）

| 失真 | DISTS | MUSIQ | DOVER 技术 | DOVER 美学 |
|---|---|---|---|---|
| clean | -0.0000 | 66.932 | -0.0885 | 0.0358 |
| blur_sigma2 | 0.1888 | 25.189 | -0.1549 | -0.0027 |
| noise_s12 | 0.1410 | 51.129 | -0.1372 | -0.0005 |
| jpeg_q20 | 0.1169 | 51.652 | -0.1205 | 0.0081 |
| shift_3_2px | 0.0145 | 66.849 | -0.1270 | 0.0467 |
| oversharp_2.5 | 0.0897 | 70.003 | -0.0986 | 0.1229 |

DISTS 对全部 5 种失真都上升，**通过**。MUSIQ / DOVER 未要求单调，但方向已记录：

- MUSIQ 对模糊最敏感（66.9 → 25.2），对 3px 平移几乎无反应（66.8）——这是
  NR 指标的正常行为，它不知道内容错位。
- **oversharp 让 MUSIQ 从 66.9 升到 70.0（高于 clean）**，同时 DISTS 从 0 升到 0.0897。
  这是本轮的"观感变好但内容变差"最直接的单点证据，也是 D4 敏感性值得做的原因。
- DOVER 技术分对 oversharp 的反应（−0.0986）小于模糊（−0.1549），
  说明它对锐化伪影比 MUSIQ 保守。

## 3. 参考实现复核

pyiqa 的 DISTS 与官方 `DISTS_pytorch/DISTS_pt.py` 在 12 对真实图像上逐对比较：

- 最大绝对差 **0.0**（容差 1e-5），12 对全部精确一致。
- 两份权重文件 SHA256 相同（`f91de6bc…`），即 pyiqa 打包的就是官方 `weights.pt`。

因此 DISTS 的数值口径无歧义，VGG16（`vgg16-397923af.pth`）与 DISTS
alpha/beta 权重都已固定并记录哈希。

## 4. 重复性

同一片段连跑两次：

- DISTS 逐帧最大差 0.0，MUSIQ 逐帧最大差 0.0
- DOVER 技术/美学分数逐位相同，采样帧索引完全相同（seed 20261119）

DOVER 的确定性靠显式播种实现：官方 `UnifiedFrameSampler` 用 `np.random.randint`
取时段偏移，`get_spatial_fragments` 用 `torch.randint` 取碎片裁剪位置，
**eval 模式同样随机**。不播种则重复性必然失败。

## 5. 旧指标回归

在本环境重算 clip 000 的 bicubic PSNR / LPIPS，与 P0 存储值比较：

| 指标 | 旧值 | 新值 | 差值 |
|---|---|---|---|
| PSNR | 25.617046 | 25.617046 | 7.0e-07 |
| LPIPS | 0.513352 | 0.513352 | 0.0 |

通过。另外 `scripts/verify_predictions.py` 在 000/002/009 上对 **27 组方法**
做了同样比较（见 `prediction_regression.json`），全部通过：LPIPS 差全为 0.0，
PSNR 差在 1e-07～4e-06 量级。这证明预测生成器与 P0 逐位一致，
新指标测的就是 P0 表格里那些帧。

## 6. 合理性抽查

| 方法 | DISTS | MUSIQ |
|---|---|---|
| gt | -0.0000 | 68.241 |
| bicubic | 0.1780 | 28.560 |
| edvr | 0.0752 | 64.527 |
| tecogan | 0.1417 | 40.301 |

方向符合预期：EDVR 比 bicubic 更接近 GT（DISTS 更低），GT 的 MUSIQ 高于 bicubic。
没有出现"明显视觉错误却有异常高分"的情况。

注意 GT 的 MUSIQ（68.2）只比 EDVR（64.5）高 3.7，说明在这批 REDS 数据上
EDVR 的观感已接近 GT——这不是"EDVR 恢复了真实纹理"，只是两者在 MUSIQ 的
特征空间里同样缺乏极端失真。

## 环境与设施

- 独立 venv，不修改共享的 `/tmp/srenv`；权重 SHA256 全部记录在 `environment.json`。
- `torch.backends.cudnn.benchmark = False`（否则 float64 7×7 conv 首次调用约 126 s）。
- 耗时（干净运行）：DISTS 全 clip 约 13 s、MUSIQ 约 9 s、DOVER 约 18 s；
  自检总耗时 434.5 s。
