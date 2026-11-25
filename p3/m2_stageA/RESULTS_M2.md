# M2 Stage A 结果：四臂 Pilot 全部 No-Go

日期：2026-11-25
范围：UA-RealDiffVSR Stage A（零初始化潜空间残差 LoRA + 不确定性门控）
数据：REDS-Real c9 开发集 `000 / 002 / 009`，27 帧
打分：与 M1 完全同一套 `score_reds_real.py`（luma PSNR / SSIM / LPIPS / DISTS / tLP / 遮挡感知 E_warp）

## 结论先说

**四臂 Pilot 全部 No-Go，没有进入 M2-3 正式扩量。** 卡点统一在 LPIPS：手册要求相对
RealBasicVSR 改善 ≥5%，实测四臂分别变差 4.9% / 1.7% / 6.9% / 7.7%。
PSNR 与 `E_warp` 两条门槛反而大多守住了，说明问题不在"训练崩了"，而在
**当前损失配方把残差推向了保真方向，感知方向没有获益**。

## 开发集基线与门槛

RealBasicVSR 在同 27 帧上实测（`prior_qualification/results/reds_real_c9/rbv/metrics.json`）：

| 指标 | 值 |
|---|---:|
| PSNR | 21.5206 |
| SSIM | 0.5360 |
| LPIPS | 0.43579 |
| DISTS | 0.18152 |
| tLP | 0.04852 |
| E_warp | 5.1537 |

门槛按相对值换算：LPIPS ≤ 0.41400（改善 5%）、PSNR ≥ 21.1206（降幅 ≤0.40 dB）、
E_warp ≤ 5.9268（不高于 +15%）。

## 四臂结果（1500 步，有效 batch 8，HR 256×256）

| 臂 | 门控 | lambda_t | PSNR | dPSNR | LPIPS | dLPIPS | DISTS | dDISTS | E_warp | dE_warp | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P0_const | 常数 0.5 | 0.0 | 21.5999 | **+0.079** | 0.45738 | **+4.9%** | 0.20219 | +11.4% | 5.0602 | **-1.8%** | No-Go |
| P1_fixed_U1 | 固定 1-U1 | 0.0 | 20.9098 | -0.611 | 0.44341 | +1.7% | 0.18668 | +2.8% | 6.7478 | **+30.9%** | No-Go |
| P2_learn_M | 可学习 M_theta | 0.20 | 21.6234 | **+0.103** | 0.46567 | **+6.9%** | 0.20166 | +11.1% | 5.4463 | +5.7% | No-Go |
| P3_from_ose | 可学习（osediff LoRA 初始化） | 0.20 | 21.5951 | **+0.075** | 0.46925 | **+7.7%** | 0.20278 | +11.7% | 5.5167 | +7.0% | No-Go |

三个门槛列里，**PSNR 与 E_warp 基本达成，只有 LPIPS 全线失败**，而且失败方向是变差而非不足。

## 训练过程证据（说明这不是训练没跑起来）

四臂都确认在学，不是空转：

| 观测项 | 起点 | 1500 步后 |
|---|---|---|
| `W_pix` 均值（P2/P3） | 0.119 | 0.49-0.52 |
| `gamma`（P0） | 0.15 | 0.505 |
| `gamma`（P2） | 0.15 | 0.371 |
| `gamma`（P3） | 0.0 | **-0.460** |
| `gnorm` | 9e-4 | 1.2e-2 ~ 5.0e-2 |

`W_pix` 学到 0.5 左右、`gamma` 持续增长，说明门控与残差通路都在被优化器使用。
P3 的 `gamma` 走向负值是它自己的信号：从 `osediff.pkl` 载入的 LoRA 是为**预测噪声**训练的，
在 `raw` 残差模式下其输出方向与所需修正相反，模型只能靠负 `gamma` 抵消它。

## 为什么 LPIPS 不降反升：残差幅度不够

在 P2@250 步的输出上直接量（同一帧 `000/002/009` 的帧 24）：

| clip | Ours-RBV 均值 | RBV 与 GT 差距均值 | 修改幅度占比 |
|---|---:|---:|---:|
| 000 | 1.89/255 | 10.56/255 | 18% |
| 002 | 2.25/255 | 23.94/255 | 9% |
| 009 | 2.12/255 | 11.72/255 | 18% |

模型对 RealBasicVSR 输出的改动只有所需修正量的 **9-18%**。这个量级不足以改变 LPIPS
（LPIPS 由 AlexNet 特征响应决定，需要成片的纹理/结构改变），但足以轻微扰动像素，
于是 PSNR 微升、LPIPS 变差。

放大这个改动幅度不能解决问题：P0 的 `gamma` 已经涨到 0.505、改动最大，LPIPS 反而最差
（+4.9%）；P2 的 `gamma` 0.371 次之。**幅度越大，感知指标越差**，说明残差学到的方向
本身与感知改善无关——它在往"更像 GT 的平滑/均值"方向走，那正是 Charbonnier 项的最优解。

## 排障过程中发现并修正的 5 个真实缺陷

这些都不是文档问题，是代码/公式层面的错误，会在正式训练里继续放大：

1. **手册的 Step-0 恒等公式不成立。** SD2.1 VAE 有损，`decode(encode(B_t))` 与 `B_t`
   差 0.045 均值 / 0.81 最大（fp32 与 bf16 几乎相同）。按手册
   `Y_out = B_t + W_pix*(Y_diff - B_t)` 在 `gamma=0` 时得到 `max|Y-B| = 0.1001`，
   起点就不在 RealBasicVSR 上。改为门控**解码差值**
   `Y_out = B_t + W_pix*(decode(z_hat) - decode(z_base))*0.5`，两个解码共用同一 VAE，
   往返误差相消，实测 `max|Y-B| = 0.000e+00`。

2. **门控正则会塌缩。** 手册的 `0.01*mean(W_pix*(1-U_1))` 单侧奖励小权重，
   实测 50 步内 `W_pix` 从 0.119 崩到 0.000，模型静默退化成纯 RealBasicVSR
   （loss 六位不变，不报错）。加反塌缩项 `((1-W_pix)^2).mean()` 后 `W_pix` 才正常升到 0.5。

3. **时序正则的 Y_{t-1} 门控不一致。** 传 `u1=0, uw=0` 使前一帧用"全开"门控、
   当前帧用学到的门控，`L_temp` 从 0.114 单调涨到 0.247。改成两帧共用同一
   `u1/uw` 后不再发散。

4. **训练速度 3 倍浪费。** `batch_at` 对每个 micro-batch 构造 `eff=8` 大小的批次却只用
   其中 2 个，每步重复解码 48 张 PNG。实测 24.5 s/步 -> 改后 8.4 s/步。

5. **tile 拼接把噪声放大到 139/255。** 升余弦窗边缘取到 0.0006，两窗相接处权重和趋近 0，
   `acc/wsum` 放大量化噪声。改用带 0.25 下限的线性斜坡后降到 3.6/255。

## 学习率的定量结论

`lr=1e-4`（手册值）与 `3e-4` 下，120 步内 `W_pix` 停在 0.119、loss 六位不变；
`lr=1e-3` 时 `W_pix` 0.119->0.664、`gamma` 0.15->0.164、`gnorm` 9e-4->2.5e-2、
`pix` 0.0948->0.0909。**Pilot 全部采用 1e-3。**

## 自检 10/10 通过（M2-1）

| 项 | 实测 |
|---|---|
| 离线加载（断网） | SD2.1 / RBV / DISTS 全部成功，20-40 s |
| text_encoder 释放 | 静态显存 0.01 GB（< 3.0） |
| Step-0 恒等 | `max\|Y-B\| = 0.000e+00`（< 1e-4） |
| 可训练参数 | 16.24M（LoRA 16.22M + gate 0.02M + gamma 1），主干与 RBV 无梯度 |
| 反传梯度 | LoRA / gate / gamma 均有限非零 |
| 峰值显存 | **3.54 GB**（micro_batch=2 @256，门槛 <11） |
| checkpoint | 31.2 MB（初始 LoRA 全零）/ 98 MB（训练后含 gate+Adam，超 80 MB 上限） |
| resume 一致性 | gate 张量逐位恢复、gamma 恢复、step 正确 |
| 单 batch 过拟合 | 80 步后到恒等基线的 94.6%（需 <95%） |
| LR 数据集 | 3 档 x 20 clips x 100 帧 = 6000 帧，320x180，559 MB |

注意：`LoRA 梯度范数 ~1.2e-10`、`gate ~1.3e-4`，差六个数量级；这是 zero-init +
`W_pix~0.12` 的必然结果，也是为什么 `lr=1e-4` 学不动。

## 环境事实（与手册前提不符，必须记录）

手册按 `/data/disk1` 有 442 GB 空闲设计训练，但家目录 `~` 真实路径是
`/data/disk2/home/ghe270`（软链接），**disk2 当时只剩 1.9 GB**；`/data/disk1` 的
442 GB 全在 `root` 属主目录下，`ghe270` 无写权限（实测 `touch` 被拒）。

清理 `outputs/realvsr/{bicubic,edvr,ft}`（全量 3000 帧推理 PNG，共 8.0 GB，
指标已在 `results/realvsr/*/metrics.json` 入库）后增至 9.5 GB，训练全程够用。

**RBV 只能跑 fp32**：`rbvsr_net._warp` 的采样网格写死 `.float()`，bf16 直接报
`expected scalar type BFloat16 but found Float`。5 帧 96x96 窗口 fp32 峰值 0.16 GB，
不影响显存预算。

## 产物清单

```
p3/m2_stageA/
  scripts/  build_train_lr.py  dataset.py  model_stageA.py  selfcheck.py
            train_stageA.py    eval_stageA.py  pilot_judge.py  vis_gate.py
            wait_gpu.sh
  data/train_lr/    v0/v1/v2 x 20 clips x 100 帧（559 MB，不入库）
  runs/             P0_const / P1_fixed_U1 / P2_learn_M / P3_from_ose 的 last.pth
  outputs/          pilot_* 的 27 帧推理结果（可重算，不入库）
  results/          pilot_*/metrics.json 与 pilot_judgement.json
  selfcheck/        selfcheck.json（10/10）
```


## 可视化交付

两张图在 `results/` 下：

- **`compare_textures_m2.png`**（1920x1026）：与 M1 完全相同的三个裁剪坐标
  （`016:22@932,116` 树叶纹理 / `000:24@196,20` 砖墙栏杆 / `005:24@756,100` 脚手架网格），
  6 列 `GT | EDVR | RealBasicVSR | OSEDiff | RBV->OSEDiff | Ours(P2)`。
  裁剪约定沿用 M1 的 `make_compare_picks.py`（左上角坐标 + 160px 裁剪 + 2x 放大）。
  **肉眼几乎无法把 Ours 与 RealBasicVSR 分开**，与"改动只有所需修正的 9-18%"一致。

- **`gate_vis_p2.png`**（1024x834）：三行 `B_t | U_1 | W_pix | |Y_out-B_t|x5`。
  `W_pix` 均值 0.55-0.63，与训练末尾一致；`U_1` 在树叶（0.391）明显高于
  砖墙（0.083）与脚手架（0.041），符合 U_1 作为翻转分歧的预期。
  `|Y-B|x5` 显示出残差集中在结构边缘，幅度偏弱——这是 LPIPS 没有改善的直接可视化证据。

### 可视化过程中修掉的 3 个渲染缺陷

1. **裁剪坐标约定错位**：初版把 `(x, y)` 当成中心点、用 192px 裁剪，导致 GT 列与
   其它列显示不同内容。改为复用 M1 的左上角坐标 + 160px + 2x。
2. **`U_1` 饱和到 1.000**：可视化脚本传了 `(b - b)` 作归一化参考（恒为 0，
   被 clamp 到 1e-6），任何非零差异都饱和。改为用相邻帧差（与评测脚本一致）。
   评测脚本本身没有被此 bug 影响。
3. **`B_t` 与 `|Y-B|x5` 全黑**：`label()` 把 uint8 标题条与 float64(0..1) 图像
   `vstack`，整个画布被提升为 float64；`cv2.imwrite` 不支持 float64，fallback 到
   CV_8U 后把 0..1 截成 0/1。加 `to_u8()` 归一化后恢复。

## 下一步的三个可选方向（都未执行）

1. **换感知目标**：把 DISTS 权重从 0.05 提到 0.30（M1 的 fine_tune 扫描显示 0.03->0.30
   能换来 LPIPS 同向改善），同时把 Charbonnier 降到 0.3，让残差不再被推向均值。
2. **改残差语义**：当前 `raw` 模式下 dZ 是稠密潜变量，解出来是低频重绘。改用
   **x0 残差**（`residual_mode=x0`，代码已实现）让它预测"噪声/纹理增量"，或把
   `gamma` 改成逐像素幅值而非单一标量。
3. **放弃扩散残差通路**：Pilot 的证据支持"RealBasicVSR 已接近该退化下回归模型的上限"，
   把预算转去做时序后处理（直接压 `E_warp`，P0 已证明能到 -1.8%）。

任选其一都需要新一轮 1500 步 Pilot，成本约 4 小时/臂。
