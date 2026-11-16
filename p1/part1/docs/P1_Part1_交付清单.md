# P1 · Part 1 交付清单（Baseline：Hand-crafted + Compact CNN）

> 依据：`Project1_VideoSuperResolution.pdf` §5.1、§6；服务器 `ghe270@121.12.5.46:10084` 实测（2026-11-11）。
> 本文所有数字都是**在 REDS val 全 30 clip × 100 帧上真实跑出来的**，不是估算。

---

## 0. PDF 对 Part 1 的原文要求

> **5.1 Part 1: The Baseline – "Hand-crafted Approach"**
> Goal: Implement a basic video upscaling pipeline using classic image processing and early deep learning models to understand the fundamental trade-offs between speed and PSNR.
>
> **1. Spatial Upsampling: Interpolation & SRCNN**
> - **Step 1: Classic Interpolation.** Implement Bicubic and Lanczos resampling as the absolute baseline.
> - **Step 2: Residual CNN.** Implement or adapt a compact EDSR baseline [3] to learn the mapping from LR to HR patches. **Use the provided pretrained checkpoint for inference** if training resources are limited; a small patch-based training run is optional.
>
> **2. Temporal Baseline: Multi-frame Averaging**
> - **Algorithm:** For each frame, perform a **weighted average of its spatial neighbors (after Bicubic upscaling)** to observe the effect of simple temporal fusion on denoising and sharpening.
> - **Advanced Idea:** Apply **Unsharp Masking** on the averaged result to enhance high-frequency edges.
>
> **Expected Result:** The EDSR baseline should improve detail reconstruction over interpolation on synthetic bicubic data, while processing each frame independently and therefore providing **no explicit temporal consistency**.

拆成可验收的 4 件事：

| # | 要交的东西 | 验收标准 |
|---|---|---|
| 1 | Bicubic / Lanczos 逐帧上采样 | 跑满 30 clip，出 PSNR/SSIM |
| 2 | EDSR（老师给的 checkpoint）推理 | 明显高于插值，且**逐帧独立** |
| 3 | 多帧加权平均 | 跑出来，并**如实报告它对 PSNR 和时域一致性的影响**（实测：两个都变差，见 §1②） |
| 4 | + Unsharp Mask | 高频边缘增强，PSNR 略有回升 |

**Part 1 的真正考点不是分数，是让你亲手做出这个矛盾**：EDSR 赢在空间细节但完全不管时间轴（实测 tLPIPS 反而最差）；朴素多帧平均想改善时域一致性，但在有运动时**适得其反**。这个矛盾正是 Part 2（EDVR 的 PCD 对齐 + TSA 融合）要解决的东西 —— 而且我在 §3 用光流对齐证明了：**是"缺对齐"而不是"融合"本身的问题**。

---

## 1. 实测结果（REDS val，30 clip × 100 帧）

LR = `val_sharp_bicubic`（320×180），GT = `val_sharp`（1280×720）。
PSNR/SSIM 在 **Y 通道**上算（REDS/EDVR 的惯例，与论文可比）；LPIPS 用 AlexNet v0.1。

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | tLPIPS ↓ |
|---|---|---|---|---|
| Bicubic | 26.403 | 0.73838 | 0.4835 | 0.0519 |
| Lanczos | 26.546 | 0.74411 | 0.4861 | 0.0507 |
| Bicubic + USM | 26.642 | 0.75173 | **0.4398** | 0.0642 |
| **EDSR (x4)** | **28.284** | **0.80665** | **0.3087** | 0.0811 |
| 3-frame avg | 23.203 | 0.63737 | 0.5595 | 0.1017 |
| 5-frame avg | 21.852 | 0.59027 | 0.6052 | 0.0893 |
| 5-frame avg + USM | 21.960 | 0.59826 | 0.5590 | 0.1006 |
| Flow-aligned 3-frame | 26.132 | 0.72861 | 0.4993 | 0.0281 |
| **Flow-aligned 5-frame** | 25.834 | 0.71951 | 0.5096 | **0.0236** |
| Flow-aligned 5-frame + USM | 26.164 | 0.73640 | 0.4692 | 0.0276 |

> 最后三行是我加的消融（见 §3），PDF 里没有，但这是把"为什么需要对齐"讲清楚的关键证据。

### 三个必须写进报告的观察

**① EDSR 稳定赢插值，30/30 个 clip 全赢。**
EDSR 相对 bicubic 的增益 **最小 +0.88 dB，中位数 +1.82 dB，最大 +4.37 dB**，无一例外。这验证了 PDF 的 Expected Result。

**② 朴素多帧平均把 PSNR 打掉了 4.55 dB，tLPIPS 也变差了（0.0519 → 0.0893，恶化 72%），30 个 clip 里只有 6 个变好。**
这是本 Part 最反直觉、也最值得写的发现：**"多帧平均能改善时域一致性"这个朴素直觉，在真实运动下是错的。** 原因是逐帧 bicubic 后直接平均，所有运动物体被叠加成重影，而重影本身在帧间并不稳定，temporal LPIPS 反而增大。这正是 Part 2 必须引入**显式对齐**（PCD）的动机。

**③ 加上光流对齐后，同样的 5 帧融合从 21.85 dB 跳到 25.83 dB（+3.98 dB），tLPIPS 从 0.0893 降到 0.0236（-73.6%），30/30 个 clip 全部改善。**
这就是 Part 1 → Part 2 的完整叙事链：**融合本身没错，错在没有对齐。**（逐 clip 增益：最小 +0.74 dB，中位数 +4.49 dB，最大 +6.95 dB。）

### EDSR 是"逐帧独立"的直接证据

EDSR 的 PSNR 最高（28.28 dB），但 tLPIPS 是**最差**的一档（0.0811，比 bicubic 差 56%），且 **30/30 个 clip 全部更差**。因为每帧独立推理，相邻帧的高频细节估计并不一致，产生 flicker。这就是 PDF 那句 "providing no explicit temporal consistency" 的定量证据 —— **不是没测出来，是测出来了而且很明显**。

值得注意的对照：flow-aligned 5 帧平均的 PSNR（25.83）低于 EDSR（28.28），但 tLPIPS（0.0236）只有 EDSR 的 **29%**。这构成一个干净的二维权衡图：横轴 PSNR、纵轴 tLPIPS，四个方法分别落在四个象限附近。

---

## 2. Vimeo-90K 测试集（可选加分项，已跑完）

`vimeo90k_test`：94 个序列、7824 个 clip，LR 112×64，GT 448×256。
**注意：每个 clip 只有 `im4` 有 GT**（`target/<seq>/<clip>/im4.png`），所以只能评中心帧，**不能算 tLPIPS**。

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Bicubic | 30.150 | 0.86186 | 0.2905 |
| Lanczos | 30.398 | 0.86629 | 0.2883 |
| Bicubic + USM | 30.409 | 0.86705 | 0.2531 |
| **EDSR (x4)** | **33.495** | **0.91578** | **0.1342** |

EDSR 相对 bicubic **+3.35 dB**，比 REDS 上的 +1.88 dB 大得多 —— 因为 Vimeo 的退化更接近 EDSR 训练时的 DIV2K bicubic 退化。这个对比本身是很好的分析素材：
**同一个模型，在不同退化分布上的增益差别接近 2 倍**，说明单帧 SR 模型的泛化高度依赖退化匹配。

---

## 3. 我额外做的消融（建议保留）

PDF 的 "Advanced Idea" 只提了 Unsharp Masking。实测 USM 的效果是**分裂的**：bicubic+USM 的 PSNR +0.24 dB、LPIPS 从 0.4835 改善到 0.4398（提升 9%），但 tLPIPS 从 0.0519 恶化到 0.0642（差 24%）。**它让单帧更好看、让时间轴更抖。** 这个反差本身就是很好的讨论材料。

但真正值得做的是**把"融合"和"对齐"拆开**：

| 变体 | 含义 | 作用 |
|---|---|---|
| `tavg3` / `tavg5` | PDF 原版：纯加权平均 | 复现 PDF 要求的基线 |
| `flowtavg3` / `flowtavg5` | 用 Farneback 光流先把邻帧 warp 到中心帧再平均 | 证明"对齐"是融合有效的必要条件 |
| `*_unsharp` | 在平均结果上做 USM（radius=1.0, amount=1.5） | 落实 PDF 的 Advanced Idea |

**一个必须写进报告的坑：光流方向。**
`cv2.calcOpticalFlowFarneback(A, B)` 给出的是"**从 A 坐标查 B**"的流（OpenCV 约定 `next(p+flow(p)) ≈ prev(p)`）。所以要把邻帧 S 对齐到目标帧 T，正确写法是

```python
f = cv2.calcOpticalFlowFarneback(gray_T, gray_S, ...)     # 注意参数顺序！
warped = cv2.remap(S, gx + f[...,0], gy + f[...,1], cv2.INTER_LINEAR)
```

写成 `remap(S, flow(S,T))` 会让误差**上升 29%**（我用一个已知 12 px 平移的合成样本标定过：正确写法 L1=1.36，错误写法 L1=34.85）。这个细节几乎所有资料都不提，踩一次要浪费半天。

**标定方法（建议写进报告附录，体现严谨性）：** 取一帧锐利图像 A，用 `cv2.remap(A, x+d, y+d')` 构造一个已知平移的 B，然后穷举 4 种"流方向 × remap 符号"组合，看哪种能把 B 从 A 里还原出来。只有 `f = fb(T, S)` + `remap(S, +f)` 和 `f = fb(S, T)` + `remap(S, -f)` 两种通过，二者数学等价。

另外光流在 **1/4 分辨率**上估计效果更好且快 58 倍：全分辨率 70.3% 误差下降 / 1000 ms 每对；1/4 分辨率 86.6% / 17 ms 每对。

**第二个坑：tLPIPS 的数值会因为光流实现差异而大幅不同。**
我第一版用错方向时，bicubic 的 tLPIPS 是 0.3089；方向修正后是 0.0519。**相差 6 倍。** 所以：
- 报告里必须写清楚光流算法（Farneback，pyramid 3 levels，window 15，iterations 3）和估计分辨率（1/4）；
- 明确写"本表的 tLPIPS 是自实现，数值不可与其他论文直接比较"，只用于**内部横向对比**。这句话能救你，避免审稿式的追问。

---

## 4. 已产出文件

| 文件 | 内容 |
|---|---|
| `p1_results/metrics_part1_final.json` | **最终结果**：10 个方法 × 30 clip 的逐 clip 指标 + 平均值 |
| `p1_results/metrics_part1.json` | 首轮结果（无光流对齐，tLPIPS 方向有误）——仅作过程留档，**不要引用其中 tLPIPS** |
| `p1_results/metrics_vimeo.json` | Vimeo-90K 7824 clip 结果 |
| `p1_figures/fig_tradeoff.png` | **主图**：PSNR vs tLPIPS 二维权衡散点（10 个方法），带对齐增益箭头 |
| `p1_figures/fig_perclip.png` | 逐 clip 柱状图：朴素平均塌陷 vs 光流对齐复原（30/30） |
| `p1_figures/fig_000_brick.png` | 砖墙纹样放大对比（6 方法并排）——**最强的定性图** |
| `p1_figures/fig_000_sign.png` | 招牌文字区放大对比 |
| `p1_figures/fig_027_detail.png` | 另一个 clip 的细节区 |
| `p1_figures/fig_full_000.png` | 全帧 + 3 倍放大两行版（适合放报告首页） |

代码在服务器 `/tmp/p1_part1_final.py`（主评测，含光流对齐）、`/tmp/p1_vimeo.py`（Vimeo）、`/tmp/flow_gt.py`（光流方向标定）。

> ⚠️ 定性图是首轮导出的样本（那一轮 tLPIPS 光流方向有 bug，但 **PSNR/LPIPS/图像内容不受影响**，图片本身完全有效）。如果要给 flow-aligned 方法也出放大图，需重跑一次样本导出。

---

## 5. 那"我现在要做什么"——按本周能落地的顺序

**已经完成的（不用再动）**：PDF 要求的 4 件事全部跑完，加上 Vimeo-90K 加分项，共 10 个方法 × 30 clip。

**你还必须亲手做的，只有两件 P0：**

**① 拍一段 wild video（半天）**
PDF §6.1 把它列为 Mandatory，而且明确说"用旧手机拍低光/低分辨率视频"来验证真实泛化。要求很低：
- 内容随意，但要**有运动**（走动、车流、摇镜头），否则测不出时域方法的差别；
- 拍 10–20 秒就够，30 fps 约 300–600 帧，取 5 个 100 帧片段即可；
- 故意选**低光或旧手机**，因为这样才能暴露"训练-测试退化不匹配"。

这是 Part 1 唯一还没有的硬性交付物，也是报告里最能出彩的定性对比（模型在合成 bicubic 上 28 dB，换真实退化会明显掉，这个 gap 值得分析）。

**② 把复原结果导出成对比视频（2–3 小时）**
PDF §6.2 要求"submit the processed video files for all Mandatory Datasets"。做法：每个方法输出帧序列 → `cv2.VideoWriter`/ffmpeg 合成 mp4 → 用 ffmpeg 横向拼接成"GT | bicubic | EDSR | flow-avg"四宫格对比视频。这一段既有交付价值，汇报录屏时也直接能用。

**然后才是写报告（1.5 页）**，素材都已经齐了。

---

## 6. 复现命令

```bash
# 环境已在 /tmp/srenv（PyTorch 2.6+cu124, BasicSR + DCN 编译完成）
cd /tmp && /tmp/srenv/bin/python /tmp/p1_part1_final.py 30    # REDS 30 clip，约 45 min（A6000）
cd /tmp && /tmp/srenv/bin/python /tmp/p1_vimeo.py 94           # Vimeo 7824 clip，约 8 min
```

**前置条件**（都已验证通过，换机器要重做）：
- `/tmp/bsr_src/basicsr/ops/dcn/deform_conv_ext.cpython-310-x86_64-linux-gnu.so` 存在
- `torchvision.transforms.functional_tensor` shim（torchvision ≥ 0.17 删了这个模块，BasicSR 还在引用）

---

## 7. 一句话总结

**Part 1 的四件事已经全部跑完，数字可以直接进论文。** 最有价值的发现不是"EDSR 比 bicubic 高 1.88 dB"，而是**朴素多帧平均在真实运动下反而破坏时域一致性（30 个 clip 只有 6 个变好），补上光流对齐后同一融合策略立刻回血 3.98 dB 且 30/30 全部改善** —— 这条从 baseline 通向 Part 2 的逻辑链，就是 Part 1 存在的意义。

接下来真正要动手的只有两件 **P0**：拍一段 wild video、把结果导出成视频。其余都是写报告的工作。

---

## 附：可选加分项（有时间再做）

| 事项 | 收益 | 成本 |
|---|---|---|
| EDSR 小规模微调 | 做"官方权重 vs 微调"消融，体现训练侧理解 | 半天（`train_sample_sharp` 切 48×48 patch，BasicSR EDSR 训练脚本，几千 iter） |
| FID | PDF 提到但未强制 | 建议**跳过**：30 clip 样本量对 FID 不稳定，报告里说明理由即可 |
| 在 Vimeo 上跑 tLPIPS | — | **做不了**：每个 clip 只有 `im4` 有 GT，无法算帧间指标 |
