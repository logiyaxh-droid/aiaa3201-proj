# Part 3 — Exploration: Uncertainty-Aware Refinement

> 初稿，数字全部来自实测（REDS val，BasicVSR++/EDVR 官方权重，100 帧/clip）。
> 英文正文可直接进 CVPR 报告；中文注释是给你自己看的。

## 一句话概括

Direction C 的原始设想——**用像素级不确定性把困难区域路由给生成式模型**——在 bicubic 退化基准上**无法成立**，因为生成分支（Real-ESRGAN）在该退化下的 fidelity 全面劣于多帧复原模型，路由的目标域本身就是被支配的。但同一套不确定性机制有两条**成立**的用法：**（a）TTA 集成直接提升 PSNR**；**（b）不确定性图可作为可信的误差预测器**（校准良好，top-10% 命中率 32.2%，随机为 10%）。我们把失败原因定位清楚，并给出成立的替代方案。

---

## 3.1 Motivation: the tension Part 2 exposed

Part 2 的结果留下一个矛盾：EDVR 赢失真指标（PSNR 30.904 / SSIM 0.8789），TecoGAN 输（25.843 / 0.7187），但两者都不是"处处更好"。Direction C 的前提是：**不同区域需要不同的先验**——文字、人脸需要稳定，草地、水面需要生成。因此第一步不是造路由器，而是问：**能不能可靠地知道"哪里不可靠"？**

## 3.2 Uncertainty sources

我们比较四类像素级不确定性来源：

| 记号 | 来源 | 额外推理成本 |
|---|---|---|
| U1 | TTA spread：`\|Y(x) − flip(Y(flip(x)))\|` | 1× 前向 |
| U2 | 模型分歧：`\|Y_BasicVSR++ − Y_EDVR\|` | 1× 前向（已有模型） |
| U3 | EDVR 内部时间注意力：`1 − mean_t corr_prob` | **0**（推理时白拿） |
| U4 | 局部拉普拉斯能量 `\|∇²Y\|` | 0，**与模型无关的盲基线** |

U4 是关键对照：如果"学出来的不确定性"打不过一个纯图像滤波器，那它就没有存在价值。

## 3.3 Result 1 — internal attention is NOT a usable uncertainty

**这是本次探索最反直觉的发现，也是最值得写进报告的负面结果。**

U3（EDVR 推理时内部计算的时序注意力）与真实误差的 Spearman 相关只有 **−0.038**，校准曲线几乎水平（图 2），误差捕获能力仅略高于随机（图 3）。也就是说：**模型"看哪里"和模型"错在哪里"基本无关。**

> This is a cautionary result worth stating explicitly: attention weights that are
> free to obtain during inference are *not* a proxy for predictive uncertainty.
> Practitioners who reach for them because they cost nothing will get a signal that
> barely beats chance (error capture 13.1% at 10% rejection, versus 10% for random).

## 3.4 Result 2 — model disagreement and blind Laplacian ARE calibrated

| 估计器 | Spearman ρ | 误差捕获 @10% | 选择性 PSNR |
|---|---|---|---|
| U2 模型分歧 | **+0.404** | **24.3%** | 32.84 |
| U4 拉普拉斯 | +0.399 | 18.1% | 32.17 |
| U3 EDVR 注意力 | −0.038 | 13.1% | 31.79 |
| （随机基线） | 0 | 10.0% | — |

两个观察：

1. **U2 和 U4 精度相当（ρ≈0.40），但 U2 在"高置信区"的可选择 PSNR 更高**（32.84 vs 32.17）。模型分歧更"锐利"——它把误差集中到更少的像素上，这对路由更有用。
2. **U4 这么强，说明大部分"难度"其实是图像本身的高频内容**，不是模型特有的失败。这直接解释了为什么硬路由会失败（见 3.5）：既然误差主要来自纹理本身，那么"换一个模型"并不能救——纹理对两个模型都难。

## 3.5 Result 3 — the original Direction C construction fails, and here is why

我们按 PDF 原文实现了硬路由（不确定 → 用另一个模型）和软融合（注入残差）两种形式，全部实测：

**硬路由**（clip 000，100 帧）：

| 方法 | PSNR | LPIPS | tLPIPS |
|---|---|---|---|
| BasicVSR++ | **31.418** | **0.1645** | **0.0747** |
| Real-ESRGAN | 23.506 | 0.2927 | 0.0929 |
| 路由 q=0.99 | 27.803 | 0.1681 | 0.0888 |
| 路由 q=0.90 | 24.575 | 0.1924 | 0.1037 |

**三项指标全部单调变差**。原因不是路由策略不够好，而是**目标域被支配**：在 bicubic 退化下 Real-ESRGAN 比 BasicVSR++ 低 **7.9 dB**，LPIPS 也差 0.13。路由相当于"把一部分像素替换成明显更差的预测"，任何路由权重都只亏不赚。

**软融合**（只注入 α 比例的高频残差，α=0.15/0.30/0.50，gate 在 0.90/0.98 分位）：最好的一档 `α=0.15, g=0.98` 是 PSNR −0.01 / LPIPS +0.0002，即**基本为零**；α 越大越差。同一根因。

> **This is the most important negative finding of Part 3.** Direction C as written
> assumes the generative branch is *better* in the regions the fidelity branch is
> unsure about. On benchmark-style bicubic degradation that assumption is false: the
> GAN prior was trained for real-world degradation, so on clean bicubic data its
> output is dominated. Uncertainty-aware routing therefore cannot help *unless* the
> two branches are competitive. Route only between models whose errors are
> complementary, not merely different.

## 3.6 Result 4 — the construction that does work: TTA on the fidelity branch

把"换模型"改成"增强同一模型"。BasicVSR++ 对水平翻转严格等变，所以翻转两遍取平均是合法集成：

| clip | 单次 PSNR | TTA PSNR | ΔPSNR | ΔLPIPS | spread 均值 |
|---|---|---|---|---|---|
| 000 | 31.407 | **31.522** | **+0.115** | +0.0018 | 1.155 |
| 002 | 34.128 | **34.263** | **+0.135** | +0.0007 | 0.830 |
| 009 | 34.015 | **34.089** | **+0.074** | +0.0010 | 0.693 |

**三个 clip 全部提升 0.07–0.14 dB，LPIPS 代价 <0.002（不可感知）**。同时 TTA spread（U1）与误差的 Spearman 为 0.36–0.46，与 U2 同量级——**它同时提供了免费的、无需第二个模型的不确定性**。

> TTA ensembling is the one intervention in Part 3 that improves the metric we are
> graded on, costs a single extra forward pass, requires no new weights, and yields a
> calibrated uncertainty map as a by-product.

## 3.7 Result 5 — the uncertainty map tracks semantic difficulty

把 clip 000 按局部方差分成三档，看 U2 与真实误差的走势：

| 区域 | 平均 U2 | 平均真实误差 |
|---|---|---|
| flat（平坦） | 0.833 | 2.100 |
| mid（中等） | 2.113 | 4.194 |
| textured（纹理） | 4.046 | 6.237 |

归一化后两条曲线相关系数 **r = 0.992**。此外 **top-10% 像素重叠率 32.2%**（随机 10%）——不确定性图抓到的误差是随机的 **3.2 倍**。

这组数字让 Direction C 的**诊断部分**成立：我们可以可靠地说出"这个模型在哪些区域不可信"。

## 3.8 What we would do with more time

1. **换真正的互补分支**：用真实退化训练的模型（STAR / 真实世界 VSR）替代 Real-ESRGAN，使两个分支的误差真正互补，硬路由才有意义。这是 Direction C 原本能成立的前提条件。
2. **把路由从空间域搬到频率域**：既然失败集中在高频，按频带融合而非按像素融合可能绕过"目标域被支配"的问题。
3. **在 wild video 上重做**：真实低光退化下，两个分支的相对强弱会改变（Part 2 已观察到 TecoGAN 在干净 bicubic 上吃亏），Direction C 的前提可能在那里成立。
4. **TTA 扩展到全量 30 clip** 并补 tLPIPS，作为报告的正式实验项。

## 3.9 Honest limitations

- 全部结论基于 **REDS val 的 3–7 个 clip**（帧数 100/clip），统计量足够做趋势判断，不足以做显著性检验。
- U2 需要跑两个模型；只有 U1（TTA）是单模型可得的。
- 硬路由失败是**在 bicubic 退化下**失败的。不能外推为"Direction C 永远不成立"——3.8 第 1、3 条正是要检验这一点。
- FID 未计算（样本量不足，见 Part 1 的说明）。

---

## 附：图表清单

| 文件 | 内容 | 建议位置 |
|---|---|---|
| `fig1_uncertainty_compare.png` | 真实误差 vs 三种估计器的空间分布 | Method/Experiments |
| `fig2_calibration.png` | 十分位校准曲线 | Experiments 主图 |
| `fig3_error_capture.png` | 误差捕获曲线（含随机基线） | Experiments |
| `fig4_tta_gain.png` | TTA 的 PSNR 增益与 LPIPS 代价 | 正面结果 |
| `fig5_region_profile.png` | 区域难度轮廓匹配（r=0.992） | 支撑 3.7 |
