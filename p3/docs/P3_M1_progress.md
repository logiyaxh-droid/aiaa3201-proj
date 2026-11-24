# M1 执行记录：基准重建与 RealBasicVSR 接入

日期：2026-11-24  
对应路线图：`p3/docs/P3_realworld_vsr_roadmap.md` 第五节 M1

## 已完成

### 1. VideoLQ 协议修正（路线图 1.2 节列出的隐患 2）

`run_edvr_vlq.py` 的两处输入错误已修正并重跑 50 段 × 5 帧：

| 原问题 | 修正 |
|---|---|
| `x[None].repeat(5,1,1,1)` 单帧重复 5 次 | 先构建全部 LR，再按真实邻帧取 5 帧窗口，仅序列两端 replicate |
| `cv2.INTER_CUBIC` 缩小时无抗锯齿 | 改用 PIL BICUBIC（缩小时展宽核） |
| 附带：`w4 = w//4 - (w//4)%2` 只保证偶数 | 改为 `// 4 * 4`，满足 EDVR 的 `h % 4 == 0` 断言 |
| `pyiqa.create_metric("niqe")` 联网取权重 | 改用本地 checkpoint 的 MUSIQ（`score_vlq_local.py`） |

结果（selfPSNR 从 20.49 升到 27.78，证明旧值主要是输入被破坏）：

| 方法 | MUSIQ ↑ | selfPSNR ↑ |
|---|---:|---:|
| 感知微调 EDVR | **52.51** | 27.44 |
| 原始 EDVR | 51.26 | **27.78** |
| 原始 EDVR（旧，作废） | 55.81 | 20.49 |

详见 `prior_qualification/results/summary/videolq.md`。

### 2. RealBasicVSR 权重与网络接入

- 权重：HF `akhaliq/RealBasicVSR_x4`，210,479,802 字节，SHA 校验通过；
  checkpoint 内是 `TwoStageBasicVSRGAN`，37.0M 参数（路线图写的 6.3M 是
  另一口径，以 checkpoint 为准）。
- 网络：未安装 mmediting（会与 DOVE/SUPIR 的 torch 2.6 环境冲突），
  按 checkpoint 的键名重新实现，`scripts/rbvsr_net.py`。
- **加载校验通过：missing 0 / unexpected 0**，这是结构正确的强证据。

实现过程中，从键名反推出的、与常见 BasicVSR 写法不同的四处：

| 细节 | 正确形式 | 写错的后果 |
|---|---|---|
| `ResidualBlocksWithInputConv.main` 嵌套 | `[conv, lrelu, Sequential(resblocks)]` | 310 个权重未加载 |
| `SPyNetBasicModule` 的 conv 命名 | `.0.conv` 而非 `.0` | 130 个权重未加载 |
| SPyNet 金字塔层数 | **6 层**（不是 5），且输入先补到 32 的倍数 | flow 达 ±11815 px |
| 传播拼接顺序 | `[当前帧, warp 后特征]` | 前期逐帧输出接近全黑 |

另外三处 `forward` 语义（对照 `basicsr/basicsr/archs/basicvsr_arch.py`）：

- `image_cleaning` 输出的是**残差**（实测 raw mean 2.2/255，输入 144.2/255，
  相加后 146.2），不是清洗帧本身；
- `fusion` 后必须 `lrelu`；
- 最终 `conv_last` 输出要与输入的 **bilinear ×4 相加**（`out += base`）。

修正后单帧前向输出均值 0.506，GT 0.555，通道相关系数 0.86–0.90。

## 已定位：16 dB 的成因是 3 处前向实现 bug（2026-11-24 复核）

原记录怀疑"`TwoStageBasicVSR` 缺了独立的第一阶段前处理"。这个怀疑不成立：
官方 `RealBasicVSRNet` 的第一阶段就是 `self.image_cleaning`（20 块），
第二阶段就是 `self.basicvsr`，`is_fix_cleaning=True` 只是在训练第二阶段时
冻住清洗模块，推理时不存在额外的前处理网络。模块一个都没缺。

逐行对照 mmediting 的 `real_basicvsr_net.py`、`basicvsr_net.py`、
`common/flow_warp.py`、`common/upsample.py`、`common/sr_backbone_utils.py`，
`rbvsr_net.py` 里确认 3 处真 bug：

| # | 位置 | 错误写法 | 官方写法 | 影响 |
|---|---|---|---|---|
| 1 | `forward` 上采样尾部 | `upsample1`/`upsample2` 后无激活 | 两次 pixel-shuffle 后各有 `lrelu(0.1)` | 两级 conv+shuffle 退化为线性映射，**主要成因** |
| 2 | `_warp` | `grid + flow.flip(1)` | `grid + flow` | dx/dy 互换，污染 SPyNet 金字塔与前后向传播 |
| 3 | `_warp` 的 `padding_mode` | 一律 `"border"` | 传播分支用 `"zeros"`，仅 SPyNet 内部用 `"border"` | 边界少量偏差 |

报告另提出的"SPyNet 最粗层多做一次 2× 上采样"**不成立**：该处 `flow`
初始化为全零，`interpolate(×2)*2` 后仍是全零，与官方 `flow_up = flow`
数值等价，只是多一次空运算。已顺手改成官方写法（`level == 0` 时不插值、
`flow` 直接按最粗层尺寸初始化），但这对 PSNR 无贡献。

### 逐项归因（RealVSR clip 016，5 帧，RGB 口径，raw PSNR）

| 变体 | PSNR | 通道偏置 /255 |
|---|---:|---|
| 原始（bug 全在） | 15.47 | [−21.6, +8.0, −12.2] |
| 仅修 #1 | 24.56 | [+4.0, +4.2, +4.6] |
| 仅修 #2 | 15.41 | [−22.7, +8.5, −8.7] |
| 仅修 #3 | 15.47 | [−21.6, +8.0, −12.2] |
| #1+#2 | 25.04 | [+3.8, +4.1, +4.5] |
| #1+#3 | 24.57 | [+4.0, +4.2, +4.6] |
| 全部修正 | **25.16**（luma） | [+3.8, +4.1, +4.5] |

结论：**bug #1 单独就解释了 9 dB 中的 9.1 dB**；#2 在 #1 之上再加 0.47 dB；
#3 在本 clip 上无可测贡献（改与不改相差 0.004 dB）。原始记录的
"[−11.9, +7.3, −19.0]/255" 三通道不一致的偏置，正是像素重排被线性折叠后
产生的通道串扰，修正 #1 后消失，变成三通道一致的均匀偏置。

### 修后指标与通道顺序对账

| 方法 | luma PSNR | SSIM |
|---|---:|---:|
| bicubic ×4 | 25.2277 | — |
| RealBasicVSR（修正后，RGB 输入） | **25.1638** | 0.8091 |
| RealBasicVSR（修正后，BGR 输入） | 25.1598 | — |

本机 luma 口径复算的 bicubic 25.2277 与 `logs/rv_bicubic.log` 完全一致，
说明度量链路可比。RGB 与 BGR 输入只差 0.004 dB，通道顺序不是问题（这一点
原记录已经排除过，现在再次确认）。目标是 24–25+ dB，已达成；修正后与
bicubic 基本持平，符合 RealBasicVSR 在强退化下"恢复细节但 PSNR 增益有限"
的常见表现。

### 已知项：残留 +4/255 均匀偏置，其中 2.6 来自数据本身

修正后的输出仍整体比 GT 亮 [+3.8, +4.1, +4.5]/255。直接量 LQ 与 GT 的均值：

| | R | G | B |
|---|---:|---:|---:|
| LQ 均值 | 144.41 | 144.57 | 144.34 |
| GT 均值 | 141.75 | 142.17 | 141.67 |
| LQ − GT | **+2.66** | **+2.40** | **+2.68** |
| bicubic − GT | +2.66 | +2.40 | +2.67 |

bicubic 的偏置与 LQ 的曝光差逐通道相同，说明这 2.6/255 是 RealVSR 采集
本身的双镜头曝光差（LQ 广角、GT 长焦），任何方法都会继承，不是网络问题。
修正后网络做到了"基本保留输入曝光"，这是正确行为。

剩下约 1.2–1.8/255 尚未归因。未验证的候选差异（本轮**未**改动，留作已知项）：

- `ResidualBlock` 用 `F.relu`，官方 `ResidualBlockNoBN` 也是 ReLU，这一处
  其实一致，不是差异。
- `PixelShufflePack` 官方 `upsample_kernel=3`、`padding=1`，本实现 `_conv`
  的 k=3 等价，一致。
- 官方 SPyNet 有 6 个 `basic_module` 但金字塔只下采样 5 次（`range(5)`）；
  本实现 `n_levels=6` 下采样 5 次，层数一致。
- ~~`img_upsample` 用输入 LR 还是 cleaned~~：**已确认一致，不是差异**。
  官方 `RealBasicVSRNet.forward` 是先把 `lqs` 原地覆盖成 cleaned
  （`lqs = (lqs + residues).view(n, t, c, h, w)`），再 `outputs =
  self.basicvsr(lqs)`，所以 `BasicVSRNet` 内部的 `img_upsample(lr_curr)`
  加的就是 cleaned。本实现用 `cleaned[i]` 是 100% 官方行为，不存在口径
  差异。实测把 `base` 换成原始未清洗 `x_i` 会掉 0.76 dB（25.04 → 24.28），
  这恰好反向印证了官方确实是先清洗再上采样。

剩下 1.2–1.8/255 的均匀亮差**不再追查**。按评测口径判断，它是
`TwoStageBasicVSRGAN` 这类 GAN 目标训练出的模型在跨域数据上的轻微亮度
漂移（GAN 感知目标本身就允许整体亮度偏离 L1 最优），不是算子实现问题；
继续装 mmcv 做逐层对账的收益不足以抵消成本。M1 以此为已知项收尾。

脚本与数据（远程 `/data/disk1/home/ghe270/p3/prior_qualification/`）：
`scripts/_diag_rbvsr_bugs.py`（逐项消融）、`scripts/_diag_rbvsr_ablate.py`
（组合归因 + 清洗残差实测）、`scripts/_diag_rbvsr_final.py`（RGB/BGR 与
bicubic 对账）、`scripts/_apply_rbvsr_fixes.py`（锚点精确匹配的补丁脚本，
可复现）。结果在 `results/rbvsr_diag/{bugs_rgb,final_check,ablate}.json`。
生产文件已改，备份 `scripts/rbvsr_net.py.bak-20261124`。

另一条实测信息：`dynamic_refine_thres` 为默认 1.0 时，清洗残差绝对均值只有
0.0095（远小于 1.0），所以清洗**第一轮就提前退出**，永远只跑 1 遍。
把循环上限从 3 改成 1 结果完全不变（25.0400 对 25.0400）。原记录里
"跑一次是剩余误差的一个原因"这一句不成立，已随之作废。
（REDS-Real 上第一轮残差是 4.807/255，比 RealVSR 高，但仍小于 1.0 的阈值，
所以同样是 1 遍退出——见下一节的档位对比。）

## M1 收尾：REDS-Real c9 两行实验与诊断（已完成）

RealVSR 的历史使命已完成（它的作用是证明实现正确：修完 bug 后 25.16 dB
与 bicubic 持平），算力转到 REDS-Real c9（30 段 × 9 帧 = 270 帧）。
所有行都在同一批帧、同一 luma-PSNR 口径下重算，可比。

### 配置选择：EMA 与清洗档位都保持默认

修完 bug 后按 30 段配对比较：

| 对比 | 均值差 | 逐段方向 | 结论 |
|---|---:|---|---|
| `generator` − `generator_ema` | **+0.0315 dB** | 24/30 段 generator 更好 | 无优势，保持 `generator` |
| `thres=255/255`(1 遍) − `thres=5/255`(≤3 遍) | **+0.0251 dB** | 25/30 段 1 遍更好 | 无优势，保持 1 遍 |

反直觉但已核实：REDS-Real 上第一轮清洗残差绝对均值是 4.807/255，确实高于
5/255 的阈值，所以 `thres=5/255` **真的跑满了 3 轮**（4.807 → 2.843 →
2.308/255），不是空操作。三轮清洗在本数据上略降 PSNR，说明这版权重对
REDS-Real 的退化程度已经"洗得过多"。Stage 1 固定 `dynamic_refine_thres=1.0`。

### 主表（REDS-Real c9，270 帧）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ | tLP↓ | d_SR | E_warp↓ | E_warp − GT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bicubic | **21.0037** | **0.5019** | 0.7271 | 0.3463 | 0.0744 | 0.1741 | **4.994** | −0.763 |
| EDVR-L (REDS) | 20.9164 | 0.4902 | 0.6351 | 0.3577 | 0.0630 | 0.2026 | 5.528 | −0.229 |
| EDVR 感知微调 | 20.8622 | 0.4840 | 0.6163 | 0.3525 | 0.0631 | — | — | — |
| RealBasicVSR | 20.8220 | 0.4989 | 0.4881 | 0.2282 | 0.0443 | 0.2076 | 6.346 | +0.589 |
| OSEDiff | 19.8122 | 0.4454 | 0.4771 | 0.2235 | 0.0479 | 0.2470 | 10.251 | +4.494 |
| **RBV → OSEDiff** | 19.5834 | 0.4747 | 0.5057 | **0.2227** | **0.0392** | 0.2249 | 8.352 | +2.595 |

GT 自身的 d_GT = 0.2388、E_warp_GT = 5.757，是这两个时序指标的下界参考。

### 三条能直接用于 M2 定配置的结论

1. **零样本串联没有把 PSNR 拉回来，反而更低。** RBV→OSEDiff 19.5834 比
   单独 RealBasicVSR 低 1.239 dB，比单独 OSEDiff 低 0.229 dB。串联不是
   "先修复再生成"的叠加收益，而是两次引入偏差。
2. **串联的时序代价是最大的。** E_warp 从 RBV 的 6.346 恶化到 8.352
   （GT 5.757），比单独 OSEDiff 的 10.251 好一些，但仍远差于 EDVR 的
   5.528 和 bicubic 的 4.994。tLP 0.0392 是全场最低，不过这是因为它把
   d_SR 从 0.2076 降到 0.2249 的同时数值缩小——注意 d_SR 与 d_GT(0.2388)
   的差距被压小了，属于"整体被抹平"而非"更稳"，E_warp 才是真实稳定性。
   所以 **Stage A 必须带时序约束**，单靠感知指标选型会选错。
3. **RealBasicVSR 是当前 PSNR/感知最平衡的一档**：PSNR 与 EDVR 持平
   （20.82 vs 20.92），LPIPS 0.488 明显低于 EDVR 0.635，DISTS 0.228 接近
   OSEDiff 0.224。它的短板是 E_warp +0.589（比 GT 更抖），这正是 M2 的
   门控/时序残差要压的部分。

对比图：`results/reds_real_c9/compare_textures.png`（三行：树叶、
砖墙+栏杆重复图案、脚手架细结构；列 = GT | EDVR | RealBasicVSR | OSEDiff |
RBV+OSEDiff，每格标注相对 GT 的局部 PSNR）。裁剪坐标由
"GT 梯度能量 + 输出相关性"双指标筛选后人工确认，避开了被完全抹平的区域。

### 新增脚本

- `scripts/rbvsr_variants.py`：同一批帧上配对比较权重前缀
  (`generator` / `generator_ema`) 与清洗档位，输出逐段差值与方向。
- `scripts/run_osediff_chain.py`：RBV→OSEDiff 串联。与 `run_osediff.py` 的
  差别是 x4 阶段换成 RealBasicVSR 输出，且
  `wavelet_color_fix(source=...)` 对齐到 RBV 输出而不是 LANCZOS。
- `scripts/score_reds_real.py`：在原有 PSNR/SSIM/LPIPS/DISTS/tLP 上补
  (a) tLP 的 `d_SR`/`d_GT` 分解，(b) 遮挡感知 E_warp——Farneback 前向
  光流 warp 后只统计前反向一致像素（`|f_fwd(p)+f_bwd(p+f_fwd(p))| ≤ 3`）
  且排除 8 px 边界，同时输出 SR 与 GT 两条，差值即 `E_warp gap`。
- `scripts/make_compare_picks.py`：拼对比图（单次绘制文字；OpenCV 5.0
  的 `putText` 在 `LINE_AA` + `thickness=3` 时有渲染缺陷，会输出重影，
  见下一节）。

### 一个环境坑（本轮踩到）

OpenCV 5.0.0 的 `cv2.putText` 在 `LINE_AA` 且 `thickness=3` 时**渲染错位**：
同一字符串画两遍做描边会得到"尾部重影 + 内部笔画缺失"。标注类脚本要么
只画一遍（文字放在不透明底色上），要么用 `thickness=1`。

## 阻塞项

- RealVSR 全量 50 段没有跑，按决定不再补（价值已由 clip 016 的实现在
  bug 修复后回到 25.16 dB 证明）。
- NIQE 权重需联网，本机无外网，VideoLQ 暂时只报 MUSIQ。
- REDS-Real 的在线退化需要 `libx264` 编码，本机无 ffmpeg（仅有
  imageio-ffmpeg 自带的二进制，可用于此）。
