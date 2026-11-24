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
- 官方在 `BasicVSRNet.forward` 里 `img_upsample` 用的是**输入 LR**，
  本实现 `base` 用的是 `cleaned[i]`。实测把 `base` 换成原始 `x_i` 会让
  PSNR 从 25.04 降到 24.28，所以用 `cleaned` 反而更接近 GT——但这与官方
  口径不同，属于"对齐官方还是对齐数值"的取舍，需要在官方实现上对账后定论。

**做数值对账的正确方式**：在能装 mmcv 的独立环境（或 CPU 上直接 import
纯 torch 的 `flow_warp` / 各模块）跑一次官方 `RealBasicVSRNet`，与本实现
在同一输入上比 `max|Δ|`。逐层比 `image_cleaning` 输出、SPyNet 输出、
`back`/`feats`，可以把剩余 1.2–1.8/255 定位到具体算子。

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

## 阻塞项

- RealBasicVSR 只在 clip 016 的 5 帧上完成单次前向验证，尚未跑完整
  RealVSR（全部 clip × 5 帧）评测，因此还没有可入表的指标行。需要的显存
  不多，但共享 A6000 常被其他用户占满（本轮三次遇到 free < 1 GB）。
- NIQE 权重需联网，本机无外网，VideoLQ 暂时只报 MUSIQ。
- REDS-Real 的在线退化需要 `libx264` 编码，本机无 ffmpeg（仅有
  imageio-ffmpeg 自带的二进制，可用于此）。
