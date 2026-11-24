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

## 未通过：端到端 PSNR 仍低于 bicubic

在 RealVSR clip 016 上（3 帧）：

| 方法 | PSNR vs GT |
|---|---:|
| bicubic ×4 | 25.18 |
| RealBasicVSR（本实现） | 16.07 |
| RealBasicVSR（EMA 权重） | 16.92 |

**已排除的原因**：权重加载不完整（0/0）、SPyNet 失效（自 flow 0.33 px、
帧间 3.19 px，合理）、通道顺序（RGB/BGR 互换无变化）、亚像素相位错位
（4×4 各相位均值 129–138，一致）、输入与 GT 不对应（相关系数 0.972）。

**剩余怀疑**：checkpoint 是 `TwoStageBasicVSR`（配置里
`num_first_stage_blocks=20, fix_first_stage=True`，且 `load_from` 指向
`experiments/165/iter_300000.pth`）。两阶段的第一阶段可能是一个独立的
x4 前处理，直接用第二阶段的 `basicvsr` 子模块前向不足以还原完整行为。
输出通道偏差呈规律性（[−11.9, +7.3, −19.0]/255）也支持"某级预处理缺失"
而不是数值 bug。

**下一步**：按 mmediting 的 `TwoStageBasicVSR` 定义核对第一阶段，
或在能装 mmcv 的独立环境里跑一次官方实现做数值对账。

## 阻塞项

- NIQE 权重需联网，本机无外网，VideoLQ 暂时只报 MUSIQ。
- REDS-Real 的在线退化需要 `libx264` 编码，本机无 ffmpeg（仅有
  imageio-ffmpeg 自带的二进制，可用于此）。
