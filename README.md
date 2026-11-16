# AIAA3201 · Introduction to Computer Vision — Project 1

REDS 官方验证集 ×4 视频超分辨率，30 clip × 100 帧。

> **本仓库只含代码 / 指标 / 图表。** 数据集、模型权重、运行环境均不入库：
> 数据由课程共享盘提供，权重按下方链接自行下载，环境按文档重建。

## 结构

一个课程项目，三个 Part，目录直接叫 `p1`/`p2`/`p3`：

```
p1 手工基线  →  p2 SOTA 复现  →  p3 探索
证明「逐帧独立」不够   对齐+融合 与 时域感知   解决遗留缺陷
                       各自的价值
```

```
p1/                  Part 1 — 手工基线：插值 / EDSR / 多帧平均
  scripts/           评测脚本（主评测、Vimeo-90K、光流标定）
  metrics/           10 方法 × 30 clip 的 PSNR / SSIM / LPIPS / Flow-warp LPIPS
  figures/           定量对比图与细节放大图
  docs/              技术拆解与交付清单
p2/                  Part 2 — SOTA 复现：EDVR + TecoGAN
  scripts/           EDVR 推理、TecoGAN 指标、出图、表格生成、2×2 演示视频合成
  metrics/           两方法的逐 clip 指标、汇总表、选片排序（visibility_rank）
  figures/           patch 级细节对比（30 clip）
  videos/            对比视频（4 clip）
  demo/              演示素材的说明与选片结论（视频本体不入库，见下）
  docs/              交付清单
p3/                  Part 3 — 探索：像素级不确定性 + 区域路由
  scripts/           不确定性估计、区域剖面、出图
  metrics/           不确定性 / 路由 / 融合 / TTA 的实验结果
  figures/           5 张报告图
  docs/              报告草稿
p0/                  P0 基础层：统一评测 / 正式 tLP / 数据与权重核查 / TecoGAN 完整性
  evalkit/           共用评测包（指标、时域、数据配对、配置、溯源）
  scripts/           核查与推理脚本（tLP 验证、网格测量、带校验的 TecoGAN 推理）
  audit/             全部审计结果 JSON
  data_manifest.md   权重 SHA256、训练数据、REDS 划分重叠核查
local-env/           本机复现环境（Windows + RTX 5070 无 nvcc 的解法）
```

## Part 1 · 手工基线

10 个方法放在同一条评测流水线上跑完，包含一个**光流方向标定**脚本：
用已知平移的合成样本自证 warp 约定，而不是靠文档相信。

`p0/` 是 P0 基础层的产物：共用评测包 `p0/evalkit/`、核查脚本 `p0/scripts/`、
审计结果 `p0/audit/`。它统一了 Part 1/2 的空间指标与 tLP、按 `clip+frame_id` 严格配对数据、
并把 TecoGAN 的预平移从“未验证的说法”换成了可复核的网格测量。

## Part 2 · SOTA 复现

EDVR 与 TecoGAN，与 Part 1 用完全相同的指标代码，因此可以并表
（PSNR/SSIM/LPIPS 口径已与旧实现逐项对齐，见 `p0/audit/selfcheck_metrics.json`）：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Flow-warp LPIPS ↓ | 正式 tLP ↓ |
|---|---|---|---|---|---|
| Bicubic（Part 1） | 26.403 | 0.7384 | 0.4835 | 0.0519 | clip 000: 0.0261 |
| Lanczos（Part 1） | 26.546 | 0.7441 | 0.4861 | 0.0507 | 待补 |
| 5 帧平均（无对齐） | 21.852 | 0.5903 | 0.6052 | 0.0893 | 待补 |
| 5 帧平均 + 光流对齐 | 25.834 | 0.7195 | 0.5096 | **0.0236** | 待补 |
| EDSR ×4（Part 1） | 28.284 | 0.8067 | 0.3087 | 0.0811 | 待补 |
| **EDVR-L ×4（Part 2）** | **30.904** | **0.8789** | **0.2039** | 0.0647 | 待补 |
| **TecoGAN（Part 2）** | 25.843 | 0.7187 | 0.3337 | 0.0800 | 待补 |

`Flow-warp LPIPS` 列就是旧表里叫 `tLPIPS` 的那个数字，定义与数值都没变，只是改了名。
正式 tLP 目前只有 clip 000 的对照（Bicubic 0.0261 vs TecoGAN+0.375 0.0047），
全量 30 clip 需要一次完整重跑，尚未进行。

## Part 3 · 探索

面对 Part 2 暴露的取舍——EDVR 在 PSNR/SSIM/**LPIPS** 上全面最好（LPIPS 0.2039 对
TecoGAN 0.3337，即 TecoGAN 在感知指标上并没有赢），而时域稳定性在原指标下无法判定——
做像素级不确定性估计与区域路由融合，方向为 PDF 建议的 *Uncertainty-Aware*。
产出 5 张报告图（不确定性对比、校准曲线、误差捕获、TTA 增益、区域剖面）。

> 早期草稿写的“TecoGAN 赢感知与时域指标”与现有数据不符：LPIPS 上 EDVR（0.2039）与 EDSR（0.3087）
> 都好于 TecoGAN（0.3337），TecoGAN 只优于 bicubic/lanczos；
> 时域上旧指标是 stride=2 的 Flow-warp LPIPS（见「指标口径」），不足以支撑该结论，
> 已在 P0 中删除。Part 3 的动机表述据此改成“取舍”而非“矛盾”。

## 演示视频（`p2/demo/`）

PDF §6.2 要求提交处理后的视频。演示片的合成脚本与选片依据在仓库里，
**视频本体不入库**（11 个 1080p 片段共约 140 MB，远超常规把二进制挡在 git 外的门槛）：

| 内容 | 位置 |
|---|---|
| 合成脚本（2×2 网格 1920×1080 / 24 fps，可选 2× 放大 loupe） | `p2/scripts/p2_demo_2x2.py` |
| 选片依据：按“GT 细节量 × bicubic 破坏程度 × EDVR 复原率”排序 30 clip | `p2/scripts/mk_vis_rank.py` → `p2/metrics/visibility_rank.json` |
| 入选 clip | `002 / 011 / 013 / 015 / 016 / 018`（另有 `contact_sheet*.png`、`picks_sheet.png`） |
| 视频本体 | 服务器 `/tmp/p2/demo/`，需要时按下方命令重新生成 |

```bash
# 需要 GT + bicubic + EDVR 帧 + TecoGAN 帧都已落盘
python p2/scripts/p2_demo_2x2.py 002 011 013 015 016 018          # 普通版
python p2/scripts/p2_demo_2x2.py --inset 002 011 013 015 016 018  # 带放大 loupe
```

> 2026-11-16 版修掉了 013 的渲染崩溃（`loupe` 裁出空区域）、改成按**解析出的 frame id** 配对帧
> 而不是按排序位置配对、缺失方法渲染成占位块而不是崩掉，并把 ffmpeg 的 stderr 收上来报错。

## 本机复现（`local-env/`）

本机没有 CUDA toolkit 也没有 MSVC，BasicSR 依赖的 `deform_conv_ext` 无法编译。
解法是用 `torchvision.ops.deform_conv2d` 转发，**并逐像素对齐服务器结果**验证等价性：

```
max abs diff   : 1.035e-03
> 1/255 的像素 : 0 / 2764800
PSNR(server, local) = 85.37 dB
clip 000 全片段 PSNR: 本地 30.486 vs 服务器 30.486  (+0.000 dB)
```

差异纯属浮点累加顺序。细节见 [`local-env/README.md`](local-env/README.md)。

## 指标口径

指标口径与踩坑记录见 [`p1/scripts/README.md`](p1/scripts/README.md)。
共用实现与验证记录见 [`p0/evalkit/`](p0/evalkit/) 与 [`p0/audit/`](p0/audit/)。

> **⚠️ 上表的 `tLPIPS` 列不是 TecoGAN 定义的 tLP。**
> 它是 `LPIPS(warp(上一帧), 当前帧)` 的 **stride=2** 自一致性指标（无 GT 也能算），
> 已在 P0 改名为 `Flow-warp LPIPS` 并保留原值不覆盖。
> 正式 tLP 是 `|LPIPS(SR[t-1],SR[t]) − LPIPS(GT[t-1],GT[t])|` 在所有相邻对上取均值，
> 需 GT、无连续 GT 时为 N/A。两者排序可以相反：clip 000 上（同片段、100 帧）Bicubic 正式 tLP = 0.0261、旧指标 0.0569。
> TecoGAN 的正式 tLP 目前只有 12 帧样例（0.0047），与全量 0.0800 期间不可直接比。
> 因此**不要**用旧列做跨方法稳定性比较，也不要用它与论文数字对比。

## 数据集与权重

完整清单（SHA256、字节数、训练数据、重叠核查）见 [`p0/data_manifest.md`](p0/data_manifest.md)。

**数据集** `/data/disk1/shared/aiaa3201/project1_video_sr/`（课程共享盘，只读）：

| 路径 | 内容 | 尺寸 |
|---|---|---|
| `val_sharp/` | REDS **官方 validation 划分** GT（original val，BasicSR regroup 后 240–269） | 30 clip × 100 帧，1280×720 |
| `val_sharp_bicubic/` | 同上的 ×4 bicubic 降质 | 320×180 |
| `train_sample_sharp/`、`train_sample_bicubic/` | 训练样本 | 20 clip |
| `vimeo90k_test/` | Vimeo-90K septuplet 测试集 | 94 序列 |

> `val_sharp/000–029` 的编号与 EDVR 论文的 REDS4（original **train** 的 `000/011/015/020`）
> **只是编号巧合**，内容完全不同：与上游 `val_sharp.zip` 逐像素一致，与 `train_sharp.zip` 全部不同
> （clips 000/011/015/020 的帧 0/49/98 共 12 帧 SHA256 比对）。所以 EDVR-REDS 权重在本数据上是未见数据。

**权重**（均不入库，课程镜像在 `/data/disk1/shared/aiaa3201/checkpoints/`）：

| 文件 | 训练数据 | 字节 | 来源 |
|---|---|---|---|
| `EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth` | **DIV2K**（非 REDS） | 6,084,475 | BasicSR 官方 EDSR ×4 |
| `EDVR_L_x4_SR_REDS_official-9f5f5039.pth` | REDS train，`val_partition=REDS4` | 82,587,950（78.8 MiB） | BasicSR 官方 EDVR-L ×4 |
| `EDVR_L_x4_SR_Vimeo90K_official-162b54e4.pth` | Vimeo-90K septuplet train | 82,850,094 | BasicSR 官方 EDVR-L ×4 |
| `TecoGAN.{index,data-00000-of-00001,meta}` | Vimeo-90K septuplets，**Gaussian** 退化 | 12,630,038 | 官方 TF1 release |

> 两点旧描述已更正：EDSR 用的是 **DIV2K** 权重而不是 REDS 权重；
> TecoGAN 官方是 **Gaussian 下采样**合成 LR，不是“真实退化训练”。

## 引用与致谢

本仓库代码为课程项目自写；其中的网络结构与权重来自原作者：

- **EDSR / EDVR / BasicSR** — [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR)（Apache-2.0）
- **TecoGAN** — 官方 TF1 实现
- **REDS 数据集** — Nah et al., *NTIRE 2019 Challenge on Video Deblurring and Super-Resolution*
- **Vimeo-90K** — Xue et al., *Video Enhancement with Task-Oriented Flow*

论文原文不随仓库分发，请从各自官方渠道获取。
