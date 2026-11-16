# AIAA3201 · Introduction to Computer Vision — Course Projects

课内项目的代码、实验指标与图表归档。

> **本仓库只含代码 / 指标 / 图表。** 数据集、模型权重、运行环境均不入库：
> 数据由课程共享盘提供，权重按 README 里的链接自行下载，环境按各项目文档重建。

## 仓库结构

```
p1/                 视频超分辨率（Video Super-Resolution）
  part1/            Part 1 — 经典 / 单帧方法基线
    scripts/        评测脚本（主评测、Vimeo-90K、光流标定）
    metrics/        10 方法 × 30 clip 的 PSNR / SSIM / LPIPS / tLPIPS
    figures/        定量对比图与细节放大图
    docs/           技术拆解与交付清单
  part2/            Part 2 — EDVR / TecoGAN 视频方法
    scripts/        EDVR 推理、TecoGAN 指标、出图、表格生成
    metrics/        两方法的逐 clip 指标与汇总表
    figures/        patch 级细节对比（30 clip）
    videos/         对比视频（4 clip）
    docs/           交付清单
  local-env/        本机复现环境（Windows + RTX 5070 无 nvcc 的解法）

p2/                 占位（未展开）
p3/                 占位（未展开）
```

## P1 · 概览

**任务**：REDS 验证集 ×4 视频超分辨率，30 clip × 100 帧。

**Part 1** 把 10 个方法放在同一条评测流水线上跑完，包含一个**光流方向标定**脚本：
用已知平移的合成样本自证 warp 约定，而不是靠文档相信。

**Part 2** 跑 EDVR 与 TecoGAN，并补齐与 Part 1 相同口径的指标。

指标口径与踩坑记录见 [`p1/part1/scripts/README.md`](p1/part1/scripts/README.md)。

## P1 · 本机复现（`local-env/`）

本机没有 CUDA toolkit 也没有 MSVC，BasicSR 依赖的 `deform_conv_ext` 无法编译。
解法是用 `torchvision.ops.deform_conv2d` 转发，**并逐像素对齐服务器结果**验证等价性：

```
max abs diff   : 1.035e-03
> 1/255 的像素 : 0 / 2764800
PSNR(server, local) = 85.37 dB
clip 000 全片段 PSNR: 本地 30.486 vs 服务器 30.486  (+0.000 dB)
```

差异纯属浮点累加顺序。细节见 [`p1/local-env/README.md`](p1/local-env/README.md)。

## 引用与致谢

本仓库的代码为课程项目自写；其中的网络结构与权重来自原作者：

- **EDSR / EDVR / BasicSR** — [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR)（Apache-2.0）
- **TecoGAN** — 官方 TF1 实现
- **REDS 数据集** — Nah et al., *NTIRE 2019 Challenge on Video Deblurring and Super-Resolution*
- **Vimeo-90K** — Xue et al., *Video Enhancement with Task-Oriented Flow*

论文原文不随仓库分发，请从各自官方渠道获取。
