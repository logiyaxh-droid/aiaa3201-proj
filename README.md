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
p2/                  Part 2 — SOTA 复现：EDVR + TecoGAN
  scripts/           EDVR 推理、TecoGAN 指标、出图、表格生成、2×2 演示视频合成
  metrics/           两方法的逐 clip 指标、汇总表、选片排序（visibility_rank）
  figures/           patch 级细节对比（30 clip）
  videos/            对比视频（4 clip）
  demo/              演示视频（6 clip × 普通/inset，1920×1080 / 24 fps）与静态预览
    wild/            自拍素材的演示视频（5 片段 × 普通/inset，同格式）
  wild_source/      5 段手机原始视频（对应 PDF §6.1 Wild Video Mandatory）
  WILD.md           自拍素材的完整评测说明、口径与结论（12 方法 × 20 片段）
p3/                  Part 3 — 探索：像素级不确定性 + 区域路由
  scripts/           不确定性估计、区域剖面、出图
  metrics/           不确定性 / 路由 / 融合 / TTA 的实验结果
  figures/           5 张报告图
  docs/              报告草稿
p0/                  P0 基础层：统一评测 / 正式 tLP / 数据与权重核查 / TecoGAN 完整性
  evalkit/           共用评测包（指标、时域、数据配对、配置、溯源）
  scripts/           核查与推理脚本（tLP 验证、网格测量、带校验的 TecoGAN 推理）
                     以及全量评测：full_eval.py / fid_full.py / aggregate_full.py / figures_full.py
  audit/             全部审计结果 JSON
  full/              全量 30 clip × 12 方法的结果（tLP / FID / 空间指标 / 图）
    metrics/         逐 clip JSON + run_config.json（口径与权重哈希）
    summary/         汇总表 table.md / table.json / deltas.json / fid.json
    figures/         4 张报告图
  data_manifest.md   权重 SHA256、训练数据、REDS 划分重叠核查
  tasks/             剩余清单任务：Vimeo Part 2 / 运动分组 / 速度基准 / 对齐诊断 / 退化失配 / 邻帧利用
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

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Flow-warp LPIPS ↓ | 正式 tLP ↓ | FID ↓ |
|---|---|---|---|---|---|---|
| Bicubic（Part 1） | 26.403 | 0.7384 | 0.4835 | 0.0519 | 0.0270 | 41.41 |
| Lanczos（Part 1） | 26.546 | 0.7441 | 0.4861 | 0.0507 | 0.0281 | 36.33 |
| Bicubic + USM | 26.642 | 0.7517 | 0.4398 | 0.0642 | 0.0159 | 35.62 |
| EDSR ×4（Part 1） | 28.284 | 0.8067 | 0.3087 | 0.0811 | **0.0072** | 16.56 |
| 3 帧平均（无对齐） | 23.203 | 0.6374 | 0.5595 | 0.1017 | 0.0895 | 65.26 |
| 5 帧平均（无对齐） | 21.852 | 0.5903 | 0.6052 | 0.0893 | 0.1351 | 97.15 |
| 5 帧平均 + USM | 21.960 | 0.5983 | 0.5590 | 0.1006 | 0.1274 | 92.15 |
| 3 帧光流对齐 | 26.132 | 0.7286 | 0.4993 | 0.0281 | 0.0387 | 46.12 |
| 5 帧光流对齐 | 25.834 | 0.7195 | 0.5096 | **0.0236** | 0.0437 | 48.67 |
| 5 帧光流对齐 + USM | 26.164 | 0.7364 | 0.4692 | 0.0276 | 0.0349 | 43.62 |
| **EDVR-L ×4（Part 2）** | **30.904** | **0.8789** | **0.2039** | 0.0647 | 0.0123 | **6.31** |
| **TecoGAN（Part 2）** | 25.843 | 0.7187 | 0.3337 | 0.0800 | 0.0106 | 40.65 |

全量实测：**30 clip × 100 帧 × 12 方法**，tLP 覆盖全部 99 个相邻帧对/clip，
FID 用全部 3000 帧。逐 clip 数字在 `p0/full/metrics/`，汇总与图在 `p0/full/`。

`Flow-warp LPIPS` 列就是旧表里叫 `tLPIPS` 的那个数字，定义与数值都没变，只是改了名。

**两个时域指标排序相反，这不是笔误。** 正式 tLP 比的是"输出抖动 vs 真值抖动"：
EDSR 的 d_SR（0.2266）最接近 d_GT（0.2304），所以 tLP 最低；而 tavg5 把帧抖没了
（d_SR 只有 0.0954），离真值的抖动最远，tLP 反而最高。**"输出更平滑"不等于"更接近真值的时域行为"**，
所以只能两个指标并列报告，不能单凭任一个说"谁最稳"。

## Part 3 · 探索

面对 Part 2 暴露的取舍——EDVR 在 PSNR/SSIM/**LPIPS**/**FID** 上全面最好
（FID 6.31 对 TecoGAN 40.65），但按正式 tLP 它比 EDSR 略差（0.0123 对 0.0072），
即**没有任何方法同时赢下所有轴**——
做像素级不确定性估计与区域路由融合，方向为 PDF 建议的 *Uncertainty-Aware*。
产出 5 张报告图（不确定性对比、校准曲线、误差捕获、TTA 增益、区域剖面）。

> 早期草稿写的“TecoGAN 赢感知与时域指标”与现有数据不符：LPIPS 上 EDVR（0.2039）与 EDSR（0.3087）
> 都好于 TecoGAN（0.3337），TecoGAN 只优于 bicubic/lanczos；FID 上 TecoGAN（40.65）
> 也只比 bicubic（41.41）好一点。全量 tLP 出来后 TecoGAN 排第 2（0.0106），
> 但那是"输出抖动接近真值抖动"的结果，与感知质量是两件事。
> Part 3 的动机据此写成“取舍”而非“矛盾”。

## 剩余清单任务（`p0/tasks/`）

运动分组、速度/显存基准、对齐诊断、TecoGAN 退化失配对照、EDVR 邻帧利用验证、
Vimeo 上的 Part 2，以及光流变体的局部图与连续帧证据。完整结论见
[`p0/tasks/`](p0/tasks/)（索引见 README，完整结论见 REPORT.md），这里只列会改变既有说法的四条：

| 结论 | 证据 |
|---|---|
| **运动越大朴素平均越糟**（ρ=−0.756），光流对齐的收益随之增大（ρ=+0.738） | 固定三分位分组，`tasks/motion/` |
| **但 EDVR 相对 EDSR 的收益与运动强度无关**（ρ=−0.012, p=0.95） | 同上；PCD 是学习式对齐，不靠"纠正大运动"得利 |
| **EDVR 确实在用邻帧**：中心帧重复五次后 PSNR 掉 2.76 dB（6/6 全负） | `tasks/edvr_neighbor/`；改变了输入分布，不等于 PCD/TSA 的单独贡献 |
| **前后向一致性 mask 四项指标全改善**（PSNR +0.07、tLP −7.5%） | `tasks/align/`；初版符号写错导致 mask 从不触发，已修 |

速度基准在共享 GPU 上跨 run 差到 6 倍，报告引用的是**跨三次运行的最小值**并附
`run_spread_ratio`；`p0/scripts/bench_merge.py` 负责合并。相对关系稳定：
bicubic 比 EDSR 快约 38 倍，EDSR 比 EDVR 快约 6.4 倍。

> 局部图用 clip 013（街景，含真实招牌文字）。脚本的自动选窗只找"最密细节"，
> 不是语义文字；004/011 选出的 "text" 窗其实是树叶，所以最终图改用 `--roi` 手工固定，
> 坐标记录在 `tasks/qual/qual_013.json`。

## 演示视频（`p2/demo/`）

PDF §6.2 要求提交处理后的视频，成品在 `p2/demo/`（12 个片段约 150 MB，6 个 clip 齐全）：

| 内容 | 位置 |
|---|---|
| 成品视频（2×2 网格：GT｜Bicubic｜EDVR｜TecoGAN，1920×1080 / 24 fps） | `p2/demo/00X_demo2x2.mp4` |
| 带 2× 放大 loupe 的版本 | `p2/demo/00X_demo2x2_inset.mp4` |
| 静态预览（6 clip 一览） | `p2/demo/picks_sheet.png` |
| 合成脚本 | `p2/scripts/p2_demo_2x2.py` |
| 选片依据：按「GT 细节量 × bicubic 破坏程度 × EDVR 复原率」排序 30 clip | `p2/scripts/mk_vis_rank.py` → `p2/metrics/visibility_rank.json` |

入选 clip：`002 / 011 / 013 / 015 / 016 / 018`，每个 clip 都有普通版与带 loupe 版各一份。
`013` 的普通版当初因裁剪窗口越界崩掉，修好后已重跑补齐。

```bash
# 需要 GT + bicubic + EDVR 帧 + TecoGAN 帧都已落盘
python p2/scripts/p2_demo_2x2.py 002 011 013 015 016 018          # 普通版
python p2/scripts/p2_demo_2x2.py --inset 002 011 013 015 016 018  # 带放大 loupe
```

> 2026-11-16 版修掉了 013 的渲染崩溃（`loupe` 裁出空区域）、改成按**解析出的 frame id** 配对帧
> 而不是按排序位置配对、缺失方法渲染成占位块而不是崩掉，并把 ffmpeg 的 stderr 收上来报错。

## 全量评测（`p0/full/`）

`p0/scripts/full_eval.py` 把 **12 个方法 × 30 clip × 100 帧**在同一条流式流水线上跑完：
推理 → 空间指标 → 正式 tLP → 旧 Flow-warp 诊断 → InceptionV3 特征，然后立刻丢帧。
Part 1 的 7 个生成类方法（EDSR、3/5 帧平均、光流对齐）本轮是**重新推理**的，
因为 2026-11-11 那次是边算边扔，磁盘上只剩 2 个 clip 的样本图。

```bash
cd ~/p0 && export TORCH_HOME=/tmp/torch_home

/tmp/srenv/bin/python scripts/full_eval.py --out ~/p0/out/full          # 全量，可断点续跑
/tmp/srenv/bin/python scripts/full_eval.py --out ~/p0/out/full \
    --clips 000 --methods bicubic --verify-clip 000                      # 冒烟 + 与 evalkit 对账
/tmp/srenv/bin/python scripts/fid_full.py ~/p0/out/full                  # FID（复用缓存特征）
/tmp/srenv/bin/python scripts/aggregate_full.py ~/p0/out/full            # 汇总表 + deltas
/tmp/srenv/bin/python scripts/figures_full.py  ~/p0/out/full             # 4 张图
```

产出：

| 文件 | 内容 |
|---|---|
| `p0/full/metrics/<clip>.json` | 每个 clip 的 12 个方法逐项指标（含 tLP 的 d_SR / d_GT 分解） |
| `p0/full/metrics/run_config.json` | 全部指标口径、权重路径、方法清单 |
| `p0/full/summary/table.md` | 上面那张主表 |
| `p0/full/summary/deltas.json` | 清单要求的 9 组配对对照 + 逐 clip 胜负计数 |
| `p0/full/summary/fid.json` | 12 个方法的 FID 与帧数 |
| `p0/full/figures/` | PSNR–tLP、PSNR–FID 散点，逐 clip tLP，逐 clip 配对增益 |

**逐 clip 胜负**（脚本自动统计，不是只报均值）：

- EDSR vs Bicubic 在 PSNR / tLP / LPIPS 上都是 **30/30** 全胜；
- flowtavg5 vs tavg5 在 PSNR / tLP / LPIPS 上都是 **30/30** 全胜（PSNR 中位 +4.49 dB）；
- EDVR vs EDSR 在 PSNR / LPIPS 上 **30/30** 全胜，**但 tLP 只有 4/30**，均值反而差 +0.005；
- TecoGAN vs Bicubic 在 PSNR 上 **0/30**（全部更差），在 tLP / LPIPS 上 **30/30** 全胜。

耗时（共享 A6000，机器 load 10–14）：单 clip 全 12 方法 4.3–17.0 min，
无争用时约 7.5 min ⇒ 30 clip 约 3.7 h，实测 5.8 h；FID 约 100 s/方法。

> 两个已定位的性能陷阱（都写进代码注释）：`torch.backends.cudnn.benchmark=True` 会让
> `evalkit.ssim_gpu` 的首次 float64 7×7 conv 花 **126 s**（cuDNN 算法搜索），关掉后 0.4 s
> 且结果逐位相同；`naive_avg` 旧的 padding 写法会把 100 帧复制成 **5.5 GB** float32，
> 改成逐 tap 累加后峰值约 1.5 GB，输出逐位一致（`p0/scripts/check_naive_avg.py`）。
> FID 的两个协方差都是 PSD，用 eigh 求平方根约 1 s，替代 `scipy.linalg.sqrtm`
> 在那儿要跑 25 min 以上（差 1.4e-14，`fid_full.py --check` 可复核）。

## 指标口径

指标口径与踩坑记录见 [`p1/scripts/README.md`](p1/scripts/README.md)。
共用实现与验证记录见 [`p0/evalkit/`](p0/evalkit/) 与 [`p0/audit/`](p0/audit/)。

> **⚠️ 上表的 `tLPIPS` 列不是 TecoGAN 定义的 tLP。**
> 它是 `LPIPS(warp(上一帧), 当前帧)` 的 **stride=2** 自一致性指标（无 GT 也能算），
> 已在 P0 改名为 `Flow-warp LPIPS` 并保留原值不覆盖。
> 正式 tLP 是 `|LPIPS(SR[t-1],SR[t]) − LPIPS(GT[t-1],GT[t])|` 在所有相邻对上取均值，
> 需 GT、无连续 GT 时为 N/A。两者排序可以相反：clip 000 上（同片段、100 帧）Bicubic 正式 tLP = 0.0261、旧指标 0.0569。
> 全量 30 clip 的正式 tLP 已完成（见上表），口径为 `cutfr=0`、`crop32=False`、`x100=False`，
> 覆盖每个 clip 的全部 99 个相邻帧对；所有方法的 tLP 用同一个 LPIPS 实例。
> 因此**不要**用旧列做跨方法稳定性比较，也不要用它与论文数字对比。

**FID 的限制**：3000 帧来自 30 个连续片段的 100 帧，视频帧高度相关，
名义样本量远大于独立场景数；不通过重复帧凑样本，也不与其它实现的 FID 混用。

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

> **两套 EDVR 权重的架构不同**（本仓库新增的核查）：REDS 版 `num_frame=5`
> （`fusion.feat_fusion.weight` 输入 640 = 5×128），Vimeo 版 `num_frame=7`
> （896 = 7×128，因为 Vimeo septuplet 是 7 帧）。把 Vimeo 权重塞进 5 帧模型会直接
> size mismatch。两套权重在 Vimeo 测试集上差 **11.7 dB**（24.49 vs 36.20，见
> [`p0/tasks/REPORT.md`](p0/tasks/REPORT.md)），所以不同权重的结果必须分两行报，
> 不能混进同一个均值。

> **"退化失配"作为 TecoGAN 落后的解释已被实验否定。** 把 LR 换成 TecoGAN 自己的
> Gaussian σ=1.5 核后，TecoGAN 只涨 3.02 dB，而 EDVR 涨 7.50 dB —— 失配并不是它的
> 主因（`p0/tasks/degradation/degradation.json`）。瓶颈是 ×4 大倍率下的重建能力。

## 引用与致谢

本仓库代码为课程项目自写；其中的网络结构与权重来自原作者：

- **EDSR / EDVR / BasicSR** — [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR)（Apache-2.0）
- **TecoGAN** — 官方 TF1 实现
- **REDS 数据集** — Nah et al., *NTIRE 2019 Challenge on Video Deblurring and Super-Resolution*
- **Vimeo-90K** — Xue et al., *Video Enhancement with Task-Oriented Flow*

论文原文不随仓库分发，请从各自官方渠道获取。

