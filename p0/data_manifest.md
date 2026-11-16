# 数据与权重清单（P0 · 任务 1 产物）

本文件是 README 对应小节的替换文本，也是对“训练范围”这一质疑的正式记录。
所有 SHA256 与字节数为 **2026-11-16 在服务器 `/data/disk1/shared/aiaa3201/` 实测**。

## 数据集

根目录 `/data/disk1/shared/aiaa3201/project1_video_sr/`（课程共享盘，只读）。

| 路径 | 内容 | 尺寸 | 与权重的训练数据是否重叠 |
|---|---|---|---|
| `val_sharp/` | REDS **官方 validation 划分**（original val，BasicSR regroup 后为 240–269） | 30 clip `000`–`029` × 100 帧，1280×720 PNG | 与 EDVR-REDS / EDSR 均不重叠 |
| `val_sharp_bicubic/` | 同上 ×4 bicubic 降质 | 320×180 | 同上 |
| `train_sample_sharp/` | REDS 训练样本 | 20 clip `000`–`019` | 属 original-train；本次未用于评测 |
| `train_sample_bicubic/` | 同上 ×4 bicubic | | |
| `vimeo90k_test/` | Vimeo-90K septuplet 测试集 | 94 序列，`low_resolution/` 112×64、`target/` 448×256 | EDVR-Vimeo / TecoGAN 的训练划分是 septuplet **train**；测试集不重叠（待核实 Vimeo 权重用的是哪个 split，见下） |

**关键澄清（P0 新增）**：课程 `val_sharp` 的 clip 编号 `000–029` 与 EDVR 论文里的 “REDS4”
（original **train** set 的 `000/011/015/020`）只是编号巧合，两者内容完全不同。依据：

1. 与上游 `snah/REDS` 的 `val_sharp.zip` 逐像素 SHA256 **完全一致**（clip 000/011/015/020 的帧 0/49/98 共 12 帧）；
2. 与上游 `train_sharp.zip`（REDS4 所在的原训练集）**全部不同**。

因此 EDVR-REDS 权重在课程 30 clip 上评测是**未见数据**。证据与脚本：`work/reds_audit.py`、
`work/reds_audit_report.json`。

## 权重

均不入库，按官方 release 下载后放到 `ckpt/`（课程镜像已提供，路径如下）。

| 用途 | 文件名 | 字节 | SHA256 | 训练数据 |
|---|---|---|---|---|
| EDSR ×4 | `EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth` | 6,084,475 | `0c287733e70d1b8b8fc5885613ecbe451e5f3010bcae0307612ef5e4aa08dd5f` | **DIV2K**（800 训练图） |
| EDVR-L ×4 | `EDVR_L_x4_SR_REDS_official-9f5f5039.pth` | 82,587,950 | `9f5f50393965194cc180e1313fd162bc85003bb4c2a58c23a434a33fdb3786a2` | REDS `train_sharp`，`val_partition=REDS4` |
| EDVR-L ×4 | `EDVR_L_x4_SR_Vimeo90K_official-162b54e4.pth` | 82,850,094 | `162b54e4935bf7e66c4a2a0cf0d3620bc4ec287eb4d4dfe1587b9ecdcc6768cb` | Vimeo-90K septuplet train |
| TecoGAN | `TecoGAN.data-00000-of-00001` | 12,128,916 | `e009bc08ec84c17bd8525252933ceafb164a743727285360259160e81a8e6961` | Vimeo-90K septuplets，**Gaussian** 退化 |
| TecoGAN | `TecoGAN.index` | 4,814 | `c100584c2ec416c2daa4175ce43c5d0ebb87f13db5c340441b8e75cee0da70d3` | 同上 |
| TecoGAN | `TecoGAN.meta` | 496,308 | `b5e126d40aced8c952b84c63a3cbe9258848a7f9a866e7e225c44ac29c979259` | 同上 |

服务器路径：`/data/disk1/shared/aiaa3201/checkpoints/{EDSR_baseline_x4,EDVR_REDS_SR_L,EDVR_Vimeo90K_SR_L,TecoGAN}/`。

### 需要修正的旧描述

- ❌ `EDSR_REDS.pth` / “BasicSR 官方 EDSR ×4 REDS 权重” → ✅ 文件名如上，训练集是 **DIV2K，不是 REDS**。
- ❌ `EDVR_REDS.pth（78.8 MB）` → ✅ 实际文件名如上，82,587,950 B = **78.8 MiB**（写 MB 会差 5%）。
- ❌ “TecoGAN 为真实退化训练” → ✅ 官方用 `tf_data_gaussDownby4(sigma=1.5)` 的 **Gaussian** 下采样管线合成 LR。

## 仍需核实

- EDVR-Vimeo / TecoGAN 的训练是否覆盖 `vimeo90k_test` 的 94 个序列：上游未随权重给出 split 文件，
  本次未取得 `sep_trainlist.txt` 做交集，标注为 **训练范围待核实**，相关结果不用于证明未见数据泛化。
- TecoGAN 官方模型的训练序列来自 28 个在线视频（`dataPrepare.py` 硬编码），与 REDS 无关；
  与“真实低光视频”实验（任务 15）也不存在重叠问题。
