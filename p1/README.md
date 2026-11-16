# P1 — Video Super-Resolution

REDS 验证集，×4 超分，30 clip × 100 帧。

| 目录 | 内容 |
|---|---|
| `part1/` | 10 个方法的统一评测（含光流标定） |
| `part2/` | EDVR / TecoGAN，同样口径 |
| `local-env/` | 本机 Windows 复现环境（无 nvcc 的 DCN 转发方案） |

## 数据集

`/data/disk1/shared/aiaa3201/project1_video_sr/`（课程共享盘，只读）：

| 路径 | 内容 | 尺寸 |
|---|---|---|
| `val_sharp/` | REDS 验证集 GT | 30 clip × 100 帧，1280×720 |
| `val_sharp_bicubic/` | 同上的 ×4 bicubic 降质 | 320×180 |
| `train_sample_sharp/`、`train_sample_bicubic/` | 训练样本 | 20 clip |
| `vimeo90k_test/` | Vimeo-90K septuplet 测试集 | 94 序列 |

## 权重

| 文件 | 来源 |
|---|---|
| `EDSR_REDS.pth` | BasicSR 官方提供的 EDSR ×4 REDS 权重 |
| `EDVR_REDS.pth` | BasicSR 官方提供的 EDVR-L ×4 REDS 权重（78.8 MB） |
| TecoGAN | 官方 TF1 release |

三者均不入库，按 BasicSR / TecoGAN 官方 README 下载后放到 `ckpt/`。
