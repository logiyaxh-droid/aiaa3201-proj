# 图注

## figures/

| 文件 | 内容 |
|---|---|
| `000_patch.png` … `029_patch.png` | 30 个 clip 各一张 patch 级细节对比（EDVR / TecoGAN / GT） |
| `perclip_psnr_edsr_vs_edvr.png` | 逐 clip：EDSR 与 EDVR 的 PSNR 差 |
| `perclip_tlpips.png` | 逐 clip tLPIPS |
| `scatter_psnr_tlpips.png` | PSNR–tLPIPS 关系 |

## videos/

| 文件 | 内容 |
|---|---|
| `000_compare.mp4`、`002_compare.mp4`、`009_compare.mp4`、`026_compare.mp4` | 并排对比片段（bicubic / EDVR / TecoGAN / GT） |
| `000_final.jpg` | clip 000 成片单帧 |

## metrics/

| 文件 | 内容 |
|---|---|
| `metrics_edvr.json` | EDVR 逐 clip 指标 |
| `metrics_tecogan.json` | TecoGAN 逐 clip 指标 |
| `part2_table.json` | 汇总表（两个方法并排） |

> 图中数值均由 `metrics/` 下的 json 直接生成，未人工调整。
