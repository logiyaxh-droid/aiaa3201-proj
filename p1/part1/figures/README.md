# 图注

`part1/figures/` — Part 1 定量对比

| 文件 | 内容 |
|---|---|
| `fig_perclip.png` | 逐 clip PSNR，方法间对比 |
| `fig_tradeoff.png` | PSNR 与 tLPIPS 的权衡散点 |
| `fig_full_000.png` | clip 000 全帧对比（多方法横排） |
| `fig_000_brick.png`、`fig_000_sign.png` | clip 000 局部细节放大 |
| `fig_027_detail.png` | clip 027 局部细节放大 |
| `000_*.png`、`027_*.png` | 单方法成对切片：`bicubic` / `bicubic_unsharp` / `lanczos` / `edsr` / `tavg3` / `tavg5` / `tavg5_unsharp` / `gt` |

`part2/figures/` — Part 2 细节对比

| 文件 | 内容 |
|---|---|
| `000_patch.png` … `029_patch.png` | 30 个 clip 各一张 patch 级对比（EDVR vs TecoGAN vs GT） |
| `perclip_psnr_edsr_vs_edvr.png` | 逐 clip：EDSR 与 EDVR 的 PSNR 差 |
| `perclip_tlpips.png` | 逐 clip tLPIPS |
| `scatter_psnr_tlpips.png` | PSNR–tLPIPS 关系 |

`part2/videos/` — 对比视频

`000_compare.mp4`、`002_compare.mp4`、`009_compare.mp4`、`026_compare.mp4`：
四个 clip 的并排对比片段，`000_final.jpg` 为 clip 000 的成片单帧。

> 图中数值均由 `metrics/` 下的 json 直接生成，未人工调整。
