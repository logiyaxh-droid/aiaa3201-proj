# 排序变化

按各指标自身的“越好越靠前”方向排序。DISTS 越低越好，MUSIQ / DOVER 越高越好。

| 名次 | dists | musiq | dover_technical | dover_aesthetic |
|---|---|---|---|---|
| 1 | edvr | edvr | edvr | edvr |
| 2 | edsr | edsr | edsr | edsr |
| 3 | tecogan | tecogan | tecogan | flowtavg5_unsharp |
| 4 | bicubic_unsharp | bicubic_unsharp | flowtavg5_unsharp | tecogan |
| 5 | flowtavg5_unsharp | flowtavg5_unsharp | flowtavg5 | bicubic_unsharp |
| 6 | lanczos | lanczos | lanczos | tavg5_unsharp |
| 7 | bicubic | bicubic | flowtavg3 | flowtavg5 |
| 8 | flowtavg3 | flowtavg3 | bicubic_unsharp | lanczos |
| 9 | flowtavg5 | flowtavg5 | bicubic | flowtavg3 |
| 10 | tavg3 | tavg3 | tavg3 | bicubic |
| 11 | tavg5_unsharp | tavg5_unsharp | tavg5_unsharp | tavg3 |
| 12 | tavg5 | tavg5 | tavg5 | tavg5 |

## 相对 PSNR / LPIPS 的名次

| 指标 | Spearman（与 PSNR） | Spearman（与 LPIPS） |
|---|---|---|
| dists | -0.874 | +0.986 |
| musiq | +0.874 | -0.986 |
| dover_technical | +0.685 | -0.839 |
| dover_aesthetic | +0.636 | -0.846 |

