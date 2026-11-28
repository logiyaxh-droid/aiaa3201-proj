# 排序变化

按各指标自身的“越好越靠前”方向排序。DISTS 越低越好，MUSIQ / DOVER 越高越好。

| 名次 | dists | musiq | dover_technical | dover_aesthetic |
|---|---|---|---|---|
| 1 | edvr | edvr | edvr | edvr |
| 2 | edsr | edsr | edsr | edsr |
| 3 | tecogan | tecogan | tecogan | tecogan |
| 4 | bicubic_unsharp | bicubic_unsharp | bicubic | bicubic_unsharp |
| 5 | bicubic | bicubic | bicubic_unsharp | bicubic |

## 相对 PSNR / LPIPS 的名次

| 指标 | Spearman（与 PSNR） | Spearman（与 LPIPS） |
|---|---|---|
| dists | -0.700 | +1.000 |
| musiq | +0.700 | -1.000 |
| dover_technical | +0.600 | -0.900 |
| dover_aesthetic | +0.700 | -1.000 |

