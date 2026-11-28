# 指标间排序一致性（逐片段 Spearman，跨方法）

每个片段内，对候选方法按两列排序求 Spearman。样本少（方法数）时 rho 波动大，只作为方向参考，不能当显著性。

| 片段 | 指标对 | n | Spearman |
|---|---|---|---|
| 000 | dists vs musiq | 5 | -1.000 |
| 000 | dists vs dover_technical | 5 | -1.000 |
| 000 | musiq vs dover_technical | 5 | +1.000 |
| 000 | dists vs dover_aesthetic | 5 | -1.000 |
| 000 | musiq vs dover_aesthetic | 5 | +1.000 |
| 002 | dists vs musiq | 5 | -1.000 |
| 002 | dists vs dover_technical | 5 | -1.000 |
| 002 | musiq vs dover_technical | 5 | +1.000 |
| 002 | dists vs dover_aesthetic | 5 | -1.000 |
| 002 | musiq vs dover_aesthetic | 5 | +1.000 |
| 009 | dists vs musiq | 5 | -1.000 |
| 009 | dists vs dover_technical | 5 | -0.900 |
| 009 | musiq vs dover_technical | 5 | +0.900 |
| 009 | dists vs dover_aesthetic | 5 | -1.000 |
| 009 | musiq vs dover_aesthetic | 5 | +1.000 |

