# 剩余清单任务（任务 6 / 7 / 8 / 9 / 12 / 13 / 14 / 16）

完整结论、口径与失败案例见 [REPORT.md](REPORT.md)。这里是文件索引。

## 目录

| 路径 | 内容 | 对应任务 |
|---|---|---|
| `motion/groups.json` | **固定分组清单**：30 clip 的光流幅值中位数、三分位归属 | 6 |
| `motion/analysis.json` | 分组均值、逐 clip 配对增益、Spearman 相关、光流对齐最差 5 例 | 6 |
| `bench/merged.json` | 三次运行合并的速度/显存表（引用 `min_s`，附 `run_spread_ratio`） | 8 |
| `bench/speed{,_clean,_final}.json` | 三次原始运行，隔数小时，用于量化共享 GPU 的噪声 | 8 |
| `align/align.json` | 已知位移标定（1/1–1/8）、FB 一致性 mask、边界越界比例 | 9 |
| `degradation/degradation.json` | Gaussian vs bicubic 退化 A/B（4 方法 × 5 clip × 25 帧） | 12 |
| `degradation/degradation_cv2_uncorrected.json` | 同实验的**初版**（cv2 无抗锯齿 LR），留作修正前后对照，**不要再引用其数字** | 12 |
| `edvr_vimeo_ab/edvr_vimeo_ab.json` | Vimeo 训练 vs REDS 训练的 EDVR，30 clip × 100 帧，归因 5 dB 差距 | 12b |
| `edvr_neighbor/neighbor.json` | 正常 5 帧 vs 中心帧重复 5 次；含丢首尾 2 帧的边界对照 | 13 |
| `vimeo/vimeo_part2.json` | Vimeo-90K 全 7824 clip 中心帧，两套 EDVR 权重分行 | 14 |
| `qual/*.png` | clip 013 的三类 ROI：六联、连续 4 帧、Part 2 五联 | 7 / 16 |
| `qual/qual_013.json` | ROI 坐标与选取规则（最终图用的是手工固定窗口） | 7 / 16 |
| `qual/qual_004.json`、`qual_011/qual_011.json` | clip 004 / 011 的 ROI 坐标与选取规则（启发式选窗，clip 无招牌时标签只是窗口规则） | 7 / 16 |
| `tecogan_ab/000_shift0_manifest.json` | TecoGAN 预平移 A/B 的 shift=0 侧：权重哈希、帧号、逐帧输出哈希 | 10 |
| `tecogan_ab/000_shift0375_manifest.json` | 同上，另一侧。**注意**：manifest 内记录的 `shift` 是 **0.25**，与目录名和 `REPORT.md` 中的 `0.375` 不符，见下方说明 | 10 |

视频在 `p2/videos/013_6up_part1_flow_f24fps.mp4`（24 fps，任务 16 要求源帧率）。

## 三条容易误读的地方

1. **`bench` 引用的是最小值**，不是中位数。同一份代码隔几小时跑，tavg3 得
   3.6 / 4.1 / 22.4 s；`median` 测的是同机其他同学的负载。带
   `unstable_between_runs: true` 的条目只能当量级看。
2. **`degradation` 已于 2026-11-18 修正并重跑。** 初版把"课程 LR"写成 cv2
   `INTER_CUBIC`，实际自带 LR 是 MATLAB 风格抗锯齿下采样（a=−0.5），两者差约
   1.5 dB。现在 `degradation.json` 是正确口径；初版留在
   `degradation_cv2_uncorrected.json` 仅供对照。`gauss` LR 仍是 cv2 实现
   （同核同尺度，不是 TecoGAN 的 TF 图，不保证逐位一致），但先做高斯模糊已限制
   带宽，影响很小。
3. **`edvr_neighbor` 的条件 B 改变了输入分布**（对齐没有东西可对齐），
   它测的是"邻帧信息对该权重的价值"，**不是** PCD 或 TSA 的单独贡献。
4. **`tecogan_ab` 的 shift 值有一处未解决的矛盾。** 目录名与 `REPORT.md` 都写
   `0.375`，但 `000_shift0375/manifest.json` 里记录的是 `shift: 0.25`，且该值与
   manifest 的哈希自洽（用 0.25 复算得到同一哈希，用 0.375 则对不上）。另外用
   相位相关量了两个目录输出的实际相对位移，中位数约 `0.36` HR px，与两者都不吻合。
   在查清之前，`REPORT.md`「`--shift 0.375` 已补偿，补回 2.664 dB」里的 `0.375`
   当作**待核实**，不要直接引用为已确认的补偿量。这件事不影响「网格错位已处理」
   这个结论方向，只影响那个具体数值的标注。

## 复算

```bash
cd ~/p0 && export TORCH_HOME=/tmp/torch_home

/tmp/srenv/bin/python scripts/motion_groups.py     --out ~/p0/out/motion
/tmp/srenv/bin/python scripts/bench_speed.py       --out ~/p0/out/bench/speed.json \
    --frames 100 --reps 5 --gap 3
/tmp/srenv/bin/python scripts/bench_merge.py       --runs <三次 speed*.json> \
    --out ~/p0/out/bench/merged.json
/tmp/srenv/bin/python scripts/align_diag.py        --out ~/p0/out/align \
    --clips 004 011 019 --frames 12
/tmp/srenv/bin/python scripts/teco_degradation.py  --out ~/p0/out/degradation_aa \
    --clips 000 004 011 019 022 --frames 25 --with-tecogan
/tmp/srenv/bin/python scripts/edvr_vimeo_ab.py     --full ~/p0/out/full \
    --out ~/p0/out/edvr_vimeo_ab
/tmp/srenv/bin/python scripts/edvr_neighbor.py     --out ~/p0/out/edvr_neighbor
/tmp/srenv/bin/python scripts/vimeo_part2.py       --out ~/p0/out/vimeo --n 0
/tmp/srenv/bin/python scripts/qualitative.py       --out ~/p0/out/qual_013 \
    --clip 013 --frames 100 --consecutive 4 --video \
    --roi text=980,300 texture=520,60 motion=380,560
```
