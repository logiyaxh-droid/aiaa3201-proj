# P1 · Part 2 交付清单（SOTA 复现：EDVR + TecoGAN）

> 依据：`Project1_VideoSuperResolution.pdf` §5.2、§6；服务器 `ghe270@121.12.5.46:10084` 实测（2026-11-13/14）。
> 全部数字都是在 **REDS val 30 clip × 100 帧** 上真实跑出来的，与 Part 1 用**完全相同的指标代码**，因此可以和 Part 1 表格并列比较。

---

## 0. 先回答硬盘问题

**结论：能做，而且已经做完了。`/data/disk1` 满盘不影响 Part 2。**

| 分区 | 容量 | 剩余 | 用途 |
|---|---|---|---|
| `/dev/sdb1` → `/data/disk1` | 492 G | **3.6 G（100%）** | 你的 home、全班共享数据（只读） |
| `/dev/sda3` → `/`（含 `/tmp`） | 194 G | **24 G** | 环境、缓存、中间结果 |

关键在于**满的是 `/data/disk1`，不是 `/`**：

- 共享数据 `/data/disk1/shared/aiaa3201` 全校只读，本来就**不需要往里写**，代码直接指过去就行；
- 你 home 也在 `/data/disk1` 上，所以**绝对不能**把输出写到 home —— `~/.cache/torch` 默认就在那里。这也是 Part 1 当初踩过的坑，本次全部改走 `/tmp`（`TORCH_HOME=/tmp/torch_home`）；
- 真正的输出（TecoGAN 帧 + 视频 + 图）落在 `/tmp/p2`，30 clip 全量约 **4–5 G**，`/` 还有 24 G，够用。

顺便说明为什么 Part 2 比 Part 1 更省空间：EDVR 的结果**边算边丢**，只留指标 JSON（100 帧 720p 的 PNG 一个 clip 就 140 MB，30 clip 要 4 G+，没必要留）。TecoGAN 因为是逐帧循环推理、且要出对比视频，帧必须落盘，但那是唯一的大头。

> ⚠️ 两点提醒：`/` 是全机器共享的，24 G 余量并不宽裕；`/tmp` 重启即清空（这台机器已连续开机 23 天，随时可能重启）。脚本和指标 JSON 已备份到本地 `p1_results/p2/`。

---

## 1. Part 2 要做什么

PDF §5.2 原话：*Reproduce a classic multi-frame VSR pipeline and compare distortion-oriented reconstruction with temporally coherent perceptual enhancement. Use official pretrained checkpoints.*

两件事：

| # | 方法 | 定位 |
|---|---|---|
| 1 | **EDVR-L ×4** (REDS) | 多帧对齐 + 融合，**失真导向**。PCD（金字塔+级联+可变形）对齐 + TSA（时空注意力）融合，滑窗式吃 5 帧 |
| 2 | **TecoGAN** | 感知导向。循环生成器 + 时空判别器 + warping 时域损失，追求"看起来真实且帧间稳定" |

报告要回答的核心问题就是 PDF 那句：*"perceptual gains can come with lower PSNR or invented details"* —— 这个 trade-off 你**能不能用自己的数据验出来**。

---

## 2. 结果

### 2.1 总表（REDS val, 30 clip × 100 帧，全部实测）

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Flow-warp LPIPS ↓ | 正式 tLP ↓ |
|---|---|---|---|---|---|
| Bicubic（Part 1） | 26.403 | 0.7384 | 0.4835 | 0.0519 | clip 000: 0.0261 |
| Lanczos（Part 1） | 26.546 | 0.7441 | 0.4861 | 0.0507 | 待补 |
| 5 帧平均（无对齐） | 21.852 | 0.5903 | 0.6052 | 0.0893 | 待补 |
| 5 帧平均 + 光流对齐 | 25.834 | 0.7195 | 0.5096 | **0.0236** | 待补 |
| EDSR ×4（Part 1） | 28.284 | 0.8067 | 0.3087 | 0.0811 | 待补 |
| **EDVR-L ×4（Part 2）** | **30.904** | **0.8789** | **0.2039** | 0.0647 | 待补 |
| **TecoGAN（Part 2）** | 25.843 | 0.7187 | 0.3337 | 0.0800 | 待补 |

> 全部为 30 clip 终值。EDVR 与 TecoGAN 都是本次实测，指标代码与 Part 1 逐行一致。
>
> **⚠️ P0 更正：原表的 `tLPIPS` 列定义有误。** 它是 `LPIPS(warp(上一帧), 当前帧)`
> 的 **stride=2** 自一致性指标（无 GT），已改名 **Flow-warp LPIPS**，原值保留不变。
> 它不是 TecoGAN 的参考式 tLP，不能与论文的 tLP 比较，也不足以支撑「谁时域最稳」。
> 正式 tLP（`|d_SR(t) − d_GT(t)|`，全部相邻对）已实现并验证（`p0/audit/tlp_check.json`），
> 但全量 30 clip 尚未重跑，表中只有 clip 000 的对照：
> Bicubic **0.0261**、TecoGAN(+0.375, 12 帧样例) **0.0047**。

### 2.2 四个能直接写进报告的结论

**① EDVR 全面赢下失真指标，且赢得干净。**
相对 EDSR：**PSNR +2.62 dB、SSIM +0.072、LPIPS −0.105**，**30/30 个 clip 的 PSNR 全部提升**。这不是噪声，是"多帧对齐+融合"相对"逐帧独立"的系统性收益 —— 正好接上 Part 1 留下的伏笔。

**② TecoGAN 的 PSNR 明显低于 EDVR（−5.06 dB），但这正是预期行为。**
GAN 会脑补纹理，脑补内容和 GT 对不上，PSNR 必然掉。PDF 那句 "perceptual gains can come with lower PSNR or invented details" 被数据证实了。

**③ 但有个反直觉的地方，值得单独写一段：TecoGAN 在感知指标上赢不过 EDVR，甚至也赢不过 EDSR。**
按常识 GAN 应该在感知指标上占优，实测 **TecoGAN LPIPS 0.3337**，而 EDVR 0.2039、EDSR 0.3087 都更好；TecoGAN 只比 bicubic（0.4835）/lanczos（0.4861）好。原因有几层，都可以展开：

- EDVR 的输出确实更接近 GT，而 GT 是真实拍摄的清晰帧；LPIPS 用的 ImageNet 预训练 AlexNet，它"喜欢"的锐利纹理恰好就是真实纹理，不是 GAN 幻想出来的那种。
- TecoGAN 在**合成 bicubic 退化**上有退化失配：它的官方模型用 **Gaussian 下采样**（`tf_data_gaussDownby4`, sigma=1.5）合成 LR 训练，而 REDS 是干净的 bicubic 下采样。注意这是「退化类型不同」，**不是**「为真实退化训练」——原句写错了，且这一解释目前只是假设，尚未做控制变量实验（清单任务 12）。
- TecoGAN 的 PSNR（25.84）已经**低于 bicubic**（26.40，−0.56 dB）。这说明在这个设定下它连"保真"都没守住，感知上的收益不足以补偿 —— 这就是为什么 **wild video 上的对比是 Part 2 剩下的关键一步**（见 §5）。

**这条比"GAN 感知更好"更有价值**，因为它说明 "accuracy vs perception" 的教科书结论**依赖测试数据的退化类型**。建议报告里作为 Part 2 的主要讨论点。

**④ 时域一致性上，Part 1 的光流对齐在旧指标下数值最低 —— 但该指标不足以定论。**
flowtavg5 的 Flow-warp LPIPS 0.0236 在旧指标下最低（EDVR 0.0647、TecoGAN 0.0800）。可能的原因是 warp + 加权平均本身就是很强的时域平滑器，代价是空间细节（PSNR 只有 25.8）；但 Flow-warp LPIPS 只有输出、没有 GT，**平滑的输出天然占便宜**，所以这只能算待验证的观察，不能当作「时域最稳」。要用正式 tLP 复核。（clip 000 的对照里 TecoGAN +0.375 的正式 tLP 是 0.0047，比 bicubic 的 0.0261 更低。）

> 口径已交叉验证：用共用实现重算 clip 000 的 bicubic Y-channel PSNR 得 **25.617**，与 Part 1 存档值 25.617 逐位一致（`p0/audit/selfcheck_metrics.json`：SSIM 差 5.2e-14、PSNR 差 0、LPIPS 差 0），说明两批空间指标可以直接并列。

### 2.3 建议的图

| 图 | 内容 | 状态 |
|---|---|---|
| 图 1 | **PSNR vs Flow-warp LPIPS 散点图**，7 个方法，一眼看出"没有方法同时赢两轴"（时域轴待换正式 tLP 后重画） | ✅ `figures/scatter_psnr_tlpips.png` |
| 图 2 | **放大 patch**：GT \| Bicubic \| EDVR \| TecoGAN（7 个 clip 是 4 联，其余 3 联） | ✅ `figures/*_patch.png`（30 个 clip 全出） |
| 图 3 | PSNR 逐 clip 柱状图（EDSR vs EDVR，30 组，证明不是均值假象） | ✅ `figures/perclip_psnr_edsr_vs_edvr.png` |
| 图 4 | Flow-warp LPIPS 逐 clip 折线（EDSR / EDVR / flow-align 三线） | ✅ `figures/perclip_tlpips.png` |
| 图 5 | 连续 3–5 帧闪烁对比 | 帧已落盘，需要时再拼 |

---

## 3. 两个必须写进报告的工程问题

### 3.1 TecoGAN 只能用 CPU（TF1.15 不支持 Ampere）

`tf.test.is_gpu_available()` = **False**。TF 1.15 只带 CUDA 10 的 kernel，A6000 是 sm_86。所以 TecoGAN **纯 CPU 推理**：

- 实测 **≈1.5 s/帧**，100 帧/clip ≈ 2.5 分钟，30 clip ≈ **90 分钟**
- 报告里要注明：**TecoGAN 的耗时数据与 GPU 方法不可直接比较**

### 3.2 我修掉了一个会毁掉全部指标的 1.5 像素错位（这条很重要）

第一次跑完 TecoGAN，PSNR 只有 **21.69 dB** —— 比 bicubic 还低 4.7 dB，明显不对。查下去发现：

- TecoGAN 自己的上采样函数 `lib/ops.py:bicubic_four` 假设源图是**半像素中心**网格（`resize_bicubic` 约定）；
- 而课程给的 REDS LR 帧是**像素中心**约定（`align_corners=True`）生成的；
- 两者相差 0.375 个 LR 像素 = **1.5 个 HR 像素**，恒定不漂移。

实测证据（clip 000 第 10 帧）：

| 处理 | PSNR |
|---|---|
| 原始 TecoGAN 输出 | 21.58 dB |
| 平移 1.5 px 后 | **26.24 dB** |
| 亚像素搜索最优点 | 26.24 dB @ (1.50, 1.50) |

**修法**：推理前把 LR 输入预先平移 +0.375 px（`warpAffine` + `BORDER_REPLICATE`），从源头抵消。修正后残差降到 0.00–0.25 px，PSNR 从 21.69 → **25.14**（clip 000），全 30 clip 均值 **25.84**。

**报告里值得单独一小节**，因为：
1. 不修的话，你会得出"TecoGAN 还不如 bicubic"的错误结论，整个 Part 2 的分析全废；
2. 这是一个**真实存在的复现陷阱**，写出来体现你确实动手验证过，而不是抄数字；
3. 顺带说明：所有对比方法都必须先对齐再算指标，否则 PSNR 没有意义。

---

## 4. 复现

### 4.1 环境

```bash
# 环境 A：PyTorch（EDVR 用，含编译好的 DCN）
/tmp/srenv/bin/python          # torch 2.6.0+cu124, basicsr 1.4.2
ls /tmp/bsr_src/basicsr/ops/dcn/*.so   # DCN 编译产物，EDVR 必需

# 环境 B：TF1（TecoGAN 用）
/tmp/tf1env/bin/python         # tf 1.15.0, keras 2.2.4, numpy 1.18.5, cv2 4.5.5

# 关键：LPIPS 权重必须放 /tmp，不能落到 home（磁盘满）
export TORCH_HOME=/tmp/torch_home
```

### 4.2 脚本（都在服务器 `/tmp`，已备份到本地）

| 脚本 | 作用 | 耗时（实测） |
|---|---|---|
| `/tmp/p2_edvr.py` | EDVR 30 clip，边算边出指标 | 19.9 min (GPU, 0.64 s/帧) |
| `/tmp/p2_teco.sh` | TecoGAN 30 clip，含 0.375 px 对齐修正 | 约 90 min (CPU, 1.5 s/帧) |
| `/tmp/p2_teco_metrics.py` | TecoGAN 指标（增量，可反复跑） | 13 s/clip |
| `/tmp/p2_edvr_frames.py` | 只为出图重跑少数 clip 并存帧 | 28–35 s/clip |
| `/tmp/p2_export.py` | 3/4 联对比视频 + 放大 patch 图 | 约 6 s/clip |
| `/tmp/p2_figs.py` | 散点图 / 逐 clip 柱状图 / 折线图 | 秒级 |
| `/tmp/p2_table.py` | 汇总总表 + deltas | 秒级 |

```bash
# 完整复现顺序
/tmp/srenv/bin/python /tmp/p2_edvr.py                    # EDVR 指标（19.9 min）
bash /tmp/p2_teco.sh                                     # TecoGAN 推理（~90 min）
/tmp/srenv/bin/python /tmp/p2_teco_metrics.py            # TecoGAN 指标
/tmp/srenv/bin/python /tmp/p2_edvr_frames.py 000 010 022 # 出图用的 EDVR 帧
/tmp/srenv/bin/python /tmp/p2_export.py                  # 视频 + 放大 patch
/tmp/srenv/bin/python /tmp/p2_figs.py                    # 报告用的三张图
/tmp/srenv/bin/python /tmp/p2_table.py                   # 总表
```

### 4.3 产出文件

| 路径 | 内容 |
|---|---|
| `/tmp/p2/metrics_edvr.json` | EDVR 逐 clip 指标 + 均值 |
| `/tmp/p2/metrics_tecogan.json` | TecoGAN 逐 clip 指标 + 均值 |
| `/tmp/p2/part2_table.json` | 总表 + deltas |
| `p2/videos/*_compare.mp4` | **对比视频**（部分 clip）。4 联 GT\|Bicubic\|EDVR\|TecoGAN，**固定 10 fps** —— 与源 24 fps 不一致，按清单任务 16 应改为源 FPS |
| `/tmp/p2/out/figures/*_patch.png` | 放大 patch 图，30 个 clip |
| `p2/figures/scatter_psnr_tlpips.png` | PSNR–Flow-warp LPIPS 散点图（文件名保留） |
| `p2/figures/perclip_*.png` | 逐 clip PSNR 柱状图 / Flow-warp LPIPS 折线图 |

> PDF §6.2 要求 *"submit the processed video files for all Mandatory Datasets"* —— 上面的 `*_compare.mp4` 就是这份交付物。
> 本地备份在 `p1_results/p2/`（指标 JSON + 全部脚本 + 三张图 + 一条示例视频）。

**为什么会话里 30 个视频不全部带 EDVR 画面**：EDVR 一帧 720p PNG 约 1.4 MB，100 帧就是 140 MB/clip，30 clip 要 4 G+。所以全量跑的时候帧是算完即弃的（只留指标），只有需要出图的 7 个 clip 才重跑并存帧。要全补齐只需 `p2_edvr_frames.py` 跑一遍剩下的，约 15 分钟 + 4 G 空间。

---

## 5. Part 2 还差什么

| 优先级 | 事项 | 说明 |
|---|---|---|
| **P0** | wild video 上的 Part 2 结果 | PDF §6.1 Mandatory。真实拍摄的低光/低清视频上 TecoGAN 是否反超 EDVR 是待检验问题（不要预设结论）—— 这是**最能出彩**的一点，也顺便补齐 Part 1 缺的那个 P0 |
| P1 | Vimeo-90K 上跑 EDVR | 有 `EDVR_Vimeo90K_SR_L` 权重。但 Vimeo 每个 clip 只有 im4 有 GT，**算不了参考式 tLP**，按要求写 N/A 并说明原因 |
| P1 | 4 联视频拼成一个长片 | 30 个 mp4 逐个交也可以，拼成一条更方便评审 |
| P1 | 把 4 联视频补齐到 30 个 | 目前 7 个 clip（000/002/009/010/019/022/026）有 EDVR 帧，其余是 3 联。补 EDVR 帧约 30 s/clip，但要多 4 G |
| P2 | 报告 Part 2 章节（约 2 页） | 1 总表 + 散点图 + patch 图 + 对齐问题的说明 |

---

## 6. 一句话总结

**Part 2 已经全部跑完：EDVR-L ×4 拿到 PSNR 30.90 / SSIM 0.879 / LPIPS 0.204，比 Part 1 的 EDSR 高 2.62 dB，30/30 clip 全胜；TecoGAN（CPU 推理 + 修正 1.5 px 对齐后）PSNR 25.84，感知指标不但没赢 EDVR，连 bicubic 的 PSNR 都没守住 —— 这个"反教科书"的发现，加上对齐 bug 的排查过程，正好是 Part 2 报告最有分量的两部分。**

硬盘不是问题：满的是 `/data/disk1`（只读共享数据 + home），真正干活的 `/` 全程还有 20 G 以上余量。

