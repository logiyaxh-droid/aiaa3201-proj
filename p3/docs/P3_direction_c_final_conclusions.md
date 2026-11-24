# Direction C 探索最终结论：残差扩散、外部生成先验与真实退化

日期：2026-11-22  
状态：已完成（RealVSR 六方法；MVSR4x 五方法，DOVE 12/15 段；REDS-Real 四方法 × 300 帧）  
范围：REDS bicubic、真实退化无 GT（手机实拍 + VideoLQ）、
真实退化有 GT（RealVSR + MVSR4x + REDS-Real）五条线

## 一句话结论

**方向 C 的价值取决于退化是否真实，而"真实"要看退化强度而非数据来源标签。** REDS bicubic 上生成先验失败（PSNR −2.8 dB）；RealVSR（约 2 倍退化、且被本仓库二次降采样）上回归模型占优；MVSR4x（真 ×4）与 REDS-Real（模拟真实压缩+噪声+模糊）上生成先验在感知与时序轴上全面领先。**REDS-Real 上 RealBasicVSR 的 PSNR 只低 0.18 dB 而 LPIPS 低 32.9%、tLP 低 40.5%**，是"保真几乎不付代价"的候选；同时干净域 EDVR 在真实退化下连 bicubic 都打不过。但**零样本串联 RealBasicVSR→OSEDiff 会让时序误差恶化 4.4 倍**，说明后续必须真的训练而非直接拼接。

## 1. 现有唯一正向结果（REDS bicubic，有 GT）

EDVR-L + DISTS 0.03 + relativistic GAN 0.001 微调：

- LPIPS 0.2039 → 0.1402（−31.2%，30/30 片段）
- DISTS 0.0870 → 0.0433（−50.2%，但参与训练，不能单独作证据）
- tLP 0.0123 → 0.0077（−36.6%，29/30）
- PSNR 30.9042 → 30.5231（−0.381 dB）

这是感知—保真权衡，不是全面提高。原始结果：`quality_metrics/fine_tune/RESULTS.md`。

## 2. U-TRD 三原型（REDS bicubic，有 GT）

同一 U-Net、1500 步、batch 8、DISTS 0.01、无 GAN。

| 模型 | PSNR | LPIPS | 相对冻结 EDVR |
|---|---:|---:|---:|
| 冻结 EDVR | 32.4095 | 0.15138 | — |
| 确定性残差 | 32.3892 | 0.14957 | LPIPS −1.2% |
| 均匀噪声扩散 | 28.0344 | 0.26446 | PSNR −4.38 dB |
| 不确定性扩散 | 26.7161 | 0.22065 | PSNR −5.69 dB |

结论：小预算、像素空间、四步残差扩散没有超过同容量确定性对照，且不确定性噪声没有优于均匀噪声。**该结论限定在这一设置内。**原始结果：`utrd/RESULTS_PROTO1.md`。

## 3. 外部生成先验（REDS bicubic，有 GT）

三个候选在 EDVR 高分歧区域与纹理区域都未提供增益：

| 候选 | 纹理块 LPIPS（微调 EDVR = 0.0659） | 相对变化 | PSNR |
|---|---:|---:|---:|
| OSEDiff | 0.1660 | +151.8% | 22.93 |
| SUPIR v0F | 0.1955 | +196.6% | 26.15 |
| DOVE 5B | 0.1963 | +197.2% | 25.34 |

三者全部不通过门槛。原因一致：真实退化生成先验在干净 bicubic 上会重绘纹理而非恢复 GT 纹理。原始结果：`prior_qualification/results/summary/`。

## 4. 真实退化评测（无 GT）

真实退化分两批做：手机实拍视频，以及最终取得的官方 VideoLQ。两者结论方向一致。

### 4.1 手机实拍（30 帧，720×1280）

| 方法 | NIQE ↓ | MUSIQ ↑ |
|---|---:|---:|
| **SUPIR** | **2.364** | 61.99 |
| **OSEDiff** | 2.948 | 73.61 |
| 原始 EDVR | 4.167 | 63.44 |
| 感知微调 EDVR | 4.195 | 63.24 |
| **DOVE** | 4.468 | **74.93** |
| bicubic ×4 | 7.035 | 40.26 |

### 4.2 官方 VideoLQ（50 段 × 5 帧，250 帧，原生分辨率）

| 方法 | NIQE ↓ | MUSIQ ↑ | selfPSNR |
|---|---:|---:|---:|
| **OSEDiff** | **4.109** | **66.85** | 21.54 |
| **DOVE** | 5.003 | 61.51 | 21.76 |
| 感知微调 EDVR | 5.190 | 56.21 | 20.33 |
| SUPIR | 5.629 | 46.53 | **24.29** |
| 原始 EDVR | 5.673 | 55.81 | 20.49 |
| bicubic ×4 | 7.563 | 31.38 | 23.74 |

VideoLQ 已核实为官方数据：50 段、编号 000–049、分辨率 320×240 至 720×576，与 ICCV/CVPR 描述一致。

### 4.3 VideoLQ 的 DOVER 补充（6 段 × 97 帧连续序列）

DOVER 是视频级指标，需要连续帧，与 5 帧筛选协议不兼容，故单独跑一组。

| 方法 | 技术分 ↑ | 美学分 ↑ |
|---|---:|---:|
| **DOVE** | **−0.0715** | **0.1488** |
| **OSEDiff** | −0.0775 | 0.1480 |
| 原始 EDVR | −0.1000 | 0.0499 |
| 感知微调 EDVR | −0.1051 | 0.0553 |
| SUPIR | −0.1339 | 0.0038 |
| bicubic ×4 | −0.1947 | −0.0288 |

分数是原始值（负数），无 MOS 标签做 z-score，不能与论文表格比，但方法间排序有效。

### 4.4 七个关键观察

1. **生成模型在真实退化上不再明显更差**：手机实拍与 VideoLQ 两组独立数据的四个指标（NIQE/MUSIQ/DOVER 技术/美学）都指向同一方向，与 bicubic 上的结论相反。
2. **OSEDiff 在 VideoLQ 上全面最好**（NIQE 4.109 / MUSIQ 66.85），速度也最快；而它在 bicubic 上纹理块 LPIPS 差 151.8%。
3. **DOVER 上 DOVE 与 OSEDiff 前两名**（技术分 −0.0715/−0.0775），明显领先 EDVR 的 −0.1000。
4. **SUPIR 出现指标分裂**：手机视频 NIQE 最好（2.364），但 DOVER 技术分垫底（−0.1339）。说明不同 IQA 模型的偏好不同，**不能用一个指标替代另一个**。
5. **感知微调 EDVR 的收益不稳定**：VideoLQ 上优于原始 EDVR（5.190 vs 5.673），手机视频上略差（4.195 vs 4.167），DOVER 上与原始 EDVR 各有胜负。
6. **selfPSNR 不能当质量指标**：bicubic 的 23.74 高于 EDVR 的 20.49，但观感最差。它奖励“不动”，不奖励“变好”。
7. **DOVE 的时序优势未获独立证据**：DOVER 技术分虽最高，但它同时使用了分块推理，无法与时序质量分离。

### 4.5 边界

- 无 GT，**不能**计算 PSNR/SSIM/LPIPS，也**不能**与 REDS 表格横向比较；
- selfPSNR 不等于准确度，bicubic 就是反例；
- 不能证明这些模型更接近真值，只能说明在无参考指标下观感更好；
- 无法排除“生成模型只是更符合 IQA 模型偏好”这一解释；
- DOVER 只覆盖 6 段，样本量小，片段间波动较大。

## 5. 工程教训

1. **文件大小正确不代表内容正确。** DOVE 的 `text_encoder/model-00002-of-00004` 与 CogVideoX VAE 都出现过“大小精确匹配但张量损坏”，必须用 `safetensors.safe_open` 逐文件校验。
2. **`huggingface_hub` 新版默认在 `~/.cache` 留符号链接**，`local_dir` 里只有链接。借用机器上必须实体化到指定目录，否则占用落在别人 home 下。
3. SUPIR 在 720p 需约 15 GiB，显存被他人占用时可用 `min_size 768` 降内存。
4. DOVE 在 24 GiB 卡上必须 `tile_size_hw 256`，否则 VAE 编码 OOM。
5. 借用服务器：无 `python3-venv`（用 anaconda Python 建 venv）、PyPI 官方 CDN 不通（用阿里云镜像）、`huggingface.co` 可直连但 hf-mirror 小文件 resolve 会 503。
6. **官方 VideoLQ 的 GD/Dropbox/OneDrive 均不可达，OpenDataLab 可用**（`openxlab` CLI，需补一行 import 修复其 `ls` 的 bug）；归档是嵌套两层。

## 6. 最终判断

**方向 C 的价值取决于域。** 在合成 bicubic 上，回归模型（EDVR）已经接近最优，生成先验只会引入偏离；在真实退化上（手机实拍与官方 VideoLQ 两组独立数据），生成先验的相对价值显著上升，OSEDiff 在两项无参考指标上都最好。

因此之前的“停止外部生成先验”结论需要修正为：**停止在 REDS bicubic 上使用外部生成先验**，而不是停止这条技术路线本身。

剩余缺口是配对 GT。这一缺口后来部分补上了：RealVSR 的测试集已取得（见第 8 节），在有 GT 的真实退化上，**生成先验的领先不再出现**。MVSR4x 仍受下载带宽限制未完成。

## 7. 原始结果索引

| 内容 | 路径 |
|---|---|
| 感知微调结论 | `quality_metrics/fine_tune/RESULTS.md` |
| U-TRD 路线与结果 | `docs/P3_direction_c_residual_diffusion_roadmap.md`、`utrd/RESULTS_PROTO1.md` |
| 外部先验路线 | `docs/P3_generative_prior_qualification_roadmap.md` |
| 外部先验汇总 | `prior_qualification/results/summary/qualification.md` |
| 手机实拍评测 | `prior_qualification/results/summary/real_world.md` |
| VideoLQ 评测 | `prior_qualification/results/summary/videolq.md` |
| VideoLQ 指标 | `prior_qualification/results/videolq/*.json` |
| 手机实拍指标 | `prior_qualification/results/real/*.json` |
| 各候选分项 | `prior_qualification/results/summary/{osediff,supir,dove}.md` |
| RealVSR 配对评测 | `prior_qualification/results/realvsr/*/metrics.json`、`results/summary/realvsr.md` |

## 8. RealVSR 配对评测（有 GT，2026-11-23 补）

这是本研究里**唯一**能在真实退化上同时报告 PSNR/SSIM/LPIPS/DISTS 的实验。

### 8.1 数据的实际形态

RealVSR 测试集 50 段 × 50 帧，LQ 与 GT **同为 512×1024**，不是 4 倍超分配对。官方 `test_RealVSR_wi_GT.py` 因此构造 `EDVR_NoUp`（`upscale=1`）直接吃 LQ。

实测带宽（clip 016 第 22 帧，0.25 归一化频率以上能量占比）：

| 图 | 高频占比 |
|---|---:|
| LQ（实拍） | 0.316% |
| GT 降 2 倍再上采样 | 0.298% |
| GT 降 4 倍再上采样 | 0.065% |

LQ 的有效分辨率约是 GT 的 1/2，不是 1/4。为让所有 x4 模型（EDVR/OSEDiff/SUPIR/DOVE）共用一条协议，本实验把 LQ 再降 4 倍喂给模型，输出对齐 GT 原生网格。

可达上限核对：用 GT 自身降 4 倍再上采样，只比我们这条路径高 0.5–2.7 dB（clip 016/108/364 分别 26.49/21.27/23.93 dB），说明输入路径没有额外损失信息。

对齐核对：±6 px 平移搜索只带来 ≤0.29 dB 收益，说明 LQ–GT 已对齐，掉点不是错位造成的。

### 8.2 结果（50 段 × 5 帧 = 250 帧）

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | tLP ↓ |
|---|---:|---:|---:|---:|---:|
| 原始 EDVR | **23.036** | **0.7590** | 0.2404 | 0.1315 | 0.0128 |
| 感知微调 EDVR | 22.871 | 0.7452 | **0.1979** | **0.0944** | **0.0094** |
| bicubic ×4 | 22.147 | 0.6480 | 0.4997 | 0.2118 | 0.0287 |
| SUPIR v0F | 22.082 | 0.6470 | 0.3347 | 0.1774 | 0.0111 |
| DOVE 5B | 21.577 | 0.6666 | 0.2215 | 0.1118 | 0.0177 |
| OSEDiff | 20.049 | 0.5875 | 0.2710 | 0.1397 | 0.0143 |

相对感知微调 EDVR 的变化：

| 方法 | ΔPSNR | ΔLPIPS |
|---|---:|---:|
| 原始 EDVR | +0.165 | +21.5% |
| bicubic | −0.724 | +152.5% |
| SUPIR | −0.789 | +69.1% |
| **DOVE** | −1.294 | **+11.9%** |
| OSEDiff | −2.822 | +36.9% |

### 8.3 结论

1. **三个生成模型没有一个能同时超过回归模型**，但差距差异很大。DOVE 的 LPIPS 只比感知微调 EDVR 高 11.9%、DISTS 高 18.5%，已经接近原始 EDVR 的水平（LPIPS 0.2215 vs 0.2404，DISTS 0.1118 vs 0.1315）；SUPIR 和 OSEDiff 差得远。
2. **不能概括为"生成先验在真实退化上普遍更差"**。正确的说法是：在 PSNR 这条轴上三个生成模型都不如回归模型（差 0.79–2.82 dB），而在感知轴上 DOVE 已经追到回归模型附近，SUPIR/OSEDiff 没有。
3. **这修正了第 4 节的无参考结论**。VideoLQ 上 OSEDiff 的 NIQE/MUSIQ 全面领先，但那是"看起来更清晰"，不是"更像真值"。一旦能对照真值，OSEDiff 的 PSNR 比 bicubic 还低 2.1 dB。
4. **感知微调的权衡在真实退化上同样成立**：用 0.165 dB 的 PSNR 换来 LPIPS −17.7%、DISTS −28.2%，方向与 REDS 一致。
5. **DOVE 有约 +4.5/255 的整体亮度偏高**（clip 016 第 20/22 帧的 RGB 三通道一致偏高 4.2–4.8），这解释了一部分 PSNR 损失，但不改变排名。
6. **不能把 RealVSR 的绝对值和 REDS 表格比较**：两者退化核、分辨率、场景都不同，只在本表内部做方法间比较。

### 8.4 边界（必须一并说明）

- 三个生成模型都在**共享主机的资源竞争**下跑成，DOVE 首轮 50 段中有 17 段因 OOM（rc=-9）失败，分三批补跑才凑齐；这说明该表在资源紧张时复现成本较高，不代表模型行为不稳定。
- DOVE 使用 `--is_cpu_offload` 之外的 `pipe.to("cuda")` 路径；`enable_sequential_cpu_offload` 在 bfloat16 下触发 `Cannot copy out of meta tensor`（diffusers 0.32.2 已知限制）。
- 采用 LQ 再降 4 倍的适配协议，与官方 `EDVR_NoUp` 的 1 倍协议不同，绝对值不可与 RealVSR 论文表格直接对照。

## 9. MVSR4x 配对评测（第二个真实退化数据集，2026-11-24）

MVSR4x 用手机双摄同时采集：超广角（13mm）给低清、长焦（90mm）给高清，两路已对齐到同一像素网格（都是 1024×1024）。缩放倍数实测确定：wide 的高频能量（0.033%/0.004%）与 tele 降 4 倍（0.029%/0.009%）最接近，**是真正的 x4 配对**（RealVSR 只有约 2 倍）。

15 段 × 5 帧：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | tLP ↓ |
|---|---:|---:|---:|---:|---:|
| bicubic ×4 | **23.649** | **0.7400** | 0.4789 | 0.3128 | 0.0250 |
| 感知微调 EDVR | 23.548 | 0.7368 | **0.4598** | **0.2889** | 0.0258 |
| 原始 EDVR | 23.282 | 0.7294 | 0.4832 | 0.3004 | 0.0323 |
| DOVE 5B（12 段） | 21.758 | 0.6905 | **0.3670** | **0.2193** | **0.0158** |
| OSEDiff | 21.159 | 0.6863 | 0.4114 | 0.2551 | 0.0290 |

**结论与 RealVSR 相反：**

1. **回归模型连 bicubic 都没超过**：原始 EDVR 的 PSNR 比 bicubic 低 0.37 dB、LPIPS 高 0.9%。而在 RealVSR 上 EDVR 高出 bicubic 0.89 dB。
2. **生成模型在感知轴上明显领先**：DOVE 的 LPIPS 比 bicubic 低 23.4%、DISTS 低 29.9%；OSEDiff 分别低 14.1% / 18.4%。PSNR 上两者仍落后 1.89 / 2.49 dB。
3. **关键差别是退化强度**：MVSR4x 是真 x4（wide 对 tele 降 4 倍），RealVSR 只有约 2 倍。退化更重时回归模型天花板更低，生成模型的相对位置更好。
4. **DOVE 的时序指标最好**（tLP 0.0158，比 bicubic 低 36.7%），是唯一在时序上明显优于回归模型的方法。

**边界**：DOVE 只有 12/15 段（缺失的 115/139/158 在运行时遇上其他用户占用 29.8 GB 显存，DOVE 需约 47 GB，直接 OOM），因此该行与其余方法不同批，严格不可直接比较；但它与 bicubic 在同一 12 段上的差值 −1.89 dB，方向上不会因缺 3 段而反转。

原始结果：`prior_qualification/results/mvsr4x/*/metrics.json`、`results/summary/mvsr4x.md`。

## 10. REDS-Real：补齐"有 GT 且退化真实"的基准（2026-11-24）

前三节的三套数据各有缺陷（干净 bicubic、无 GT、RealVSR 被二次降采样），
无法干净地回答"真实退化下生成先验是否有用"。本节用一个新基准直接对照。

做法：取**干净的** REDS val GT，施加 RealBasicVSR 二阶真实退化（模糊 → 随机缩放 →
高斯/泊松噪声 → JPEG → 视频压缩，整条链两遍），再降到 GT/4，保留 GT。
退化强度已标定：REDS-Real LR 的拉普拉斯能量 15.11，低于 REDS 官方 bicubic LR 的 25.64，
即确实更平滑（退化更重）。数据清单 sha256 `8ab95add…` 已冻结。

30 段 × 10 帧（300 帧）：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ |
|---|---:|---:|---:|---:|
| bicubic ×4 | **20.547** | **0.4934** | 0.7397 | 0.3702 |
| 原始 EDVR | 20.500 | 0.4848 | 0.6605 | 0.3747 |
| 感知微调 EDVR | 20.450 | 0.4790 | 0.6373 | 0.3642 |
| OSEDiff | 19.477 | 0.4330 | **0.5204** | **0.2703** |

相对 bicubic：EDVR 的 PSNR −0.047、LPIPS −10.7%；感知微调 −0.097 / −13.8%；
OSEDiff **−1.070 / −29.6%**。

**连续 9 帧子集**（30 段 × 9 帧 = 270 帧，可算 tLP；帧 20–28）：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | tLP ↓ | E_warp ↓ |
|---|---:|---:|---:|---:|---:|---:|
| bicubic ×4 | **21.004** | **0.5019** | 0.7271 | 0.3463 | 0.0744 | — |
| 原始 EDVR | 20.916 | 0.4902 | 0.6351 | 0.3577 | 0.0630 | — |
| 感知微调 EDVR | 20.862 | 0.4840 | 0.6163 | 0.3525 | 0.0631 | — |
| **RealBasicVSR** | 20.822 | 0.4989 | 0.4881 | 0.2282 | 0.0443 | **0.589** |
| DOVE 5B | 20.856 | 0.5023 | 0.5149 | 0.2392 | 0.0526 | — |
| OSEDiff | 19.812 | 0.4454 | **0.4771** | **0.2235** | 0.0479 | — |
| RealBasicVSR→OSEDiff | 19.583 | 0.4747 | 0.5057 | 0.2227 | **0.0392** | 2.595 |

相对 bicubic：EDVR −0.088 / −12.7% / +3.3% / −15.3%；感知微调 −0.142 / −15.2% / +1.8% / −15.2%；
RealBasicVSR **−0.182 / −32.9% / −34.1% / −40.5%**；DOVE −0.148 / −29.2% / −30.9% / −29.3%；
OSEDiff **−1.192 / −34.4% / −35.5% / −35.7%**；串联 −1.421 / −30.4% / −35.7% / −47.3%。

**结论：**

1. **干净域训练的 EDVR 在真实退化下连 bicubic 都打不过**（20.916 vs 21.004；
   10 帧子集与全量 3000 帧方向一致），而它在干净 REDS 上高出 bicubic 4.5 dB。
   之前的强基线是"退化域匹配"的产物，不是通用能力。
2. **生成先验在真实退化上三轴同时领先**：OSEDiff 的 LPIPS 低 34.4%、DISTS 低 35.5%、
   **tLP 低 35.7%**，PSNR 只低 1.19 dB。这与第 3 节 REDS clean bicubic 上
   "PSNR −2.8 dB、LPIPS +36.9%"方向完全相反。
3. **时序轴也翻了**：干净域上 EDVR 的 tLP 优于 OSEDiff，真实退化上 OSEDiff 的 tLP
   比 EDVR 低 24.0%（0.0479 vs 0.0630）。"生成模型必然闪烁"的印象不成立。
4. **RealBasicVSR 是"保真几乎不付代价"的那一个**：PSNR 只比 bicubic 低 0.18 dB，
   而 LPIPS 低 32.9%、DISTS 低 34.1%、tLP 低 40.5%，`E_warp` 只比 GT（5.757）高 0.589。
   它是唯一在保真轴上几乎不损失的候选。
5. **零样本串联 RealBasicVSR→OSEDiff 未按路线图预期生效**：PSNR 再掉 1.24 dB、
   `E_warp` 从 0.589 恶化到 2.595（4.4 倍），只有 tLP 略好（0.0392 vs 0.0443）。
   让扩散模型在清洗结果上再重绘一次会重新引入时序不稳定——这正是路线图
   Stage A/B 要解决的问题，但**零样本串联本身不是解**，M2 必须真的训练 LoRA 与门控。
6. **第 8 节 RealVSR 上回归模型占优的原因得到解释**：RealVSR 的 LQ 被再降 4 倍后，
   输入已接近干净 bicubic 域；一旦退化是真正的压缩+噪声+模糊，生成先验的价值
   在保真之外的每一条轴上立刻显现。

原始结果：`prior_qualification/results/reds_real/*_s10/metrics.json`、`results/summary/reds_real.md`。

## 11. 工程教训（本轮新增）

1. **单连接限速是这批大文件下不动的真正原因**：hf-mirror 单连接约 64 KB/s，改成 HTTP Range 多线程分段（16 路）后同一文件达 37 MB/s，2.5 GB 从"11 小时"变成 2 分钟。脚本 `scripts/pdl.py`。
2. **`content-length` 会骗人**：CLIP2 先前记录的 4.1 GB 其实是旧的不完整文件大小，真实为 10.16 GB（`x-linked-size`）。下载前先读 `x-linked-size`。
3. **按大小判断完整性不够**：SUPIR 的 SDXL 与 CLIP2 都曾"大小接近"但 `safe_open` 报 `incomplete metadata`。必须逐文件用 `safe_open` 打开验证。
4. **共享主机 OOM 的隐蔽来源是 fp32 骨架**：`instantiate_from_config` 默认 fp32，SDXL UNet 单独就约 10 GB；31 GB 内存机器上直接被 OOM reaper 杀掉。改用 `torch.set_default_dtype(torch.float16)` 构建骨架可绕开（`scripts/supir_lean.py`）。
5. **`SDXL_CLIP1_PATH` 必须指向目录而非单文件**，`CLIPTokenizer.from_pretrained` 不接受单文件；需要把 open_clip 的 `text_model.*` 张量另存为 `pytorch_model.bin` 并配套 tokenizer 文件。
6. **项目服务器 `/data/disk1/home` 是 `/data/disk2/home` 的软链接**，`df` 显示的 442 GB 可用属于不可写的 `/data/disk1/shared`，实际家目录在已满的 disk2 上。
