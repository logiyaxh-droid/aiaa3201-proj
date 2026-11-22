# Direction C 探索最终结论：残差扩散、外部生成先验与真实退化

日期：2026-11-22  
状态：已完成  
范围：REDS bicubic（有 GT）与真实退化（手机实拍 + 官方 VideoLQ，均无 GT）两条线

## 一句话结论

**方向 C 在 REDS bicubic 上失败，但在真实退化上不失败，且已在官方 VideoLQ 上复现。** 之前的负结论只对合成 bicubic 域成立，不能外推为“生成先验普遍无用”。

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

剩余缺口是配对 GT：RealVSR / MVSR4x 目前仍下载不到（仅 Google Drive / 百度网盘），因此无法证明“更接近真值”，只能停留在无参考观感层面。

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
