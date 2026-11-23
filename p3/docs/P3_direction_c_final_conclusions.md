# Direction C 探索最终结论：残差扩散、外部生成先验与真实退化

日期：2026-11-22  
状态：已完成（RealVSR 部分仅含 OSEDiff，SUPIR/DOVE 受下载带宽限制未完成）  
范围：REDS bicubic（有 GT）、真实退化无 GT（手机实拍 + 官方 VideoLQ）、真实退化有 GT（RealVSR）三条线

## 一句话结论

**方向 C 在 REDS bicubic 上失败；在无参考的真实退化评测中领先；但一旦在真实退化上引入配对真值（RealVSR），领先消失。** 因此“生成先验在真实退化上更好”这个说法只在无参考指标下成立，不能等同于“更接近真值”。

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
| OSEDiff | 20.049 | 0.5875 | 0.2710 | 0.1397 | 0.0143 |

相对感知微调 EDVR 的变化：

| 方法 | ΔPSNR | ΔLPIPS |
|---|---:|---:|
| 原始 EDVR | +0.165 | +21.5% |
| bicubic | −0.724 | +152.5% |
| OSEDiff | **−2.822** | +36.9% |

### 8.3 结论

1. **在有 GT 的真实退化上，OSEDiff 不占优**：PSNR 比 bicubic 还低 2.1 dB，且 50/50 段全部更差；LPIPS/DISTS 也不如感知微调 EDVR。
2. **这直接修正了第 4 节的无参考结论**。VideoLQ 上 OSEDiff 的 NIQE/MUSIQ 全面领先，但那是“看起来更清晰”，不是“更像真值”。一旦能对照真值，生成模型偏离真值的幅度反而更大。
3. **感知微调的权衡在真实退化上同样成立**：用 0.165 dB 的 PSNR 换来 LPIPS −17.7%、DISTS −28.2%，方向与 REDS 一致。
4. **不能把 RealVSR 的绝对值和 REDS 表格比较**：两者退化核、分辨率、场景都不同，只在本表内部做方法间比较。

### 8.4 边界（必须一并说明）

- SUPIR 与 DOVE **未完成**：SUPIR 缺 CLIP2（10.16 GB），DOVE 缺 CogVideoX 分片（约 20 GB）。两者在当前网络下（HF 大文件单连接 8 B/s–1.6 KB/s）无法取得，因此本表只有 OSEDiff 一个生成模型。
- 因此“生成模型在真实退化上更远离真值”目前**只有一个生成模型样本**，不能外推为所有生成先验的普遍结论。
- 采用 LQ 再降 4 倍的适配协议，与官方 `EDVR_NoUp` 的 1 倍协议不同，绝对值不可与 RealVSR 论文表格直接对照。

## 9. 工程教训（本轮新增）

1. **单连接限速是这批大文件下不动的真正原因**：hf-mirror 单连接约 64 KB/s，改成 HTTP Range 多线程分段（16 路）后同一文件达 37 MB/s，2.5 GB 从"11 小时"变成 2 分钟。脚本 `scripts/pdl.py`。
2. **`content-length` 会骗人**：CLIP2 先前记录的 4.1 GB 其实是旧的不完整文件大小，真实为 10.16 GB（`x-linked-size`）。下载前先读 `x-linked-size`。
3. **按大小判断完整性不够**：SUPIR 的 SDXL 与 CLIP2 都曾"大小接近"但 `safe_open` 报 `incomplete metadata`。必须逐文件用 `safe_open` 打开验证。
4. **共享主机 OOM 的隐蔽来源是 fp32 骨架**：`instantiate_from_config` 默认 fp32，SDXL UNet 单独就约 10 GB；31 GB 内存机器上直接被 OOM reaper 杀掉。改用 `torch.set_default_dtype(torch.float16)` 构建骨架可绕开（`scripts/supir_lean.py`）。
5. **`SDXL_CLIP1_PATH` 必须指向目录而非单文件**，`CLIPTokenizer.from_pretrained` 不接受单文件；需要把 open_clip 的 `text_model.*` 张量另存为 `pytorch_model.bin` 并配套 tokenizer 文件。
6. **项目服务器 `/data/disk1/home` 是 `/data/disk2/home` 的软链接**，`df` 显示的 442 GB 可用属于不可写的 `/data/disk1/shared`，实际家目录在已满的 disk2 上。
