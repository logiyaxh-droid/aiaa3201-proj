# Direction C 探索最终结论：残差扩散、外部生成先验与真实退化

日期：2026-11-22  
状态：已完成  
范围：REDS bicubic（有 GT）与手机实拍真实退化（无 GT）两条线

## 一句话结论

**方向 C 在 REDS bicubic 上失败，但在真实退化上不失败。** 之前的负结论只对合成 bicubic 域成立，不能外推为“生成先验普遍无用”。

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

## 4. 真实退化评测（手机实拍，无 GT）

**这是本轮最重要的新结果。** 官方 VideoLQ 的所有公开渠道在该服务器均不可达（Google Drive/Dropbox 不通、OneDrive 失效、HF 同名镜像实为 RealISVSR 配对数据、ModelScope 只有元数据），因此改用 `p2/wild_source` 手机实拍视频：1080×1920、30 帧中心裁切 720×1280、统一 ×4 下采样后恢复。

| 方法 | NIQE ↓ | MUSIQ ↑ | 显存 | 30 帧耗时 |
|---|---:|---:|---:|---:|
| **SUPIR** | **2.364** | 61.99 | 12.43 GiB | 3.6 min |
| **OSEDiff** | 2.948 | 73.61 | 4.55 GiB | 35 s |
| 原始 EDVR | 4.167 | 63.44 | ~8 GiB | 25 s |
| 感知微调 EDVR | 4.195 | 63.24 | ~8 GiB | 23 s |
| **DOVE** | 4.468 | **74.93** | 分块 | 27 s |
| bicubic ×4 | 7.035 | 40.26 | — | 秒级 |

### 四个关键观察

1. **生成模型在真实退化上不再明显更差**：三者 NIQE 全部优于 EDVR，两个 MUSIQ 也优于 EDVR。与 bicubic 上的结论方向相反。
2. **NIQE 与 MUSIQ 排序不一致**：SUPIR 的 NIQE 最好但 MUSIQ 最低；DOVE 的 MUSIQ 最高但 NIQE 最差。不能合并成单一排名。
3. **感知微调 EDVR 在真实退化上不占优**：与原始 EDVR 基本持平（NIQE 4.195 vs 4.167）。它在 bicubic 上的 LPIPS 优势没有迁移。
4. **DOVE 的时序优势未能确认**：因 30 帧被拆成 4 段 9 帧推理，存在段间不连续，不能宣称时序最好。

### 边界

- 无 GT，**不能**计算 PSNR/SSIM/LPIPS，也**不能**与 REDS 表格横向比较；
- 数据不是 VideoLQ，不能写成交叉验证 VideoLQ 的结论；
- 不能证明这些模型更接近真值。

## 5. 工程教训

1. **文件大小正确不代表内容正确。** DOVE 的 `text_encoder/model-00002-of-00004` 与 CogVideoX VAE 都出现过“大小精确匹配但张量损坏”，必须用 `safetensors.safe_open` 逐文件校验。
2. **`huggingface_hub` 新版默认在 `~/.cache` 留符号链接**，`local_dir` 里只有链接。借用机器上必须实体化到指定目录，否则占用落在别人 home 下。
3. SUPIR 在 720p 需约 15 GiB，显存被他人占用时可用 `min_size 768` 降内存。
4. DOVE 在 24 GiB 卡上必须 `tile_size_hw 256`，否则 VAE 编码 OOM。
5. 借用服务器：无 `python3-venv`（用 anaconda Python 建 venv）、PyPI 官方 CDN 不通（用阿里云镜像）、`huggingface.co` 可直连但 hf-mirror 小文件 resolve 会 503。

## 6. 最终判断

**方向 C 的价值取决于域。** 在合成 bicubic 上，回归模型（EDVR）已经接近最优，生成先验只会引入偏离；在真实退化上，生成先验的相对价值显著上升。

因此之前的“停止外部生成先验”结论需要修正为：**停止在 REDS bicubic 上使用外部生成先验**，而不是停止这条技术路线本身。

若要继续，必须先取得带 GT 的真实退化配对数据（RealVSR / MVSR4x 目前下载不到），否则只能停留在无参考观感层面，无法证明“更接近真值”。

## 7. 原始结果索引

| 内容 | 路径 |
|---|---|
| 感知微调结论 | `quality_metrics/fine_tune/RESULTS.md` |
| U-TRD 路线与结果 | `docs/P3_direction_c_residual_diffusion_roadmap.md`、`utrd/RESULTS_PROTO1.md` |
| 外部先验路线 | `docs/P3_generative_prior_qualification_roadmap.md` |
| 外部先验汇总 | `prior_qualification/results/summary/qualification.md` |
| 真实退化评测 | `prior_qualification/results/summary/real_world.md` |
| 真实退化指标 | `prior_qualification/results/real/*.json` |
| 各候选分项 | `prior_qualification/results/summary/{osediff,supir,dove}.md` |
