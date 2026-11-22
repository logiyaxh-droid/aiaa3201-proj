# 真实退化数据评测：四个方法在手机实拍视频上的表现

日期：2026-11-22  
状态：已完成，结果与 REDS bicubic 上的结论方向相反  
数据：手机实拍 1080×1920 竖屏视频，30 帧、720×1280 中心裁切、无 GT

## 1. 为什么用这份数据

VideoLQ 的官方渠道在本服务器全部不可达：Google Drive、Dropbox 网络不可达，OneDrive 分享链接返回 Error，HuggingFace 上的同名镜像实为 RealISVSR 的同尺寸降质配对数据，ModelScope 镜像只有元数据没有图像。

因此改用 `p2/wild_source` 中的手机实拍视频作为真实退化素材。这与 VideoLQ 的定位一致（真实低质、无 GT、无场景切换），但**不是 VideoLQ 本身**，报告里不能写成交叉验证 VideoLQ 的结论。

## 2. 评测协议

- 源视频 1080×1920、738 帧、30 fps，取第 120–419 帧内 30 个均匀间隔帧；
- 中心裁切 720×1280（竖屏），保存为无损 PNG；
- 所有方法统一输入：该帧 bicubic 下采样 ×4 到 180×320，再恢复到 720×1280；
- 指标：**NIQE**（越低越好）、**MUSIQ**（越高越好），均为无参考指标；
- 没有任何 GT，所以**不能**计算 PSNR / SSIM / LPIPS / DISTS，也不能与 REDS 表格横向比较。

方法配置：

| 方法 | 配置 | 显存 | 30 帧耗时 |
|---|---|---|---|
| bicubic ×4 | OpenCV INTER_CUBIC | — | 秒级 |
| EDVR | 官方 REDS 权重，五帧窗口，真实邻帧 | ~8 GiB | 24.7 s |
| 感知微调 EDVR | DISTS 0.03 + GAN 0.001，900 步 | ~8 GiB | 23.0 s |
| OSEDiff | SD2.1 + 官方 osediff.pkl，空提示，seed 42 | 4.55 GiB | 35.4 s |
| SUPIR | v0F fp16，20 EDM steps，min_size 768 | 12.43 GiB | 3.6 min |
| DOVE | 5B Stage-2，tile 256，chunk 9，`--upscale 1` | 分块后可跑 | 27 s（4 段） |

## 3. 结果

| 方法 | NIQE ↓ | MUSIQ ↑ |
|---|---:|---:|
| **SUPIR** | **2.364** | 61.99 |
| **OSEDiff** | 2.948 | 73.61 |
| 原始 EDVR | 4.167 | 63.44 |
| 感知微调 EDVR | 4.195 | 63.24 |
| **DOVE** | 4.468 | **74.93** |
| bicubic ×4 | 7.035 | 40.26 |

### 关键观察

**1. 生成模型在真实退化上不再是“明显更差”。**  
在 REDS bicubic 上，OSEDiff/SUPIR/DOVE 的纹理块 LPIPS 比感知微调 EDVR 差 1.5～2 倍。在真实手机视频上，三个生成模型的 NIQE 全部优于 EDVR，MUSIQ 中两个也优于 EDVR。**这与之前的结论方向相反，说明域匹配是关键变量。**

**2. NIQE 与 MUSIQ 排序不一致，不能合并成单一排名。**  
SUPIR 的 NIQE 最低（2.364）但 MUSIQ 只有 61.99；DOVE 的 MUSIQ 最高（74.93）但 NIQE 4.468，比原始 EDVR 还差。两个指标衡量不同性质：NIQE 偏自然统计特性，MUSIQ 偏学习到的感知质量。必须并列报告。

**3. 感知微调 EDVR 在真实退化上不占优。**  
NIQE 4.195 vs 原始 EDVR 4.167，MUSIQ 63.24 vs 63.44，两者几乎持平且微调版略差。它在 REDS bicubic 上的 LPIPS −31.2% 优势没有迁移到真实退化。这是本次最有价值的发现之一。

**4. DOVE 的时序机制在这里没有体现出优势。**  
它的 MUSIQ 最高，但 NIQE 最差，且因为分段推理（4 段 9 帧、含重叠）存在段间不连续风险。这一点必须如实说明，不能宣称它时序最好。

## 4. 这些结果能说明什么、不能说明什么

**能说明：**

- 在匹配的真实退化数据上，外部生成先验的相对价值与 REDS bicubic 上完全不同；
- 之前“生成先验普遍不可用”的结论只对 bicubic 域成立，不能外推到真实退化；
- 现有感知微调 EDVR 的收益是域特异的。

**不能说明：**

- 不能说明这些模型在 VideoLQ 上同样占优（数据不是 VideoLQ）；
- 不能说明它们“更接近真值”，因为没有任何 GT；
- 不能用 NIQE/MUSIQ 的数值与 REDS 的 PSNR/LPIPS 比较；
- 不能断言 DOVE 时序更好，分段推理本身引入了不连续。

## 5. 工程记录

本次重建环境的实际问题，供后续复用：

- 借用服务器无 `python3-venv`，用 anaconda 的 Python 建 venv；
- PyPI 官方源的 `files.pythonhosted.org` 不可达，改用阿里云镜像；
- `huggingface.co` 在该服务器**可直连**（与原服务器相反），但 hf-mirror 的小文件 resolve 会 503；
- 必须把大权重实体化到 `/mnt/sda1/shidanhe/`，`huggingface_hub` 新版默认在 `~/.cache` 留符号链接；
- **两个权重文件大小正确但内容损坏**：DOVE 的 `text_encoder/model-00002-of-00004` 与之前的 CogVideoX VAE。用 `safetensors.safe_open` 逐个校验是必要的，不能只看大小；
- SUPIR 在 720p 需要 15 GiB 左右，本机被别人占用 7.9 GiB 时只能用 `min_size 768` 降内存；
- DOVE 在 24 GiB 卡上必须 `tile_size_hw 256`，否则 VAE 编码 OOM。

## 6. 结论

**方向 C 在真实退化上没有像 bicubic 那样失败，这是本轮最重要的发现。**

但这不足以直接推出“应该用生成式路由”，因为：

1. 没有 GT，无法证明生成模型更接近真实；
2. NIQE/MUSIQ 指向不同结论；
3. 三者在真实退化上的优势需要配对数据验证（RealVSR / MVSR4x 目前拿不到）。

下一步若要继续，需要先取得带 GT 的真实退化数据，或者把目标明确改成“无参考观感提升”，并接受这一限制。
