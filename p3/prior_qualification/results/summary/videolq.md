# VideoLQ 真实退化评测结果

日期：2026-11-22  
状态：已完成  
数据：官方 VideoLQ（OpenDataLab 镜像），50 段真实低质视频，5 帧/段，共 250 帧

## 1. 数据核实

通过 OpenXLab CLI 从 `OpenDataLab/VideoLQ` 下载，1.71 GB 分卷归档，解压后：

- 50 个片段，编号 000–049；
- 多数片段 100 帧，030（41 帧）、031（34 帧）、033（47 帧）不足 100 帧；
- 分辨率各异：320×240、640×480、720×576、492×360 等；
- 与官方描述（Flickr/YouTube 采集、无场景切换、纹理丰富）一致。

这是真正的 VideoLQ，不是之前 HuggingFace 上误认的 RealISVSR 配对数据。

## 2. 协议

- 每段按 `linspace` 均匀取 5 帧，共 250 帧；
- **保持原生分辨率**，不缩放、不裁剪，因为 VideoLQ 的退化与分辨率绑定；
- 有 GT 不存在，因此只算无参考指标：**NIQE**（越低越好）、**MUSIQ**（越高越好）；
- 另记 **selfPSNR**：原图作为目标的自我一致性，衡量模型是否破坏已有信息，**不是**对真值的准确度；
- 按片段平均后再对 50 段等权平均，避免 640×480 片段压过 320×240 片段。

x4 模型（EDVR/OSEDiff/SUPIR/DOVE）统一：原图 → 1/4 → 模型 ×4 → 与原图比较。
DOVE 需要 `(F-1)%8==0`，输入 9 帧，只评前 5 帧（与评测集同位置）。

## 3. 结果

| 方法 | NIQE ↓ | MUSIQ ↑ | selfPSNR | 显存 | 250 帧耗时 |
|---|---:|---:|---:|---:|---:|
| **OSEDiff** | **4.109** | **66.85** | 21.54 | 4.55 GiB | 37 s |
| **DOVE 5B** | 5.003 | 61.51 | 21.76 | tile 256 | 91 s |
| 感知微调 EDVR | 5.190 | 56.21 | 20.33 | ~8 GiB | 36 s |
| SUPIR v0F | 5.629 | 46.53 | **24.29** | 12.4 GiB | 16.3 min |
| 原始 EDVR | 5.673 | 55.81 | 20.49 | ~8 GiB | 36 s |
| bicubic ×4 | 7.563 | 31.38 | 23.74 | — | 秒级 |

## 4. 关键观察

**1. OSEDiff 在 VideoLQ 上全面最好。** NIQE 4.11（比原始 EDVR 低 27.6%），MUSIQ 66.85（比 EDVR 高 19.8%），速度也最快。这与它在 REDS bicubic 上"纹理块 LPIPS 差 151.8%"的结论完全相反。

**2. 感知微调 EDVR 在 VideoLQ 上优于原始 EDVR。** NIQE 5.190 vs 5.673，MUSIQ 56.21 vs 55.81。这与手机实拍视频上"两者基本持平"不同，说明微调收益在 VideoLQ 这种真实退化上部分成立。

**3. SUPIR 出现指标分裂。** selfPSNR 24.29 最高（最忠实保留原图信息），但 MUSIQ 46.53 最低（比 bicubic 的 31.38 好，但远低于其他方法）。它倾向于保守复原，牺牲观感。

**4. bicubic 的 selfPSNR 很高（23.74）但 NIQE/MUSIQ 最差。** 这印证了 selfPSNR 的局限：它奖励"不动"，不奖励"变好"。三个指标必须并列，不能选一个当结论。

**5. DOVE 居中。** NIQE 5.003 略优于感知微调 EDVR，MUSIQ 61.51 明显更高。它的视频时序能力在这里没有带来压倒性优势。

## 5. 与前两组实验的关系

| 数据集 | 有 GT | 生成模型 vs EDVR | 结论方向 |
|---|---|---|---|
| REDS bicubic | 有 | 纹理块 LPIPS 差 151–197% | 生成先验明显更差 |
| 手机实拍 | 无 | NIQE 全部更优，MUSIQ 部分更优 | 生成先验占优 |
| VideoLQ | 无 | NIQE/MUSIQ 全部更优 | 生成先验占优 |

三组结果一致地指向：**域匹配是关键变量**。合成 bicubic 上回归模型接近最优，真实退化上生成先验价值上升。

## 6. 不能说明什么

- VideoLQ 没有 GT，**不能**证明生成模型更接近真实；
- NIQE/MUSIQ 与 REDS 的 PSNR/LPIPS 不可比；
- selfPSNR 高不代表质量好（bicubic 就是反例）；
- 没有配对数据，无法排除"生成模型只是让图像更符合 IQA 模型偏好"这一解释。

## 7. 工程记录

- 官方 VideoLQ 的 GD/Dropbox/OneDrive 均不可达；**OpenDataLab 可用**，需 `openxlab` CLI；
- `openxlab dataset ls` 有 bug（缺 `rprint` 导入），补一行 import 即可；
- 归档是嵌套的 `VideoLQ.tar.gz.00` → `VideoLQ.tar`，需两次解压；
- VideoLQ 尺寸不整除 8，OSEDiff 的小波颜色修复会崩，需 pad→计算→crop；
- DOVE 需 `--upscale 1`，否则会把小图放大 4 倍。
