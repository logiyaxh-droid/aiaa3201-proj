# P3 感知质量指标（DISTS / MUSIQ / DOVER）

本轮只回答一个问题：**在 PSNR / SSIM 结果不变的前提下，现有 12 个方法在结构、纹理、
自然观感和视频技术质量上分别差在哪里。** 先测量，不先选改进算法。

计划与判读规则见 `../docs/P3_perceptual_quality_roadmap.md`；本目录只放实现、结果与结论。

## 适用范围（先说清楚边界）

- 只覆盖 **REDS `val_sharp` 的 30 个 clip × 100 帧 × ×4 bicubic 退化**。
- 不构成跨域泛用性结论：没有真实退化数据集、没有自拍视频、没有其他生成模型。
- MUSIQ / DOVER 不需要 GT，它们的分数是"观感"参照，**不是"更接近真实"的证明**。
  GT 自身的分数也只是数据域参照，不是满分上限。

## 目录

```text
configs/protocol.json      冻结的协议：数据、方法、每个指标的输入与聚合口径
scripts/common.py          帧配对、颜色/range、权重哈希、三个指标的封装
scripts/predictions.py     预测帧生成（与 P0 完全同参，逐位可复现）
scripts/selfcheck.py       阶段 A 的六项自检（第 3.3 节）
scripts/verify_predictions.py  生成器逐位回归（新环境 PSNR/LPIPS vs 旧 JSON）
scripts/run_metrics.py     分 scope 跑指标，按 (clip, method) 断点续跑
scripts/summarize.py       从原始 JSON 生成所有表与排序对照
scripts/stageD*.py         区域归因、运动分组、时序异常、敏感性
selfcheck/                 自检产物
results/<scope>/<metric>/<clip>.json
summary/<scope>/            主表、配对差值、排序变化、一致性
cases/<scope>/             代表片段与局部诊断
logs/                       每次运行的日志
weights/                    权重文件（含 SHA256 记录）
vendor/                     DOVER 官方源码、DISTS 官方实现
```

## 环境

独立 venv，**不修改**共享的 `/tmp/srenv`：

```bash
cd p3/quality_metrics
./venv/bin/python scripts/selfcheck.py
```

- venv 用 `--without-pip` 创建，通过 `.pth` 只读复用 `/tmp/srenv` 的 torch 2.6.0
  与 basicsr（`/tmp/bsr_src`），新包 `pip install --target` 装进 venv 自身目录。
- 因此这个 venv **依赖 `/tmp` 存活**：`/tmp/srenv` 或 `/tmp/bsr_src` 被清理后需要重建。
  重建步骤见下节。
- `huggingface_hub` 固定 0.25.2：2.4.0 依赖 `httpx2`，在当前环境不可用。

重建：

```bash
python3 -m venv --without-pip venv
SP=venv/lib/python3.10/site-packages
echo /tmp/srenv/lib/python3.10/site-packages > $SP/_basesrenv.pth
echo /tmp/bsr_src > $SP/_basicsr_src.pth
/tmp/srenv/bin/pip install --no-deps --target $SP pyiqa==0.1.13 timm einops pandas \
    huggingface_hub==0.25.2 decord scikit-video
```

## 权重（SHA256 记录在 selfcheck/environment.json）

| 指标 | 文件 | 来源 |
|---|---|---|
| DISTS | `DISTS_weights-f5e65c96.pth` | pyiqa 权重库；与官方 `weights.pt` **逐字节相同** |
| MUSIQ 主 | `musiq_koniq_ckpt-e95806b9.pth` | koniq10k |
| MUSIQ 敏感性 | `musiq_paq2piq_ckpt-364c0c84.pth` | paq2piq，仅敏感性检查 |
| DOVER | `DOVER.pth` | 官方完整版，**不是 Mobile** |
| 骨干 | `vgg16-397923af.pth`、`convnext_tiny_1k_224_ema.pth` | torchvision / fbaipublicfiles |

网络限制（实测）：`github.com` 的 git/HTTPS 直连超时；`raw.githubusercontent.com`、
`ghfast.top`、`hf-mirror.com`、`download.pytorch.org`、`dl.fbaipublicfiles.com` 可用。

## 指标口径（与 protocol.json 一致）

- **DISTS**：预测与 GT 都按 RGB uint8 读入转 `[0,1]` float，**不缩放、不裁边**，逐帧。
  越低越好。`selfcheck` 中 pyiqa 实现与官方 `DISTS_pytorch` 在 12 对图上最大差
  **0.0**（容差 1e-5）。
- **MUSIQ**：只输入预测帧，使用库自带多尺度预处理（longer-side 224/384），逐帧。
  越高越好。GT 帧同样计算，单独作为数据域参照。
- **DOVER**：以完整 100 帧 clip 为输入，`dover.yml` 默认配置（技术分支
  clip_len 32 / interval 2 / num_clips 3 + 7×7 个 32×32 碎片；美学分支
  224×224 / num_clips 1）。技术分与美学分**分别保存**，融合分仅记录、不替代。
  每 clip 一个值。

### DOVER 的两个实现要点

1. **不走视频解码。** 官方 loader 用 `decord.VideoReader(mp4)`；我们换成读 PNG 目录的
   兼容 reader，采样、分块、归一化仍全部走官方代码。用交付视频（H.264）解码会改变
   纹理与时序统计，正是这三个指标要测的东西。
2. **必须固定随机种子。** 官方 `UnifiedFrameSampler` 用 `np.random.randint` 取时段偏移，
   `get_spatial_fragments` 用 `torch.randint` 取碎片裁剪位置，**eval 模式也不确定**。
   当前固定 seed=20261119 并写入每个结果 JSON；同种子重复得分逐位相同（自检第 4 项）。
3. **原始分数是负数**（例如 bicubic 技术分约 −0.13）。官方评估脚本会对整组预测做
   z-score 再映射到 GT MOS 尺度；我们没有 MOS 标签，因此只报告原始分支分数。
   z-score 是线性变换，**不改变方法间排序**，但绝对值和论文表格不可直接比较。

### 聚合

主表是 **30 个 clip 级均值的均值**，括号内为 clip 间标准差。不使用 3000 帧混合均值。
缺失单元格留空并计入 `coverage`，**不写 0**。

## 复算（帧不长期保存）

12 方法 × 30 clip 的预测 PNG 约 40 GB，磁盘放不下，因此沿用 P0 的流式做法：
生成一帧、测三指标、丢弃。复算命令：

```bash
./venv/bin/python scripts/run_metrics.py --scope reds30 --metrics dists musiq dover
./venv/bin/python scripts/summarize.py --scope reds30
```

按 `(clip, method)` 断点续跑：每个完成的组合原子写入
`results/<scope>/<metric>/<clip>.json`，被杀最多损失一个片段。

已存在的预测缓存会被优先使用：bicubic（仅 000/027）、tecogan（全 30）、
edvr（部分片段），其余现场生成。

**预测生成器已逐位回归验证**：`scripts/verify_predictions.py` 在 000/002/009 上
重算 PSNR 与 LPIPS 并与 P0 存储值比较，27 组全部通过，LPIPS 差 0.0、
PSNR 差约 1e-6（容差 1e-3 / 1e-6）。

## 已知坑

- `torch.backends.cudnn.benchmark` 必须为 `False`：开启时 `evalkit.ssim_gpu` 首次
  float64 7×7 conv 会花约 126 s 做 cuDNN 算法搜索。
- 共享 GPU 的耗时取**多次干净运行的最小值**，不用中位数。
- basicsr / EDSR / EDVR 需要 `torchvision.transforms.functional_tensor` 兼容垫片，
  脚本里已按 P0 的做法注入。
- 输出重定向到文件时 Python 默认块缓冲，需要 `PYTHONUNBUFFERED=1` 才能实时看日志。
