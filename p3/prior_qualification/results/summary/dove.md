# DOVE 资格测试结果

日期：2026-11-22  
判定：**不通过，停止**

## 配置

借用服务器 s11（RTX 4090 24GB），独立 venv（Python 3.11，torch 2.6.0+cu124，diffusers 0.34.0）。
官方 `zhengchen1999/DOVE` 5B Stage-2 权重（transformer 5 分片 + text_encoder 4 分片 + CogVideoX1.5-5B VAE）。
输入 9 帧 320×180（满足 DOVE 的 `(F-1) % 8 == 0`），内部 ×4 到 768×1280，输出裁回 720×1280。
空间 tile 256、时间 chunk 9、`--is_vae_st`、固定 seed 42、空提示。

## 实测资源

| 项 | 值 |
|---|---|
| 显存 | 全图 720p 编码会 OOM；tile 256 可稳定运行 |
| 单段 9 帧耗时 | 约 8 秒 |
| 权重磁盘 | 41.33 GB（不含 VAE） |
| 结论 | 24 GB 显存可跑，但必须分块 |

## 一个必须先记录的工程故障

第一次下载的 CogVideoX VAE（862,388,596 字节，大小与官方一致）**内容是损坏的**：436 个张量中有 259 个全零，`encoder.conv_in.conv.weight` 绝对值和为 0.0。这导致 VAE 编码输出恒为全零，DOVE 输出塌陷成 std≈8.9 的近似纯色图。

重新下载后 `conv_in` 绝对值和 590.04、零张量 0 个，输出 std 恢复到 74.12。**所有基于第一次 VAE 的 DOVE 结果都无效，已作废。** 这个教训是：文件大小正确不代表内容正确，大权重必须校验张量非零。

## 结果（000/002/009，frame 40–48，n=27，与参照严格同帧）

| 指标 | 原始 EDVR | 感知微调 EDVR | DOVE | 相对微调 |
|---|---:|---:|---:|---:|
| PSNR | 32.3524 | 31.8832 | 25.3416 | −6.542 dB |
| SSIM | 0.9109 | 0.9013 | 0.7096 | — |
| 全帧 LPIPS | 0.1547 | 0.1021 | 0.2829 | +177% |
| 纹理块 LPIPS | 0.0993 | 0.0660 | 0.1963 | **+197.2%** |
| 高分歧块 LPIPS | 0.0906 | 0.0603 | 0.1933 | **+220.3%** |
| 平坦块 LPIPS | 0.0855 | 0.0492 | 0.1679 | +241% |
| tLP | 0.0089 | 0.0054 | 0.0083 | +54.7% |

三个片段方向一致。

## 结论

DOVE 是本轮唯一真正使用视频时序的候选，tLP（0.0083）确实明显好于 OSEDiff（0.0232）和 SUPIR（0.0336），接近原始 EDVR（0.0089）。**它的时序优势是真实的。**

但空间质量失败：PSNR 低 6.54 dB，纹理块 LPIPS 差近 3 倍。按门槛第 1 条（纹理块 LPIPS 改善 ≥10%），实际恶化 197.2%，**不通过**。

原因与 OSEDiff、SUPIR 相同：DOVE 面向真实退化训练，其生成先验在干净 bicubic 上会重绘纹理而非恢复 GT 纹理。时序机制解决不了这个问题。

## 复算

```bash
# 借用服务器
cd /mnt/sda1/shidanhe/utrd_qual
./venv_dove/bin/python mkvid3.py
cd dove/DOVE
HF_ENDPOINT=https://huggingface.co PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /mnt/sda1/shidanhe/utrd_qual/venv_dove/bin/python -u inference_script.py \
  --input_dir /mnt/sda1/shidanhe/utrd_qual/dove/input_vid3 \
  --model_path /mnt/sda1/shidanhe/utrd_qual/dove/weights \
  --output_path /mnt/sda1/shidanhe/utrd_qual/dove/outputs/dove9 \
  --png_save --is_vae_st --tile_size_hw 256 256 --overlap_hw 32 32 \
  --chunk_len 9 --overlap_t 0
# 原服务器评分
cd /data/disk1/home/ghe270/p3/prior_qualification
../quality_metrics/venv/bin/python scripts/score_subset.py --name dove9 \
  --dir outputs/dove9_720 --start 40 --end 49
```
