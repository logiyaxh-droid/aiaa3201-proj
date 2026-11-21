# SUPIR 资格测试结果

日期：2026-11-22  
判定：**不通过，停止**

## 配置

借用服务器 s11（RTX 4090 24GB），独立 venv（Python 3.11，torch 2.6.0+cu124）。
官方 `Fanghua-Yu/SUPIR`，SUPIR-v0F fp16，20 EDM steps，固定 seed 1234，固定提示词，
未加载 LLaVA（改为固定文本，属路线文档允许的工程缩减并记录）。
输入 LR 320×180 先 LANCZOS ×4 到 1280×720，与 OSEDiff 同一预处理。

## 实测资源

| 项 | 值 |
|---|---|
| 显存峰值 | 15.28 GiB |
| 单帧耗时 | 9.2 s |
| 30 帧总耗时 | 4.6 min |
| 磁盘 | 约 21.6 GB（SDXL 7.11 + CLIP2 10.16 + SUPIR 2.66 + CLIP1 1.71） |

在 24 GB 的 4090 上可以运行，此前「24 GB 不够」的担心被实测否定；真正的限制是磁盘，不是显存。

## 结果（000/002/009，frame 40–49，n=30）

| 指标 | 感知微调 EDVR | SUPIR | 相对差 |
|---|---:|---:|---:|
| PSNR | 31.9032 | 26.1450 | −5.758 dB |
| SSIM | 0.9017 | 0.7423 | — |
| 全帧 LPIPS | 0.1016 | 0.2286 | +125% |
| 纹理块 LPIPS | 0.0659 | 0.1955 | **+196.6%** |
| 高分歧块 LPIPS | 0.0602 | 0.1915 | **+218.1%** |
| 平坦块 LPIPS | 0.0485 | 0.0935 | +93% |

三个片段方向一致。纹理块与高分歧块的恶化都比 OSEDiff 更大。

## 结论

SUPIR 在本项目的干净 bicubic 退化上同样严重偏离 GT，且比 OSEDiff 更差。它与 StableSR、Real-ESRGAN、OSEDiff 属同一模式：面向真实退化的强生成先验，在合成 bicubic 上会重绘纹理，而不是恢复 GT 纹理。

按门槛第 1 条（纹理块 LPIPS 改善 ≥10%），实际恶化 196.6%，**不通过**。不进入 60 帧时序测试，不进入条件化设计。

## 复算

```bash
# 借用服务器
cd /mnt/sda1/shidanhe/utrd_qual
./venv/bin/python -u run_supir_frames.py --steps 20
# 原服务器评分
cd /data/disk1/home/ghe270/p3/prior_qualification
../quality_metrics/venv/bin/python scripts/score_candidate.py --name supir --dir outputs/supir
```
