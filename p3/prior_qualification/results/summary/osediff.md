# 生成先验资格测试结果

## OSEDiff

日期：2026-11-21  
判定：**不通过，停止**

### 配置

- 官方仓库 `cswry/OSEDiff` main 分支
- SD 2.1 Base fp16 UNet（865.9M 参数）
- 官方 `osediff.pkl`（rank 4 LoRA）
- 空提示（未加载 RAM：该权重 5.63 GB，对资格测试性价比低）
- 固定 seed 42，wavelet 颜色修复
- 输入：LR 320×180 先 LANCZOS ×4 到 1280×720，与官方 test_osediff.py 一致
- 输出：1280×720 RGB PNG

### 实测资源

| 项 | 值 |
|---|---|
| 显存峰值 | 4.55 GiB |
| 单帧耗时（热） | 2.4～3.4 s |
| 30 帧总耗时 | 1.5 min |
| 磁盘（SD21 + OSEDiff） | 5.5 GB |

### 结果（000/002/009，frame 40–49，n=30）

| 指标 | 感知微调 EDVR | OSEDiff | 相对差 |
|---|---:|---:|---:|
| PSNR | 31.9032 | 22.9259 | −8.977 dB |
| SSIM | 0.9017 | 0.6631 | — |
| 全帧 LPIPS | 0.1016 | 0.2318 | +128% |
| 纹理块 LPIPS | 0.0659 | 0.1660 | **+151.8%** |
| 高分歧块 LPIPS | 0.0602 | 0.1656 | **+175.1%** |
| 平坦块 LPIPS | 0.0485 | 0.1193 | +146% |
| tLP | 0.0077（全量参照） | 0.0232 | 明显更差 |

三个片段方向一致，无一片段例外。

### 结论

OSEDiff 在本项目的干净 bicubic 退化上完全不具竞争力，且差距不是 10% 量级，而是 1.5～2.8 倍。这符合"真实退化生成先验在合成 bicubic 上域失配"的既有判断（StableSR、Real-ESRGAN 同一模式）。

按路线文档第 2.5 节门槛，第 1 条（纹理块 LPIPS 至少改善 10%）就不满足，且方向相反。**不进入 60 帧时序测试，不进入条件化设计。**

### 未做的补救

未尝试：加载 RAM 提示词、调 cfg、换 align_method、多次采样挑最好。原因是差距过大，这些调整不可能补回 1.5 倍 LPIPS 差距；继续尝试只会消耗时间，且违反"一次只改一个因素并回答明确问题"的原则。

### 复算

```bash
cd /data/disk1/home/ghe270/p3/prior_qualification
envs/osediff/bin/python scripts/run_osediff.py --name osediff --no-ram
../quality_metrics/venv/bin/python scripts/score_candidate.py --name osediff --dir outputs/osediff
```
