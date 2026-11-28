# MVSR4x 配对评测：第二个真实退化数据集

日期：2026-11-24  
数据：MVSR4x 测试集（CVPRW 2023），15 段 × 5 帧 = 75 帧  
脚本：`prior_qualification/scripts/{build_mvsr_subset.py, run_realvsr.py, run_osediff_realvsr.py, run_dove_realvsr.py, score_realvsr.py}`  
原始指标：`prior_qualification/results/mvsr4x/*/metrics.json`

## 数据形态（与 RealVSR 不同）

MVSR4x 用手机双摄同时拍摄：超广角（13mm 等效）给低清，长焦（90mm 等效）给高清。两路已经对齐到**同一像素网格**（都是 1024×1024），所以缩放倍数不能从文件尺寸看出来，必须实测。

高频能量（0.25 归一化频率以上占比，clip 003 / 065 第 40 帧）：

| 图像 | clip 003 | clip 065 |
|---|---:|---:|
| tele 原生 | 0.078% | 0.030% |
| wide 原生 | 0.033% | 0.004% |
| tele 降 2 倍再上采样 | 0.059% | 0.023% |
| tele 降 4 倍再上采样 | **0.029%** | **0.009%** |

wide 与 tele 降 4 倍最接近，**说明 MVSR4x 是真正的 x4 配对**（RealVSR 只有约 2 倍）。

因此本实验把 wide 帧降 4 倍（PIL BICUBIC，缩小时带抗锯齿）作为 LQ，各 x4 模型重建回 1024×1024 的 tele 网格。

## 结果

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | tLP ↓ | 片段数 |
|---|---:|---:|---:|---:|---:|---:|
| bicubic ×4 | **23.649** | **0.7400** | 0.4789 | 0.3128 | 0.0250 | 15 |
| 感知微调 EDVR | 23.548 | 0.7368 | **0.4598** | **0.2889** | 0.0258 | 15 |
| 原始 EDVR | 23.282 | 0.7294 | 0.4832 | 0.3004 | 0.0323 | 15 |
| DOVE 5B | 21.758 | 0.6905 | 0.3670 | 0.2193 | **0.0158** | 12 |
| OSEDiff | 21.159 | 0.6863 | 0.4114 | 0.2551 | 0.0290 | 15 |

相对 bicubic：

| 方法 | ΔPSNR | ΔLPIPS |
|---|---:|---:|
| 感知微调 EDVR | −0.101 | −4.0% |
| 原始 EDVR | −0.367 | +0.9% |
| DOVE | −1.891 | −23.4% |
| OSEDiff | −2.490 | −14.1% |

## 结论

1. **在这个数据集上，回归模型连 bicubic 都没超过。** 原始 EDVR 的 PSNR 比 bicubic 低 0.37 dB、LPIPS 高 0.9%；感知微调也只比 bicubic 好 4.0% 的 LPIPS，同时 PSNR 低 0.10 dB。这与 RealVSR 上"EDVR 高出 bicubic 0.89 dB"完全不同。
2. **生成模型在感知轴上明显领先，在 PSNR 轴上明显落后。** DOVE 的 LPIPS 比 bicubic 低 23.4%、DISTS 低 29.9%；OSEDiff 的 LPIPS 低 14.1%、DISTS 低 18.4%。但两者 PSNR 分别低 1.89 / 2.49 dB。
3. **这与 RealVSR 的结论方向相反**：RealVSR 上生成模型感知轴也没赢（DOVE 的 LPIPS 仍比感知微调 EDVR 差 11.9%），MVSR4x 上 DOVE 的 LPIPS 已经超过所有回归模型。
4. **不能说"某个数据集上生成先验更好"作为普遍结论。** 两个数据集的关键差别：MVSR4x 是真正的 x4 退化（wide 对 tele 降 4 倍），RealVSR 只有约 2 倍。退化更重时，回归模型的天花板更低，生成模型的相对位置更好。
5. **DOVE 的时序指标最好**（tLP 0.0158，比 bicubic 低 36.7%），是唯一在时序上明显优于回归模型的方法。

## 未完成

DOVE 只跑了 12/15 段。缺失的 3 段（115、139、158）在运行期间遇上其他用户占用 29.8 GB 显存（可用仅 18 GB），DOVE 的 bfloat16 权重需要约 47 GB，直接 CUDA OOM（`rc=1`，日志明确记录 "GPU has a total capacity of 47.41 GiB of which 13.69 MiB is free"）。

补齐方式：等显存空闲后重跑同一条命令，脚本会跳过已完成的 12 段：

```bash
cd p3/prior_qualification
RVQ_DS=mvsr4x venv/bin/python scripts/run_dove_realvsr.py
RVQ_DATA=$PWD/data/mvsr4x venv/bin/python scripts/score_realvsr.py --name dove --dir outputs/realvsr/dove
```

因此 DOVE 这一行的 12 段与其余方法的 15 段不是同一批片段，**严格来说不可直接比较**；但它与 bicubic 在同一 12 段上的差值为 −1.89 dB，方向上不会因缺少 3 段而反转。

## 复现

```bash
# 1. 取数据并解包（分卷 zip 需先拼接，中央目录在最后一卷）
bash scripts/dl_mvsr.sh

# 2. 构建子集
python scripts/build_mvsr_subset.py --root mvsr_test/MVSR4x \
    --out data/mvsr4x --frames 40 41 42 43 44

# 3. 推理与打分
cd p3/prior_qualification
VENV=p3/quality_metrics/venv/bin/python
export RVQ_DATA=$PWD/data/mvsr4x
$VENV scripts/run_realvsr.py --name bicubic --method bicubic
$VENV scripts/run_realvsr.py --name edvr    --method edvr
$VENV scripts/run_realvsr.py --name ft      --method edvr --finetuned true
$VENV scripts/run_osediff_realvsr.py --name osediff_mv
RVQ_DS=mvsr4x $VENV scripts/run_dove_realvsr.py
for n in bicubic edvr ft osediff dove; do
  $VENV scripts/score_realvsr.py --name $n --dir outputs/realvsr/$n
done
```

## 分卷 zip 的解包方法（踩过的坑）

这个 40.18 GB 归档是标准分卷 zip（z01–z18 各 2 GiB，末尾 `.zip` 存中央目录），但：

- 直接 `zipfile.ZipFile("joined.zip")` 报 “zipfiles that span multiple disks are not supported”；
- 中央目录里的 `cdoff` 是相对首卷的 32 位偏移（1523523311），拼接后按它定位得到的是错误位置；
- EOCD/Zip64 EOCD 里的偏移同样不可信。

可行做法：在拼接文件里搜索 `MVSR4x/tele_test/` 字符串定位第一个测试条目的本地头（本数据在偏移 22,431,678,099），从那里顺序遍历本地头直到中央目录起点（40,178,228,975），边遍历边解压。`dl_mvsr.sh` 里没有再实现这套逻辑，请以本文档描述为准。
