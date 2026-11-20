# -*- coding: utf-8 -*-
"""Second pass: fix the summary paragraph and the DOVER section."""
import io

P = "/data/disk1/home/ghe270/p3/quality_metrics/CONCLUSIONS.md"
s = io.open(P, encoding="utf-8").read()

old = """四个新指标与 LPIPS 的排序几乎完全一致（Spearman 0.986），**没有推翻任何既有排名**；
它们提供的是同一排序下的**可解释性**——所有方法普遍缺高频纹理，EDVR 缺得最少；
TecoGAN 在结构和纹理两个方向都更差，并不存在"纹理更自然"的优势。"""

new = """四个新指标与 LPIPS 的排序几乎完全一致（Spearman 0.986），**没有推翻任何既有排名**；
它们提供的是同一排序下的**可解释性**——所有方法普遍缺高频纹理，EDVR 缺得最少。
TecoGAN 在结构和纹理两个方向都更差，并不存在"纹理更自然"的优势。

缺陷的落点很具体：**纹理区块贡献了 81%～83% 的总绝对误差**（与方法无关），
平坦区反而是唯一接近及格的区域。真正的方法间差异不在空间域而在时间域——
tavg 系列的劣势随运动增强接近翻倍，空间方法则几乎不变。"""

assert old in s, "summary paragraph not found"
s = s.replace(old, new, 1)

old2 = """**DOVER 技术分与 tLP 方向一致但不冗余。** DOVER 技术分与 PSNR 的 Spearman 只有 +0.685，
明显低于 DISTS 的 −0.874，说明它确实捕捉了像素误差之外的成分。但与 tLP 的对照显示：

- DOVER 技术分最差的 6 个"片段×方法"全部来自 **同一个片段 016**（`d3_temporal.md`），
  即它主要反映片段难度，方法间差异在其次；
- 同一片段内，DOVER 技术分与 tLP 排序一致：tavg5（tLP 0.0660）最差、
  bicubic_unsharp（tLP 0.0298）最好。"""

new2 = """**DOVER 技术分与 tLP 方向一致但不冗余。** DOVER 技术分与 PSNR 的 Spearman 只有 +0.685，
明显低于 DISTS 的 −0.874，说明它确实捕捉了像素误差之外的成分。两点具体观察：

- 全部 30 个片段的 DOVER 技术分里，最差的 6 个"片段×方法"**全部来自同一个片段 016**
  （`d3_temporal.md`）。但片段间均值的离散（标准差 0.013）**小于**同一片段内方法间的
  极差（均值 0.084），所以它仍主要在片段内区分方法，不是纯粹在挑难题片段。
- 同一片段内，DOVER 技术分与 tLP 排序一致：tavg5（tLP 0.0660）最差、
  bicubic_unsharp（tLP 0.0298）最好。"""

assert old2 in s, "dover paragraph not found"
s = s.replace(old2, new2, 1)

old3 = """1. **DISTS 不是可分解的逐像素指标。** 本报告的块级 DISTS 归类为"局部裁切诊断"，
   不是整帧主分数，不能与主表数值混用。"""
new3 = """1. **DISTS 不是可分解的逐像素指标。** 本报告的块级统计全部来自裁切诊断，
   不是整帧主分数，不能与主表数值混用；`d1_attribution` 里的 hf_ratio / MAE
   是块内直接测得的量，不是块级 DISTS。"""
assert old3 in s, "limitation 1 not found"
s = s.replace(old3, new3, 1)

io.open(P, "w", encoding="utf-8").write(s)
print("patched summary + dover + limitation")
