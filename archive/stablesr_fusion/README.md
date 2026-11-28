# StableSR + EDVR 输出级融合（快速验证）

**结论：不通过，停止该路线。** 完整判据、逐方法数值与复现入口见
[`PROBE_CONCLUSION.md`](PROBE_CONCLUSION.md)。

一句话概括：在 REDS ×4、StableSR 官方 200 步单帧配置下，把冻结的 StableSR 生成结果
与 EDVR 做输出级融合，对像素保真没有可利用的收益——最优固定混合相对 EDVR 仅
**+0.00045 dB**（门槛 +0.05 dB），用 GT 的连续 alpha oracle 上限也只有 **+0.020 dB**。

## 目录

| 路径 | 内容 |
|---|---|
| `PROBE_CONCLUSION.md` | 结论文档：设置、结果表、三条判读、复现入口 |
| `probe/probe_results.json` | 完整结果为 JSON（4 片段 × 40 帧，含固定混合扫描与两种 oracle） |
| `scripts/cache_edvr_probe.py` | 生成 EDVR 逐帧缓存 |
| `scripts/cache_stablesr_probe.py` | 生成 StableSR 逐帧缓存（200 DDPM 步） |
| `scripts/score_probe.py` | 打分：固定混合、空间/连续 oracle，写 JSON 与 markdown 摘要 |

## 未入库的内容

以下都是可再生的中间产物或第三方依赖，按仓库 `.gitignore` 的既有取舍不入库：

- `probe/cache/`（约 241 MB 的 EDVR 与 StableSR 逐帧缓存）；
- `venv/`、`pydeps/`、`vendor/`、`clip_src/`、`smoke/` 等运行环境与第三方源码；
- `clip-1.0.zip`、`setuptools-*.whl` 等下载件。

复算时按 `PROBE_CONCLUSION.md`「复现」一节，先跑两个 `cache_*.py` 生成缓存，
再用 `score_probe.py` 打分即可重建 `probe_results.json`。
