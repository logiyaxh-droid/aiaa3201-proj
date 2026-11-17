# 全量 30 clip 评测结果

主表、tLP 分解、配对对照、逐 clip 胜负与 FID 见 [REPORT.md](REPORT.md)。

- metrics/：每个 clip 一个 JSON（12 个方法的 PSNR/SSIM/LPIPS/正式 tLP/Flow-warp LPIPS，
  以及 tLP 的 d_SR、d_GT 分解）+ un_config.json（口径与权重路径）
- summary/：	able.md（主表）、	able.json、deltas.json（配对对照 + 逐 clip 胜负）、id.json
- igures/：PSNR–tLP、PSNR–FID 散点、逐 clip tLP、逐 clip 配对增益
