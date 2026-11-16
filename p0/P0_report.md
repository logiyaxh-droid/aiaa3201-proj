# P0 基础层：数据/权重核查、统一评测、正式 tLP、TecoGAN 预平移与完整性

本轮只做清单里的 **任务 1–3、10–11、17**（按你选择的范围），算力按「小样本冒烟、≤30 min GPU」执行。
所有数字都来自本次重跑，脚本与中间结果在服务器 `~/p0/`，可直接复算；代码与审计 JSON 已同步到仓库
`p0/` 目录。

旧的 `/tmp/p1_part1/*.json`、`/tmp/p2/*.json` **未被覆盖**，已原样备份为 `~/p0/out/metrics/legacy_*.json`。

---

## 0. 结论速览

| 项 | 结论 | 证据 |
|---|---|---|
| 课程 `val_sharp` 是哪个划分 | **官方 validation 划分（原 val 240–269），不是 REDS4**。课程目录 `000..029` 与 REDS4 的 `000/011/015/020` 只是编号巧合 | 与上游 `val_sharp.zip` 逐像素 SHA256 **完全相同**；与 `train_sharp.zip` **全部不同** |
| EDVR 权重与测试数据是否重叠 | **不重叠**。训练时 `val_partition=REDS4` 挖掉了 original-train 的 000/011/015/020，课程 30 clip 全在 original-val | BasicSR `reds_dataset.py` 划分逻辑 + 上面的内容比对 + clip 000 无记忆痕迹 |
| EDSR 权重训练数据 | **DIV2K**（800 张），README 写的 “EDSR_REDS.pth” 是错的 | 文件名 `EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth` + ModelZoo |
| 旧 “tLPIPS” 是什么 | `LPIPS(warp(prev), cur)` 的 **stride=2** 自一致性指标，**不是** TecoGAN 的 tLP；无 GT 也能算 | 复现误差 1.5e-10 |
| 正式 tLP 是否实现并验证 | 已实现并逐项验证（GT→0、无 GT→N/A、逐对交叉校验、×100、cutfr、crop32、缺帧断邻）| `p0/audit/tlp_check.json` 全项 pass |
| TecoGAN +0.375 预平移 | **数值对，理由写反了**。`bicubic_four` 是“边缘对齐”网格（LR 像素 i ↔ HR 4i），与 cv2 差 **0.376 LR px**；+0.375 恰好抵消 | 斜坡精确测量 + 真实 `bicubic_four` PSNR 扫描峰值 0.34–0.35 + 12 帧 A/B **+2.66 dB** |
| TecoGAN 缓存是否可信 | 旧 `.done` 不绑定任何配置。新流程绑定 manifest 哈希，命中时**逐帧校验内容 SHA256** | 故意篡改 1 帧同尺寸像素 → 被识别并要求重跑 |
| 空间指标换实现后是否要重跑 | **不需要**。与旧实现逐位一致 | 100 帧 × 2 clip：PSNR/SSIM 差 0，LPIPS 差 ≤2e-9 |

---

## 1. 任务 1：数据与权重核查

### 1.1 权重清单（2026-11-16 实测）

| 键 | 文件 | 字节 | SHA256 |
|---|---|---|---|
| edsr | `EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth` | 6,084,475 | `0c287733e70d1b8b8fc5885613ecbe451e5f3010bcae0307612ef5e4aa08dd5f` |
| edvr_reds | `EDVR_L_x4_SR_REDS_official-9f5f5039.pth` | 82,587,950 | `9f5f50393965194cc180e1313fd162bc85003bb4c2a58c23a434a33fdb3786a2` |
| edvr_vimeo | `EDVR_L_x4_SR_Vimeo90K_official-162b54e4.pth` | 82,850,094 | `162b54e4935bf7e66c4a2a0cf0d3620bc4ec287eb4d4dfe1587b9ecdcc6768cb` |
| tecogan | `TecoGAN.data-00000-of-00001` | 12,128,916 | `e009bc08ec84c17bd8525252933ceafb164a743727285360259160e81a8e6961` |
| tecogan | `TecoGAN.index` | 4,814 | `c100584c2ec416c2daa4175ce43c5d0ebb87f13db5c340441b8e75cee0da70d3` |
| tecogan | `TecoGAN.meta` | 496,308 | `b5e126d40aced8c952b84c63a3cbe9258848a7f9a866e7e225c44ac29c979259` |

来源：课程共享盘 `/data/disk1/shared/aiaa3201/checkpoints/` 镜像（2026-11-05 上传）。
文件名里的 8 位十六进制就是 BasicSR 官方 release 的 tag，与 `docs/ModelZoo.md` 一致。
TecoGAN 是官方 TF1 release 的 checkpoint 三件套（不是冻结 pb）。

**训练数据：**

| 权重 | 训练数据 | 依据 |
|---|---|---|
| EDSR x4 (M, 64feat, 16blocks) | DIV2K 800 张训练图 | 官方文件名 + ModelZoo（Set5 30.1821 / Set14 26.7528 / DIV2K100 28.9679）|
| EDVR-L x4 (REDS) | REDS `train_sharp`，`val_partition=REDS4` | `options/train/EDVR/train_EDVR_L_x4_SR_REDS.yml`、`reds_dataset.py:65` |
| EDVR-L x4 (Vimeo) | Vimeo-90K septuplet train | 官方命名规范 |
| TecoGAN | Vimeo-90K septuplets，**Gaussian** 退化 `tf_data_gaussDownby4(sigma=1.5)` | `lib/ops.py:349`、`dataPrepare.py` |

> 清单里“TecoGAN 为真实退化训练”确实是错的，官方是 Gaussian 下采样合成 LR。
> 本轮只改描述；对应的控制变量实验（任务 12）不在本轮范围。

### 1.2 课程 `val_sharp` 是哪个划分 —— 内容比对

编号本身说明不了问题（000 在两个划分里都有），所以比内容。方法：对上游 HuggingFace `snah/REDS`
的两个 zip **只用 HTTP range 请求**读中央目录 + 目标条目，全程 19 MB 流量，没有落盘
4.3 GB / 34 GB。（服务器访问不到 HuggingFace，这段在本机执行。）

| 上游归档 | clip 000/011/015/020 的帧 0、49、98 | 判定 |
|---|---|---|
| `val_sharp.zip`（官方 validation） | 12/12 帧 SHA256 **完全一致** | ✅ 课程数据 = 官方 validation |
| `train_sharp.zip`（原训练集） | 12/12 帧 **全部不同** | ✅ 与 REDS4 所在的 original-train 无关 |

报告 `p0/audit/reds_audit_report.json`，脚本 `p0/scripts/reds_audit.py`（原写在 `~/` 下的 `work/`，
入库时归到 `p0/` 下）。

### 1.3 EDVR 权重与测试数据重叠吗

不重叠，两边同时确认：

1. **训练侧**：BasicSR 训练集把 REDS4 的 `000/011/015/020` 从 original-train 剔除
   （`reds_dataset.py:65-70`）；这 4 个 clip 属 original-train，课程 30 clip 属 original-val，定义上不相交。
2. **数据侧**：课程 `val_sharp` 与上游 `train_sharp.zip` 逐像素不同（§1.2）。
3. **旁证**：若课程 clip 000 真是 EDVR 训过的那个，它应明显优于其它 clip；实测 clip 000 的
   EDVR PSNR 30.49 dB，反而**低于**其余 26 个 clip 的均值 31.29 dB，没有记忆痕迹。

**顺手自查**：`val_sharp` 3000 帧 vs `train_sample_sharp` 2000 帧的 MD5 全量比对，
跨划分重复 **0**，划分内部重复也是 0。

### 1.4 README 已修正的三处

| 旧 | 新 |
|---|---|
| `EDSR_REDS.pth`｜官方 EDSR ×4 **REDS** 权重 | `EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth`｜**DIV2K** |
| `EDVR_REDS.pth`（78.8 MB） | `EDVR_L_x4_SR_REDS_official-9f5f5039.pth`（82,587,950 B = 78.8 **MiB**） |
| `val_sharp/` REDS 验证集 | REDS **官方 validation 划分**，并注明与 REDS4 只是编号巧合 |

完整清单见仓库 `p0/data_manifest.md`（含 SHA256、训练数据、仍需核实项）。

---

## 2. 任务 3 + 17：统一评测入口与最小工程整理

新增 `~/p0/evalkit/`（已同步到仓库 `p0/evalkit/`），五个模块，Part 1/2 都调它，不再各抄一份：

| 模块 | 职责 |
|---|---|
| `metrics.py` | PSNR/SSIM/LPIPS 唯一实现；口径写成 `CONVENTIONS` 常量随结果存 |
| `temporal.py` | 正式 tLP + 旧 flow-warp LPIPS（改名保留）；帧对规则、`crop_8x8` 移植 |
| `data.py` | 按 `clip + frame_id` 严格配对；缺帧/多帧/尺寸不符**直接抛错** |
| `config.py` | 配置默认值 + 覆盖，`out.root` 必填（去掉 `/tmp` 隐式依赖）|
| `provenance.py` | 权重 SHA256、训练数据说明、代码哈希、环境快照 |

### 2.1 一致性验证

6 帧真实 REDS 帧（clip 000）：SSIM vs skimage 最大差 **5.2e-14**；PSNR vs 闭式差 **0.0**；
LPIPS 新旧封装差 **0.0**；`evaluate_stack` 与逐项一致。

**全量回归（100 帧 × clip 000/027）**，用旧 JSON 的 bicubic 逐 clip 值对照：

| clip | PSNR 新 vs 旧 | SSIM 新 vs 旧 | LPIPS 新 vs 旧 |
|---|---|---|---|
| 000 | 25.6170 vs 25.6170（差 **0**） | 0.709609 vs 0.709609（差 **0**） | 0.51335 vs 0.51335（差 −2.1e-09） |
| 027 | 25.1068 vs 25.1068（差 **0**） | 0.733556 vs 0.733556（差 **0**） | 0.49752 vs 0.49752（差 +1.8e-09） |

结论：**空间指标定义没变，旧空间数字仍可比，不需要重跑全部推理。**

### 2.2 工程整理做了什么

- 路径全参数化：数据根、权重、输出根、device、clip 列表、指标开关，无写死 `/tmp`。
- 输出前 `os.makedirs`，不再依赖已存在的 `/tmp` 目录。
- **推理与离线指标分离**：`run_tecogan.py` 只产帧；打分脚本只读帧算指标，补 tLP/FID 不必重跑推理。
- 每个结果目录写 `run_config.json` + `manifest.json`（权重哈希、代码哈希、帧清单、设备）。
- 旧 JSON 一律另存 `legacy_*`，不覆盖。

---

## 3. 任务 2：正式 tLP 与旧指标的关系

### 3.1 实现

```text
d_SR(t) = LPIPS(SR[t-1], SR[t])
d_GT(t) = LPIPS(GT[t-1], GT[t])
tLP(t)  = |d_SR(t) - d_GT(t)|
```

- 在**所有**合法相邻帧对上取均值（clip 000 共 99 对，不是 stride=2 的 50 对）。
- 相邻 = `id[k+1] == id[k]+1`；缺帧断邻，不跨 clip。
- 没有连续 GT 时返回 `null` 并写原因，**不返回 0**。
- 开关全部记进结果：`x100`、`cutfr`（TecoGAN 用 2）、`crop32`（移植 `crop_8x8`，720×1280 → 704×1248）。

### 3.2 验证（`p0/audit/tlp_check.json`，全项 pass）

| 检查 | 结果 |
|---|---|
| A. 用 GT 当输出 | tLP = **0.0**（99 对） |
| B. GT 每隔一帧取（无连续对）| `value=null`, `n_pairs=0`，附原因 |
| C. 向量化 vs 逐对 batch=1 朴素循环 | 差 **6.2e-09**（GPU LPIPS 不同 batch 非逐位一致，容差 1e-6）|
| D1. `x100` | 比值 **100.000000** |
| D2. `cutfr=2` | 99 → **95** 对；0.02628（全对 0.02610）|
| D3. `crop32` | 704×1248；0.02236 |
| E. 旧指标复现 | 旧 `tlpips=0.056854682564735415`；新 stride=2 → `0.056854682713747020`，差 **1.5e-10** |
| F. 缺第 4 帧 | 99 → **97** 对（断邻而非桥接）|

### 3.3 旧数字怎么读

旧 JSON 的 `tlpips` **全部是 stride=2 的 flow-warp LPIPS**，衡量“输出序列自身有多稳”，
与 GT 质量无关，**平滑的输出天然占便宜**（Bicubic 25.86 dB 拿 0.0519，EDSR 28.28 dB 却是 0.0811）。
所以：

- 旧列**不能**当作 TecoGAN 的 tLP 与论文比较；
- 不能单凭它的低值断言“某方法最稳定”；
- clip 000 的对照很直观：Bicubic 正式 tLP **0.0261**（100 帧），TecoGAN(+0.375) **0.0047**（前 12 帧样例），两者都不等于旧指标给出的 0.0569 / 0.0800。

---

## 4. 任务 10：TecoGAN 预平移复核

### 4.1 旧注释的说法

`p2_teco.sh` 声称 `bicubic_four` 假设“半像素中心网格”、课程 LR 是 `align_corners=True`，
两者差 1.5 HR px ≈ 3.9 dB，所以输入 +0.375 LR px 抵消。**值碰对了，归因是反的**，
而且“课程 LR 是 align_corners=True”这句从未验证。

### 4.2 精确测量网格（斜坡 + 脉冲）

斜坡对任何核和为 1 的三次插值都是精确解，所以每个输出样本的**内容位置**可直接读出；脉冲给出整数映射：

```
脉冲：输入 (row=4, col=10) → 输出峰值 (16, 40) = (4×4, 10×4)，值 1.0000
斜坡：out[j] 的隐含输入坐标 = j/4（j=0→0.000, j=4→1.000, j=76→19.000）
```

| 实现 | 内容位置 = origin + step·j（LR 单位）| 判定 |
|---|---|---|
| **`bicubic_four` (TF 1.15)** | origin **+0.0002 / −0.0004**，step **0.250000** | 边缘对齐：LR 像素 i 正好落在 HR 4i |
| cv2 `INTER_CUBIC` | origin **−0.3757 / −0.3735**，step 0.250004 | 半像素中心 |
| torch `align_corners=False` | 同 cv2 | 同 cv2 |
| torch `align_corners=True` | origin +0.0006 / −0.0012，step 0.2474/0.2448 | 角点对齐（步长也变）|
| `bicubic_four` vs cv2 | 图像位移 **+0.3759 / +0.3731 LR px**（+1.50 / +1.49 HR px）| 与旧注释量级一致 |

即：**`bicubic_four` 是边缘对齐（origin 0），课程 LR（cv2 语义）是半像素中心（origin −0.375）**。
输入整体右移 0.375 LR px，内容位置 +0.375，恰好补上 → 两套坐标对上。
旧注释把“哪边是哪种网格”写反了，但 **0.375 这个数值是对的**。

顺带纠正一个常见误解：cv2 与 torch `align_corners=False` 在本例里是**同一套**半像素中心网格，
真正不同的那一套是 `align_corners=True`。

### 4.3 三条独立路线互相印证

| 路线 | 最优预平移 |
|---|---|
| 斜坡几何（精确、与数据无关）| **+0.376 / +0.373 LR px** |
| 真实 `bicubic_four` 输出 vs GT 的 PSNR 扫描（−0.6…+0.8，步长 0.05）| 网格峰值 **+0.35**，抛物线顶点 **+0.3436**；PSNR 18.9 → **25.25**（+6.4 dB）|
| 真实 TecoGAN 推理 A/B（clip 000 前 12 帧）| **+0.375 vs 0：22.030 → 24.694（+2.664 dB）** |

三条都对上，说明 0.375 不是“在测试集上试出来的好数字”，而是网格约定的必然结果。
第二条与 0.376 的小差距来自 `bicubic_four` 的边界复制和 GT 自身，不影响结论。

### 4.4 A/B 完整指标（12 帧，同一协议）

| 配置 | PSNR | SSIM | LPIPS | 正式 tLP | 帧对 |
|---|---|---|---|---|---|
| 原始 LR（shift=0）| 22.030 | 0.5736 | 0.2299 | 0.0232 | 11 |
| +0.375 预平移 | **24.694** | **0.6581** | 0.2872 | **0.0047** | 11 |

注意：平移后 **LPIPS 变差**（0.2299 → 0.2872）而 PSNR/tLP 大幅变好。这是 12 帧小样本，
且 GAN 输出的锐度对 LPIPS 很敏感，**不能**据此下“感知变差”的结论 —— 要等全量重跑。

### 4.5 最终配置与遗留

- **最终使用 `shift = 0.375`**，理由换成可复核的网格测量。
- 记录事实：0.375 最早是在测试样本上选出的；现在有独立几何依据，且在两个不同的帧集合上一致。
- **仍需注意**：TecoGAN 官方用**它自己的 Gaussian 退化 LR**，不是课程 bicubic LR，
  因此在课程数据上的 PSNR 与论文不可比（属任务 12 范围）。

---

## 5. 任务 11：TecoGAN 完整性与缓存

旧问题：不检查退出码；不校验帧数/文件名/尺寸就 `touch .done`；`.done` 不绑定权重或配置；
固定 `LR/tc` + `results/tc` 共享 scratch 会被并发运行互相覆盖；指标脚本对缺帧只打印一行然后继续算。

新 `run_tecogan.py` 逐条对应：

| 旧问题 | 新行为 |
|---|---|
| 不看退出码 | 检查返回码；非 0 写 `.failed` 并保留 `run.log`，绝不写 `.done` |
| 不校验输出 | 期望帧 id 由 LR 推出；要求**恰好**该集合、全部可读、尺寸 = LR×4 |
| `.done` 不绑定配置 | `manifest.json` 记权重 SHA256 ×4、shift、帧 id、LR 尺寸、`num_resblock`、设备、TecoGAN 源码哈希；`.done` 存 manifest 哈希，不匹配即重跑 |
| 缓存可能已损坏 | 命中时**重新校验**：帧集合 + 尺寸 + **逐帧内容 SHA256** |
| 共享 scratch | 每次运行独立 `LR/<tag>`、`results/<tag>`，跑完删除 |
| 帧号映射隐式 | TecoGAN 以 1-based 位置命名 `output_%04d`，显式映射回 frame id 后与输入 id 断言比对 |

**实测：**

```
首次运行      -> OK: 12 frames, manifest a61b84143c17
再运行        -> cache hit
篡改第 5 帧（同尺寸、只改像素）-> stale cache discarded: frame content changed since it was written: [5]
                              -> 自动重跑并恢复，OK
改 --shift 0.25 -> cache key changed (0c75cc6a1d35 -> 0295f09f530b), re-running
```

说明：第一版校验只看数量与尺寸，被这个篡改测试**当场发现是漏洞**并补上内容哈希 ——
这条测试要保留，因为“同尺寸错误帧”正是旧 `.done` 最可能放进去的东西。

指标侧的方法学要求（丢帧时所有方法用同一列表、不用不连续帧算相邻指标）已在
`evalkit.data.match_clip` 与 `evalkit.temporal.pairs_of` 落实：缺帧默认**报错**，
要排除必须显式 `allow_missing_pred=True` 并把 `missing` 写进结果。

---

## 6. 本轮没做的事（明确留白）

- 任务 4（FID）、5（完整消融）、6（运动分组）、7（光流变体局部图）、8（速度/显存）、9（对齐诊断）；
- 任务 12（TecoGAN 退化失配对照）、13（EDVR 邻帧利用）、14（Vimeo 上的 Part 2）、15（真实低光视频）、16（Part 2 可见时域证据）；
- **全量重跑**：本轮只用 clip 000 的 12 帧做 A/B。REDS 30 clip × 100 帧的正式 tLP 与 FID 需要一次
  完整重跑（约 45 min 推理 + 指标），留给下一轮。
- 因此 Part 1/2 的**正式 tLP 列目前只有 clip 000 的示例**，还没有全量数字。
- 文档里对应的错误说法已在仓库中改掉（README、`p1/docs/P1_详细拆解.md` 的 8.3/8.4 节、
  `P1_Part1_交付清单.md`、`P2_交付清单.md`），旧数值一律保留。

---

## 7. 复算入口

```bash
cd ~/p0
export TORCH_HOME=/tmp/torch_home

# 指标一致性（任务 3）
/tmp/srenv/bin/python -m evalkit.selfcheck --clip 000 --frames 6 --out out/audit/selfcheck_metrics.json

# 正式 tLP 的全部验证（任务 2）
/tmp/srenv/bin/python scripts/check_tlp.py --clip 000 --frames 100 \
    --pred-dir out/pred/bicubic/000 --old-json out/metrics/legacy_part1.json \
    --old-method bicubic --out out/audit/tlp_check.json

# bicubic_four 的网格（任务 10）
/tmp/tf1env/bin/python scripts/probe_grid.py --part tf  --out out/audit/grid_tf.json
/tmp/srenv/bin/python  scripts/probe_grid.py --part ref --out out/audit/grid_ref.json
/tmp/srenv/bin/python  scripts/probe_grid.py --part compare \
    --tf out/audit/grid_tf.json --ref out/audit/grid_ref.json --out out/audit/grid_convention.json
/tmp/tf1env/bin/python scripts/identify_grid.py            # 脉冲 + 斜坡原始读数
/tmp/tf1env/bin/python scripts/scan_bicubic_four.py --clip 000 --frame 2 --out out/audit/bf_shift_scan.json

# TecoGAN 预平移 A/B（任务 10 + 11）
/tmp/srenv/bin/python scripts/run_tecogan.py --lru <LR clip dir> \
    --out out/tecogan_ab/000_shift0    --shift 0     --limit 12 --tag ab0
/tmp/srenv/bin/python scripts/run_tecogan.py --lru <LR clip dir> \
    --out out/tecogan_ab/000_shift0375 --shift 0.375 --limit 12 --tag ab375
```

## 8. 文件清单

| 路径 | 内容 |
|---|---|
| `p0/evalkit/{metrics,temporal,data,config,provenance,selfcheck}.py` | 共用评测包 |
| `p0/scripts/check_tlp.py` | 正式 tLP 的 A–F 六项验证 |
| `p0/scripts/probe_grid.py` | 斜坡/相位测量各上采样器的网格 |
| `p0/scripts/identify_grid.py` | 脉冲 + 斜坡原始读数（只打印，不解释）|
| `p0/scripts/scan_bicubic_four.py` | 真实 `bicubic_four` 的平移–PSNR 曲线 |
| `p0/scripts/run_tecogan.py` | 带完整性校验与缓存键的 TecoGAN 推理 |
| `p0/scripts/teco_shift_ab.py` | 预平移 A/B 的指标打分 |
| `p0/scripts/make_baselines.py` | bicubic/lanczos/USM 预测帧 |
| `p0/audit/*.json` | 全部审计结果 |
| `p0/data_manifest.md` | 权重 SHA256 + 训练数据 + 重叠核查（已并入 README 数据集与权重小节） |
| `p0/scripts/reds_audit.py`, `p0/audit/reds_audit_report.json` | REDS 划分内容比对（HTTP range）|
| `p0/全量tLP_FID_耗时预估.md` | 全量 30 clip 正式 tLP + FID 的实测吞吐与两档预算 |

---

## 9. 文档修正清单（旧结论 → 新表述）

本轮把仓库里与新证据冲突的说法一并改掉，**数值一律保留**，只改标签和结论：

| 文件 | 旧 | 新 |
|---|---|---|
| `README.md` | EDSR 是 REDS 权重 | DIV2K 权重（含 SHA256、字节数） |
| `README.md` | `tLPIPS` 列 | `Flow-warp LPIPS`，并加框说明它不是参考式 tLP |
| `README.md` | 「TecoGAN 赢感知与时域指标」 | LPIPS 上 EDVR 0.2039 / EDSR 0.3087 都优于 TecoGAN 0.3337，TecoGAN 只优于 bicubic/lanczos |
| `README.md` | 「REDS 验证集」 | 「REDS **官方 validation 划分**」，并说明与 REDS4 只是编号巧合 |
| `p1/docs/P1_详细拆解.md` §8.3 | 把 flow-warp 公式当作 TecoGAN 的 tLPIPS | 拆成 (a) 参考式 tLP、(b) Flow-warp LPIPS、(c) 旧表列是什么、(d) 光流方向标定 |
| `p1/docs/P1_详细拆解.md` §8.4 | 「FID 可选，时间紧可跳过」 | FID 是作业 Mandatory 项，不能跳过；给出固定配置与限制写法 |
| `p1/docs/P1_详细拆解.md` §6 | 「TecoGAN 赢感知和时域指标，这个矛盾就是 Part 3 的出发点」 | 与实测不符，改为「EDVR 在全部已测指标上领先；时域缺可靠证据」，Part 3 动机改为「取舍」 |
| `p1/docs/P1_Part1_交付清单.md` | 表头 `tLPIPS`、多处「tLPIPS 变差」 | 表头与措辞改 `Flow-warp LPIPS`；补口径说明（PSNR/SSIM 不能直接对标论文） |
| `p2/docs/P1_Part2_交付清单.md` | 表头 `tLPIPS`；「TecoGAN 为真实退化训练」；「flowtavg5 tLPIPS 至今无人超越」 | 表头改 `Flow-warp LPIPS` + 新增正式 tLP 列（仅 clip 000 有值）；退化描述改为 Gaussian 且标注为未验证假设；时域结论改为「待正式 tLP 复核」 |
| `p2/docs/P1_Part2_交付清单.md` | 对比视频「10 fps」未标注问题 | 注明与源 24 fps 不一致，按任务 16 应改为源 FPS |

新的 `p0/data_manifest.md` 是 README 数据/权重小节的完整版，可直接替换。