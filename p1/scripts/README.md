# P1 · Part 1 脚本备份与环境重建说明

> 备份时间：2026-11-11 18:43。三个脚本已通过 SHA256 校验，与服务器 `/tmp` 上的版本逐字节一致。

---

## 文件清单

| 文件 | 作用 | 依赖 |
|---|---|---|
| `p1_part1_final.py` | **主评测**：10 个方法 × 30 clip（REDS val），含光流对齐 | EDSR 权重、lpips、cv2 |
| `p1_vimeo.py` | Vimeo-90K 测试集评测（7824 clip，仅中心帧） | 同上 |
| `flow_gt.py` | **光流方向标定**：用已知平移的合成样本验证 warp 约定 | 仅 cv2 |

SHA256（本地 = 服务器）：

```
p1_part1_final.py  7095184a58a4b1183c44fdf9f8097be6e4214df6514df53190b7870f916ba3eb
p1_vimeo.py        211d7b6d08c4c7ae0013a0efcb3a43309b4cc631abbe127def1fa1f9f1f64df3
flow_gt.py         e882205013cf568b2059f128c68ef9136ef1667aebe988cadacea394252cfb8e
```

---

## ⚠️ 为什么需要这份文档

**`/tmp` 重启就会清空。** 脚本备份了，但 `/tmp/srenv`（Python 环境）和 `/tmp/bsr_src`（BasicSR + 编译好的 DCN 扩展）会一起消失，只拷回脚本是跑不起来的。

判断环境是否还在：

```bash
ls /tmp/srenv/bin/python                              # 存在？
ls /tmp/bsr_src/basicsr/ops/dcn/*.so                  # DCN 编译产物，EDVR 才需要；Part 1 不用
```

Part 1 其实**只需要 torch + cv2 + lpips + BasicSR 的 EDSR 定义**，不需要 DCN（那是 Part 2 的 EDVR 才要）。

---

## 重建步骤（从零开始）

```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4
export TORCH_CUDA_ARCH_LIST="8.6"

# 1. 建环境（必须 conda-forge，anaconda channel 有 ToS 问题）
/opt/miniconda3/bin/conda create -y -p /tmp/srenv --override-channels -c conda-forge python=3.10

# 2. 先装 opencv！否则装 basicsr 时 setup.py 的 get_hash() 会 import 失败
/tmp/srenv/bin/pip install -q opencv-python-headless

# 3. torch cu124（和系统 nvcc 12.4 对齐，不要用 cu128）
/tmp/srenv/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. 其余依赖
/tmp/srenv/bin/pip install -q pyyaml lmdb scikit-image tqdm addict future requests yapf lpips imageio-ffmpeg

# 5. BasicSR 源码安装（不能 pip install basicsr）
curl -sSL -o /tmp/bsr.tgz https://codeload.github.com/XPixelGroup/BasicSR/tar.gz/refs/heads/master
tar xzf /tmp/bsr.tgz -C /tmp && mv /tmp/BasicSR-master /tmp/bsr_src
printf "version = '1.4.2'\n" > /tmp/bsr_src/basicsr/version.py      # 源码无 .git，必须手写
/tmp/srenv/bin/pip install -q --no-build-isolation --no-deps -e /tmp/bsr_src

# 6.（仅 Part 2 需要）编译 DCN
export BASICSR_EXT=True          # ← 没有 --cuda_ext 这个参数！
cd /tmp/bsr_src && /tmp/srenv/bin/python setup.py build_ext --inplace
```

**验证环境可用：**

```bash
/tmp/srenv/bin/python -c "
import sys, types, torchvision
m = types.ModuleType('torchvision.transforms.functional_tensor')
from torchvision.transforms import functional as F
for n in dir(F): setattr(m, n, getattr(F, n))
sys.modules['torchvision.transforms.functional_tensor'] = m
import torch, cv2, lpips
from basicsr.archs.edsr_arch import EDSR
print('OK', torch.__version__, torch.cuda.is_available(), cv2.__version__)
"
```

> `torchvision.transforms.functional_tensor` 这个 shim 是**必需的**：torchvision ≥ 0.17 删了该模块，而 BasicSR 仍在 import。三个脚本都在开头打了这个补丁。

---

## 运行

```bash
scp p1_part1_final.py p1_vimeo.py flow_gt.py ghe270@121.12.5.46:/tmp/

# 主评测，30 clip，A6000 上约 45 min
cd /tmp && /tmp/srenv/bin/python /tmp/p1_part1_final.py 30

# Vimeo，7824 clip，约 8 min
cd /tmp && /tmp/srenv/bin/python /tmp/p1_vimeo.py 94

# 光流方向标定（改流水线前先跑这个，确认约定没写反）
cd /tmp && /tmp/srenv/bin/python /tmp/flow_gt.py
```

**参数约定**（`p1_part1_final.py <LIMIT> <START>`）：第一个参数是 clip 数，第二个是起始下标。只跑最后一个参数就能快速冒烟测试。

---

## 三个脚本里最容易改错的地方

**① 光流方向（`p1_part1_final.py` 的 `fb_target_to_source`）**

```python
f = cv2.calcOpticalFlowFarneback(gray_T, gray_S, ...)   # 目标在前、源在后
warped = cv2.remap(S, gx + f[...,0], gy + f[...,1], ...)
```

写成 `fb(S, T)` 会让误差**上升 29%**。改 `flow_gt.py` 里的参数顺序重跑一遍就能自证。

**② SSIM 必须用 `ssim_gpu`，不要换回 skimage**

`ssim_gpu` 是对 `skimage.metrics.structural_similarity(win_size=7, gaussian_weights=False, use_sample_covariance=True)` 的精确复刻，实测在真实 720p 帧上误差 5e-14，速度快 30 倍（66 ms vs 1975 ms / 20 帧）。换回 skimage 会让整轮跑从 45 min 变成 3.5 h。

**③ 光流分辨率用 1/4（`FLOW_SCALE = 4`）**

全分辨率对齐效果反而更差（70.3% vs 86.6% 误差下降）且慢 58 倍。

---

## 对应产出

| 脚本 | 输出 |
|---|---|
| `p1_part1_final.py` | `metrics_part1_final.json`（10 方法 × 30 clip） |
| `p1_vimeo.py` | `metrics_vimeo.json`（7824 clip） |
| `flow_gt.py` | 仅标定，无输出文件 |
