# P1 本地复现环境（备选方案，已验证）

> 建立于 2026-11-14。目的：服务器磁盘/网络不可用时，能在本机 `NI-DEV` 上继续跑 P1 的 Part 2。
> **EDVR 已做到与服务器逐位等价（clip 000 PSNR 30.486 = 服务器 30.486）**；TecoGAN 有平台限制，见 §4。

## 1. 机器与关键结论

| 项 | 实测值 |
|---|---|
| GPU | NVIDIA RTX 5070，12 GB，**sm_120 (Blackwell)** |
| CPU / RAM | Intel Ultra 5 250KF，18 核 / 23.5 GB |
| torch | 2.11.0+**cu128**（arch list 含 sm_120） |
| EDVR 速度 | **0.28–0.30 s/帧**，30 clip ≈ **14 分钟**（比服务器 A6000 快一倍） |
| EDVR 显存 | 1.5 GB 峰值（12 GB 绰绰有余） |
| 环境体积 | 6.2 GB（D 盘余量 103 GB） |

**核心结论：本地能完全替代服务器跑 Part 2 的 EDVR 部分，而且更快。**

## 2. 三个必须知道的坑

### 2.1 没有 CUDA toolkit → 不能编译 DCN → 但根本不需要编译

服务器上 EDVR 依赖 `BASICSR_EXT=True` 编译出的 `deform_conv_ext`。本机没有 `nvcc`，
也没有 MSVC 构建工具，编译这条路走不通。

**解法**：`torchvision.ops.deform_conv2d` 自带预编译的可变形卷积 kernel，cu128 轮子已覆盖 sm_120。
写一个 shim 冒充 `deform_conv_ext` 转发即可（`shim\deform_conv_ext.py`）。

偏移量内存布局两边一致（都是每个 tap 交错 `(y, x)`），**但这点不是靠文档相信的**——
用同一输入在服务器（编译版）和本机（shim）各跑一次 EDVR，逐像素比对：

```
max abs diff : 1.035e-03
mean abs diff: 3.063e-05
> 1/255 的像素: 0 / 2764800
PSNR(server, local) = 85.37 dB
clip 000 全片段 PSNR: 本地 30.486 vs 服务器 30.486  (+0.000 dB)
```

差异纯属浮点累加顺序，**没有量化差异**。

### 2.2 RTX 5070 是 sm_120，普通 torch 轮子不支持

必须装 cu128 的 wheel：

```powershell
& $env:Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

装完必须验证 `sm_120` 在 `torch.cuda.get_arch_list()` 里，否则 kernel 会静默 fallback 到 CPU 或直接报错。

### 2.3 原生 Windows 上 TensorFlow 没有 GPU（影响 TecoGAN）

`tensorflow` 2.11 之后**不再提供原生 Windows 的 GPU 支持**，装 2.20 得到的是 `gpus []`。
TecoGAN 官方代码用的是 `tf.contrib`（TF1 专属），而 TF1.15 只有 **cp37** 的 Windows 轮子；
本机只有 3.11 / 3.12，`uv python install 3.7` 也拿不到 Windows 构建。

## 3. 怎么用（EDVR，已可用）

```powershell
$py = 'D:\workspace\p1_local\env\Scripts\python.exe'
# 1) 确认 basicsr 源码路径已挂上（装环境时写过 .pth，重装 venv 后需重做）
#    内容一行： D:\workspace\p1_local\basicsr_src\basicsr-1.4.2
#    shim 要覆盖到 basicsr/ops/dcn/deform_conv_ext.py（安装时已复制）
& $py -c "import sys,types,torch,torchvision; m=types.ModuleType('torchvision.transforms.functional_tensor'); from torchvision.transforms import functional as F; [setattr(m,n,getattr(F,n)) for n in dir(F)]; sys.modules['torchvision.transforms.functional_tensor']=m; from basicsr.archs.edvr_arch import EDVR; print('ok')"
```

`basicsr` 还缺几个非核心依赖（`addict/future/lmdb/yapf/tb-nightly`）。当前只用 `archs`
和 `ops`，不装也能跑；若要跑完整训练管线再补。

数据只需按需下载 —— 30 个 clip 全量约 **4.3 GB**（LR 10.9 MB + GT 145 MB 每 clip），
而 `val_sharp` 是 720p PNG，**只下要算的那几个 clip 即可**。

## 4. TecoGAN 的两条可行路线

本机能跑 EDVR，但 TecoGAN 在原生 Windows 上**跑不起来官方 TF1 代码**。两条路：

| 路线 | 做法 | 代价 |
|---|---|---|
| A. WSL2 | WSL 里装 Python 3.7 + TF 1.15，走 CUDA 直通 | 要装 WSL（本机已有 `Ubuntu`，见 `AGENTS.md`）；TF1.15 的 CUDA 是 10.x，需要 cuDNN 7.6，与 WSL 里的现代驱动兼容性要试 |
| B. PyTorch 重实现 | 换 `skycrapers/TecoGAN-PyTorch` | **权重是自己训的，不能声称复现官方 TecoGAN**，报告里必须注明 |

**实际建议：TecoGAN 留在服务器上跑。** 服务器那边已经跑完 30 clip（90 分钟），
结果已落盘归档。本地只接管 EDVR 就足够了 —— EDVR 是 Part 2 的主体（PSNR/SSIM/LPIPS 全线最好），
而且本地跑得快一倍。

## 5. 数据与权重

| 路径 | 内容 |
|---|---|
| `ckpt\EDVR_REDS.pth` | EDVR-L x4 REDS 官方权重，78.8 MB |
| `data\val_sharp_bicubic\000\` | clip 000 的 100 帧 LR（验证用） |
| `data\val_sharp\000\` | clip 000 的 100 帧 GT（验证用） |
| `shim\deform_conv_ext.py` | **关键文件**，torchvision 转接层，含完整注释 |

权重和验证数据来自服务器共享盘（只读）。重装环境时 `shim\deform_conv_ext.py` 必须重新复制到
`env\Lib\site-packages\` 下的 basicsr 目录，否则 EDVR 会报 `cannot import deform_conv_ext`。
