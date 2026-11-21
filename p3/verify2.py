"""Load the official BasicVSR++ (REDS4, c64n7) checkpoint into the BasicSR arch
that is already installed in /tmp/bsr_src, then score one REDS clip end-to-end.

Three name differences between mmediting and BasicSR are bridged:
  1. generator. * prefix on every tensor            -> strip
  2. upsampleN.upsample_conv  (conv 64->256, then     -> upconvN (identical shapes;
     F.pixel_shuffle inside the module)                 arch applies pixel_shuffle)
  3. spynet.basic_module.L.basic_module.I.conv.W      -> spynet.basic_module.L.basic_module.{2I}.W
     (mmediting keeps the conv nested one level deeper)  loaded separately as spynet.pth
"""
import os
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import sys, types, glob, time, json, re
import numpy as np
import torch
import cv2

import torchvision
_m = types.ModuleType('torchvision.transforms.functional_tensor')
from torchvision.transforms import functional as F
for _n in dir(F):
    setattr(_m, _n, getattr(F, _n))
sys.modules['torchvision.transforms.functional_tensor'] = _m

from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus
from basicsr.archs.spynet_arch import SpyNet

torch.backends.cudnn.benchmark = True

B = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT_ROOT = B + "/val_sharp"
LR_ROOT = B + "/val_sharp_bicubic"
CK = "/data/disk2/home/ghe270/p3/ckpt/bvsrpp_reds4.pth"
SPY = "/data/disk2/home/ghe270/p3/ckpt/spynet_bsr.pth"
CLIP = sys.argv[1] if len(sys.argv) > 1 else "000"


def build_spynet():
    sd = torch.load(CK, map_location="cpu", weights_only=False)["state_dict"]
    pat = re.compile(r"^generator\.spynet\.basic_module\.(\d+)\.basic_module\.(\d+)\.conv\.(weight|bias)$")
    out = {}
    for k, v in sd.items():
        mm = pat.match(k)
        if mm:
            lvl, idx, p = mm.groups()
            out["basic_module.%s.basic_module.%d.%s" % (lvl, int(idx) * 2, p)] = v
    torch.save({"params": out}, SPY)
    return len(out)


if not os.path.exists(SPY):
    print("extracting SPyNet from the generator: %d tensors" % build_spynet(), flush=True)


def psnr_y(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(10.0 * np.log10(255.0 ** 2 / np.maximum((d * d).mean(axis=(1, 2)), 1e-12))))


t0 = time.time()
sd = torch.load(CK, map_location="cpu", weights_only=False)["state_dict"]
new = {}
for k, v in sd.items():
    if k == "step_counter":
        continue
    k2 = k[len("generator."):] if k.startswith("generator.") else k
    if k2.startswith("spynet."):
        continue
    k2 = k2.replace("upsample1.upsample_conv", "upconv1").replace("upsample2.upsample_conv", "upconv2")
    new[k2] = v

m = BasicVSRPlusPlus(mid_channels=64, num_blocks=7, is_low_res_input=True, spynet_path=SPY)
miss, unexp = m.load_state_dict(new, strict=False)
miss = [x for x in miss if not x.startswith("spynet.")]
print("missing:", miss, flush=True)
print("unexpected:", unexp, flush=True)
print("params M %.2f | loaded in %.1fs" % (sum(p.numel() for p in m.parameters()) / 1e6, time.time() - t0), flush=True)
m = m.cuda().eval()
for p in m.parameters():
    p.requires_grad_(False)

lr_files = sorted(glob.glob(os.path.join(LR_ROOT, CLIP, "*.png")))
gt_files = sorted(glob.glob(os.path.join(GT_ROOT, CLIP, "*.png")))
n = len(lr_files)
lq = np.stack([cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in lr_files])
gt = np.stack([cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in gt_files])
x = torch.from_numpy(lq).float().div_(255.0).permute(0, 3, 1, 2).unsqueeze(0).cuda()

t0 = time.time()
with torch.no_grad():
    y = m(x)
dt = time.time() - t0
out = y[0].clamp_(0, 1).mul_(255).round_().byte().permute(0, 2, 3, 1).cpu().numpy()
py = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb)[:, :, 0] for f in out])
gy = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb)[:, :, 0] for f in gt])
print("clip %s frames=%d out=%s inference %.1fs (%.3f s/frame)" % (CLIP, n, out.shape, dt, dt / n), flush=True)
print("PSNR(Y) %.3f dB" % psnr_y(py, gy), flush=True)
print("peak mem %.2f GB" % (torch.cuda.max_memory_allocated() / 2 ** 30), flush=True)
print("OK", flush=True)
