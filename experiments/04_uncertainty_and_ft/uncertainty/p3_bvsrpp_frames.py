"""Part 3 / Direction C -- dump BasicVSR++ frames for the clips we already have
EDVR frames for, so the two branches can be compared pixel-by-pixel offline.

Kept deliberately narrow: 7 clips x 100 frames at 720p is ~1.4 GB, and every
later experiment (uncertainty calibration, routing, fusion) reuses these dumps
instead of paying for inference again.
"""
import os
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import sys, types, glob, time, re
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
LR_ROOT = B + "/val_sharp_bicubic"
GT_ROOT = B + "/val_sharp"
CK = "/data/disk2/home/ghe270/p3/ckpt/bvsrpp_reds4.pth"
SPY = "/data/disk2/home/ghe270/p3/ckpt/spynet_bsr.pth"
OUT = "/data/disk2/home/ghe270/p3/bvsrpp_frame"


def build_spynet():
    sd = torch.load(CK, map_location="cpu", weights_only=False)["state_dict"]
    pat = re.compile(r"^generator\.spynet\.basic_module\.(\d+)\.basic_module\.(\d+)\.conv\.(weight|bias)$")
    out = {}
    for k, v in sd.items():
        mm = pat.match(k)
        if mm:
            lvl, idx, p = mm.groups()
            out["basic_module.%s.basic_module.%d.%s" % (lvl, int(idx) * 2, p)] = v
    os.makedirs(os.path.dirname(SPY), exist_ok=True)
    torch.save({"params": out}, SPY)


if not os.path.exists(SPY):
    build_spynet()

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

model = BasicVSRPlusPlus(mid_channels=64, num_blocks=7, is_low_res_input=True, spynet_path=SPY)
miss, unexp = model.load_state_dict(new, strict=False)
assert not [x for x in miss if not x.startswith("spynet.")], miss
assert not unexp, unexp
model = model.cuda().eval()
for p in model.parameters():
    p.requires_grad_(False)

CHUNK = 40  # 100 frames at 720p peaks at 23 GB; keep the footprint neighbourly


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for clip in (sys.argv[1:] or ["000"]):
        d = os.path.join(OUT, clip)
        if os.path.exists(os.path.join(d, ".done")):
            print("skip", clip, flush=True)
            continue
        os.makedirs(d, exist_ok=True)
        files = sorted(glob.glob(os.path.join(LR_ROOT, clip, "*.png")))
        n = len(files)
        lq = np.stack([cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files])
        t0 = time.time()
        for a in range(0, n, CHUNK):
            b = min(n, a + CHUNK)
            x = torch.from_numpy(lq[a:b]).float().div_(255.0).permute(0, 3, 1, 2).unsqueeze(0).cuda()
            with torch.no_grad():
                y = model(x)
            o = y[0].clamp_(0, 1).mul_(255).round_().byte().permute(0, 2, 3, 1).cpu().numpy()
            for i in range(a, b):
                cv2.imwrite(os.path.join(d, "%08d.png" % i),
                            cv2.cvtColor(o[i - a], cv2.COLOR_RGB2BGR))
            del x, y, o
            torch.cuda.empty_cache()
        open(os.path.join(d, ".done"), "w").write("ok")
        print("%s %d frames %.0fs" % (clip, n, time.time() - t0), flush=True)
    print("done ->", OUT, flush=True)
