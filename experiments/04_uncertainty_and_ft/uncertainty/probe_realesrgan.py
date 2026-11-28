"""Feasibility probe for the generative branch of Direction C.

RRDBNet (BasicSR arch, already installed) + RealESRGAN x4 weights pulled from
hf-mirror.com (github releases stall at ~20 kB/s from this box).
Reports PSNR/LPIPS against GT so the hallucination cost is measured, not assumed.
"""
import os
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import sys, types, glob, time
import numpy as np
import torch
import cv2

import torchvision
_m = types.ModuleType('torchvision.transforms.functional_tensor')
from torchvision.transforms import functional as F
for _n in dir(F):
    setattr(_m, _n, getattr(F, _n))
sys.modules['torchvision.transforms.functional_tensor'] = _m

from basicsr.archs.rrdbnet_arch import RRDBNet
import lpips

B = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT_ROOT = B + "/val_sharp"
LR_ROOT = B + "/val_sharp_bicubic"
OUT = "/data/disk2/home/ghe270/p3/gen"
os.makedirs(OUT, exist_ok=True)

lp = lpips.LPIPS(net="alex").cuda().eval()
for p in lp.parameters():
    p.requires_grad_(False)


def _t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).cuda().float().div_(127.5).sub_(1.0).permute(0, 3, 1, 2)


def lpips_mean(a, b, bs=25):
    v = []
    for i in range(0, len(a), bs):
        with torch.no_grad():
            d = lp(_t(a[i:i + bs]), _t(b[i:i + bs]))
        v.extend(d.reshape(-1).cpu().numpy().tolist())
    return float(np.mean(v))


def psnr_y(py, gy):
    d = py.astype(np.float64) - gy.astype(np.float64)
    return float(np.mean(10.0 * np.log10(255.0 ** 2 / np.maximum((d * d).mean(axis=(1, 2)), 1e-12))))


def build(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "params_ema" in ck:
        sd = ck["params_ema"]
    elif isinstance(ck, dict) and "params" in ck:
        sd = ck["params"]
    else:
        sd = ck
    m = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    m.load_state_dict(sd, strict=True)
    return m.cuda().eval()


def run(m, clip, nframes, keep=False):
    lrf = sorted(glob.glob(os.path.join(LR_ROOT, clip, "*.png")))[:nframes]
    gtf = sorted(glob.glob(os.path.join(GT_ROOT, clip, "*.png")))[:nframes]
    outs, gts = [], []
    t0 = time.time()
    for f, g in zip(lrf, gtf):
        lr = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt = cv2.cvtColor(cv2.imread(g), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(lr).permute(2, 0, 1).unsqueeze(0).cuda()
        with torch.no_grad():
            y = m(x)
        o = y[0].clamp_(0, 1).mul_(255).round_().byte().permute(1, 2, 0).cpu().numpy()
        if o.shape != gt.shape:
            o = cv2.resize(o, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        outs.append(o)
        gts.append(gt)
    dt = time.time() - t0
    oy = np.stack([cv2.cvtColor(o, cv2.COLOR_RGB2YCrCb)[:, :, 0] for o in outs])
    gy = np.stack([cv2.cvtColor(g, cv2.COLOR_RGB2YCrCb)[:, :, 0] for g in gts])
    r = dict(clip=clip, frames=len(outs), psnr=psnr_y(oy, gy),
             lpips=lpips_mean(np.stack(outs), np.stack(gts)), sec_per_frame=dt / len(outs))
    print("%s frames=%d PSNR(Y) %.3f LPIPS %.4f %.3f s/frame"
          % (clip, r["frames"], r["psnr"], r["lpips"], r["sec_per_frame"]), flush=True)
    if keep:
        d = os.path.join(OUT, clip)
        os.makedirs(d, exist_ok=True)
        for i, o in enumerate(outs):
            cv2.imwrite(os.path.join(d, "%08d.png" % i), cv2.cvtColor(o, cv2.COLOR_RGB2BGR))
    return r


if __name__ == "__main__":
    p = "/data/disk2/home/ghe270/p3/ckpt/RealESRGAN_x4_ai-forever.pth"
    m = build(p)
    print("params M %.2f" % (sum(q.numel() for q in m.parameters()) / 1e6), flush=True)
    keep = os.environ.get("KEEP", "0") == "1"
    for clip in (sys.argv[1:] or ["000"]):
        run(m, clip, 100 if keep else 12, keep=keep)
    del m
    torch.cuda.empty_cache()
    print("OK", flush=True)
