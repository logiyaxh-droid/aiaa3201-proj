#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TecoGAN metrics for the WILD clips. Same metric code as everything else.

Reads the frames p2_teco_wild.sh produced and writes metrics_tecogan_wild.json.
usage: python p2_teco_metrics_wild.py GT_ROOT OUT
"""
import os
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import sys, glob, json
import numpy as np
import cv2
import torch
import torchvision
import types

_mod = types.ModuleType('torchvision.transforms.functional_tensor')
from torchvision.transforms import functional as _F
for _n in dir(_F):
    setattr(_mod, _n, getattr(_F, _n))
sys.modules['torchvision.transforms.functional_tensor'] = _mod

import lpips

DEV = "cuda"
FLOW_SCALE = 4
_K = torch.ones(1, 1, 7, 7, dtype=torch.float64, device=DEV) / 49.0
_COV = 49.0 / 48.0


def ssim_gpu(a_u8, b_u8, data_range=255.0):
    a = torch.from_numpy(np.ascontiguousarray(a_u8)).to(DEV).double().unsqueeze(1)
    b = torch.from_numpy(np.ascontiguousarray(b_u8)).to(DEV).double().unsqueeze(1)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    f = lambda x: torch.nn.functional.conv2d(x, _K)
    ux, uy = f(a), f(b)
    uxx = _COV * (f(a * a) - ux * ux)
    uyy = _COV * (f(b * b) - uy * uy)
    uxy = _COV * (f(a * b) - ux * uy)
    m = ((2 * ux * uy + C1) * (2 * uxy + C2)) / ((ux ** 2 + uy ** 2 + C1) * (uxx + uyy + C2))
    return m.mean(dim=(1, 2, 3)).cpu().numpy()


_net = lpips.LPIPS(net="alex").to(DEV).eval()
for p in _net.parameters():
    p.requires_grad_(False)


def _t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).to(DEV).float().div_(127.5).sub_(1.0).permute(0, 3, 1, 2)


def lpips_mean(a, b, bs=25):
    vals = []
    for i in range(0, len(a), bs):
        with torch.no_grad():
            d = _net(_t(a[i:i + bs]), _t(b[i:i + bs]))
        vals.extend(d.reshape(-1).cpu().numpy().tolist())
    return float(np.mean(vals))


def psnr_mean(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(10.0 * np.log10(255.0 ** 2 /
                 np.maximum((d * d).mean(axis=tuple(range(1, d.ndim))), 1e-12))))


def fb_target_to_source(target_gray, source_gray, scale=FLOW_SCALE):
    h, w = target_gray.shape
    ts = cv2.resize(target_gray, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    ss = cv2.resize(source_gray, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    f = cv2.calcOpticalFlowFarneback(ts, ss, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) * float(scale)


def warp_onto(src, target_gray, src_gray, gx, gy):
    f = fb_target_to_source(target_gray, src_gray)
    return cv2.remap(src, gx + f[..., 0], gy + f[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def tlpips(frames, grays, gx, gy, stride=2):
    al, bl = [], []
    for i in range(1, len(frames), stride):
        al.append(warp_onto(frames[i - 1], grays[i], grays[i - 1], gx, gy))
        bl.append(frames[i])
    return lpips_mean(np.stack(al), np.stack(bl))


if __name__ == "__main__":
    GT_ROOT, OUT = sys.argv[1], sys.argv[2]
    per = {}
    clips = sorted(d for d in os.listdir(GT_ROOT) if os.path.isdir(os.path.join(GT_ROOT, d)))
    for clip in clips:
        gtf = sorted(glob.glob(os.path.join(GT_ROOT, clip, "*.png")))
        tef = sorted(glob.glob(os.path.join(OUT, clip, "*.png")))
        tef = [f for f in tef if os.path.basename(f)[0].isdigit()]
        n = min(len(gtf), len(tef))
        if n < 10:
            print("skip", clip, len(gtf), len(tef))
            continue
        gt = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in gtf[:n]]
        te = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in tef[:n]]
        H, W = gt[0].shape[:2]
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        gtY = np.stack([cv2.cvtColor(g, cv2.COLOR_RGB2YCrCb)[:, :, 0] for g in gt])
        pY = np.stack([cv2.cvtColor(p, cv2.COLOR_RGB2YCrCb)[:, :, 0] for p in te])
        gr = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in te]
        row = {"psnr": psnr_mean(pY, gtY), "ssim": float(ssim_gpu(pY, gtY).mean()),
               "lpips": lpips_mean(te, np.stack(gt)), "tlpips": tlpips(te, gr, gx, gy)}
        per[clip] = row
        print("%s psnr %.2f ssim %.4f lpips %.4f tlpips %.4f" %
              (clip, row["psnr"], row["ssim"], row["lpips"], row["tlpips"]), flush=True)
        json.dump({"method": "tecogan", "per_clip": per},
                  open(os.path.join(OUT, "metrics_tecogan_wild.json"), "w"), indent=2)
    av = {k: float(np.mean([r[k] for r in per.values()])) for k in ("psnr", "ssim", "lpips", "tlpips")}
    print("\n===== TecoGAN wild =====")
    for k, v in av.items():
        print("%-8s %.4f" % (k, v))
    json.dump({"method": "tecogan", "avg": av, "per_clip": per},
              open(os.path.join(OUT, "metrics_tecogan_wild.json"), "w"), indent=2)
