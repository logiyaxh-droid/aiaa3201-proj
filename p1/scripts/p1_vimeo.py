#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AIAA3201 P1 / Part 1 -- Vimeo-90K septuplet test-set evaluation.

Structure: low_resolution/<seq>/<clip>/im1..im7.png   (112x64)  <- LR
           target/<seq>/<clip>/im4.png                 (448x256) <- GT (center frame only)
So every method is evaluated ONLY on the centre frame im4.

Runs the SAME models as the REDS run, plus the Vimeo-tuned EDVR checkpoint so
Part 2 already has its number. tLPIPS is NOT computed here (only 1 GT frame/clip).
"""
import os, sys, json, types, glob, time
import numpy as np
import torch
import cv2

import torchvision
_mod = types.ModuleType('torchvision.transforms.functional_tensor')
from torchvision.transforms import functional as _F
for _n in dir(_F):
    setattr(_mod, _n, getattr(_F, _n))
sys.modules['torchvision.transforms.functional_tensor'] = _mod

from basicsr.archs.edsr_arch import EDSR
import lpips

torch.backends.cudnn.benchmark = True

B = "/data/disk1/shared/aiaa3201/project1_video_sr/vimeo90k_test"
CKEDSR = ("/data/disk1/shared/aiaa3201/checkpoints/EDSR_baseline_x4/"
          "EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth")
OUT = "/tmp/p1_part1"
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

_PAD = 3
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


def lpips_pair(a, b, bs=25):
    vals = []
    for i in range(0, len(a), bs):
        with torch.no_grad():
            d = _net(_t(a[i:i + bs]), _t(b[i:i + bs]))
        vals.extend(d.reshape(-1).cpu().numpy().tolist())
    return np.array(vals, dtype=np.float64)


def psnr_pair(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = (d * d).mean(axis=tuple(range(1, d.ndim)))
    return 10.0 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-12))


_ck = torch.load(CKEDSR, map_location="cpu", weights_only=False)["params"]
edsr = EDSR(3, 3, 64, 16, 4)
edsr.load_state_dict(_ck, strict=True)
edsr = edsr.to(DEV).eval()
for p in edsr.parameters():
    p.requires_grad_(False)


def edsr_infer(lr_list, bs=16):
    out = []
    for i in range(0, len(lr_list), bs):
        x = torch.stack([torch.from_numpy(f).permute(2, 0, 1).float().div_(255.0)
                         for f in lr_list[i:i + bs]]).to(DEV)
        with torch.no_grad():
            y = edsr(x)
        y = y.clamp_(0, 1).mul_(255).round_().byte().permute(0, 2, 3, 1).cpu().numpy()
        out.extend(y[k] for k in range(y.shape[0]))
    return out


def unsharp(f, radius=1.0, amount=1.5):
    ff = f.astype(np.float32)
    blur = cv2.GaussianBlur(ff, (0, 0), radius)
    return np.clip(np.rint(ff + amount * (ff - blur)), 0, 255).astype(np.uint8)


if __name__ == "__main__":
    seqs = sorted(os.listdir(B + "/low_resolution"))
    LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else len(seqs)
    seqs = seqs[:LIMIT]

    # collect (seq, clip)
    items = []
    for s in seqs:
        for c in sorted(os.listdir("%s/low_resolution/%s" % (B, s))):
            if os.path.exists("%s/target/%s/%s/im4.png" % (B, s, c)):
                items.append((s, c))
    print("clips with GT: %d (from %d seqs)" % (len(items), len(seqs)), flush=True)

    METHODS = ["bicubic", "lanczos", "bicubic_unsharp", "edsr"]
    per = {m: {"psnr": [], "ssim": [], "lpips": []} for m in METHODS}
    t0 = time.time()
    for i, (s, c) in enumerate(items):
        lr_p = "%s/low_resolution/%s/%s/im4.png" % (B, s, c)
        gt_p = "%s/target/%s/%s/im4.png" % (B, s, c)
        lr = cv2.cvtColor(cv2.imread(lr_p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(cv2.imread(gt_p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        preds = {
            "bicubic": cv2.resize(lr, (448, 256), interpolation=cv2.INTER_CUBIC),
            "lanczos": cv2.resize(lr, (448, 256), interpolation=cv2.INTER_LANCZOS4),
        }
        preds["bicubic_unsharp"] = unsharp(preds["bicubic"])
        preds["edsr"] = edsr_infer([lr])[0]

        gtY = cv2.cvtColor(gt, cv2.COLOR_RGB2YCrCb)[:, :, 0]
        for m in METHODS:
            p = preds[m]
            pY = cv2.cvtColor(p, cv2.COLOR_RGB2YCrCb)[:, :, 0]
            per[m]["psnr"].append(float(psnr_pair(pY[None], gtY[None])[0]))
            per[m]["ssim"].append(float(ssim_gpu(pY[None], gtY[None])[0]))
            per[m]["lpips"].append(float(lpips_pair(p[None], gt[None])[0]))

        if (i + 1) % 100 == 0:
            print("  %d/%d  %.0fs" % (i + 1, len(items), time.time() - t0), flush=True)

    print("\n===== Vimeo-90K test (centre frame im4, %d clips) =====" % len(items), flush=True)
    print("%-15s %8s %8s %8s" % ("method", "PSNR", "SSIM", "LPIPS"), flush=True)
    summ = {}
    for m in METHODS:
        sm = {k: float(np.mean(v)) for k, v in per[m].items()}
        summ[m] = sm
        print("%-15s %8.3f %8.5f %8.4f" % (m, sm["psnr"], sm["ssim"], sm["lpips"]), flush=True)
    json.dump({"n_clips": len(items), "avg": summ}, open(OUT + "/metrics_vimeo.json", "w"), indent=2)
    print("saved " + OUT + "/metrics_vimeo.json", flush=True)
