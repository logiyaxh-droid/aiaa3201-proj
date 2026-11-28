#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Part 2 step 2 -- metrics + deliverable videos for the TecoGAN output.

Reads /tmp/p2/tecogan/<clip>/*.png (written by p2_teco.sh), scores them with the
exact same metric code as Part 1 / the EDVR run, then optionally writes the
side-by-side comparison video required by the PDF ("submit the processed video
files for all Mandatory Datasets").

Idempotent: re-running only evaluates clips whose .done marker exists but that
are not yet in the JSON, so it can be called repeatedly while TecoGAN runs.
"""
import os
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import sys, types, json, glob, time
import numpy as np
import torch
import cv2

import torchvision
_mod = types.ModuleType('torchvision.transforms.functional_tensor')
from torchvision.transforms import functional as _F
for _n in dir(_F):
    setattr(_mod, _n, getattr(_F, _n))
sys.modules['torchvision.transforms.functional_tensor'] = _mod

import lpips

DEV = "cuda"
B = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT_ROOT = B + "/val_sharp"
TE = "/tmp/p2/tecogan"
OUT = "/tmp/p2"
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


def psnr_mean(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(10.0 * np.log10(255.0 ** 2 /
                 np.maximum((d * d).mean(axis=tuple(range(1, d.ndim))), 1e-12))))


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


def fb_target_to_source(tg, sg, scale=FLOW_SCALE):
    h, w = tg.shape
    if scale == 1:
        return cv2.calcOpticalFlowFarneback(tg, sg, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    ts = cv2.resize(tg, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    ss = cv2.resize(sg, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    f = cv2.calcOpticalFlowFarneback(ts, ss, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) * float(scale)


def warp_onto(src, tg, sg, gx, gy):
    f = fb_target_to_source(tg, sg)
    return cv2.remap(src, gx + f[..., 0], gy + f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def tlpips(frames, grays, gx, gy, stride=2):
    al, bl = [], []
    for i in range(1, len(frames), stride):
        al.append(warp_onto(frames[i - 1], grays[i], grays[i - 1], gx, gy))
        bl.append(frames[i])
    return lpips_mean(np.stack(al), np.stack(bl))


def read_rgb(p):
    a = cv2.imread(p)
    return None if a is None else cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def evaluate(arr, gt):
    H, W = gt[0].shape[:2]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    gtY = np.stack([cv2.cvtColor(g, cv2.COLOR_RGB2YCrCb)[:, :, 0] for g in gt])
    pY = np.stack([cv2.cvtColor(p, cv2.COLOR_RGB2YCrCb)[:, :, 0] for p in arr])
    gr = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in arr]
    return {"psnr": psnr_mean(pY, gtY), "ssim": float(ssim_gpu(pY, gtY).mean()),
            "lpips": lpips_mean(arr, np.stack(gt)), "tlpips": tlpips(arr, gr, gx, gy)}


def write_video(clip, methods, path, fps=10, max_frames=100):
    """4-up grid: GT | bicubic | EDVR | TecoGAN -- the figure the PDF asks for."""
    tiles = {m: sorted(glob.glob(v)) for m, v in methods.items()}
    n = min([len(v) for v in tiles.values()] + [max_frames])
    if n < 2:
        return None
    im = cv2.imread(tiles["gt"][0])
    h, w = im.shape[:2]
    sc = 640.0 / w                      # keep the file small
    hh, ww = int(h * sc) // 2 * 2, 640
    cols = len(tiles)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ww * cols, hh))
    for k in range(n):
        row = []
        for m in methods:
            a = cv2.imread(tiles[m][k])
            if a is None:
                a = np.zeros((h, w, 3), np.uint8)
            a = cv2.resize(a, (ww, hh), interpolation=cv2.INTER_AREA)
            cv2.putText(a, m, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            row.append(a)
        vw.write(np.hstack(row))
    vw.release()
    return path


if __name__ == "__main__":
    do_video = "--video" in sys.argv
    f = os.path.join(OUT, "metrics_tecogan.json")
    state = json.load(open(f)) if os.path.exists(f) else {"method": "tecogan", "per_clip": {}}
    per = state["per_clip"]

    clips = sorted(d for d in os.listdir(TE) if os.path.isdir(os.path.join(TE, d)))
    todo = [c for c in clips if c not in per and os.path.exists(os.path.join(TE, c, ".done"))]
    print("tecogan done-clips=%d cached=%d todo=%d" % (len(clips), len(per), len(todo)), flush=True)

    for clip in todo:
        t0 = time.time()
        gtf = sorted(glob.glob(os.path.join(GT_ROOT, clip, "*.png")))
        gt = [read_rgb(p) for p in gtf]
        arr = [read_rgb(os.path.join(TE, clip, os.path.basename(p))) for p in gtf]
        keep = [i for i, a in enumerate(arr) if a is not None and a.shape[:2] == gt[i].shape[:2]]
        if len(keep) < len(gt):
            print("  clip %s: using %d/%d frames (missing/short)" % (clip, len(keep), len(gt)), flush=True)
        row = evaluate([arr[i] for i in keep], [gt[i] for i in keep])
        per[clip] = row
        print("[%s] %4.0fs psnr %.2f ssim %.4f lpips %.4f tlpips %.4f" %
              (clip, time.time() - t0, row["psnr"], row["ssim"], row["lpips"], row["tlpips"]), flush=True)
        json.dump(state, open(f, "w"), indent=2)

    if per:
        keys = ["psnr", "ssim", "lpips", "tlpips"]
        avg = {k: float(np.mean([v[k] for v in per.values()])) for k in keys}
        state["avg"] = avg
        state["n_clips"] = len(per)
        json.dump(state, open(f, "w"), indent=2)
        print("\n===== TecoGAN (CPU) | %d clips =====" % len(per), flush=True)
        print("%-8s %8s %8s %8s %8s" % ("method", "PSNR", "SSIM", "LPIPS", "tLPIPS"), flush=True)
        print("%-8s %8.3f %8.5f %8.4f %8.4f" % ("tecogan", avg["psnr"], avg["ssim"],
                                                avg["lpips"], avg["tlpips"]), flush=True)
