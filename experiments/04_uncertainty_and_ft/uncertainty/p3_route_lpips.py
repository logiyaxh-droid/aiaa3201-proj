"""Part 3 / Direction C -- router evaluated on the axis it can actually win.

The PSNR probe (p3_route_probe.py) showed the shape of the problem:

  * BasicVSR++ wins PSNR by ~0.9 dB over EDVR, so a router whose *destination*
    is the weaker fidelity model can only lose on PSNR. The oracle bound
    (+1.3 dB) is real but unreachable by that construction.
  * "Model disagreement" is nevertheless a genuine error proxy
    (Spearman 0.40-0.51 against |prediction - GT|), which is the one thing
    Direction C actually needs.

So the router has to point at the *generative* branch, and it has to be scored
on perception, not on PSNR. This script does that:

  fidelity branch   BasicVSR++ (REDS4 official)      -- steady, sometimes soft
  generative branch Real-ESRGAN x4                   -- sharp textures, invents
  uncertainty       mean |Y_fid - Y_gen|, per pixel   -- free, 2 forward passes
  fusion            Y  := where(U > q) generative else fidelity
                    CbCr := fidelity (colour drift from the GAN is not the point)

and reports LPIPS / tLPIPS of the fused video against q, alongside both
single-model baselines. If any q beats both branches on LPIPS while losing less
than ~0.3 dB PSNR, Direction C has its headline result.
"""
import os
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import sys, types, glob, json
import numpy as np
import torch
import cv2

import torchvision
_m = types.ModuleType('torchvision.transforms.functional_tensor')
from torchvision.transforms import functional as F
for _n in dir(F):
    setattr(_m, _n, getattr(F, _n))
sys.modules['torchvision.transforms.functional_tensor'] = _m

import lpips

GT_ROOT = "/data/disk1/shared/aiaa3201/project1_video_sr/val_sharp"
BVSR = "/data/disk2/home/ghe270/p3/bvsrpp_frame"
GEN = "/data/disk2/home/ghe270/p3/gen"
OUT = "/data/disk2/home/ghe270/p3"

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


def fb_to_src(ts, ss, scale=4):
    h, w = ts.shape
    f = cv2.calcOpticalFlowFarneback(cv2.resize(ts, (w // scale, h // scale), interpolation=cv2.INTER_AREA),
                                     cv2.resize(ss, (w // scale, h // scale), interpolation=cv2.INTER_AREA),
                                     None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) * float(scale)


def tlpips(frames, gx, gy):
    """LPIPS(warp(prev -> cur), cur), the Part 1/2 definition, no GT required."""
    gr = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    a, b = [], []
    for i in range(1, len(frames), 2):
        f = fb_to_src(gr[i], gr[i - 1])
        a.append(cv2.remap(frames[i - 1], gx + f[..., 0], gy + f[..., 1],
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        b.append(frames[i])
    return lpips_mean(np.stack(a), np.stack(b))


def psnr_y(py, gy):
    d = py.astype(np.float64) - gy.astype(np.float64)
    return float(np.mean(10.0 * np.log10(255.0 ** 2 / np.maximum((d * d).mean(axis=(1, 2)), 1e-12))))


def load(clip):
    g = sorted(glob.glob(os.path.join(GT_ROOT, clip, "*.png")))
    f = sorted(glob.glob(os.path.join(BVSR, clip, "*.png")))
    e = sorted(glob.glob(os.path.join(GEN, clip, "*.png")))
    n = min(len(g), len(f), len(e))
    return ([cv2.cvtColor(cv2.imread(g[i]), cv2.COLOR_BGR2RGB) for i in range(n)],
            [cv2.cvtColor(cv2.imread(f[i]), cv2.COLOR_BGR2RGB) for i in range(n)],
            [cv2.cvtColor(cv2.imread(e[i]), cv2.COLOR_BGR2RGB) for i in range(n)], n)


if __name__ == "__main__":
    clips = sys.argv[1:] or ["000"]
    res = {}
    for clip in clips:
        G, Fid, Gen, n = load(clip)
        if n < 10:
            print("skip", clip, "(need generated frames)", flush=True)
            continue
        H, W = G[0].shape[:2]
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

        def score(name, frames):
            py = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb)[:, :, 0] for f in frames])
            gy_ = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb)[:, :, 0] for f in G])
            return dict(method=name, psnr=round(psnr_y(py, gy_), 3),
                        lpips=round(lpips_mean(np.stack(frames), np.stack(G)), 4),
                        tlpips=round(tlpips(frames, gx, gy), 4))

        rows = [score("basicvsr++", Fid), score("realesrgan", Gen)]

        # per-pixel disagreement, computed once, reused for every q
        U = [np.abs(cv2.cvtColor(Fid[i], cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.int16) -
                    cv2.cvtColor(Gen[i], cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.int16)) for i in range(n)]
        for q in (0.99, 0.97, 0.95, 0.90, 0.80, 0.60):
            fused = []
            for i in range(n):
                m = U[i] > np.quantile(U[i], q)
                y = np.where(m, cv2.cvtColor(Gen[i], cv2.COLOR_RGB2YCrCb)[:, :, 0],
                             cv2.cvtColor(Fid[i], cv2.COLOR_RGB2YCrCb)[:, :, 0])
                cc = cv2.cvtColor(Fid[i], cv2.COLOR_RGB2YCrCb)
                cc[:, :, 0] = y
                fused.append(cv2.cvtColor(cc, cv2.COLOR_YCrCb2RGB))
            s = score("route_q%.2f" % q, fused)
            s["routed"] = round(1.0 - q, 3)
            rows.append(s)

        res[clip] = rows
        print("\n== clip %s ==" % clip, flush=True)
        for r in rows:
            print("  %-14s PSNR %7.3f  LPIPS %.4f  tLPIPS %.4f%s"
                  % (r["method"], r["psnr"], r["lpips"], r["tlpips"],
                     "" if "routed" not in r else "  (routed %.0f%%)" % (r["routed"] * 100)), flush=True)

    json.dump(res, open(os.path.join(OUT, "route_lpips.json"), "w"), indent=2)
    print("\nOK -> route_lpips.json", flush=True)
