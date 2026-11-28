"""Direction C, second cut: soft confidence-weighted fusion instead of hard routing.

The hard-router probe (route_lean.json) lost on every axis, and the reason is
mechanical: Real-ESRGAN's bicubic-degradation output is 7.9 dB below BasicVSR++
and even its LPIPS is worse, so *replacing* luma with it can only hurt. A router
that swaps whole pixels therefore has no headroom here -- the destination is
simply dominated.

Soft fusion keeps the fidelity branch as the base and only injects a fraction of
the generative branch's high-frequency residual, gated by disagreement:

    Y = Y_fid + alpha * g * (Y_gen - Y_fid),      g = smoothstep((U - lo)/(hi - lo))
    CbCr = fidelity

alpha in {0.15, 0.3, 0.5}, gate width from the disagreement quantiles. This is
the form Direction C actually implies ("use the generative model for uncertain
textures"), and it degrades gracefully: alpha = 0 recovers BasicVSR++ exactly.

Reports PSNR(Y)/LPIPS/tLPIPS per (alpha, gate) on one clip.
Run: python p3_soft_fuse.py 000
"""
import os
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import sys, glob, json
import numpy as np
import torch
import cv2

import lpips

GT_ROOT = "/data/disk1/shared/aiaa3201/project1_video_sr/val_sharp"
BVSR = "/data/disk2/home/ghe270/p3/bvsrpp_frame"
GEN = "/data/disk2/home/ghe270/p3/gen"
OUT = "/data/disk2/home/ghe270/p3/soft_fuse.json"

ALPHAS = (0.15, 0.30, 0.50)
GATES = (0.90, 0.98)   # gate starts at this quantile of the disagreement map

net = lpips.LPIPS(net="alex").cuda().eval()
for p in net.parameters():
    p.requires_grad_(False)


def to_t(bgr):
    return torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).cuda().float() \
        .div_(127.5).sub_(1.0).permute(2, 0, 1).unsqueeze(0)


def lp1(a, b):
    with torch.no_grad():
        return float(net(to_t(a), to_t(b)).reshape(-1)[0])


def ly(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]


def psnr1(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return 10.0 * np.log10(255.0 ** 2 / max((d * d).mean(), 1e-12))


def flow_prev_to_cur(prev, cur, scale=4):
    h, w = prev.shape
    f = cv2.calcOpticalFlowFarneback(
        cv2.resize(cur, (w // scale, h // scale), interpolation=cv2.INTER_AREA),
        cv2.resize(prev, (w // scale, h // scale), interpolation=cv2.INTER_AREA),
        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) * float(scale)


def run_clip(clip):
    gtf = sorted(glob.glob(os.path.join(GT_ROOT, clip, "*.png")))
    bvf = sorted(glob.glob(os.path.join(BVSR, clip, "*.png")))
    gnf = sorted(glob.glob(os.path.join(GEN, clip, "*.png")))
    n = min(len(gtf), len(bvf), len(gnf))
    combos = [("fidelity", 0.0, 0.0), ("generative", 1.0, 0.0)]
    combos += [("soft_a%.2f_g%.2f" % (a, g), a, g) for a in ALPHAS for g in GATES]
    acc = {k: dict(psnr=0.0, lp=0.0, tlp=0.0, tn=0) for k, _, _ in combos}
    prev, gxgy = {}, None

    for i in range(n):
        gt = cv2.imread(gtf[i])
        fid = cv2.imread(bvf[i])
        gen = cv2.imread(gnf[i])
        if gxgy is None:
            H, W = gt.shape[:2]
            gxgy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        gx, gy = gxgy

        fy = ly(fid).astype(np.float32)
        gey = ly(gen).astype(np.float32)
        res = gey - fy
        U = np.abs(res)
        ycc = cv2.cvtColor(fid, cv2.COLOR_BGR2YCrCb)
        gty = ly(gt)

        frame = {"fidelity": fid, "generative": gen}
        for name, a, gq in combos:
            if name in ("fidelity", "generative"):
                continue
            lo = np.quantile(U, gq)
            hi = max(lo + 1e-3, float(U.max()))
            g = np.clip((U - lo) / (hi - lo), 0.0, 1.0)
            g = g * g * (3.0 - 2.0 * g)          # smoothstep, avoids blocky seams
            cc = ycc.copy()
            cc[:, :, 0] = np.clip(fy + a * g * res, 0, 255).round().astype(np.uint8)
            frame[name] = cv2.cvtColor(cc, cv2.COLOR_YCrCb2BGR)

        for k, f in frame.items():
            A = acc[k]
            A["psnr"] += psnr1(ly(f), gty)
            A["lp"] += lp1(f, gt)
            if k in prev:
                fl = flow_prev_to_cur(ly(prev[k]), ly(f))
                warped = cv2.remap(prev[k], gx + fl[..., 0], gy + fl[..., 1],
                                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                A["tlp"] += lp1(warped, f)
                A["tn"] += 1
            prev[k] = f
        if i % 25 == 0:
            print("  %s [%3d/%d]" % (clip, i, n), flush=True)

    return [dict(method=k, psnr=round(v["psnr"] / n, 3), lpips=round(v["lp"] / n, 4),
                 tlpips=round(v["tlp"] / max(1, v["tn"]), 4)) for k, v in acc.items()]


if __name__ == "__main__":
    res = {}
    for clip in (sys.argv[1:] or ["000"]):
        print("== clip", clip, flush=True)
        rows = run_clip(clip)
        res[clip] = rows
        base = [r for r in rows if r["method"] == "fidelity"][0]
        for r in rows:
            d = r["psnr"] - base["psnr"]
            dl = r["lpips"] - base["lpips"]
            print("  %-18s PSNR %7.3f (%+5.2f)  LPIPS %.4f (%+.4f)  tLPIPS %.4f"
                  % (r["method"], r["psnr"], d, r["lpips"], dl, r["tlpips"]), flush=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print("OK ->", OUT)
