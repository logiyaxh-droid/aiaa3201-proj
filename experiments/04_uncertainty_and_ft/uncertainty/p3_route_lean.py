"""Part 3 / Direction C -- the router evaluated on the axis it can actually win.

Memory-lean rewrite: the box has ~4 GB free (other users hold the rest), so
nothing is stacked in RAM. One frame at a time, running accumulators only.

Branches
  fidelity    BasicVSR++ (REDS4 official c64n7)   steady, occasionally soft
  generative  Real-ESRGAN x4                      sharp textures, invents detail
  uncertainty mean |Y_fid - Y_gen| per pixel      free -- both branches are
                                                  computed anyway
Fusion: keep fidelity's chroma, take the generative luma only where the two
branches disagree above the q-quantile.

Scored on PSNR(Y), LPIPS and tLPIPS. tLPIPS uses the Part 1/2 convention
(Farneback at 1/4 res, f = flow(cur, prev), remap(prev, +f)) and needs no GT.

Run: python p3_route_lean.py 000 002 009
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
OUT = "/data/disk2/home/ghe270/p3/route_lean.json"
QS = (0.99, 0.97, 0.95, 0.90, 0.80, 0.60)

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
    names = ["basicvsr++", "realesrgan"] + ["route_q%.2f" % q for q in QS]
    acc = {k: dict(psnr=0.0, lp=0.0, tlp=0.0, tn=0) for k in names}
    gxgy = None
    prev = {}

    for i in range(n):
        gt = cv2.imread(gtf[i])
        fid = cv2.imread(bvf[i])
        gen = cv2.imread(gnf[i])
        if gxgy is None:
            H, W = gt.shape[:2]
            gxgy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        gx, gy = gxgy

        frame = {"basicvsr++": fid, "realesrgan": gen}
        uy = np.abs(ly(fid).astype(np.int16) - ly(gen).astype(np.int16))
        fid_ycc = cv2.cvtColor(fid, cv2.COLOR_BGR2YCrCb)
        gen_y = ly(gen)
        for q in QS:
            m = uy > np.quantile(uy, q)
            cc = fid_ycc.copy()
            cc[:, :, 0] = np.where(m, gen_y, fid_ycc[:, :, 0])
            frame["route_q%.2f" % q] = cv2.cvtColor(cc, cv2.COLOR_YCrCb2BGR)

        gty = ly(gt)
        for k, f in frame.items():
            a = acc[k]
            a["psnr"] += psnr1(ly(f), gty)
            a["lp"] += lp1(f, gt)
            if k in prev:
                fl = flow_prev_to_cur(ly(prev[k]), ly(f))
                warped = cv2.remap(prev[k], gx + fl[..., 0], gy + fl[..., 1],
                                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                a["tlp"] += lp1(warped, f)
                a["tn"] += 1
            prev[k] = f
        if i % 25 == 0:
            print("  %s [%3d/%d]" % (clip, i, n), flush=True)
        del frame, uy, fid_ycc, gen_y, gty

    rows = [dict(method=k, psnr=round(v["psnr"] / n, 3), lpips=round(v["lp"] / n, 4),
                 tlpips=round(v["tlp"] / max(1, v["tn"]), 4))
            for k, v in acc.items()]
    return rows


if __name__ == "__main__":
    res = {}
    for clip in (sys.argv[1:] or ["000"]):
        print("== clip", clip, flush=True)
        rows = run_clip(clip)
        res[clip] = rows
        for r in rows:
            print("  %-14s PSNR %7.3f  LPIPS %.4f  tLPIPS %.4f"
                  % (r["method"], r["psnr"], r["lpips"], r["tlpips"]), flush=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print("OK ->", OUT)
