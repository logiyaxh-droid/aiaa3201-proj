# -*- coding: utf-8 -*-
"""EMA vs generator, and cleaning-pass budget, on REDS-Real c9.

Two questions, answered on the same 30-clip x 9-frame subset the rest of the
study uses:

  1. after the forward bugs are fixed, is `generator_ema` better than
     `generator`?  The checkpoint carries both (320 keys each, meta iter=150000,
     is_use_ema=True).
  2. `dynamic_refine_thres` 1.0 (255/255, one pass) vs 5/255 (~0.0196, which
     lets the loop reach its 3-pass bound).  REDS-Real is heavily video-compressed,
     so the cleaning module may have more residue to remove than on RealVSR.

PSNR is luma PSNR on the GT grid, the same metric the rest of the study uses.
Frames are processed per clip in one forward, so the propagation is identical
across variants and the comparison is paired.

Usage:
  python rbvsr_variants.py --variants gen,ema --frames 9 --limit 30
  python rbvsr_variants.py --variants gen_thres5 --thres 0.0196
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, "/data/disk1/home/ghe270/p0")

os.environ.setdefault(
    "TORCH_HOME", "/data/disk1/home/ghe270/p3/quality_metrics/weights/torch_home")

from evalkit import metrics as EM  # noqa: E402

CKPT = os.path.join(ROOT, "weights_rbvsr", "RealBasicVSR_x4.pth")
DATA = os.environ.get("RBVSR_DATA") or os.path.join(ROOT, "data", "reds_real_c9")
LQ4X = os.path.join(DATA, "LQ4x")
GT = os.path.join(DATA, "GT")


def read_rgb(p):
    a = cv2.imread(p)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def state_dict(prefix):
    import rbvsr_net as RN

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    sub = {k[len(prefix) + 1:]: v for k, v in sd.items()
           if k.startswith(prefix + ".")}
    if not sub:
        raise SystemExit("no keys under prefix %r" % prefix)
    net = RN.RealBasicVSR()
    missing, unexpected = net.load_state_dict(sub, strict=False)
    return net, missing, unexpected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="gen,ema")
    ap.add_argument("--thres", type=float, default=None,
                    help="override dynamic_refine_thres (e.g. 0.0196 = 5/255)")
    ap.add_argument("--frames", type=int, default=9)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", default="",
                    help="write the prediction of this variant as PNG (name)")
    args = ap.parse_args()

    clips = sorted(d for d in os.listdir(LQ4X)
                   if os.path.isdir(os.path.join(LQ4X, d)))
    if args.limit:
        clips = clips[:args.limit]

    # Preload so every variant sees the same bytes.
    data = []
    for clip in clips:
        ids = sorted(os.path.splitext(f)[0] for f in os.listdir(
            os.path.join(LQ4X, clip)) if f.lower().endswith(".png"))[:args.frames]
        lq = np.stack([read_rgb(os.path.join(LQ4X, clip, i + ".png")) for i in ids])
        gt = np.stack([read_rgb(os.path.join(GT, clip, i + ".png")) for i in ids])
        data.append((clip, ids, lq, gt))
    print("clips %d  frames/clip %d  LQ %s  GT %s"
          % (len(data), args.frames, data[0][2].shape[1:3], data[0][3].shape[1:3]),
          flush=True)

    results = {}
    for name in args.variants.split(","):
        prefix = "generator_ema" if name.endswith("ema") else "generator"
        net, missing, unexpected = state_dict(prefix)
        # per-variant cleaning budget: suffix _t5 means 5/255, _t1 means 255/255
        thres = args.thres
        if thres is None:
            if name.endswith("thres5"):
                thres = 5.0 / 255.0
            elif name.endswith("thres255") or name.endswith("thres1"):
                thres = 255.0 / 255.0
        if thres is not None:
            net.dynamic_refine_thres = thres
        net = net.cuda().eval()
        print("%-12s prefix=%-14s thres=%.4f missing=%d unexpected=%d"
              % (name, prefix, net.dynamic_refine_thres, len(missing),
                 len(unexpected)), flush=True)

        t0 = time.time()
        per_clip, all_p, all_g = {}, [], []
        for clip, ids, lq, gt in data:
            x = torch.from_numpy(lq).permute(0, 3, 1, 2).float().div_(255.0).cuda()
            with torch.no_grad():
                y = net(x).clamp(0, 1).mul(255.0).permute(0, 2, 3, 1)
                y = y.round().byte().cpu().numpy()
            pY = np.stack([EM.luma(a) for a in y])
            gY = np.stack([EM.luma(b) for b in gt])
            per_clip[clip] = float(EM.psnr_mean(pY, gY))
            all_p.append(pY)
            all_g.append(gY)
            if args.dump and name == args.dump:
                for fid, im in zip(ids, y):
                    dst = os.path.join(ROOT, "outputs", "reds_real_c9",
                                       "%s_%s" % (args.dump, prefix), clip,
                                       fid + ".png")
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    cv2.imwrite(dst, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            del x
            torch.cuda.empty_cache()
        m = float(EM.psnr_mean(np.concatenate(all_p), np.concatenate(all_g)))
        results[name] = {"prefix": prefix,
                         "dynamic_refine_thres": net.dynamic_refine_thres,
                         "psnr_mean": m,
                         "psnr_per_clip": per_clip,
                         "missing": len(missing), "unexpected": len(unexpected),
                         "elapsed_sec": round(time.time() - t0, 1)}
        print("%-12s PSNR %.4f  (%.1fs)" % (name, m, time.time() - t0), flush=True)
        del net
        torch.cuda.empty_cache()

    # paired comparison, so noise from shared-GPU contention cannot flip it
    names = list(results)
    if len(names) == 2:
        a, b = results[names[0]]["psnr_per_clip"], results[names[1]]["psnr_per_clip"]
        d = [a[c] - b[c] for c in a]
        results["delta"] = {"a": names[0], "b": names[1],
                           "mean": float(np.mean(d)),
                           "min": float(np.min(d)), "max": float(np.max(d)),
                           "clips_a_better": int(sum(v > 0 for v in d)),
                           "n": len(d)}
        print("delta %s - %s : %.4f dB  range [%.4f, %.4f]  %d/%d clips"
              % (names[0], names[1], np.mean(d), np.min(d), np.max(d),
                 sum(v > 0 for v in d), len(d)), flush=True)

    dst = os.path.join(ROOT, "results", "rbvsr_diag", "variants_%s.json"
                       % args.variants.replace(",", "_"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump({"data": DATA, "frames": args.frames, "results": results},
              open(dst, "w"), indent=1)
    print("wrote", dst, flush=True)


if __name__ == "__main__":
    main()
