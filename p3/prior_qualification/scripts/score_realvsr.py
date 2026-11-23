# -*- coding: utf-8 -*-
"""Score a candidate directory against RealVSR GT, paired with the baselines.

Same metric stack as run_realvsr.py (PSNR on luma, SSIM, LPIPS, DISTS, tLP) so
the generative rows sit directly next to bicubic / EDVR / fine-tuned EDVR.

Usage:
  python score_realvsr.py --name osediff --dir outputs/realvsr/osediff
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, "/data/disk1/home/ghe270/p0")
sys.path.insert(0, "/data/disk1/home/ghe270/p3/quality_metrics/scripts")

os.environ.setdefault(
    "TORCH_HOME", "/data/disk1/home/ghe270/p3/quality_metrics/weights/torch_home")

DATA = os.path.join(ROOT, "data", "realvsr")
GT = os.path.join(DATA, "GT")


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def read_rgb(p):
    a = cv2.imread(p)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    from evalkit import metrics as EM
    from evalkit import temporal as ET
    import common as C

    lp = EM.LPIPS(device="cuda", torch_home="/tmp/torch_home")
    dists = C.DistsMetric(device="cuda")

    clips = sorted(d for d in os.listdir(GT) if os.path.isdir(os.path.join(GT, d)))
    rows, missing = {}, []
    for clip in clips:
        ids = sorted(os.path.splitext(f)[0]
                     for f in os.listdir(os.path.join(GT, clip))
                     if f.lower().endswith(".png"))
        preds, gts, keep = [], [], []
        for fid in ids:
            pp = os.path.join(args.dir, clip, fid + ".png")
            if not os.path.exists(pp):
                missing.append("%s/%s" % (clip, fid))
                continue
            p = read_rgb(pp)
            g = read_rgb(os.path.join(GT, clip, fid + ".png"))
            if p.shape != g.shape:
                raise SystemExit("%s/%s size %s vs GT %s"
                                 % (clip, fid, p.shape, g.shape))
            preds.append(p)
            gts.append(g)
            keep.append(int(fid))
        if not preds:
            continue

        pY = np.stack([EM.luma(x) for x in preds])
        gY = np.stack([EM.luma(x) for x in gts])
        lp_vals = lp.pair(np.stack(preds), np.stack(gts))
        d_vals = dists.per_frame(preds, gts)
        row = {
            "psnr": float(EM.psnr_mean(pY, gY)),
            "ssim": float(EM.ssim_gpu(pY, gY, device="cuda").mean()),
            "lpips": float(lp_vals.mean()),
            "dists": float(d_vals.mean()),
            "n_frames": len(preds),
            "per_frame": {"ids": keep, "lpips": lp_vals.tolist()},
        }
        pairs = ET.pairs_of(list(range(len(keep))), 0)
        if pairs:
            d_sr = lp.pair(np.stack([preds[a] for a, _ in pairs]),
                           np.stack([preds[b] for _, b in pairs]))
            d_gt = lp.pair(np.stack([gts[a] for a, _ in pairs]),
                           np.stack([gts[b] for _, b in pairs]))
            row["tlp"] = float(np.abs(d_sr - d_gt).mean())
        else:
            row["tlp"] = None
        rows[clip] = row
        print("%s %s PSNR %.4f SSIM %.4f LPIPS %.5f DISTS %.5f tLP %s"
              % (args.name, clip, row["psnr"], row["ssim"], row["lpips"],
                 row["dists"],
                 ("%.5f" % row["tlp"]) if row["tlp"] is not None else "N/A"),
              flush=True)

    if not rows:
        raise SystemExit("no scored frames under %s" % args.dir)

    mean = {}
    for k in ("psnr", "ssim", "lpips", "dists", "tlp"):
        vals = [r[k] for r in rows.values() if r[k] is not None]
        mean[k] = float(np.mean(vals)) if vals else None
    mean["n_clips"] = len(rows)
    mean["n_frames"] = sum(r["n_frames"] for r in rows.values())

    out = {"name": args.name, "dir": os.path.abspath(args.dir),
           "missing": missing, "clip_means": rows, "mean": mean,
           "protocol": {
               "input": "RealVSR LQ downsampled 4x",
               "output_grid": "RealVSR GT native 512x1024",
               "metrics": "PSNR(luma)/SSIM/LPIPS/DISTS/tLP against GT",
           }}
    dump(os.path.join(ROOT, "results", "realvsr", args.name, "metrics.json"), out)
    print("\n%s MEAN %s" % (args.name, json.dumps(mean)), flush=True)


if __name__ == "__main__":
    main()
