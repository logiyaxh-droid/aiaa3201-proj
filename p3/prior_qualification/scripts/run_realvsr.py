# -*- coding: utf-8 -*-
"""RealVSR (ICCV 2021) paired evaluation: the first real-degradation set with GT.

Why this protocol
-----------------
RealVSR ships LQ and GT at the *same* pixel size (512x1024).  The LQ frame is
captured by a wide lens, the GT by a tele lens, then aligned.  Bandwidth
measurement on clip 016 frame 22 (high-frequency energy above 0.25 of Nyquist,
percent of total spectral energy):

    LQ (real capture)     0.316
    GT  bicubic-down x2   0.298     <- matches LQ
    GT  bicubic-down x4   0.065

so the effective resolution ratio is about 2x, not 4x.  The official RealVSR
test script (codes/test_RealVSR_wi_GT.py) therefore builds EDVR_NoUp with
upscale=1 and feeds LQ directly.

Our x4 checkpoints (REDS EDVR-L, fine-tuned EDVR, OSEDiff, SUPIR, DOVE) are all
x4 models and cannot consume a same-size pair.  To keep every method on one
protocol AND comparable with the earlier REDS/VideoLQ numbers, we downsample LQ
by 4 with anti-aliased bicubic and let each x4 model reconstruct back to the GT
grid.  The LQ-as-is row is reported separately as the "do nothing" reference.

Metrics need GT, so this is the only experiment in the whole study where
PSNR/SSIM/LPIPS/DISTS on real degradation are meaningful.

Inputs : data/realvsr/LQ4x/<clip>/<frame>.png   128x256   (LQ / 4)
         data/realvsr/GT/<clip>/<frame>.png     512x1024
Output : results/realvsr/<name>/metrics.json + outputs/realvsr/<name>/

Usage:
  python run_realvsr.py --name edvr        --method edvr
  python run_realvsr.py --name ft          --method edvr --finetuned true
  python run_realvsr.py --name bicubic     --method bicubic
  python run_realvsr.py --name lq          --method identity
"""
from __future__ import annotations

import argparse
import glob
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
sys.path.insert(0, "/data/disk1/home/ghe270/p3/quality_metrics/scripts")

os.environ.setdefault(
    "TORCH_HOME", "/data/disk1/home/ghe270/p3/quality_metrics/weights/torch_home")

DATA = os.path.join(ROOT, "data", "realvsr")
LQ4X = os.path.join(DATA, "LQ4x")
GT = os.path.join(DATA, "GT")
SCALE = 4


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def read_rgb(path):
    a = cv2.imread(path)
    if a is None:
        raise RuntimeError("unreadable %s" % path)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def write_rgb(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def clips():
    return sorted(d for d in os.listdir(LQ4X)
                  if os.path.isdir(os.path.join(LQ4X, d)))


def frames(clip):
    return sorted(os.path.splitext(f)[0]
                  for f in os.listdir(os.path.join(LQ4X, clip))
                  if f.lower().endswith(".png"))


# ---------------------------------------------------------------- methods

def build_edvr(finetuned):
    import qa_common as Q

    return Q.build_edvr(finetuned=finetuned)


def infer_edvr(net, lr_frames):
    """Centre-frame EDVR over the clip, five-frame replicate-at-edge windows."""
    import qa_common as Q

    return Q.infer_clip(net, lr_frames)


def bicubic_up(lr):
    h, w = lr.shape[:2]
    return cv2.resize(lr, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------- scoring

def score(preds, gts, ids):
    from evalkit import metrics as EM
    from evalkit import temporal as ET

    lp = EM.LPIPS(device="cuda", torch_home="/tmp/torch_home")
    import common as C
    dists = C.DistsMetric(device="cuda")

    pY = np.stack([EM.luma(x) for x in preds])
    gY = np.stack([EM.luma(x) for x in gts])
    lp_vals = lp.pair(np.stack(preds), np.stack(gts))
    dists_vals = dists.per_frame(preds, gts)

    row = {
        "psnr": float(EM.psnr_mean(pY, gY)),
        "ssim": float(EM.ssim_gpu(pY, gY, device="cuda").mean()),
        "lpips": float(lp_vals.mean()),
        "dists": float(dists_vals.mean()),
        "n_frames": len(ids),
        "per_frame": {"ids": ids,
                      "psnr": [float(EM.psnr_mean(pY[i:i + 1], gY[i:i + 1]))
                               for i in range(len(ids))],
                      "lpips": lp_vals.tolist()},
    }
    pairs = ET.pairs_of(ids, 0)
    if pairs:
        d_sr = lp.pair(np.stack([preds[a] for a, _ in pairs]),
                       np.stack([preds[b] for _, b in pairs]))
        d_gt = lp.pair(np.stack([gts[a] for a, _ in pairs]),
                       np.stack([gts[b] for _, b in pairs]))
        row["tlp"] = float(np.abs(d_sr - d_gt).mean())
    else:
        row["tlp"] = None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--method", required=True, choices=("edvr", "bicubic"))
    ap.add_argument("--finetuned", default="false")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    finetuned = args.finetuned.lower() in ("1", "true", "yes")
    net = build_edvr(finetuned) if args.method == "edvr" else None
    out_dir = os.path.join(ROOT, "outputs", "realvsr", args.name)

    cl = clips()
    if args.limit:
        cl = cl[:args.limit]
    t0 = time.time()
    rows, all_pred, all_gt, all_ids = {}, [], [], []
    for clip in cl:
        ids = frames(clip)
        lr_list = [read_rgb(os.path.join(LQ4X, clip, i + ".png")) for i in ids]
        gt_list = [read_rgb(os.path.join(GT, clip, i + ".png")) for i in ids]

        if args.method == "edvr":
            preds = infer_edvr(net, lr_list)
        else:
            preds = [bicubic_up(x) for x in lr_list]

        for fid, p in zip(ids, preds):
            write_rgb(os.path.join(out_dir, clip, fid + ".png"), p)

        if preds[0].shape[:2] != gt_list[0].shape[:2]:
            raise SystemExit("size mismatch %s vs %s"
                             % (preds[0].shape[:2], gt_list[0].shape[:2]))

        row = score(preds, gt_list, [int(i) for i in ids])
        rows[clip] = row
        all_pred += preds
        all_gt += gt_list
        all_ids += [int(i) for i in ids]
        print("%s %s PSNR %.4f SSIM %.4f LPIPS %.5f DISTS %.5f tLP %s (%d frames)"
              % (args.name, clip, row["psnr"], row["ssim"], row["lpips"],
                 row["dists"],
                 ("%.5f" % row["tlp"]) if row["tlp"] is not None else "N/A",
                 len(ids)), flush=True)

    keys = ("psnr", "ssim", "lpips", "dists", "tlp")
    mean = {}
    for k in keys:
        vals = [r[k] for r in rows.values() if r[k] is not None]
        mean[k] = float(np.mean(vals)) if vals else None
    mean["n_clips"] = len(rows)
    mean["n_frames"] = sum(r["n_frames"] for r in rows.values())

    res = {"name": args.name, "method": args.method, "finetuned": finetuned,
           "scale": SCALE, "clip_means": rows, "mean": mean,
           "protocol": {
               "input": "RealVSR LQ downsampled 4x, anti-aliased bicubic",
               "output_grid": "RealVSR GT native",
               "lq_gt_same_size": True,
               "effective_resolution_ratio": 2,
               "note": "x4 models reconstruct from LQ/4 so every method shares "
                       "one protocol; the LQ-as-is row is the do-nothing "
                       "reference",
           },
           "elapsed_sec": round(time.time() - t0, 1)}
    dump(os.path.join(ROOT, "results", "realvsr", args.name, "metrics.json"), res)
    print("\n%s MEAN %s" % (args.name, json.dumps(mean)), flush=True)


if __name__ == "__main__":
    main()
