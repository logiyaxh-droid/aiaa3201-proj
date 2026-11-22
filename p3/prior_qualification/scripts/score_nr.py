# -*- coding: utf-8 -*-
"""No-reference evaluation for real-world footage (no GT available).

Metrics that do NOT need a reference:
  NIQE   lower is better; classical naturalness measure
  MUSIQ  higher is better; learned perceptual quality

Because there is no GT, this protocol cannot show "closer to the true image".
It only measures naturalness / perceptual quality, which is exactly why the
report must not compare these numbers with the REDS PSNR/LPIPS table.

Usage:
  python score_nr.py --name edvr --dir outputs/real/edvr
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

BASE = "/mnt/sda1/shidanhe/utrd_qual/data/wild"
OUT_ROOT = "/mnt/sda1/shidanhe/utrd_qual/results/real"


def load_ids(clip):
    meta = json.load(open(os.path.join(BASE, "meta.json"), encoding="utf-8"))
    return meta, [f["out_id"] for f in meta["frames"]]


def read_rgb(path):
    import cv2
    a = cv2.imread(path)
    if a is None:
        raise RuntimeError("unreadable %s" % path)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def score_dir(method, clip, in_dir, metrics):
    meta, ids = load_ids(clip)
    preds, used = [], []
    for i in ids:
        p = os.path.join(in_dir, "%08d.png" % i)
        if not os.path.exists(p):
            continue
        preds.append(read_rgb(p))
        used.append(i)
    if not preds:
        raise SystemExit("no frames found under %s" % in_dir)
    shapes = {p.shape for p in preds}
    if len(shapes) != 1:
        raise SystemExit("inconsistent shapes: %s" % shapes)

    row = {"method": method, "clip": clip, "n_frames": len(preds),
           "size": list(preds[0].shape[:2]), "missing": len(ids) - len(preds)}
    x = torch.stack([torch.from_numpy(p).permute(2, 0, 1).float().div(255)
                     for p in preds]).cuda()
    for name, net in metrics.items():
        with torch.no_grad():
            vals = []
            for i in range(0, len(x), 8):
                v = net(x[i:i + 8])
                vals.append(v.reshape(-1).cpu().numpy())
        arr = np.concatenate(vals)
        row[name] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)),
                     "median": float(np.median(arr)), "n": int(arr.size)}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--clip", default="wild01")
    args = ap.parse_args()

    import pyiqa
    metrics = {
        "niqe": pyiqa.create_metric("niqe", device="cuda"),
        "musiq": pyiqa.create_metric("musiq", device="cuda"),
    }
    for m in metrics.values():
        m.eval()

    row = score_dir(args.name, args.clip, args.dir, metrics)
    os.makedirs(OUT_ROOT, exist_ok=True)
    dst = os.path.join(OUT_ROOT, "%s.json" % args.name)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=1, ensure_ascii=False)
    print("%-12s frames %d  NIQE %.4f (lower better)  MUSIQ %.4f (higher better)"
          % (args.name, row["n_frames"], row["niqe"]["mean"], row["musiq"]["mean"]),
          flush=True)
    print("->", dst, flush=True)


if __name__ == "__main__":
    main()
