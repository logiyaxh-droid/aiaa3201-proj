# -*- coding: utf-8 -*-
"""Score a method on VideoLQ.

VideoLQ has no ground truth.  What we can measure:
  * no-reference: NIQE (lower better), MUSIQ (higher better);
  * self-consistency: the native frame is treated as the target, so a model that
    degrades it should score worse.  This is NOT accuracy against truth.

Aggregation: per clip mean, then unweighted mean across clips.  Resolutions
differ, so a pooled frame average would let 640x480 clips dominate 320x240 ones.

Usage:
  python score_vlq.py --name edvr --dir outputs/vlq/edvr
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

BASE = "/mnt/sda1/shidanhe/utrd_qual"
SRC = os.path.join(BASE, "data/vlq_eval")
OUT = os.path.join(BASE, "results/vlq")


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

    import pyiqa
    dev = "cuda"
    niqe = pyiqa.create_metric("niqe", device=dev).eval()
    musiq = pyiqa.create_metric("musiq", device=dev).eval()

    clips = sorted(d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d)))
    rows, miss = {}, []
    for clip in clips:
        sd = os.path.join(SRC, clip)
        od = os.path.join(args.dir, clip)
        ids = sorted(f for f in os.listdir(sd) if f.endswith(".png"))
        preds, refs = [], []
        for f in ids:
            p = os.path.join(od, f)
            if not os.path.exists(p):
                miss.append("%s/%s" % (clip, f))
                continue
            preds.append(read_rgb(p))
            refs.append(read_rgb(os.path.join(sd, f)))
        if not preds:
            continue
        shapes = {x.shape for x in preds}
        if len(shapes) != 1:
            raise SystemExit("%s: inconsistent shapes %s" % (clip, shapes))
        x = torch.stack([torch.from_numpy(p).permute(2, 0, 1).float().div(255)
                         for p in preds]).to(dev)
        r = torch.stack([torch.from_numpy(p).permute(2, 0, 1).float().div(255)
                         for p in refs]).to(dev)
        with torch.no_grad():
            nq = float(niqe(x).reshape(-1).mean().item())
            mq = float(musiq(x).reshape(-1).mean().item())
            mse = float(((x - r) ** 2).mean().item())
        rows[clip] = {"niqe": nq, "musiq": mq, "n": len(preds),
                      "size": list(preds[0].shape[:2]),
                      "self_psnr": float(10 * np.log10(1.0 / max(mse, 1e-12)))}
    if not rows:
        raise SystemExit("no comparable frames under %s" % args.dir)

    def mean(key):
        return float(np.mean([v[key] for v in rows.values()]))

    out = {"method": args.name, "dir": os.path.abspath(args.dir),
           "n_clips": len(rows), "n_frames": sum(v["n"] for v in rows.values()),
           "missing": miss,
           "niqe": mean("niqe"), "musiq": mean("musiq"),
           "self_psnr": mean("self_psnr"),
           "note": "VideoLQ has no GT: NIQE/MUSIQ are no-reference, self_psnr is "
                   "agreement with the native frame, not accuracy against truth.",
           "per_clip": rows}
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, "%s.json" % args.name)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("%-10s clips %d frames %d | NIQE %.4f | MUSIQ %.4f | selfPSNR %.4f"
          % (args.name, out["n_clips"], out["n_frames"], out["niqe"],
             out["musiq"], out["self_psnr"]), flush=True)


if __name__ == "__main__":
    main()
