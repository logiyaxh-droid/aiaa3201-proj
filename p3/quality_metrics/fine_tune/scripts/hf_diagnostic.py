# -*- coding: utf-8 -*-
"""Is the perceptual gain real texture, or just added high-frequency noise?

DISTS was used as a training loss, so its own improvement proves nothing.  This
script compares a candidate's frames against the baseline and the GT on the same
32x32 grid used in stage D and asks three separate questions:

  1. hf_ratio (pred high-frequency energy / GT high-frequency energy)
     baseline EDVR sits near 0.5 on textured blocks.  A healthy improvement moves
     this toward 1.0.  A value above ~1.1 means energy was invented, not recovered.
  2. hf_corr (correlation between the block's high-pass residual and the GT's)
     This is the discriminative test: noise raises energy but does not correlate
     with the GT's missing detail.  Real recovery raises both.
  3. flat-region noise (hf_ratio on low-texture blocks)
     A global sharpening trick raises this; genuine texture synthesis should not.

Runs on a fixed frame subset so the numbers are comparable across checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = "/data/disk1/home/ghe270/p3/quality_metrics"
sys.path.insert(0, os.path.join(ROOT, "fine_tune", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import cv2
import numpy as np
import common as C
import evaluate as E


def highpass(x):
    """Laplacian of the grey channel: what the hf metrics measure."""
    g = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return cv2.Laplacian(g, cv2.CV_32F, ksize=3)


def block_stats(pred, gt, bs=32):
    rows = []
    h, w = gt.shape[:2]
    hp_p, hp_g = highpass(pred), highpass(gt)
    for y in range(0, h - bs + 1, bs):
        for x in range(0, w - bs + 1, bs):
            a = hp_p[y:y + bs, x:x + bs].ravel()
            b = hp_g[y:y + bs, x:x + bs].ravel()
            ea = float(np.sqrt((a ** 2).mean()))
            eb = float(np.sqrt((b ** 2).mean()))
            denom = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
            corr = float((a * b).sum() / denom) if denom > 0 else 0.0
            rows.append({"ea": ea, "eb": eb, "corr": corr,
                         "g_energy": eb, "mae": float(
                             np.abs(pred[y:y + bs, x:x + bs].astype(np.float32)
                                    - gt[y:y + bs, x:x + bs].astype(np.float32)).mean())})
    return rows


def summarize(rows, thresh):
    tex = [r for r in rows if r["g_energy"] > thresh]
    flat = [r for r in rows if r["g_energy"] <= thresh]
    out = {}
    for name, sel in (("textured", tex), ("flat", flat), ("all", rows)):
        if not sel:
            continue
        ea = np.array([r["ea"] for r in sel])
        eb = np.array([r["eb"] for r in sel])
        out[name] = {"n": len(sel),
                     "hf_ratio": float(ea.mean() / eb.mean()),
                     "hf_corr": float(np.mean([r["corr"] for r in sel])),
                     "mae": float(np.mean([r["mae"] for r in sel]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="name=path pairs, or name=baseline")
    ap.add_argument("--clips", nargs="*", default=["000"])
    ap.add_argument("--frames", nargs="*", type=int, default=[10, 40, 70, 95])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ckpts = []
    for spec in args.checkpoints:
        name, _, path = spec.partition("=")
        ckpts.append((name, path or "baseline"))

    results = {}
    for clip in args.clips:
        gt_dir = os.path.join(C.GT_ROOT, clip)
        lr_dir = os.path.join(C.LR_ROOT, clip)
        ids = C.clip_frame_ids(gt_dir)
        gt = C.read_clip(gt_dir, ids)
        lr = C.read_clip(lr_dir, ids)

        # GT high-frequency third-quartile threshold, fixed on GT only so the
        # textured/flat split cannot depend on any prediction.
        gt_e = [float(np.sqrt((highpass(g) ** 2).mean())) for g in gt[:20]]
        thresh = float(np.quantile(gt_e, 0.5))

        for name, path in ckpts:
            net, meta = E.load_checkpoint(path, args.device)
            net.eval()
            pred = E.predict(net, [lr[i] for i in args.frames], args.device)
            rows = []
            for k, i in enumerate(args.frames):
                rows += block_stats(pred[k], gt[i])
            s = summarize(rows, thresh)
            results.setdefault(name, {})[clip] = {
                "threshold": thresh, "checkpoint": meta, **s}
            print("%-14s %s textured hf_ratio %.3f corr %.3f | flat hf_ratio %.3f "
                  "corr %.3f | all mae %.3f"
                  % (name, clip, s["textured"]["hf_ratio"], s["textured"]["hf_corr"],
                     s["flat"]["hf_ratio"], s["flat"]["hf_corr"],
                     s["all"]["mae"]), flush=True)
            del net
            import torch
            torch.cuda.empty_cache()

    path = os.path.join(ROOT, "fine_tune", "results", "hf_diagnostic.json")
    C.write_json(path, {"frames": args.frames, "results": results})
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
