# -*- coding: utf-8 -*-
"""Add LPIPS + DISTS to the MVSR4x cross-domain numbers.

Why a separate script rather than editing eval_cross.py
-------------------------------------------------------
The 5-frame MVSR4x outputs for all four models are already on disk from the
earlier run (they were written under /tmp/m2b_cross and re-generated per run),
so rescoring only needs the metric networks -- about 1 minute for 15 clips,
versus ~6 minutes to re-run inference.  Keeping it separate also means the
already-published PSNR/E_warp numbers cannot drift.

Protocol note (important for the report)
----------------------------------------
These runs restore each checkpoint and score MVSR4x at its native 1024x1024.
LPIPS uses the AlexNet trunk exactly as `evalkit.metrics.LPIPS` does for the
REDS-Real table, so the two tables are comparable in protocol even though the
datasets differ.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
M2B = os.path.dirname(HERE)
sys.path.insert(0, M2B)

DATA = "/data/disk1/home/ghe270/p3/prior_qualification/data/mvsr4x"
RESULTS = os.path.join(M2B, "results")


def rd(p):
    a = cv2.imread(p)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", default="/tmp/m2b_cross")
    ap.add_argument("--clips", nargs="*", default=None)
    args = ap.parse_args()

    for p in ("/data/disk1/home/ghe270/p3/quality_metrics/scripts",
              "/data/disk1/home/ghe270/p3/quality_metrics",
              "/data/disk1/home/ghe270/p0"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from evalkit import metrics as EM
    import common as C

    dev = os.environ.get("RVQ_DEV", "cuda")
    lp = EM.LPIPS(device=dev, torch_home="/tmp/torch_home")
    dists = C.DistsMetric(device=dev)

    clips = args.clips or sorted(d for d in os.listdir(os.path.join(DATA, "GT"))
                                 if os.path.isdir(os.path.join(DATA, "GT", d)))
    out = {}
    for arm in sorted(os.listdir(args.pred_root)):
        root = os.path.join(args.pred_root, arm)
        if not os.path.isdir(root):
            continue
        lp_v, di_v, n = [], [], 0
        for clip in clips:
            pd = os.path.join(root, clip)
            gd = os.path.join(DATA, "GT", clip)
            if not os.path.isdir(pd):
                continue
            files = sorted(f for f in os.listdir(pd) if f.endswith(".png"))
            preds, gts = [], []
            for f in files:
                gp = os.path.join(gd, f)
                if not os.path.exists(gp):
                    continue
                a, b = rd(os.path.join(pd, f)), rd(gp)
                if a.shape != b.shape:
                    continue
                preds.append(a)
                gts.append(b)
            if not preds:
                continue
            lp_v.append(float(lp.pair(np.stack(preds), np.stack(gts)).mean()))
            di_v.append(float(dists.per_frame(preds, gts).mean()))
            n += len(preds)
        if lp_v:
            out[arm] = {"lpips": float(np.mean(lp_v)),
                        "dists": float(np.mean(di_v)),
                        "n_clips": len(lp_v), "n_frames": n}
            print("%-22s LPIPS %.5f  DISTS %.5f  (%d clips, %d frames)"
                  % (arm, out[arm]["lpips"], out[arm]["dists"],
                     len(lp_v), n), flush=True)

    dst = os.path.join(RESULTS, "cross_mvsr_lpips.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote", dst)


if __name__ == "__main__":
    main()
