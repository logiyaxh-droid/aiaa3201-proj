#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Part 1 baselines that need no network: bicubic / lanczos / +USM.

Kept as a tiny standalone tool so the metric and tLP checks have real prediction
frames to compare against the old JSONs without re-running EDSR or the flow
fusion.  Frames are written as ``<out>/<method>/<clip>/<08d>.png`` (uint8 RGB,
same size as GT), which is what ``evalkit.data.match_clip`` expects.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evalkit import data  # noqa: E402


def unsharp(f, radius=1.0, amount=1.5):
    import cv2

    ff = f.astype(np.float32)
    return np.clip(np.rint(ff + amount * (ff - cv2.GaussianBlur(ff, (0, 0), radius))),
                   0, 255).astype(np.uint8)


def main() -> int:
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/disk1/shared/aiaa3201/project1_video_sr")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help="frames per clip, 0 = all")
    ap.add_argument("--methods", nargs="*", default=["bicubic", "lanczos", "bicubic_unsharp"])
    a = ap.parse_args()

    clips = a.clips or data.list_clips(os.path.join(a.root, "val_sharp"))
    for clip in clips:
        gt_i = data.index_dir(os.path.join(a.root, "val_sharp", clip))
        lr_i = data.index_dir(os.path.join(a.root, "val_sharp_bicubic", clip))
        ids = sorted(set(gt_i) & set(lr_i))
        if a.limit:
            ids = ids[: a.limit]
        H, W = data.read_rgb(gt_i[ids[0]]).shape[:2]
        for m in a.methods:
            d = os.path.join(a.out, m, clip)
            os.makedirs(d, exist_ok=True)
        for i in ids:
            lr = data.read_rgb(lr_i[i])
            res = {
                "bicubic": cv2.resize(lr, (W, H), interpolation=cv2.INTER_CUBIC),
                "lanczos": cv2.resize(lr, (W, H), interpolation=cv2.INTER_LANCZOS4),
            }
            res["bicubic_unsharp"] = unsharp(res["bicubic"])
            for m in a.methods:
                cv2.imwrite(os.path.join(a.out, m, clip, "%08d.png" % i),
                            cv2.cvtColor(res[m], cv2.COLOR_RGB2BGR))
        print("clip %s frames=%d -> %s" % (clip, len(ids), a.out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
