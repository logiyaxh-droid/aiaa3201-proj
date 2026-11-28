# -*- coding: utf-8 -*-
"""Make a frame subset of a paired dataset so expensive models are affordable.

REDS-Real has 30 clips x 100 frames = 3000 frames.  OSEDiff runs at roughly
8 frames/minute on the shared A6000 (measured), so the full set would take about
six hours per generative model — and DOVE is slower still.

This writes a new root that symlinks every clip but exposes only N frames per
clip, so every method (cheap or expensive) is scored on exactly the same frames.
Metrics are recomputed on the subset rather than sliced from the full-set JSON,
because per-clip means over different frame counts are not comparable.

Usage:
  python make_frame_subset.py --src ../data/reds_real --out ../data/reds_real_s10 \
      --frames 20 30 40 50 60 70 80 90 95 99
"""
from __future__ import annotations

import argparse
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    args = ap.parse_args()

    for split in ("GT", "LQ"):
        src_split = os.path.join(args.src, split)
        if not os.path.isdir(src_split):
            continue
        for clip in sorted(os.listdir(src_split)):
            sd = os.path.join(src_split, clip)
            if not os.path.isdir(sd):
                continue
            dd = os.path.join(args.out, split, clip)
            shutil.rmtree(dd, ignore_errors=True)
            os.makedirs(dd, exist_ok=True)
            ids = sorted(os.path.splitext(f)[0] for f in os.listdir(sd)
                         if f.lower().endswith(".png"))
            for i in args.frames:
                if i >= len(ids):
                    continue
                shutil.copy(os.path.join(sd, ids[i] + ".png"),
                            os.path.join(dd, ids[i] + ".png"))
        print("%s done" % split, flush=True)

    # the runner expects an LQ4x name; REDS-Real already stores quarter-size LR
    link = os.path.join(args.out, "LQ4x")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link) if os.path.islink(link) else shutil.rmtree(link)
    if os.path.isdir(os.path.join(args.out, "LQ")):
        os.symlink("LQ", link)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
