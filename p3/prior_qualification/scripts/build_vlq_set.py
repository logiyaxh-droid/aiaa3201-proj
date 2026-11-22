# -*- coding: utf-8 -*-
"""Build the VideoLQ evaluation set.

VideoLQ characteristics that force design decisions:
  * 50 real low-quality clips, resolutions differ (320x240 .. 720x576);
  * most clips have 100 frames, 030/031/033 have fewer;
  * there is NO ground truth, so only no-reference metrics apply.

Protocol chosen here, and why:
  * take 10 evenly spaced frames per clip (long clips would cost too much GPU
    for a screening test);
  * do NOT resize or crop: each clip keeps its native resolution so we measure
    the model on the degradation it was actually given.  Metrics are computed
    per frame, then averaged per clip, then across clips (unweighted).
  * the models that need x4 I/O (EDVR, OSEDiff, SUPIR, DOVE) still get the
    native frame as their target; they internally upsample/downsample.  This is
    stated in the report.

Output:
  data/vlq_eval/<clip>/<frame>.png    native-resolution frames
  data/vlq_eval/meta.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import cv2
import numpy as np

SRC = "/mnt/sda1/shidanhe/utrd_qual/data/videolq_final/VideoLQ"
OUT = "/mnt/sda1/shidanhe/utrd_qual/data/vlq_eval"
NF = 10


def clip_ids(d):
    ids = []
    for n in sorted(os.listdir(d)):
        if n.lower().endswith(".png"):
            m = re.findall(r"(\d+)", n)
            if m:
                ids.append(int(m[-1]))
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=0,
                    help="0 = all 50 clips")
    ap.add_argument("--frames", type=int, default=NF)
    args = ap.parse_args()

    all_clips = sorted(d for d in os.listdir(SRC)
                       if os.path.isdir(os.path.join(SRC, d)))
    use = all_clips if args.clips <= 0 else all_clips[:args.clips]
    meta = {"source": SRC, "n_clips": len(use), "frames_per_clip": args.frames,
            "clips": {}}
    for clip in use:
        d = os.path.join(SRC, clip)
        ids = clip_ids(d)
        if not ids:
            continue
        pick = np.linspace(0, len(ids) - 1, args.frames).round().astype(int)
        chosen = [ids[i] for i in sorted(set(pick.tolist()))]
        od = os.path.join(OUT, clip)
        os.makedirs(od, exist_ok=True)
        saved = []
        for k, fid in enumerate(chosen):
            src = glob.glob(os.path.join(d, "*%08d.png" % fid)) or \
                  glob.glob(os.path.join(d, "%08d.png" % fid))
            if not src:
                cand = sorted(glob.glob(os.path.join(d, "*.png")))
                src = [cand[fid]] if fid < len(cand) else []
            if not src:
                continue
            im = cv2.imread(src[0])
            if im is None:
                continue
            dst = os.path.join(od, "%08d.png" % k)
            cv2.imwrite(dst, im)
            saved.append({"out_id": k, "src": os.path.basename(src[0]),
                          "size": [im.shape[1], im.shape[0]]})
        if saved:
            meta["clips"][clip] = {"n": len(saved), "size": saved[0]["size"],
                                   "frames": saved}
        print("%s %d frames %s" % (clip, len(saved),
                                   saved[0]["size"] if saved else "NA"), flush=True)
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print("clips done:", len(meta["clips"]), flush=True)


if __name__ == "__main__":
    main()
