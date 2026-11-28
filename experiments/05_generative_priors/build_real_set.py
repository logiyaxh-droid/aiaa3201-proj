# -*- coding: utf-8 -*-
"""Build the real-world evaluation set from local phone video.

This is a stand-in for VideoLQ, which is currently unreachable from this
server (Google Drive / Dropbox / OneDrive all fail, the HF mirror contains a
different dataset, and ModelScope has metadata only).

Design decisions that keep the four methods comparable:
  * centre-crop to a fixed 1280x720 window so EDVR's x4 requirement and the
    diffusion models' spatial tiling see the same content;
  * take 30 evenly spaced frames from a stable segment (no scene changes);
  * save as lossless PNG, never decode the H.264 output for scoring.

Layout:
  data/wild/frames/<clip>/<frame>.png        1280x720 source frames
  data/wild/meta.json                        clip, frame ids, crop box
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

BASE = "/mnt/sda1/shidanhe/utrd_qual/data/wild"
# Portrait source (1080x1920).  Crop 720x1280 so the EDVR x4 input is 180x320
# (both divisible by 4) and the output keeps the source's native pixel count.
W, H = 720, 1280
NF = 30


def center_crop(frame):
    h, w = frame.shape[:2]
    if h < H or w < W:
        return None
    x = (w - W) // 2
    y = (h - H) // 2
    return frame[y:y + H, x:x + W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="VID_01.mp4")
    ap.add_argument("--clip", default="wild01")
    ap.add_argument("--start", type=int, default=120,
                    help="skip the first N frames (handheld settle)")
    ap.add_argument("--span", type=int, default=300,
                    help="sample NF frames inside this many frames")
    args = ap.parse_args()

    path = os.path.join(BASE, args.video)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("cannot open %s" % path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("source %dx%d, %d frames" % (src_w, src_h, total), flush=True)

    last = min(total, args.start + args.span)
    span = max(last - args.start, NF)
    ids = np.linspace(args.start, last - 1, NF).round().astype(int).tolist()

    out_dir = os.path.join(BASE, "frames", args.clip)
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for k, fid in enumerate(ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, frame = cap.read()
        if not ok:
            print("frame %d unreadable, skipped" % fid, flush=True)
            continue
        crop = center_crop(frame)
        if crop is None:
            raise SystemExit("source smaller than %dx%d" % (W, H))
        dst = os.path.join(out_dir, "%08d.png" % k)
        cv2.imwrite(dst, crop)
        saved.append({"out_id": k, "src_frame": int(fid),
                      "path": dst})
    cap.release()

    import scipy.ndimage as ndi
    stats = []
    for s in saved:
        g = cv2.imread(s["path"], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        stats.append({"lap": float(np.abs(ndi.laplace(g)).mean()),
                      "std": float(g.std())})
    meta = {"clip": args.clip, "source": path, "source_size": [src_w, src_h],
            "source_frames": total, "crop": [W, H], "n": len(saved),
            "frames": saved,
            "sharpness": {"lap_mean": float(np.mean([s["lap"] for s in stats])),
                          "std_mean": float(np.mean([s["std"] for s in stats]))},
            "note": "stand-in for VideoLQ; real-world handheld footage, no GT"}
    with open(os.path.join(BASE, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print("saved %d frames to %s" % (len(saved), out_dir), flush=True)
    print("sharpness lap_mean %.2f  std_mean %.2f"
          % (meta["sharpness"]["lap_mean"], meta["sharpness"]["std_mean"]),
          flush=True)


if __name__ == "__main__":
    main()
