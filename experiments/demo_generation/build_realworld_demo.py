# -*- coding: utf-8 -*-
"""Compose the M4 2x2 demo video from the generated 720p branches.

Layout (one panel per method, 960x540 each -> 1920x1080):
    +-------------+------------------+
    |    EDVR     |   RealBasicVSR   |
    +-------------+------------------+
    |   OSEDiff   |  Ours (blend_sqrt)|
    +-------------+------------------+

Why 2x2 and not 5-up: at 1080p a 5-column strip drops every panel to 384px wide,
which destroys exactly the texture detail the video is meant to show.

Two encodes per clip, same as Part 2:
  plain : full 720p frames
  inset : same, plus a 2x magnified loupe of a fixed busy region, drawn on every
          panel so the viewer can compare the SAME pixels across methods.

cv2.putText note: OpenCV 5.0.0 misrenders LINE_AA at thickness>=3, so labels are
drawn ONCE at thickness=1 on an opaque dark strip.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

# No system ffmpeg on this host; Part 2 used the binary bundled with
# imageio-ffmpeg and that is the only libx264 build available here.
try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:                                  # pragma: no cover
    FFMPEG = "ffmpeg"

PW, PH = 960, 540
FPS = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX
CW, CH, CSC = 192, 108, 2          # loupe crop and zoom
LBL_H = 26


def rd(p):
    a = cv2.imread(p)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return a


def label(img, text):
    strip = np.full((LBL_H, img.shape[1], 3), 32, np.uint8)
    cv2.putText(strip, text, (6, 19), FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([strip, img])


def busy_box(gray, w, h, cands):
    """Pick the candidate window with the most high-frequency energy."""
    best, beste = None, -1.0
    g = cv2.resize(gray, None, fx=1.0, fy=1.0)
    if g.shape[1] < w or g.shape[0] < h:
        return (max(0, (g.shape[1] - w) // 2), max(0, (g.shape[0] - h) // 2))
    for (x, y) in cands:
        x = max(0, min(x, g.shape[1] - w))
        y = max(0, min(y, g.shape[0] - h))
        patch = g[y:y + h, x:x + w].astype(np.float32)
        e = cv2.Laplacian(patch, cv2.CV_32F).var()
        if e > beste:
            best, beste = (x, y), e
    return best


def loupe(img, x, y):
    h, w = img.shape[:2]
    x = max(0, min(x, w - CW)); y = max(0, min(y, h - CH))
    c = img[y:y + CH, x:x + CW]
    z = cv2.resize(c, (CW * CSC, CH * CSC), interpolation=cv2.INTER_NEAREST)
    return z


def place_loupe(panel, z, box):
    ph = panel.shape[0]
    px, py = PW - z.shape[1] - 10, ph - z.shape[0] - 10
    panel[py:py + z.shape[0], px:px + z.shape[1]] = z
    cv2.rectangle(panel, (px - 2, py - 2),
                  (px + z.shape[1] + 1, py + z.shape[0] + 1), (255, 255, 255), 1)
    x, y = box
    cv2.rectangle(panel, (x, y), (x + CW, y + CH), (255, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/p3_demo/out")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inset", action="store_true")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--cands", default="",
                    help="json list of [x,y] candidate loupe origins")
    args = ap.parse_args()

    BR = [("edvr", "EDVR"), ("rbv", "RealBasicVSR"),
          ("osediff", "OSEDiff"), ("ours", "Ours (blend_sqrt)")]
    ids = sorted(f for f in os.listdir(os.path.join(args.root, "rbv", args.clip))
                 if f.endswith(".png"))[:args.frames]
    if not ids:
        raise SystemExit("no frames")

    # choose the loupe window once, from the GT-free RBV frame (stable target)
    box = (0, 0)
    if args.inset:
        ref = cv2.imread(os.path.join(args.root, "rbv", args.clip, ids[0]),
                         cv2.IMREAD_GRAYSCALE)
        ref = cv2.resize(ref, (PW, PH - LBL_H), interpolation=cv2.INTER_AREA)
        if args.cands:
            cands = json.loads(args.cands)
        else:
            # dense grid: the busiest window is picked from the panel itself,
            # so the loupe never lands on an empty patch of sky or road
            cands = [(x, y)
                     for y in range(0, PH - LBL_H - CH, 40)
                     for x in range(0, PW - CW, 40)]
        box = busy_box(ref, CW, CH, cands)
        print("loupe box", box, flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    h, w = args.out.rsplit(".", 1)[0], args.out.rsplit(".", 1)[1]
    cmd = [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", "%dx%d" % (PW * 2, PH * 2), "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    for k, fn in enumerate(ids):
        panels = {}
        for tag, _ in BR:
            p = os.path.join(args.root, tag, args.clip, fn)
            if not os.path.exists(p):
                panels[tag] = label(np.full((PH - LBL_H, PW, 3), 40, np.uint8),
                                    "%s MISSING" % tag)
                continue
            # panel = LBL_H label strip + image; the two must add up to PH,
            # otherwise the stacked 2x2 frame is taller than the declared
            # 1080p input size and ffmpeg silently squeezes it.
            img = cv2.resize(rd(p), (PW, PH - LBL_H),
                             interpolation=cv2.INTER_AREA)
            if args.inset:
                place_loupe(img, loupe(img, box[0], box[1]), box)
            panels[tag] = label(img, tag)
        top = np.hstack([panels["edvr"], panels["rbv"]])
        bot = np.hstack([panels["osediff"], panels["ours"]])
        frame = np.vstack([top, bot])
        assert frame.shape[:2] == (PH * 2, PW * 2), frame.shape
        proc.stdin.write(frame.tobytes())
        if k % 25 == 0:
            print("  %d/%d" % (k, len(ids)), flush=True)

    proc.stdin.close()
    err = proc.stderr.read().decode("utf-8", "replace")
    rc = proc.wait()
    if rc != 0:
        print(err[-2000:], flush=True)
        if os.path.exists(args.out):
            os.remove(args.out)
        raise SystemExit("ffmpeg failed rc=%d" % rc)
    print("wrote %s (%.1f MB)" % (args.out,
                                  os.path.getsize(args.out) / 2**20), flush=True)


if __name__ == "__main__":
    main()
