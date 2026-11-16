#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Part 2 step 3 -- qualitative figures + deliverable comparison videos.

Everything is written to /tmp/p2/out (never to ~, because /data/disk1 is full).
The PDF requires "processed video files for all Mandatory Datasets", so the main
product here is a 4-up MP4 per clip:  GT | bicubic | EDVR | TecoGAN.

Also writes the enlarged-patch PNGs used in the report, cropped from a region
chosen by a gradient-energy heuristic (busy areas show the methods apart).
"""
import os
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import sys, glob, json
import numpy as np
import cv2

B = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT_ROOT = B + "/val_sharp"
LR_ROOT = B + "/val_sharp_bicubic"
TE = "/tmp/p2/tecogan"
ED = "/tmp/p2/edvr_frame"
OUT = "/tmp/p2/out"
VID = os.path.join(OUT, "videos")
FIG = os.path.join(OUT, "figures")
os.makedirs(VID, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

K = {"gt": 0, "bicubic": 0, "edvr": 0, "tecogan": 0}


def load(p, size=None):
    a = cv2.imread(p)
    if a is None:
        return None
    if size and (a.shape[1], a.shape[0]) != size:
        a = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
    return a


def pick_patch(gt, w=320, h=180):
    """Busiest w x h window by gradient energy -- detail-rich regions separate methods."""
    g = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY).astype(np.float32)
    e = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    ei = np.abs(e)
    # box-sum energy via integral image
    s = cv2.integral(ei)
    H, W = ei.shape
    if h >= H or w >= W:
        return 0, 0
    best, bxy = -1, (0, 0)
    for y in range(0, H - h, 16):
        for x in range(0, W - w, 16):
            tot = s[y + h, x + w] - s[y, x + w] - s[y + h, x] + s[y, x]
            if tot > best:
                best, bxy = tot, (x, y)
    return bxy


def make_figure(clip, frame=2, scale=3):
    gtp = "%s/%s/%08d.png" % (GT_ROOT, clip, frame)
    gt = load(gtp)
    if gt is None:
        return None
    H, W = gt.shape[:2]
    x, y = pick_patch(cv2.resize(gt, (W // 2, H // 2)))      # search at half res
    x, y = x * 2, y * 2
    pw, ph = 320, 180

    tiles = [("GT", gt)]
    lr = load("%s/%s/%08d.png" % (LR_ROOT, clip, frame))
    if lr is not None:
        tiles.append(("Bicubic", cv2.resize(lr, (W, H), interpolation=cv2.INTER_CUBIC)))
    edv = load("%s/%s/%08d.png" % (ED, clip, frame))
    if edv is not None:
        tiles.append(("EDVR", edv))
    teo = load("%s/%s/%08d.png" % (TE, clip, frame))
    if teo is not None:
        tiles.append(("TecoGAN", teo))

    crops, labels = [], []
    for name, img in tiles:
        if img is None:
            continue
        c = img[y:y + ph, x:x + pw]
        c = cv2.resize(c, (pw * scale, ph * scale), interpolation=cv2.INTER_NEAREST)
        cv2.putText(c, name, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        crops.append(c)
        labels.append(name)
    if not crops:
        return None
    fig = np.hstack(crops)
    p = os.path.join(FIG, "%s_patch.png" % clip)
    cv2.imwrite(p, fig)
    return p


def make_video(clip, fps=10):
    gtf = sorted(glob.glob(os.path.join(GT_ROOT, clip, "*.png")))
    if not gtf:
        return None
    teof = sorted(glob.glob(os.path.join(TE, clip, "*.png")))
    if len(teof) < 10:
        return None
    edf = sorted(glob.glob(os.path.join(ED, clip, "*.png")))
    if edf:
        n = min(len(gtf), len(teof), len(edf))
    else:
        n = min(len(gtf), len(teof))
    base = load(gtf[0])
    H, W = base.shape[:2]
    tw = 640
    th = int(H * (tw / float(W))) // 2 * 2

    path = os.path.join(VID, "%s_compare.mp4" % clip)
    ncols = 4 if edf else 3
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (tw * ncols, th))
    if not vw.isOpened():
        print("video writer failed for", clip)
        return None

    for i in range(n):
        gt = load(gtf[i], (tw, th))
        lr = load("%s/%s/%s" % (LR_ROOT, clip, os.path.basename(gtf[i])))
        bic = cv2.resize(lr, (tw, th), interpolation=cv2.INTER_CUBIC) if lr is not None else np.zeros((th, tw, 3), np.uint8)
        teo = load(teof[i], (tw, th))
        if teo is None:
            teo = np.zeros((th, tw, 3), np.uint8)
        row = []
        panels = [("GT", gt), ("Bicubic", bic)]
        if edf:
            e = load(edf[i], (tw, th))
            panels.append(("EDVR", e if e is not None else np.zeros((th, tw, 3), np.uint8)))
        panels.append(("TecoGAN", teo))
        for nm, img in panels:
            t = img.copy()
            cv2.putText(t, nm, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            row.append(t)
        vw.write(np.hstack(row))
    vw.release()
    return path


if __name__ == "__main__":
    which = [a for a in sys.argv[1:] if not a.startswith("-")]
    clips = which or sorted(d for d in os.listdir(TE) if os.path.isdir(os.path.join(TE, d)))
    for c in clips:
        if not os.path.exists(os.path.join(TE, c, ".done")):
            print("skip %s (not finished)" % c)
            continue
        p = make_figure(c)
        v = make_video(c)
        print("%s figure=%s video=%s" % (c, os.path.basename(p) if p else "-",
                                         os.path.basename(v) if v else "-"), flush=True)
    print("out dir:", OUT)
