#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ground-truth calibration of Farneback warping convention.

Take a sharp frame A, and a known-shifted copy B (B(x) = A(x + d), i.e. content
moves by -d). Then check which convention recovers B from A.
"""
import numpy as np
import cv2
import glob

B_DIR = "/data/disk1/shared/aiaa3201/project1_video_sr"
H, W = 720, 1280
gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))


def fb(a, b):
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def remap(img, f, sign=+1):
    return cv2.remap(img, gx + sign * f[..., 0], gy + sign * f[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


A = cv2.cvtColor(cv2.imread(sorted(glob.glob(B_DIR + "/val_sharp/000/*.png"))[0]), cv2.COLOR_BGR2RGB)
# B is A sampled at (x + d): content appears shifted by -d
for dx, dy in [(12, 7), (-9, 5)]:
    Bframe = cv2.remap(A, gx + dx, gy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ga = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(Bframe, cv2.COLOR_RGB2GRAY)
    f_ab = fb(ga, gb)   # calcOpticalFlowFarneback(A, B)
    print("\n=== true sampling shift d=(%d,%d); expect flow ~ (%.0f,%.0f) if flow maps A->B coords ===" % (dx, dy, -dx, -dy))
    print("   mean flow from fb(A,B): (%.2f, %.2f)" % (f_ab[..., 0].mean(), f_ab[..., 1].mean()))
    for name, f, sgn in [("fb(A,B) remap(+f)", f_ab, +1),
                         ("fb(A,B) remap(-f)", f_ab, -1),
                         ("fb(B,A) remap(+f)", fb(gb, ga), +1),
                         ("fb(B,A) remap(-f)", fb(gb, ga), -1)]:
        rec = remap(A, f, sgn)      # try to reconstruct B from A
        l1 = np.abs(rec.astype(np.float32) - Bframe.astype(np.float32)).mean()
        raw = np.abs(A.astype(np.float32) - Bframe.astype(np.float32)).mean()
        print("   %-18s L1=%.3f (no-warp %.3f)  -> %s" % (name, l1, raw, "OK" if l1 < raw else "WRONG"))
