#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bit-exactness check for the memory-lean naive_avg against the 2026-11-11 version."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/p0/scripts"))
import importlib.util
import numpy as np
import cv2

spec = importlib.util.spec_from_file_location("fe", os.path.expanduser("~/p0/scripts/full_eval.py"))
fe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fe)

for clip in ["000", "001"]:
    ids, gtp = fe.clip_frames(clip)
    lr = fe.lr_frames(clip, ids)
    gt = [fe.rd_rgb(p) for p in gtp]
    H, W = gt[0].shape[:2]
    bic = [cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC) for x in lr]

    new = fe.naive_avg(bic, (1, 2, 3, 2, 1))

    wt = np.asarray((1, 2, 3, 2, 1), np.float32)
    wt /= wt.sum()
    k, n = 2, len(bic)
    f = np.stack([x.astype(np.float32) for x in bic])
    pad = np.concatenate([f[:1]] * k + [f] + [f[-1:]] * k, 0)
    old = np.clip(np.rint(sum(wt[i] * pad[i:i + n] for i in range(len(wt)))), 0, 255).astype(np.uint8)

    d = np.abs(new.astype(np.int16) - old.astype(np.int16))
    print("%s  max_abs_pixel_diff=%d  frames_differing=%d/%d"
          % (clip, int(d.max()), int((d.reshape(n, -1).max(1) > 0).sum()), n))


# also the 3-tap variant
ids, gtp = fe.clip_frames("000")
lr = fe.lr_frames("000", ids)
gt = [fe.rd_rgb(p) for p in gtp]
H, W = gt[0].shape[:2]
bic = [cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC) for x in lr]
new3 = fe.naive_avg(bic, (1, 2, 1))
wt = np.asarray((1, 2, 1), np.float32); wt /= wt.sum(); k, n = 1, len(bic)
f = np.stack([x.astype(np.float32) for x in bic])
pad = np.concatenate([f[:1]] * k + [f] + [f[-1:]] * k, 0)
old3 = np.clip(np.rint(sum(wt[i] * pad[i:i + n] for i in range(len(wt)))), 0, 255).astype(np.uint8)
d3 = np.abs(new3.astype(np.int16) - old3.astype(np.int16))
print("tavg3  max_abs_pixel_diff=%d  frames_differing=%d/%d"
      % (int(d3.max()), int((d3.reshape(n, -1).max(1) > 0).sum()), n))
