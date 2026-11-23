# -*- coding: utf-8 -*-
"""Check what the RealVSR x4 protocol can actually reach.

Question: OSEDiff loses ~2.1 dB to bicubic on all 50 clips.  Is that the model,
or is the input path (LQ downsampled 4x) throwing away information that no x4
model can recover?

Two references per frame:
  A) bicubic up from our LQ/4                 -- the baseline we compare against
  B) GT down 4 (area) then bicubic up 4       -- the ceiling of any x4 pipeline
"""
import os

import cv2
import numpy as np

GT = "data/realvsr/GT"
LQ = "data/realvsr/LQ4x"


def psnr(a, b):
    return 10 * np.log10(255.0 ** 2 / ((a - b) ** 2).mean())


for clip in ["016", "108", "364", "492"]:
    gt = cv2.cvtColor(cv2.imread(os.path.join(GT, clip, "00022.png")),
                      cv2.COLOR_BGR2RGB).astype(np.float32)
    lq = cv2.cvtColor(cv2.imread(os.path.join(LQ, clip, "00022.png")),
                      cv2.COLOR_BGR2RGB)
    H, W = gt.shape[:2]
    up = cv2.resize(lq, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    small = cv2.resize(gt, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    up2 = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    print("%s  ours(from LQ/4) %.3f   ceiling(GT/4 roundtrip) %.3f"
          % (clip, psnr(up, gt), psnr(up2, gt)))
