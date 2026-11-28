# -*- coding: utf-8 -*-
"""Is the 3.6e-3 step-0 gap a real regression or just PNG quantisation?

The M1 reference on disk is 8-bit PNG, so any comparison against it is bounded
below by 1/255 = 3.9e-3 (half a level).  A gap of 3.6e-3 is right at that floor,
which means the two paths may be bit-identical in float and only differ after
rounding.

This script compares three ways:
  a) my float output vs the M1 PNG (float vs uint8)  -> should be ~1/255
  b) my float output vs a fresh fp32 forward of rbvsr_net  -> should be ~0
  c) my float output rounded to uint8 vs the M1 PNG -> should be 0 in most pixels
"""
import os
import sys

import cv2
import numpy as np
import torch

HERE = '/data/disk1/home/ghe270/p3/m2b_rbv_ft/scripts'
sys.path.insert(0, HERE)
M2B = os.path.dirname(HERE)
sys.path.insert(0, M2B)

from model_rbv_ft import UARBVSR, load_frozen_rbvsr  # noqa: E402

DATA = "/data/disk1/home/ghe270/p3/prior_qualification/data/reds_real_c9"
RBV_OUT = "/data/disk1/home/ghe270/p3/prior_qualification/outputs/reds_real_c9"


def rd(p):
    a = cv2.imread(p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB) if a is not None else None


def main():
    torch.manual_seed(0)
    lq_dir = os.path.join(DATA, "LQ", "000")
    ids = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(lq_dir)
                 if f.endswith('.png'))
    lq = np.stack([rd(os.path.join(lq_dir, "%08d.png" % i)) for i in ids])
    t = torch.from_numpy(lq).permute(0, 3, 1, 2).float().cuda() / 255.

    model = UARBVSR("cuda")
    with torch.no_grad():
        mine = model.infer_sequence(t)                 # my trainable path
        ref = load_frozen_rbvsr()
        fresh = ref(t.float()).clamp(0, 1)             # original module

    ref_png = np.stack([rd(os.path.join(RBV_OUT, "gen_thres255_generator",
                                       "000", "%08d.png" % i)) for i in ids])
    ref_png_t = torch.from_numpy(ref_png).permute(0, 3, 1, 2).float().cuda() / 255.

    a = (mine - ref_png_t).abs().max().item()
    b = (mine - fresh).abs().max().item()
    mc = (mine.clamp(0, 1) * 255).round()
    rc = (ref_png_t * 255).round()
    c_eq = float((mc == rc).float().mean())
    c_max = float((mc - rc).abs().max())
    print("(a) mine vs M1 PNG (float vs uint8)      max|d| = %.4e  (1/255 = %.4e)"
          % (a, 1 / 255))
    print("(b) mine vs fresh fp32 rbvsr_net forward max|d| = %.4e" % b)
    print("(c) mine(rounded) vs M1 PNG              identical %.4f%%  max level diff %.0f"
          % (c_eq * 100, c_max))


if __name__ == "__main__":
    main()
