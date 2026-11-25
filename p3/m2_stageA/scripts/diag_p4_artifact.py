# -*- coding: utf-8 -*-
"""Where do the cyan/blue blotches in P4 clip 016 come from?

Suspects, in order of likelihood:
  a) the neg_raw sign flip: dZ = -unet(...)  amplifies a channel-biased output
  b) the gate opening on high-u1 foliage (u1 0.391 on clip 016) letting a bad
     residual through at full strength
  c) a decode-side artefact from bf16 at 720p

This measures the residual's per-channel statistics and checks whether the
blotches correlate with u1.
"""
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, '/data/disk1/home/ghe270/p3/m2_stageA/scripts')
import model_stageA as M

DATA = '/data/disk1/home/ghe270/p3/prior_qualification/data/reds_real_c9'
OUT = '/data/disk1/home/ghe270/p3/m2_stageA/outputs/pilot_p4'


def main():
    clip = "016"
    fid = 22
    fn = "%08d.png" % fid
    rb = cv2.cvtColor(cv2.imread(os.path.join(
        '/data/disk1/home/ghe270/p3/prior_qualification/outputs/reds_real_c9',
        'gen_thres255_generator', clip, fn)), cv2.COLOR_BGR2RGB).astype(np.float32)
    # our output lives at outputs/pilot_p4/016/00000022.png
    o = cv2.cvtColor(cv2.imread(os.path.join(OUT, clip, fn)),
                     cv2.COLOR_BGR2RGB).astype(np.float32)
    d = o - rb
    print("clip %s frame %d" % (clip, fid))
    print("  |Ours-RBV| mean %.3f  max %.1f" % (np.abs(d).mean(), np.abs(d).max()))
    for i, ch in enumerate("RGB"):
        print("  channel %s: mean %+.3f  std %.3f  min %.1f  max %.1f"
              % (ch, d[..., i].mean(), d[..., i].std(),
                 d[..., i].min(), d[..., i].max()))
    # is the artefact blue-ish? measure only large-|d| pixels
    mag = np.abs(d).max(axis=2)
    big = mag > 40
    print("  pixels |d|>40: %.2f%%" % (100 * big.mean()))
    if big.any():
        for i, ch in enumerate("RGB"):
            print("    at those pixels, channel %s mean %+.1f" % (ch, d[..., i][big].mean()))
    # x,y crop of the worst block
    ys, xs = np.where(big)
    if len(ys):
        print("  worst block bbox y[%d..%d] x[%d..%d]"
              % (ys.min(), ys.max(), xs.min(), xs.max()))


if __name__ == "__main__":
    main()
