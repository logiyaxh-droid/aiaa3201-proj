# -*- coding: utf-8 -*-
"""Why did the U1 blend give back most of the perceptual gain?

The blend beat both parents on PSNR (20.891) and E_warp (5.637) but fell back to
LPIPS 0.478 from B1's 0.457.  Two candidate explanations:

  a) U1 is mostly HIGH, so the blend is nearly B1 everywhere -> contradicts the
     result (we got RBV-like LPIPS), so this is unlikely.
  b) U1 is mostly LOW over exactly the regions LPIPS cares about, so the blend
     switches back to RealBasicVSR where the perceptual gain lived.

This measures the actual weight distribution and, per region, how much of the
frame each model contributes.
"""
import os
import sys

import cv2
import numpy as np

U1 = "/tmp/m2b_out/u1"
B1 = "/tmp/m2b_out/full_b1"
RBV = "/tmp/m2b_out/full_rbv"
DATA = "/data/disk1/home/ghe270/p3/prior_qualification/data/reds_real_c9"


def rd(p, gray=False):
    a = cv2.imread(p, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return a if gray else cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def main():
    clips = sorted(d for d in os.listdir(U1) if os.path.isdir(os.path.join(U1, d)))
    us, ds_b1, ds_rbv = [], [], []
    for clip in clips:
        for f in sorted(os.listdir(os.path.join(U1, clip))):
            u = rd(os.path.join(U1, clip, f), gray=True).astype(np.float32) / 255.
            us.append(u)
    u_all = np.concatenate([x.ravel() for x in us])
    print("U1 distribution over %d frames (all pixels)" % len(us))
    for q in (1, 5, 25, 50, 75, 95, 99):
        print("  p%-3d %.4f" % (q, float(np.percentile(u_all, q))))
    print("  mean %.4f" % float(u_all.mean()))
    print("  fraction u<0.25: %.2f%%   u>0.75: %.2f%%"
          % (100 * float((u_all < 0.25).mean()),
             100 * float((u_all > 0.75).mean())))

    # how much does each parent differ from its own blend, inside flat vs tex
    flat_gap_b1 = []
    tex_gap_b1 = []
    for clip in clips[:10]:
        for f in sorted(os.listdir(os.path.join(U1, clip))):
            u = rd(os.path.join(U1, clip, f), gray=True).astype(np.float32)
            b1 = rd(os.path.join(B1, clip, f)).astype(np.float32)
            rb = rd(os.path.join(RBV, clip, f)).astype(np.float32)
            d = np.abs(b1 - rb).mean(axis=2)
            m_flat = u < 64
            m_tex = u > 191
            if m_flat.any():
                flat_gap_b1.append(float(d[m_flat].mean()))
            if m_tex.any():
                tex_gap_b1.append(float(d[m_tex].mean()))
    print("\n|B1 - RBV| per-pixel gap (proxy for how much blending matters):")
    print("  inside flat  (u<0.25): %.3f/255" % np.mean(flat_gap_b1))
    print("  inside tex   (u>0.75): %.3f/255" % np.mean(tex_gap_b1))


if __name__ == "__main__":
    main()
