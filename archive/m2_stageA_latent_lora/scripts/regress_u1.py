# -*- coding: utf-8 -*-
"""Is the first frame still anomalous after the fix, or is u1 just small in general?

The guide's pass window (0.05-0.40) was written against the BUGGY warp, where
U_warp was computed against a transposed neighbour and therefore inflated.
After fixing the warp, U_1/U_warp are genuinely smaller everywhere, so a mean
of 0.038 is not automatically a failure -- the question is whether frame 0
differs from the other frames.  That is what this script measures.
"""
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, '/data/disk1/home/ghe270/p3/m2_stageA/scripts')
import model_stageA as M

torch.manual_seed(0)
DATA = '/data/disk1/home/ghe270/p3/prior_qualification/data/reds_real_c9'


def main():
    m = M.StageA("cuda", torch.bfloat16, lora_rank=4, osediff_init=True,
                 gamma_init=0.10, residual_mode="neg_raw")
    m.set_train_mode(False)

    for clip in ("000", "002", "009"):
        lq_dir = os.path.join(DATA, "LQ", clip)
        ids = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(lq_dir)
                     if f.endswith('.png'))
        lq = np.stack([cv2.cvtColor(cv2.imread(os.path.join(lq_dir,
                                                            '%08d.png' % i)),
                                   cv2.COLOR_BGR2RGB) for i in ids])
        t = torch.from_numpy(lq).permute(0, 3, 1, 2).float().cuda() / 255.
        with torch.no_grad():
            full = m.rbvsr(t.float()).to(m.dtype)
            fullf = torch.flip(m.rbvsr(torch.flip(t, dims=[-1]).float()),
                               dims=[-1]).to(m.dtype)
        rows = []
        for k in range(len(ids)):
            kn = (k - 1) if k > 0 else min(1, len(ids) - 1)
            b_t, b_prev, bf = full[k:k + 1], full[kn:kn + 1], fullf[k:k + 1]
            lr_t, lr_prev = t[k:k + 1], t[kn:kn + 1]
            u_ref = (b_t - b_prev).abs().mean(1, keepdim=True).flatten().float() \
                .quantile(0.95).clamp_min(1e-6).to(b_t.dtype)
            with torch.no_grad():
                u1, uw, _, _ = m._uncertainty(b_t, b_prev, bf, lr_t, lr_prev,
                                              u_ref)
            rows.append((k, float(u1.mean()), float(uw.mean()),
                         float(u_ref)))
        print("\nclip %s" % clip, flush=True)
        for k, u1m, uwm, ref in rows:
            tag = " <-- FIRST (old code gave 1.000)" if k == 0 else ""
            print("  frame %d  u1 %.4f  uw %.4f  u_ref %.4f%s"
                  % (k, u1m, uwm, ref, tag), flush=True)
        rest = [r[1] for r in rows[1:]]
        print("  frame0 u1 = %.4f | mean(other frames) = %.4f | delta %+.1f%%"
              % (rows[0][1], np.mean(rest),
                 (rows[0][1] - np.mean(rest)) / max(1e-9, np.mean(rest)) * 100),
              flush=True)


if __name__ == "__main__":
    main()
