# -*- coding: utf-8 -*-
"""3-minute regression after the 7 fixes (guide section 3)."""
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, '/data/disk1/home/ghe270/p3/m2_stageA/scripts')
import model_stageA as M

torch.manual_seed(0)
LQ = '/data/disk1/home/ghe270/p3/prior_qualification/data/reds_real_c9/LQ/000'


def main():
    m = M.StageA("cuda", torch.bfloat16, lora_rank=4, osediff_init=True,
                 gamma_init=0.10, residual_mode="neg_raw")
    m.set_train_mode(False)
    print("built: residual=%s gamma=%.3f" % (m.residual_mode, float(m.gamma)),
          flush=True)

    ids = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(LQ)
                 if f.endswith('.png'))
    lq = np.stack([cv2.cvtColor(cv2.imread(os.path.join(LQ, '%08d.png' % i)),
                               cv2.COLOR_BGR2RGB) for i in ids])
    t = torch.from_numpy(lq).permute(0, 3, 1, 2).float().cuda() / 255.
    with torch.no_grad():
        full = m.rbvsr(t.float()).to(m.dtype)
        fullf = torch.flip(m.rbvsr(torch.flip(t, dims=[-1]).float()),
                           dims=[-1]).to(m.dtype)

    print("\n[check 2] first-frame u1 must not saturate (0.05-0.40)", flush=True)
    for k, label in ((0, "frame0 (uses next as neighbour)"), (1, "frame1")):
        kn = 1 if k == 0 else k - 1
        b_t = full[k:k + 1]
        bf = fullf[k:k + 1]
        b_prev = full[kn:kn + 1]
        lr_t, lr_prev = t[k:k + 1], t[kn:kn + 1]
        u_ref = (b_t - b_prev).abs().mean(1, keepdim=True).flatten().float() \
            .quantile(0.95).clamp_min(1e-6).to(b_t.dtype)
        with torch.no_grad():
            u1, uw, _, _ = m._uncertainty(b_t, b_prev, bf, lr_t, lr_prev, u_ref)
        mean = float(u1.mean())
        ok = 0.05 <= mean <= 0.40
        print("  %-30s u1 mean %.4f  %s"
              % (label, mean, "PASS" if ok else "FAIL"), flush=True)

    # check 1 is in warp_verify; re-assert the gate direction here
    print("\n[check 4] u1 gate direction: w == u1 (high texture -> open)",
          flush=True)
    with torch.no_grad():
        b_t = full[1:2]
        b_prev = full[0:1]
        lr_t, lr_prev = t[1:2], t[0:1]
        u_ref = (b_t - b_prev).abs().mean(1, keepdim=True).flatten().float() \
            .quantile(0.95).clamp_min(1e-6).to(b_t.dtype)
        u1, uw, _, _ = m._uncertainty(b_t, b_prev, fullf[1:2], lr_t, lr_prev,
                                      u_ref)
        y, w, _ = m.refine(b_t[:, :, :256, :256], u1[:, :, :256, :256],
                           uw[:, :, :256, :256], "u1", 0.5)
    d = (w - u1[:, :, :256, :256]).abs().max()
    print("  max|W_pix - u1| = %.3e  %s"
          % (float(d), "PASS" if float(d) < 1e-5 else "FAIL"), flush=True)
    print("  W_pix mean %.4f (should track u1, not 1-u1)"
          % float(w.mean()), flush=True)


if __name__ == "__main__":
    main()
