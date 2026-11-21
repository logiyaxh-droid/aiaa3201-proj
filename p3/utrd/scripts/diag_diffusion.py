# -*- coding: utf-8 -*-
"""Post-mortem: is the diffusion arm's failure a mechanism result or a bug?

Checks on one real validation crop:
  1. baseline PSNR
  2. for t=1..4, feed the *true* training state and measure eps MAE and the
     one-step x0 PSNR.  A correct model must be best at small t.
  3. compare training-state variance with the DDIM initial-state variance
  4. print the trained epsilon loss at each t (what the model was optimised for)
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, "/data/disk1/home/ghe270/p0")

import torch
import torch.nn.functional as F

import model as M
import train as T

torch.manual_seed(11)
torch.backends.cudnn.benchmark = False
E = M.FrozenEDVR("cuda")
GT = "/data/disk1/shared/aiaa3201/project1_video_sr/val_sharp/000"
LR = "/data/disk1/shared/aiaa3201/project1_video_sr/val_sharp_bicubic/000"
ids = T.frame_ids(GT)[40:45]
to_t = lambda a: torch.from_numpy(a).float().div(255).permute(2, 0, 1)
lr = torch.stack([to_t(T.read_rgb(os.path.join(LR, "%08d.png" % i))) for i in ids])[None].cuda()
base, feat, u = M.edvr_outputs(E, lr)
up = F.interpolate(lr[:, 2], scale_factor=4, mode="bilinear", align_corners=False)
gt = to_t(T.read_rgb(os.path.join(GT, "%08d.png" % ids[2])))[None].cuda()

print("base PSNR %.4f" % float(10 * torch.log10(1 / ((base - gt) ** 2).mean())))
print("gt resid abs %.5f" % float((gt - base).abs().mean()))
c = M.noise_map("uncertain", u)
print("c mean %.4f max %.4f" % (float(c.mean()), float(c.max())))
print("AB", M.AB.tolist())

for name in ("uncertain_proto1", "uniform_proto1"):
    net = M.ResidualUNet().cuda()
    ck = torch.load(os.path.join(ROOT, "runs", name, "step_001500.pth"),
                    map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"])
    net.eval()
    print("\n==", name)
    for mode in ("uncertain", "uniform"):
        cm = M.noise_map(mode, u)
        for t in range(1, 5):
            eps = torch.randn_like(base)
            tt = base.new_full((1,), float(t))
            ab = M.AB[t].cuda()
            x = ab.sqrt() * (gt - base) + (1 - ab).sqrt() * cm * eps
            with torch.no_grad():
                ep = net(x, tt, base, up, feat, u)
            r0 = M.x0_from_eps(x, ep, base, cm, ab)
            out = (base + r0).clamp(0, 1)
            psnr = float(10 * torch.log10(1 / ((out - gt) ** 2).mean()))
            print("  t=%d eps_mae %.4f r0_abs %.4f one_step_psnr %.3f"
                  % (t, float((ep - eps).abs().mean()), float(r0.abs().mean()), psnr))

# variance check: training t=4 vs DDIM init
ab4 = M.AB[4]
print("\nvar train t4 multiplier %.4f, ddim init %.4f"
      % (float((1 - ab4).sqrt()), float((1 - M.AB[M.STEPS]).sqrt())))
