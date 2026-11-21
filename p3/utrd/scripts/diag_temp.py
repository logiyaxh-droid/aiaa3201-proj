# -*- coding: utf-8 -*-
"""Is the diffusion failure a sampling-budget artefact?

For the SAME trained checkpoints, compare:
  * 4-step DDIM with the true training schedule
  * 4-step DDIM but starting from a much lower noise (temperature sweep)
  * 1-step output directly from t=1 state (best case: least noise)

If quality recovers as the starting noise shrinks, the model learned something
and the failure is sampling calibration.  If not, the mechanism itself failed
within this budget.
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

torch.manual_seed(5)
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


def run(net, mode, temp):
    c = M.noise_map(mode, u)
    g = torch.Generator(device="cuda"); g.manual_seed(17)
    amp = (1 - M.AB[M.STEPS]).sqrt() * temp
    x = torch.randn(base.shape, generator=g, device="cuda") * (c * amp)
    with torch.no_grad():
        for t in range(M.STEPS, 0, -1):
            tt = base.new_full((1,), float(t))
            eps = net(x, tt, base, up, feat, u)
            ab = M.AB[t].cuda()
            x0 = (x - (1 - ab).sqrt() * c * eps) / ab.sqrt()
            if t == 1:
                x = x0
            else:
                x = M.AB[t - 1].cuda().sqrt() * x0 + (1 - M.AB[t - 1]).cuda().sqrt() * c * eps
    out = (base + x).clamp(0, 1)
    return float(10 * torch.log10(1 / ((out - gt) ** 2).mean()))


for name, mode in (("uniform_proto1", "uniform"), ("uncertain_proto1", "uncertain")):
    net = M.ResidualUNet().cuda()
    ck = torch.load(os.path.join(ROOT, "runs", name, "step_001500.pth"),
                    map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"]); net.eval()
    print("\n==", name)
    for temp in (1.0, 0.5, 0.25, 0.1, 0.0):
        print("  start-noise x%.2f  PSNR %.3f" % (temp, run(net, mode, temp)))
