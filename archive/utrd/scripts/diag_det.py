# -*- coding: utf-8 -*-
"""Quick diagnostic: does the deterministic arm actually learn texture, or just
predict a near-zero residual?  Prints fixed-batch loss, residual magnitude and
output range over 60 steps so a degenerate solution is visible immediately.
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

torch.manual_seed(3)
torch.backends.cudnn.benchmark = False

device = "cuda"
edvr = M.FrozenEDVR(device)
ds = T.WindowDataset(crop=64, length=32, seed=5)
loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False,
                                     num_workers=0, collate_fn=T.collate)
batch = next(iter(loader))
lr, gt = batch[0].to(device), batch[1].to(device)
base, feat, u = M.edvr_outputs(edvr, lr)
lr_up = F.interpolate(lr[:, 2], scale_factor=4, mode="bilinear", align_corners=False)

gt_res = gt - base
print("GT residual abs mean %.5f  std %.5f  max %.3f"
      % (gt_res.abs().mean(), gt_res.std(), gt_res.abs().max()), flush=True)
print("base PSNR vs GT %.4f" % float(10 * torch.log10(1 / ((base - gt) ** 2).mean())),
      flush=True)

net = M.ResidualUNet(base=64).to(device)
opt = torch.optim.Adam(net.parameters(), lr=3e-5)
for step in range(1, 61):
    pred = net(base, base.new_zeros(base.shape[0]), base, lr_up, feat,
               torch.zeros_like(u))
    out = base + pred
    loss = T.charbonnier(out.clamp(0, 1), gt)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 100.0)
    opt.step()
    if step in (1, 5, 10, 20, 40, 60):
        print("step %3d loss %.5f res_abs %.5f out_range [%.3f, %.3f] gn %.2f"
              % (step, float(loss), float(pred.abs().mean()),
                 float(out.min()), float(out.max()), float(gn)), flush=True)
