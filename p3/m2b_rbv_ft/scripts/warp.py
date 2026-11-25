# -*- coding: utf-8 -*-
"""Occlusion-aware flow warp, carrying over the M2 Stage A fixes verbatim.

These three functions are the *fixed* versions validated at the end of M2 Stage A:

  * `indexing="ij"` returns (rows, cols) = (y, x).  Binding the first value to a
    variable named `gx` and adding it to the flow's x channel transposed the
    warp 90 degrees.  Measured on a 96x96 crop: no-warp frame difference 0.1444,
    the buggy warp gave 0.1977 (motion compensation made things worse); after
    the fix 0.1182 (-18%), and on a full 320x180 frame 0.0786 -> 0.0221 (-72%).
  * the centre-crop guard was `if W > HR_OUT // 4` (W > 64), which is also true
    for the 96x96 training crop and the 320x180 inference frame, so training was
    reading only the top-left 64x64 of its own flow field.  It is now
    `if H == LR_CROP and W == LR_CROP`.
  * the flow scale used `s = h / flow_width` (height over width): 4.0 for a
    square crop by luck, 720/320 = 2.25 on a full frame, 720/64 = 11.25 once the
    crop bug was stacked on it.  It is now separate `sx` / `sy`.

`selfcheck_rbv.py` item 4 re-verifies the property (warped difference must be
lower than unwarped) rather than trusting this copy.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

LR_CROP = 96
HR_OUT = 256
BR = (LR_CROP - HR_OUT // 4) // 2      # 16 LR pixels of border


def flow_and_occ(spynet, lr_t, lr_prev, device="cuda"):
    """SpyNet flow (prev->cur, LR px) + forward-backward occlusion mask.

    Returns the flow/occ pair centre-cropped to the 64x64 region that maps onto
    the 256x256 HR output when the input is the 96x96 training crop, and the
    full field otherwise (whole-frame inference).
    """
    n = lr_t.shape[0]
    fwd = [spynet(lr_t[i:i + 1], lr_prev[i:i + 1]) for i in range(n)]
    bwd = [spynet(lr_prev[i:i + 1], lr_t[i:i + 1]) for i in range(n)]
    f_fwd = torch.cat(fwd, 0)
    f_bwd = torch.cat(bwd, 0)

    H, W = f_fwd.shape[-2:]
    gy, gx = torch.meshgrid(
        torch.arange(H, device=device, dtype=f_fwd.dtype),
        torch.arange(W, device=device, dtype=f_fwd.dtype),
        indexing="ij")
    bx = (gx[None, None] + f_fwd[:, 0:1]).clamp(0, W - 1)
    by = (gy[None, None] + f_fwd[:, 1:2]).clamp(0, H - 1)
    nx = (bx / max(W - 1, 1)) * 2 - 1
    ny = (by / max(H - 1, 1)) * 2 - 1
    grid = torch.cat([nx, ny], dim=1).permute(0, 2, 3, 1)
    b_back = F.grid_sample(f_bwd, grid, mode="bilinear",
                           padding_mode="border", align_corners=True)
    # consistency test is a SUM: |f_fwd(p) + f_bwd(p + f_fwd(p))| ~ 0.
    # Written as a subtraction it reports ~14 px on identical flow and the mask
    # never fires (a silent M1 bug).
    fb = (f_fwd + b_back).abs().sum(dim=1, keepdim=True)
    occ = (fb <= 3.0).to(f_fwd.dtype)

    if H == LR_CROP and W == LR_CROP:
        sl = slice(BR, BR + HR_OUT // 4)
        return f_fwd[:, :, sl, sl], occ[:, :, sl, sl]
    return f_fwd, occ


def warp_hr(x, flow_lr, dtype=None):
    """Warp an HR tensor with an LR flow, honouring separate x/y scales."""
    if dtype is not None:
        x = x.to(dtype)
    b, _, h, w = x.shape
    sx = float(w) / float(flow_lr.shape[-1])
    sy = float(h) / float(flow_lr.shape[-2])
    f = F.interpolate(flow_lr.to(x.dtype), size=(h, w), mode="bilinear",
                      align_corners=False)
    f = torch.cat([f[:, 0:1] * sx, f[:, 1:2] * sy], dim=1)
    gy, gx = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=x.dtype),
        torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
    px = (gx[None, None] + f[:, 0:1]).clamp(0, w - 1)
    py = (gy[None, None] + f[:, 1:2]).clamp(0, h - 1)
    nx = (px / max(w - 1, 1)) * 2 - 1
    ny = (py / max(h - 1, 1)) * 2 - 1
    grid = torch.cat([nx, ny], dim=1).permute(0, 2, 3, 1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border",
                         align_corners=True)


def uncertainty_u1(b_ref, b_ref_flip):
    """Flip-disagreement uncertainty in [0,1], normalised by ITS OWN 95th pct.

    M2 Stage A normalised U_1 by the neighbouring-frame difference instead, and
    low-texture clips (000) came out with the whole frame below 0.05 -- the gate
    never opened.  Normalising each image by its own q95 puts flat regions at
    0.0-0.15 and textured regions at 0.7-1.0 regardless of clip content, which
    is what makes a FIXED threshold meaningful across clips.
    """
    u = (b_ref - b_ref_flip).abs().mean(dim=1, keepdim=True)
    u = F.avg_pool2d(u, 3, 1, 1)
    q = torch.quantile(u.flatten(1), 0.95, dim=1).clamp_min(1e-6)
    return (u / q[:, None, None, None]).clamp(0.0, 1.0)


def pix_weight(u):
    """Fidelity weight: flat regions locked at 1.0, texture allowed down to 0.4."""
    return 1.0 - 0.6 * u


def adv_weight(u):
    """Adversarial weight: flat regions capped at 0.15, texture opened to 1.0."""
    return 0.15 + 0.85 * u
