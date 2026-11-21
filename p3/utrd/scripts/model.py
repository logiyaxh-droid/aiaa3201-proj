# -*- coding: utf-8 -*-
"""U-TRD prototype networks.

One residual U-Net is shared by all three arms.  They differ only in what the
network is asked to predict:

  det       : residual R = GT - B directly (no noise, no timestep)
  uniform   : epsilon of a 4-step residual diffusion with scalar noise level
  uncertain : same diffusion, per-pixel noise level from EDVR flip disagreement

The official REDS EDVR-L backbone is frozen in eval mode.  Its 128-channel
reconstruction feature is captured by a forward hook, so this prototype cannot
quietly fine-tune the baseline and attribute the gain to the new model.
"""
from __future__ import annotations

import math
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# BasicSR's EDVR imports this removed torchvision shim.
_mod = types.ModuleType("torchvision.transforms.functional_tensor")
from torchvision.transforms import functional as _F  # noqa: E402

for _n in dir(_F):
    setattr(_mod, _n, getattr(_F, _n))
sys.modules["torchvision.transforms.functional_tensor"] = _mod

CK_EDVR = ("/data/disk1/shared/aiaa3201/checkpoints/EDVR_REDS_SR_L/"
           "EDVR_L_x4_SR_REDS_official-9f5f5039.pth")
NUM_FEAT = 128
# alpha-bar for t = 0..4.  t=0 is the clean residual, t=4 the noisiest state.
# t=4 keeps a small non-zero residual coefficient so the one-step x0 recovery
# is not a division by zero; training samples t=1..4 from this same schedule.
AB = torch.tensor([1.0, 0.75, 0.50, 0.25, 0.05], dtype=torch.float32)
STEPS = 4
U_FLOOR = 0.1
MODES = ("det", "uniform", "uncertain")


class FrozenEDVR(nn.Module):
    """Official REDS EDVR-L, eval-only, returns output and pre-upsample feature."""

    def __init__(self, device="cuda"):
        super().__init__()
        from basicsr.archs.edvr_arch import EDVR

        net = EDVR(num_in_ch=3, num_out_ch=3, num_feat=NUM_FEAT, num_frame=5,
                   deformable_groups=8, num_extract_block=5,
                   num_reconstruct_block=40, center_frame_idx=2,
                   hr_in=False, with_predeblur=False, with_tsa=True)
        state = torch.load(CK_EDVR, map_location="cpu", weights_only=False)["params"]
        net.load_state_dict(state, strict=True)
        self.net = net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device
        self._feat = None
        self._hook = self.net.reconstruction.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        self._feat = output

    def forward(self, lr):
        """lr [B,5,3,H,W] -> out [B,3,4H,4W], feat [B,128,H,W]."""
        self._feat = None
        with torch.no_grad():
            out = self.net(lr.to(self.device))
        if self._feat is None:
            raise RuntimeError("EDVR reconstruction hook did not fire")
        return out, self._feat


class TimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(),
                                 nn.Linear(dim, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) *
                          torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float()[:, None] * freqs[None] * (2 * math.pi)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return self.mlp(emb)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim=128):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.emb = nn.Linear(emb_dim, cout * 2)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(emb).chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ResidualUNet(nn.Module):
    """Small conditional U-Net at HR resolution (training crop 256)."""

    def __init__(self, base=64, feat_ch=16, emb_dim=128):
        super().__init__()
        self.feat_proj = nn.Conv2d(NUM_FEAT, feat_ch, 1)
        in_ch = 3 + 3 + 3 + feat_ch + 1   # state, B, LR-up, EDVR feature, U
        self.stem = nn.Conv2d(in_ch, base, 3, 1, 1)
        self.temb = TimeEmbedding(emb_dim)
        self.d1 = ResBlock(base, base, emb_dim)
        self.d2 = ResBlock(base, base * 2, emb_dim)
        self.d3 = ResBlock(base * 2, base * 4, emb_dim)
        self.mid = ResBlock(base * 4, base * 4, emb_dim)
        self.u3 = ResBlock(base * 8, base * 2, emb_dim)
        self.u2 = ResBlock(base * 4, base, emb_dim)
        self.u1 = ResBlock(base * 2, base, emb_dim)
        self.out_norm = nn.GroupNorm(8, base)
        self.out = nn.Conv2d(base, 3, 3, 1, 1)

    def forward(self, x, t, base_img, lr_up, feat, u):
        fp = self.feat_proj(feat)
        fp = F.interpolate(fp, size=x.shape[-2:], mode="nearest")
        h = self.stem(torch.cat([x, base_img, lr_up, fp, u], dim=1))
        emb = self.temb(t)
        h1 = self.d1(h, emb)
        h2 = self.d2(F.avg_pool2d(h1, 2), emb)
        h3 = self.d3(F.avg_pool2d(h2, 2), emb)
        m = self.mid(F.avg_pool2d(h3, 2), emb)
        m = F.interpolate(m, size=h3.shape[-2:], mode="bilinear", align_corners=False)
        h = self.u3(torch.cat([m, h3], dim=1), emb)
        h = F.interpolate(h, size=h2.shape[-2:], mode="bilinear", align_corners=False)
        h = self.u2(torch.cat([h, h2], dim=1), emb)
        h = F.interpolate(h, size=h1.shape[-2:], mode="bilinear", align_corners=False)
        h = self.u1(torch.cat([h, h1], dim=1), emb)
        return self.out(F.silu(self.out_norm(h)))


def edvr_outputs(edvr: FrozenEDVR, lr):
    """Quantised base B, feature F, and flip-disagreement uncertainty U in [0,1]."""
    b_raw, feat = edvr(lr)
    b = b_raw.clamp(0, 1).mul(255).round().div(255)
    bf_raw, _ = edvr(torch.flip(lr, dims=[4]))
    bf = torch.flip(bf_raw, dims=[3]).clamp(0, 1).mul(255).round().div(255)
    u = (b - bf).abs().mean(dim=1, keepdim=True)
    u = F.avg_pool2d(u, 3, 1, 1)
    q = torch.quantile(u.flatten(1), 0.95, dim=1).clamp_min(1e-6)
    u = (u / q[:, None, None, None]).clamp(0, 1)
    return b, feat, u


def noise_map(mode: str, u: torch.Tensor):
    """Per-pixel noise amplitude.

    uniform keeps each sample's RMS equal to the uncertain map's RMS, so the
    comparison removes spatial variation rather than reducing total noise.
    """
    c = U_FLOOR + (1.0 - U_FLOOR) * u
    if mode == "uncertain":
        return c
    if mode == "uniform":
        rms = c.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        return rms.expand_as(c)
    return torch.zeros_like(c)


def make_state(mode, base_img, gt, u):
    """Training state and target for one arm.  Returns dict for the loss step."""
    bsz = base_img.shape[0]
    if mode == "det":
        zeros = base_img.new_zeros(bsz)
        return {"x": base_img, "t": zeros, "c": torch.zeros_like(u),
                "eps": None, "ab": None, "target": gt - base_img}
    c = noise_map(mode, u)
    eps = torch.randn_like(base_img)
    t = torch.randint(1, STEPS + 1, (bsz,), device=base_img.device)
    ab = AB.to(base_img.device)[t][:, None, None, None]
    x = ab.sqrt() * (gt - base_img) + (1 - ab).sqrt() * c * eps
    return {"x": x, "t": t, "c": c, "eps": eps, "ab": ab,
            "target": gt - base_img}


def x0_from_eps(x, eps_hat, base_img, c, ab):
    """Recover the clean residual from a predicted epsilon."""
    return (x - (1 - ab).sqrt() * c * eps_hat) / ab.sqrt()


def ddim_sample(net, mode, base_img, lr_up, feat, u, seed=1234):
    """Fixed 4-step deterministic residual sampling -> [B,3,H,W] in [0,1]."""
    net.eval()
    g = torch.Generator(device=base_img.device)
    g.manual_seed(seed)
    c = noise_map(mode, u)
    # Match the training marginal at t=STEPS: variance (1-alpha_bar_T)*c^2.
    amp = (1.0 - AB[STEPS]).sqrt().to(base_img.device)
    x = torch.randn(base_img.shape, generator=g, device=base_img.device,
                    dtype=base_img.dtype) * (c * amp)
    with torch.no_grad():
        for t in range(STEPS, 0, -1):
            tt = base_img.new_full((base_img.shape[0],), float(t))
            eps = net(x, tt, base_img, lr_up, feat, u)
            ab = AB[t].to(base_img.device)
            x0 = (x - (1 - ab).sqrt() * c * eps) / ab.sqrt()
            if t == 1:
                x = x0
            else:
                ab_prev = AB[t - 1].to(base_img.device)
                x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * c * eps
    return (base_img + x).clamp(0, 1)


def forward_one(net, mode, base_img, lr_up, feat, u, seed=1234):
    """Inference output for one arm, before uint8 quantisation."""
    if mode == "det":
        net.eval()
        with torch.no_grad():
            r = net(base_img, base_img.new_zeros(base_img.shape[0]),
                    base_img, lr_up, feat, torch.zeros_like(u))
        return (base_img + r).clamp(0, 1)
    if mode in ("uniform", "uncertain"):
        return ddim_sample(net, mode, base_img, lr_up, feat, u, seed=seed)
    raise ValueError("unknown mode: %s" % mode)
