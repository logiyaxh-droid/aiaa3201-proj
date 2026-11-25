# -*- coding: utf-8 -*-
"""M2-B model: Uncertainty-Aware RealBasicVSR (pixel-space, multi-frame).

Why pixel space this time
-------------------------
M2 Stage A put a diffusion residual on top of RealBasicVSR's output and lost on
every arm.  The measured cause was not a coding mistake: the SD 2.1 VAE is 8x
lossy (round-trip error 0.045 mean / 0.81 max, i.e. ~11.5/255), which is the same
order as RealBasicVSR's remaining error (10-24/255), so a latent residual cannot
carry the correction.  `neg_raw` additionally turned the pretrained noise
predictor's channel bias into a cyan colour cast (R mean -10.8 vs B -1.6).

This model never leaves pixel space:

    LR 5-frame window
      -> image_cleaning        FROZEN  (no grad)
      -> spynet flow           FROZEN  (no grad)
      -> forward/backward propagation + fusion + upsampling   TRAINABLE
      -> 5 HR frames, centre-cropped to 256x256

Because RealBasicVSR emits all five frames in one forward, the temporal term is
an *end-to-end* multi-frame loss here: with idx=[1,2,3] both the (1->2) and
(2->3) pairs carry gradient, instead of Stage A's one-sided no-grad `Y_{t-1}`.

Freezing rationale
------------------
`image_cleaning` does denoising/deblocking on the LR input and `spynet` supplies
the only flow estimate; both are fragile pretrained priors and the guide's
`lr=3e-5` regime is far too small to re-learn them. Freezing them also means the
uncertainty map U_1, which is computed from a separate frozen reference model,
cannot drift while the propagation branch trains -- so the gate is a fixed
spatial prior, not a moving target.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

PRIOR = "/data/disk1/home/ghe270/p3/prior_qualification"
M2A = "/data/disk1/home/ghe270/p3/m2_stageA"
sys.path.insert(0, os.path.join(PRIOR, "scripts"))
sys.path.insert(0, os.path.join(M2A, "scripts"))

RBVSR_CKPT = os.path.join(PRIOR, "weights_rbvsr", "RealBasicVSR_x4.pth")

TRAINABLE_PREFIXES = (
    "basicvsr.forward_resblocks", "basicvsr.backward_resblocks",
    "basicvsr.fusion", "basicvsr.upsample1", "basicvsr.upsample2",
    "basicvsr.conv_hr", "basicvsr.conv_last",
)


def load_frozen_rbvsr(fp32=True):
    """A frozen RealBasicVSR, used only as the U_1 reference.

    fp32 on purpose: `rbvsr_net._warp` builds its sampling grid with a hard
    `.float()`, so a bf16 forward raises "expected scalar type BFloat16 but
    found Float". The reference is no_grad, so fp32 costs ~150 MB, not more.
    """
    from rbvsr_net import load_rbvsr as _load
    net, missing, _ = _load(RBVSR_CKPT, device="cpu")
    if missing:
        raise RuntimeError("RBV missing keys: %s" % list(missing)[:5])
    net = net.to("cuda", torch.float32).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


class UARBVSR(nn.Module):
    """Trainable propagation branch + frozen cleaning/spynet + frozen U_ref."""

    def __init__(self, device="cuda", freeze_ref=True):
        super().__init__()
        self.device = torch.device(device)
        from rbvsr_net import load_rbvsr as _load
        net, missing, _ = _load(RBVSR_CKPT, device="cpu")
        if missing:
            raise RuntimeError("RBV missing keys: %s" % list(missing)[:5])
        self.net = net.to(self.device, torch.float32)

        # freeze cleaning + spynet; train propagation and upsampling
        for name, p in self.net.named_parameters():
            p.requires_grad_(any(name.startswith(x) for x in TRAINABLE_PREFIXES))
        self.net.image_cleaning.eval()
        self.net.basicvsr.spynet.eval()

        self.ref = load_frozen_rbvsr() if freeze_ref else None
        self.device_name = device

    # -- parameter accounting --------------------------------------------
    def trainable_parameters(self):
        return [p for p in self.net.parameters() if p.requires_grad]

    def n_trainable(self):
        return sum(p.numel() for p in self.trainable_parameters())

    def frozen_report(self):
        """Per-group requires_grad, for the self-check to assert on."""
        groups = {}
        for name, p in self.net.named_parameters():
            key = name.split(".")[0] if not name.startswith("basicvsr.") else \
                ".".join(name.split(".")[:2])
            groups.setdefault(key, [0, 0])
            groups[key][0] += p.numel()
            groups[key][1] += int(p.requires_grad)
        return groups

    # -- forward ---------------------------------------------------------
    @torch.no_grad()
    def reference_forward(self, lr5):
        """5 HR frames from the frozen reference, plus the flip branch."""
        b = self.ref(lr5.float())                                    # [5,3,4h,4w]
        bf = torch.flip(self.ref(torch.flip(lr5, dims=[-1]).float()),
                        dims=[-1])
        return b, bf

    @torch.no_grad()
    def clean_and_flow(self, lr5):
        """Frozen cleaning + the flow fields needed for propagation.

        Mirrors RealBasicVSR.forward's internals.  The cleaning pass accumulates
        at most 3 residual passes and exits early once the mean absolute
        residual drops below dynamic_refine_thres (measured 0.0095 on a real
        frame, so it always exits after the first pass).
        """
        n = lr5.shape[0]
        x = lr5.float()
        cleaned = [x[i:i + 1].clone() for i in range(n)]
        for _ in range(3):
            res = [self.net.image_cleaning(cleaned[i]) for i in range(n)]
            for i in range(n):
                cleaned[i] = cleaned[i] + res[i]
            mean_res = torch.stack([r.abs().mean() for r in res]).mean()
            if float(mean_res) < self.net.dynamic_refine_thres:
                break
        spy = self.net.basicvsr.spynet
        fwd_flow = [None] * n      # flow(cur -> prev), used by forward branch
        bwd_flow = [None] * n      # flow(cur -> next), used by backward branch
        for i in range(1, n):
            fwd_flow[i] = spy(cleaned[i], cleaned[i - 1])
        for i in range(n - 1):
            bwd_flow[i] = spy(cleaned[i], cleaned[i + 1])
        return cleaned, fwd_flow, bwd_flow

    def forward_trainable(self, lr5):
        """5 HR frames with gradient flowing through the propagation branch."""
        from rbvsr_net import _warp
        cleaned, fwd_flow, bwd_flow = self.clean_and_flow(lr5)
        n = len(cleaned)
        bv = self.net.basicvsr

        back = [None] * n
        feat = torch.zeros(1, 64, cleaned[0].shape[2], cleaned[0].shape[3],
                           dtype=cleaned[0].dtype, device=cleaned[0].device)
        for i in range(n - 1, -1, -1):
            if i < n - 1:
                feat = _warp(feat, bwd_flow[i])
            feat = bv.backward_resblocks(torch.cat([cleaned[i], feat], 1))
            back[i] = feat

        feats = [None] * n
        feat = torch.zeros(1, 64, cleaned[0].shape[2], cleaned[0].shape[3],
                           dtype=cleaned[0].dtype, device=cleaned[0].device)
        for i in range(n):
            if i > 0:
                feat = _warp(feat, fwd_flow[i])
            feat = bv.forward_resblocks(torch.cat([cleaned[i], feat], 1))
            feats[i] = feat

        outs = []
        for i in range(n):
            x = bv.fusion(torch.cat([back[i], feats[i]], 1))
            x = F.leaky_relu(x, negative_slope=0.1, inplace=True)
            x = F.leaky_relu(bv.upsample1(x), negative_slope=0.1, inplace=True)
            x = F.leaky_relu(bv.upsample2(x), negative_slope=0.1, inplace=True)
            x = F.leaky_relu(bv.conv_hr(x), negative_slope=0.1, inplace=True)
            x = bv.conv_last(x)
            base = F.interpolate(cleaned[i], scale_factor=4, mode="bilinear",
                                 align_corners=False)
            outs.append(x + base)
        return torch.cat(outs, 0)                     # [5,3,4h,4w]

    @torch.no_grad()
    def infer_sequence(self, lr_all, chunk=9):
        """Whole-clip inference (no grad): one forward over up to `chunk` frames.

        The propagation branch is fully convolutional in time only through
        feature propagation, so a 9-frame clip is a single forward -- 270 frames
        take well under 2 minutes.  Chunking keeps memory bounded for long clips.
        """
        outs = []
        n = lr_all.shape[0]
        for s in range(0, n, chunk):
            seg = lr_all[s:s + chunk].float()
            outs.append(self.net(seg).clamp(0, 1))
        return torch.cat(outs, 0)
