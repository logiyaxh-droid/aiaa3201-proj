# -*- coding: utf-8 -*-
"""RealBasicVSR inference implemented directly against the official weights.

The upstream implementation lives in mmediting, which drags in mmcv and a
specific PyTorch range; installing it next to the DOVE/SUPIR stack would break
that environment.  The network is small enough to re-declare here, and the
layer names below are taken from the checkpoint's own keys rather than from
memory, so `load_state_dict` is the check that the reimplementation is right.

Graph (from the key names in `RealBasicVSR_x4.pth`):

    image_cleaning      20 residual blocks, 64 ch, cleans compression/noise
    spynet              6 basic modules, predicts flow between two frames
    forward_resblocks   20 residual blocks, input is [64 feat | 3 warped]
    backward_resblocks  same, run in reverse time
    fusion              1x1 conv, 128 -> 64
    upsample1/2         pixel-shuffle x4, 64 -> 256 -> 64
    conv_hr, conv_last  3x3 refine, 64 -> 3

Weights: 37.0M parameters in total (the roadmap's "6.3M" referred to a different
counting of the propagation body only; the checkpoint is the authority).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(in_ch, out_ch, k=3, s=1):
    return nn.Conv2d(in_ch, out_ch, k, s, k // 2)


class ResidualBlock(nn.Module):
    """Two 3x3 convs inside one residual connection, no norm (BasicVSR style)."""

    def __init__(self, mid_channels=64):
        super().__init__()
        self.conv1 = _conv(mid_channels, mid_channels)
        self.conv2 = _conv(mid_channels, mid_channels)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x), inplace=True))


class ResidualBlocksWithInputConv(nn.Module):
    """`main` is [conv, act, Sequential(resblocks), final_add].

    The nesting is not cosmetic: the checkpoint names the blocks
    `main.2.0.conv1` and the conv `main.0`, so a flat Sequential would leave
    310 weights unloaded and 302 "unexpected".
    """

    def __init__(self, in_channels, mid_channels=64, num_blocks=30):
        super().__init__()
        self.main = nn.Sequential(
            _conv(in_channels, mid_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Sequential(*[ResidualBlock(mid_channels)
                            for _ in range(num_blocks)]),
        )

    def forward(self, x):
        return self.main(x)


class _OneConv(nn.Module):
    """Single conv exposed as `.conv`, matching the checkpoint's key layout."""

    def __init__(self, in_ch, out_ch, k=7):
        super().__init__()
        self.conv = _conv(in_ch, out_ch, k=k)

    def forward(self, x):
        return self.conv(x)


class SPyNetBasicModule(nn.Module):
    """Five 7x7 convs with ReLU between them, named `.0.conv` ... `.4.conv`.

    The checkpoint wraps each conv in a module called `conv`; registering the
    Conv2d directly names it `basic_module.<i>.weight` and nothing loads.
    """

    def __init__(self):
        super().__init__()
        chans = [(8, 32), (32, 64), (64, 32), (32, 16), (16, 2)]
        self.basic_module = nn.ModuleList(
            [_OneConv(a, b) for a, b in chans])

    def forward(self, x):
        n = len(self.basic_module)
        for i, blk in enumerate(self.basic_module):
            x = blk(x)
            if i < n - 1:
                x = F.relu(x, inplace=False)
        return x


class SPyNet(nn.Module):
    """Coarse-to-fine flow with six pyramid levels.

    Mirrors basicsr/basicsr/archs/spynet_arch.py, which is the reference this
    checkpoint was trained against.  Two details matter and both were wrong in
    the first draft:
      * the inputs are first resized up to a multiple of 32 so every pooling
        level divides evenly, and the flow is scaled back by w/w_floor afterwards;
      * the pyramid is built *coarse first* (`insert(0, ...)`) and the loop walks
        it from the smallest level, so `flow` starts at the coarsest size.
    Getting either wrong yields flows in the thousands of pixels on a
    zero-motion pair, which is what the earlier version did.
    """

    def __init__(self, n_levels=6):
        super().__init__()
        self.n_levels = n_levels
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])
                             .view(1, 3, 1, 1).clone())
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])
                             .view(1, 3, 1, 1).clone())
        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(n_levels)])

    def _preprocess(self, x):
        return (x - self.mean) / self.std

    def process(self, ref, supp):
        import math
        ref = [self._preprocess(ref)]
        supp = [self._preprocess(supp)]
        for _ in range(self.n_levels - 1):
            ref.insert(0, F.avg_pool2d(ref[0], 2, 2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(supp[0], 2, 2, count_include_pad=False))

        flow = ref[0].new_zeros(ref[0].size(0), 2,
                                ref[0].size(2), ref[0].size(3))
        for level in range(len(ref)):
            if level == 0:
                up = flow
            else:
                up = F.interpolate(flow, scale_factor=2, mode="bilinear",
                                   align_corners=True) * 2.0
            flow = self.basic_module[level](torch.cat([
                ref[level],
                _warp(supp[level], up, padding_mode="border"), up], 1)
                ) + up
        return flow

    def forward(self, ref, supp):
        import math
        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)
        ref = F.interpolate(ref, size=(h_floor, w_floor), mode="bilinear",
                            align_corners=False)
        supp = F.interpolate(supp, size=(h_floor, w_floor), mode="bilinear",
                             align_corners=False)
        flow = F.interpolate(self.process(ref, supp), size=(h, w),
                             mode="bilinear", align_corners=False)
        flow[:, 0] *= float(w) / float(w_floor)
        flow[:, 1] *= float(h) / float(h_floor)
        return flow


def _zero_feat(ref, channels):
    """Zero feature map with the spatial size of `ref` (first/last frame)."""
    return torch.zeros(ref.shape[0], channels, ref.shape[2], ref.shape[3],
                       dtype=ref.dtype, device=ref.device)


def _warp(x, flow, padding_mode="zeros"):
    """Backward warp: sample x at (grid + flow).

    `padding_mode` follows mmedit.models.common.flow_warp: the default is
    "zeros" for the feature-propagation branches, while SPyNet internally
    warps the supporting *image* with "border".
    """
    b, _, h, w = x.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=x.device),
                            torch.arange(w, device=x.device), indexing="ij")
    grid = torch.stack((gx, gy), 0).float()
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
    grid = grid + flow
    grid[:, 0] = 2 * grid[:, 0] / max(w - 1, 1) - 1
    grid[:, 1] = 2 * grid[:, 1] / max(h - 1, 1) - 1
    return F.grid_sample(x, grid.permute(0, 2, 3, 1), align_corners=True,
                         mode="bilinear", padding_mode=padding_mode)


class PixelShufflePack(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        self.upsample_conv = _conv(in_ch, out_ch * scale * scale)
        self.scale = scale

    def forward(self, x):
        x = self.upsample_conv(x)
        return F.pixel_shuffle(x, self.scale)


class RealBasicVSR(nn.Module):
    def __init__(self, mid_channels=64, n_cleaning=20,
                 n_forward=20, n_backward=20, dynamic_refine_thres=1.0):
        super().__init__()
        self.image_cleaning = nn.Sequential(
            ResidualBlocksWithInputConv(3, mid_channels, n_cleaning),
            # the checkpoint has a trailing 3x3 conv back to RGB
            # (`image_cleaning.1.weight`), i.e. the cleaning stage is residual
            # in the sense of adding to the LR, not of ending on features
            _conv(mid_channels, 3))
        self.basicvsr = nn.Module()
        self.basicvsr.spynet = SPyNet()
        self.basicvsr.forward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, n_forward)
        self.basicvsr.backward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, n_backward)
        self.basicvsr.fusion = _conv(mid_channels * 2, mid_channels, k=1)
        self.basicvsr.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2)
        self.basicvsr.upsample2 = PixelShufflePack(mid_channels, mid_channels, 2)
        self.basicvsr.conv_hr = _conv(mid_channels, mid_channels)
        self.basicvsr.conv_last = _conv(mid_channels, 3)
        self.dynamic_refine_thres = dynamic_refine_thres

    def forward(self, lqs):
        """lqs: [N, 3, H, W] in [0,1].  Returns [N, 3, 4H, 4W].

        Mirrors basicsr/basicsr/archs/basicvsr_arch.py::forward.  Three details
        there are easy to get wrong and all three matter:
          * the propagation concat is `[current frame, warped feature]`, not the
            other way round;
          * `fusion` is followed by leaky ReLU (output of fusion itself is not
            the feature);
          * the final RGB prediction is added to a bilinear x4 of the *cleaned*
            frames.  Upstream overwrites `lqs` with the cleaned result before
            calling `self.basicvsr(lqs)`, so `img_upsample` inside BasicVSRNet
            also sees the cleaned frames; using `cleaned[i]` here is exact
            upstream behaviour, not an approximation.
        """
        n = lqs.shape[0]
        # Official loop: up to 3 cleaning passes, accumulating the residual into
        # the frames, stopping early once the mean absolute residual drops below
        # dynamic_refine_thres (255/255 by default).  Measured on a real frame the
        # residual abs mean is ~0.0095, far below that threshold, so the loop
        # always exits after the first pass.  The bound is kept for parity with
        # upstream so a lower threshold can enable extra passes.
        cleaned = [lqs[i:i + 1].clone() for i in range(n)]
        for it in range(3):
            res = [self.image_cleaning(cleaned[i]) for i in range(n)]
            for i in range(n):
                cleaned[i] = cleaned[i] + res[i]
            mean_res = torch.stack([r.abs().mean() for r in res]).mean()
            if float(mean_res) < self.dynamic_refine_thres:
                break

        fw = self.basicvsr.forward_resblocks
        bw = self.basicvsr.backward_resblocks

        # backward branch runs first in the reference (it fills out_l for every t)
        back = [None] * n
        feat_prop = torch.zeros(1, 64, cleaned[0].shape[2], cleaned[0].shape[3],
                                dtype=cleaned[0].dtype, device=cleaned[0].device)
        for i in range(n - 1, -1, -1):
            x_i = cleaned[i]
            if i < n - 1:
                flow = self.basicvsr.spynet(cleaned[i], cleaned[i + 1])
                feat_prop = _warp(feat_prop, flow)
            feat_prop = bw(torch.cat([x_i, feat_prop], 1))
            back[i] = feat_prop

        # forward branch
        feats = [None] * n
        feat_prop = torch.zeros(1, 64, cleaned[0].shape[2], cleaned[0].shape[3],
                                dtype=cleaned[0].dtype, device=cleaned[0].device)
        for i in range(n):
            x_i = cleaned[i]
            if i > 0:
                flow = self.basicvsr.spynet(cleaned[i], cleaned[i - 1])
                feat_prop = _warp(feat_prop, flow)
            feat_prop = fw(torch.cat([x_i, feat_prop], 1))
            feats[i] = feat_prop

        outs = []
        for i in range(n):
            x = self.basicvsr.fusion(torch.cat([back[i], feats[i]], 1))
            x = F.leaky_relu(x, negative_slope=0.1, inplace=True)
            x = F.leaky_relu(self.basicvsr.upsample1(x),
                             negative_slope=0.1, inplace=True)
            x = F.leaky_relu(self.basicvsr.upsample2(x),
                             negative_slope=0.1, inplace=True)
            x = F.leaky_relu(self.basicvsr.conv_hr(x), negative_slope=0.1,
                             inplace=True)
            x = self.basicvsr.conv_last(x)
            base = F.interpolate(cleaned[i], scale_factor=4, mode="bilinear",
                                 align_corners=False)
            outs.append(x + base)
        return torch.cat(outs, 0)


def load_rbvsr(ckpt_path, device="cuda"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    sd = {k[len("generator."):]: v for k, v in sd.items()
          if k.startswith("generator.")}
    net = RealBasicVSR()
    missing, unexpected = net.load_state_dict(sd, strict=False)
    net = net.to(device).eval()
    return net, missing, unexpected
