# -*- coding: utf-8 -*-
"""M2 Stage A model: zero-init latent-residual LoRA + uncertainty gate.

The mechanism, in one line
--------------------------
    Y_out = B_t + W_pix * ( Decode( z_base + gamma * M_lat * dZ ) - B_t )

where B_t is the frozen RealBasicVSR output for the current frame.  At
initialisation gamma = 0, so Y_out is *bit-identical* to B_t and training starts
at the M1 RealBasicVSR operating point (20.822 dB / LPIPS 0.4881 / E_warp 6.346)
instead of falling into the -1.24 dB hole that zero-shot RBV->OSEDiff fell into.

Why the pieces are frozen
-------------------------
M1 established that a second full diffusion redraw destroys what RealBasicVSR
already restored (PSNR -1.24 dB, E_warp 6.346 -> 8.352).  Stage A therefore
never re-generates the image: it only predicts a small latent *correction*, and
the gate decides per pixel how much of the decoded correction to keep.  Only
three things train: the UNet LoRA, the gate head, and the scalar gamma.

Numerical choice that matters
-----------------------------
`residual_mode`:
  raw   dZ = UNet(z_base, t=999, c_empty)            (guide default)
  x0    dZ = x0_from_eps(...) - z_base               (OSEDiff's own semantics)
At t=999 alpha_bar_T is ~0.0047, so x0_from_eps divides by sqrt(alpha_bar) and
amplifies by ~14.6x.  That amplification is exactly the U-TRD style blow-up the
guide warns about, so `raw` is the default.  Both are kept because the P3 arm
initialises LoRA from osediff.pkl, whose weights were trained to predict *noise*,
not a latent residual; whichever mode keeps the step-0 |dZ| in a sane range wins.
The magnitude of both is measured in selfcheck.py rather than assumed.

Geometry (why 96 in, 256 out)
-----------------------------
RBV is convolutional with flow warping at its patch borders.  Feeding exactly
80x80 and taking all 320x320 would expose border effects.  So: crop 96x96 from
LR, RBV -> 384x384, then centre-crop 256x256.  The discarded 8px ring (LR) /
64px ring (HR) absorbs the border artefacts, and no 720p RBV cache is needed on
disk.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

P3 = "/data/disk1/home/ghe270/p3"
PRIOR = os.path.join(P3, "prior_qualification")
sys.path.insert(0, os.path.join(PRIOR, "scripts"))
sys.path.insert(0, os.path.join(P3, "quality_metrics", "scripts"))
sys.path.insert(0, "/data/disk1/home/ghe270/p0")
# OSEDiff needs diffusers 0.25; keep the shim in exactly one place.
_OLD = os.path.join(PRIOR, "site_old")
if os.path.isdir(os.path.join(_OLD, "diffusers")):
    sys.path.insert(0, _OLD)

SD21 = os.path.join(PRIOR, "weights", "SD21BASE")
OSEDIFF_PKL = os.path.join(PRIOR, "vendor", "OSEDiff-main", "preset", "models", "osediff.pkl")
RBVSR_CKPT = os.path.join(PRIOR, "weights_rbvsr", "RealBasicVSR_x4.pth")

LR_CROP = 96        # LR pixels fed to RBV
HR_OUT = 256        # centre crop from the 384x384 RBV output
BR = (LR_CROP - HR_OUT // 4) // 2   # 16: border in LR pixels


# --------------------------------------------------------------------- RBV

def load_rbvsr(device="cuda", dtype=torch.float32):
    """Frozen RealBasicVSR in eval mode.

    Kept in fp32 on purpose.  `rbvsr_net._warp` builds its sampling grid with a
    hard `.float()`, so a bf16 forward raises
        RuntimeError: expected scalar type BFloat16 but found Float
    (measured).  M1 validated this module in fp32 only, and rewriting the M1
    warp is not worth it: a 5-frame 96x96 window peaks at 0.16 GB in fp32, which
    is 2% of the 10 GB budget.  bf16 buys nothing here.
    """
    from rbvsr_net import load_rbvsr as _load
    net, missing, unexpected = _load(RBVSR_CKPT, device="cpu")
    if missing:
        raise RuntimeError("RBV missing keys: %s" % list(missing)[:5])
    net = net.to(device=device, dtype=dtype).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


# ------------------------------------------------------------------- SD 2.1

def empty_prompt_embed(device="cuda", dtype=torch.bfloat16):
    """Precompute the CLIP embedding of "" once, then drop the text encoder.
    Saves ~3.5 GB (1.3 GB of weights plus its activation workspace)."""
    from transformers import CLIPTextModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(SD21, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(SD21, subfolder="text_encoder").to(device)
    with torch.no_grad():
        ids = tok("", max_length=tok.model_max_length, padding="max_length",
                  truncation=True, return_tensors="pt").input_ids.to(device)
        emb = te(ids)[0]
    emb = emb.to(dtype).detach()
    del te
    torch.cuda.empty_cache()
    return emb          # [1, 77, 1024]


def build_unet(train_lora=True, gradient_checkpointing=True, device="cuda",
               dtype=torch.bfloat16, osediff_init=False, lora_rank=16):
    """SD2.1 UNet, backbone frozen, LoRA trainable.

    osediff_init=True reproduces the *exact* adapter layout of osediff.pkl
    (three named adapters plus the rank stored in the checkpoint); otherwise a
    fresh r=lora_rank adapter is attached whose lora_B is zero, so the UNet
    output starts unchanged.
    """
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(SD21, subfolder="unet")
    unet.requires_grad_(False)
    from peft import LoraConfig

    if osediff_init:
        ck = torch.load(OSEDIFF_PKL, map_location="cpu", weights_only=False)
        r = ck["rank_unet"]
        for name, key in (("default_encoder", "unet_lora_encoder_modules"),
                          ("default_decoder", "unet_lora_decoder_modules"),
                          ("default_others", "unet_lora_others_modules")):
            cfg = LoraConfig(r=r, init_lora_weights="gaussian",
                             target_modules=ck[key])
            unet.add_adapter(cfg, adapter_name=name)
        loaded = 0
        for n, p in unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                if n in ck["state_dict_unet"]:
                    p.data.copy_(ck["state_dict_unet"][n])
                    loaded += 1
        unet.set_adapter(["default_encoder", "default_decoder", "default_others"])
        del ck
        print("[unet] osediff LoRA copied into %d tensors (rank %d)" % (loaded, r))
    else:
        pats = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
                "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in",
                "ff.net.2", "ff.net.0.proj"]
        names = [n.replace(".weight", "") for n, _ in unet.named_parameters()
                 if "bias" not in n and "norm" not in n
                 and any(pt in n for pt in pats)]
        cfg = LoraConfig(r=lora_rank, lora_alpha=lora_rank,
                         init_lora_weights=True,   # lora_B = 0 -> identity
                         target_modules=names)
        unet.add_adapter(cfg, adapter_name="default")
        unet.set_adapter(["default"])

    for n, p in unet.named_parameters():
        p.requires_grad_(train_lora and "lora_" in n)

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    unet = unet.to(device=device, dtype=dtype)
    return unet


def build_vae(device="cuda", dtype=torch.bfloat16, gradient_checkpointing=True):
    """Frozen VAE.  The *decoder* must stay differentiable: the loss reaches
    z_hat through it, so its weights are frozen but its graph is not."""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(SD21, subfolder="vae")
    vae.requires_grad_(False)
    if gradient_checkpointing:
        try:
            vae.enable_gradient_checkpointing()
        except Exception:
            pass
    vae = vae.to(device=device, dtype=dtype)
    return vae


# ------------------------------------------------------------------ gate head

class GateHead(nn.Module):
    """U_t -> (W_pix, M_lat).

    Input 5 channels: [B_t (3), U_1, U_warp].
    Final conv bias = -2.0 so W_pix starts near sigmoid(-2) = 0.12: the model
    begins almost identity and has to *earn* its edits.
    """

    def __init__(self, ch=32, in_ch=5, lat_stride=8, bias_init=-2.0):
        super().__init__()
        self.lat_stride = lat_stride
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1), nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(inplace=True),
        )
        self.pix_head = nn.Conv2d(ch, 1, 1)
        nn.init.zeros_(self.pix_head.weight)
        nn.init.constant_(self.pix_head.bias, bias_init)

    def forward(self, x):
        feat = self.body(x)
        w_pix = torch.sigmoid(self.pix_head(feat))
        if self.lat_stride > 1:
            m_lat = F.avg_pool2d(w_pix, self.lat_stride, self.lat_stride,
                                 ceil_mode=True)
        else:
            m_lat = w_pix
        return w_pix, m_lat


# --------------------------------------------------------------- the model

class _Cap:
    """Collects every module's input/output during one no_grad pass."""

    def __init__(self):
        self.pairs = []

    def hook(self, name):
        def fn(mod, inp, out):
            if inp and torch.is_tensor(inp[0]) and torch.is_tensor(out):
                self.pairs.append((name, inp[0].detach(), out.detach()))
        return fn


class StageA(nn.Module):
    def __init__(self, device="cuda", dtype=torch.bfloat16, lora_rank=16,
                 osediff_init=False, gate_ch=32, gamma_init=0.0,
                 residual_mode="raw", grad_ckpt=True):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.residual_mode = residual_mode

        self.rbvsr = load_rbvsr(self.device, torch.float32)
        self.vae = build_vae(self.device, dtype, grad_ckpt)
        self.unet = build_unet(True, grad_ckpt, self.device, dtype,
                               osediff_init, lora_rank)
        self.gate = GateHead(ch=gate_ch).to(self.device, dtype)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init),
                                               device=self.device,
                                               dtype=torch.float32))
        self.emb = empty_prompt_embed(self.device, dtype)   # [1,77,1024]
        self.scale = float(getattr(self.vae.config, "scaling_factor", 0.18215))

    # -- helpers ----------------------------------------------------------
    def trainable_parameters(self):
        ps = [p for p in self.unet.parameters() if p.requires_grad]
        ps += [p for p in self.gate.parameters() if p.requires_grad]
        ps += [self.gamma]
        return ps

    def n_trainable(self):
        return sum(p.numel() for p in self.trainable_parameters())

    def _pooled_emb(self, b):
        return self.emb.expand(b, -1, -1) if self.emb.shape[0] != b else self.emb

    @torch.no_grad()
    def rbv_window(self, lr5):
        """lr5: [B,5,3,H,W] in [0,1] -> [B,5,3,4H,4W] bf16.

        RealBasicVSR.forward takes ONE sequence ([N,3,H,W]) and propagates over
        it, so the batch and the 5-frame window are unrolled here.  Doing it as
        a single fused batch would silently mix two independent clips into one
        bidirectional propagation, which is a correctness bug, not just a shape
        error -- the flow between clip A's last frame and clip B's first frame
        does not exist.
        """
        outs = [self.rbvsr(lr5[i].float()) for i in range(lr5.shape[0])]
        return torch.stack(outs, 0).to(self.dtype)

    def _flow_and_occ(self, lr_t, lr_prev):
        """SpyNet flow (prev->cur, LR px) plus an occlusion mask on the same
        96x96 grid, then centre-cropped to the 64x64 region that maps onto the
        256x256 HR output.

        The consistency test is a SUM, not a difference:
            |f_fwd(p) + f_bwd(p + f_fwd(p))| <= tol
        Written as a subtraction it reports ~14 px on identical flow and the
        mask never fires -- a M1 bug that fails silently.
        """
        spy = self.rbvsr.basicvsr.spynet
        # SpyNet is single-pair only: unroll the batch.
        n = lr_t.shape[0]
        fwd = [spy(lr_t[i:i + 1], lr_prev[i:i + 1]) for i in range(n)]
        bwd = [spy(lr_prev[i:i + 1], lr_t[i:i + 1]) for i in range(n)]
        f_fwd = torch.cat(fwd, 0)           # warps prev -> cur
        f_bwd = torch.cat(bwd, 0)           # warps cur  -> prev
        # Works at any LR size: training feeds 96x96 crops, inference feeds the
        # whole 320x180 LR frame.  Hard-coding LR_CROP here made full-frame
        # inference raise "size of tensor a (96) must match b (320)".
        #
        # Grid convention: meshgrid(arange(H), arange(W), indexing="ij") gives
        # an (H, W) pair.  Note that for 320x180 the earlier call passed W first
        # purely because the crop was square and the bug was invisible there.
        H, W = f_fwd.shape[-2:]
        gx, gy = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=f_fwd.dtype),
            torch.arange(W, device=self.device, dtype=f_fwd.dtype),
            indexing="ij")
        bx = (gx[None, None] + f_fwd[:, 0:1]).clamp(0, W - 1)
        by = (gy[None, None] + f_fwd[:, 1:2]).clamp(0, H - 1)
        nx = (bx / max(W - 1, 1)) * 2 - 1
        ny = (by / max(H - 1, 1)) * 2 - 1
        grid = torch.cat([nx, ny], dim=1).permute(0, 2, 3, 1)
        b_back = F.grid_sample(f_bwd, grid, mode="bilinear",
                               padding_mode="border", align_corners=True)
        fb = (f_fwd + b_back).abs().sum(dim=1, keepdim=True)   # sum, see above
        occ = (fb <= 3.0).to(f_fwd.dtype)
        # centre crop only when the LR input is the training crop size
        if W > HR_OUT // 4:
            sl = slice(BR, BR + HR_OUT // 4)
            return f_fwd[:, :, sl, sl], occ[:, :, sl, sl]
        return f_fwd, occ

    def _warp_hr(self, x, flow_lr_c):
        """Warp an HR tensor with a centre-cropped LR flow (LR px units)."""
        b, _, h, w = x.shape
        s = h / flow_lr_c.shape[-1]               # 4 at train, 4 at inference
        # The grid must carry x's dtype: grid_sample requires input and grid to
        # match.  arange defaults to int64, so the cast has to be explicit.
        f = F.interpolate(flow_lr_c.to(x.dtype), size=(h, w),
                          mode="bilinear", align_corners=False) * s
        # (H, W) order: meshgrid(arange(h), arange(w)) -- see the note in
        # _flow_and_occ; passing w first only worked for square crops.
        gx, gy = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=x.dtype),
            torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
        px = (gx[None, None] + f[:, 0:1]).clamp(0, w - 1)
        py = (gy[None, None] + f[:, 1:2]).clamp(0, h - 1)
        nx = (px / (w - 1)) * 2 - 1
        ny = (py / (h - 1)) * 2 - 1
        grid = torch.cat([nx, ny], dim=1).permute(0, 2, 3, 1)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border",
                             align_corners=True)

    def _uncertainty(self, b_t, b_prev, b_t_flip, lr_t, lr_prev, u_ref):
        """U_1 (flip disagreement) and U_warp (motion-compensated residual)."""
        u1 = (b_t - b_t_flip).abs().mean(dim=1, keepdim=True)
        u1 = F.avg_pool2d(u1, 3, 1, 1)
        u1 = (u1 / u_ref.clamp_min(1e-6)).clamp(0, 1)

        f_c, occ_c = self._flow_and_occ(lr_t, lr_prev)
        warped = self._warp_hr(b_prev.to(self.dtype), f_c.to(self.dtype))
        occ = F.interpolate(occ_c.float(),
                            size=(b_t.shape[-2], b_t.shape[-1]),
                            mode="nearest").to(b_t.dtype)
        uw = (b_t - warped).abs().mean(dim=1, keepdim=True) * occ
        uw = F.avg_pool2d(uw, 3, 1, 1)
        uw = (uw / u_ref.clamp_min(1e-6)).clamp(0, 1)
        return u1, uw, f_c, occ

    # -- forward ----------------------------------------------------------
    def refine(self, b_t, u1, uw, gate_mode="learn", fixed_w=0.5):
        """One latent-residual step for one frame batch.

        `b_t` is [B,3,256,256] in [0,1].  Returns (Y_out, W_pix, extras).
        """
        b, _, h, w = b_t.shape
        x = b_t.to(self.dtype) * 2 - 1            # VAE wants [-1,1]

        with torch.no_grad():                     # encoder is frozen
            z_base = (self.vae.encode(x).latent_dist.mode()
                      * self.scale).detach()

        dZ = self.unet(z_base, torch.tensor(999, device=self.device),
                       encoder_hidden_states=self._pooled_emb(b)).sample
        if self.residual_mode == "x0":
            ab = self.vae_alpha_bar(z_base)
            x0 = (z_base - (1 - ab).sqrt() * dZ) / ab.sqrt()
            dZ = x0 - z_base

        if gate_mode == "learn":
            w_pix, m_lat = self.gate(torch.cat([b_t.to(self.dtype), u1, uw], 1))
        elif gate_mode == "const":
            w_pix = torch.full((b, 1, h, w), float(fixed_w),
                               device=self.device, dtype=self.dtype)
            m_lat = F.avg_pool2d(w_pix, 8, 8, ceil_mode=True)
        elif gate_mode == "u1":
            w_pix = (1.0 - u1).to(self.dtype)
            m_lat = F.avg_pool2d(w_pix, 8, 8, ceil_mode=True)
        else:
            raise ValueError(gate_mode)

        z_hat = z_base + self.gamma.to(self.dtype) * m_lat * dZ

        # Gate the *decoded change*, not the absolute decoded image.
        #
        # The guide writes Y_out = B_t + W_pix*(Y_diff - B_t), which is only
        # identity at gamma=0 if Y_diff == B_t.  It is not: the SD2.1 VAE is
        # lossy, and decode(encode(B_t)) differs from B_t by 0.045 mean /
        # 0.81 max (measured on a real frame, identical in fp32 and bf16).
        # Multiplied by W_pix ~ 0.119 that alone produced max|Y-B| = 0.1001,
        # i.e. the model would start 0.10 away from RealBasicVSR instead of on
        # it.  Both decodes share the same VAE, so subtracting them cancels the
        # round-trip error exactly and leaves only the residual's effect.
        with torch.no_grad():
            y_base = self.vae.decode(z_base.to(self.dtype) / self.scale).sample
        y_diff = self.vae.decode(z_hat.to(self.dtype) / self.scale).sample

        w_pix = w_pix.to(b_t.dtype)
        delta = (y_diff - y_base).to(b_t.dtype) * 0.5   # unit-scale -> [0,1]
        y_out = b_t + w_pix * delta
        return y_out, w_pix, {"z_base": z_base, "dZ": dZ,
                              "y_diff": ((y_diff + 1) / 2).clamp(0, 1),
                              "y_base": ((y_base + 1) / 2).clamp(0, 1),
                              "delta": delta, "m_lat": m_lat}

    def vae_alpha_bar(self, ref):
        """alpha_bar at t=999, for the x0 residual mode."""
        from diffusers import DDPMScheduler
        if not hasattr(self, "_sched"):
            sh = DDPMScheduler.from_pretrained(SD21, subfolder="scheduler")
            self._sched_ab = sh.alphas_cumprod
        return self._sched_ab[999].to(ref.device, ref.dtype)

    @torch.no_grad()
    def prev_output(self, b_prev, u1, uw, gate_mode="learn", fixed_w=0.5):
        """Y_{t-1} for the temporal regulariser: no graph, near-zero memory."""
        y, _, _ = self.refine(b_prev, u1, uw, gate_mode, fixed_w)
        return y

    def set_train_mode(self, train=True):
        self.train(train)
        self.rbvsr.eval()
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.rbvsr.parameters():
            p.requires_grad_(False)
        if train:
            self.unet.train()
            self.gate.train()
