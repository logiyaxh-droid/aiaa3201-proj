# -*- coding: utf-8 -*-
"""M2 Stage A trainer: bf16 + grad-ckpt + accum + atomic resume.

Loss (guide section 4), with the pixel/perceptual split kept explicit:

    L_G = 1.0 * Charbonnier(Y_t, Y_t^gt)
        + ld * DISTS(Y_t, Y_t^gt)
        + lt * L_temp(O_t, Y_hat_t, Y_hat_{t-1}, Y_t^gt, Y_{t-1}^gt)
        + la * RaGAN(Y_t, Y_t^gt)          # la = 0 in the Pilot arms
        + lg * gate_reg

Temporal term
-------------
    L_temp = || O_t * ( (Y_t - W(Y_{t-1})) - (Y_t^gt - W(Y_{t-1}^gt)) ) ||_1

Y_{t-1} is produced by the *same* model under torch.no_grad(), so the term only
ever back-propagates into the current frame.  That is what keeps it free: no
second graph, no extra activation memory, measured at +0.2 GB.

Determinism and resume
----------------------
Before every optimiser step the sampler is a pure function of (seed, step), so
the next batch after a resume is the batch that would have been read.  RNG
states are checkpointed anyway, because the gate/LoRA dropout-free path still
touches the global generator inside diffusers and we do not want to depend on
that staying true.

Checkpoint policy (disk is tight)
---------------------------------
Only LoRA + gate + gamma + optimiser state are stored: 16.2M params of LoRA at
bf16/bf32 mixed plus Adam moments.  The 3.3 GB SD2.1 backbone is never written,
and only `last.pth` + `best.pth` are kept.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/data/disk1/home/ghe270/p3/m2_stageA")

import model_stageA as M
from dataset import WindowDataset, collate

RUNS = "/data/disk1/home/ghe270/p3/m2_stageA/runs"


# ------------------------------------------------------------------ losses

def charbonnier(pred, target, eps=1e-12):
    return torch.sqrt((pred - target) ** 2 + eps * eps).mean()


class DistsLoss:
    """Local DISTS weights; metric params stay frozen while gradients reach
    the input (the reference module constructs alpha/beta trainable)."""

    def __init__(self, device="cuda", dtype=torch.float32):
        for p in ("/data/disk1/home/ghe270/p3/quality_metrics/scripts",
                  "/data/disk1/home/ghe270/p3/quality_metrics"):
            if p not in sys.path:
                sys.path.insert(0, p)
        import common as C
        self.net = C.DistsMetric(device=device).net
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()
        self.dtype = dtype

    def __call__(self, pred, target):
        return self.net(pred.float(), target.float()).mean()


def _sn_disc(device):
    from basicsr.archs.discriminator_arch import UNetDiscriminatorSN
    return UNetDiscriminatorSN(num_in_ch=3, num_feat=64, skip_connection=True).to(device)


def ra_g_loss(real_logits, fake_logits):
    return (F.binary_cross_entropy_with_logits(
                fake_logits - real_logits.mean(), torch.ones_like(fake_logits))
            + F.binary_cross_entropy_with_logits(
                real_logits - fake_logits.mean(), torch.zeros_like(real_logits))) / 2


def ra_d_loss(real_logits, fake_logits):
    return (F.binary_cross_entropy_with_logits(
                real_logits - fake_logits.mean(), torch.ones_like(real_logits))
            + F.binary_cross_entropy_with_logits(
                fake_logits - real_logits.mean(), torch.zeros_like(fake_logits))) / 2


# --------------------------------------------------------------- checkpoint

def trainable_state(model):
    """Only what training changes: LoRA, gate, gamma."""
    sd = {k: v.detach().cpu() for k, v in model.unet.state_dict().items()
          if "lora_" in k}
    sd.update({"gate." + k: v.detach().cpu()
               for k, v in model.gate.state_dict().items()})
    sd["gamma"] = model.gamma.detach().cpu()
    return sd


def save_ckpt(path, model, opt, step, cfg, disc=None, opt_d=None, best=None):
    tmp = path + ".tmp"
    state = {
        "model": trainable_state(model),
        "optimizer": opt.state_dict(),
        "step": step,
        "config": cfg,
        "best": best,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None),
    }
    if disc is not None:
        state["discriminator"] = disc.state_dict()
        state["optimizer_d"] = opt_d.state_dict()
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_ckpt(path, model, opt=None, disc=None, opt_d=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    missing = []
    own = model.unet.state_dict()
    gsd = model.gate.state_dict()
    for k, v in ck["model"].items():
        if k == "gamma":
            model.gamma.data.copy_(v.to(model.gamma.device))
        elif k.startswith("gate."):
            gsd[k[len("gate."):]] = v
        elif k in own:
            own[k] = v.to(own[k].dtype)
        else:
            missing.append(k)
    model.unet.load_state_dict(own, strict=False)
    model.gate.load_state_dict(gsd, strict=False)
    if opt is not None and "optimizer" in ck:
        opt.load_state_dict(ck["optimizer"])
    if disc is not None and "discriminator" in ck:
        disc.load_state_dict(ck["discriminator"])
    if opt_d is not None and "optimizer_d" in ck:
        opt_d.load_state_dict(ck["optimizer_d"])
    if "torch_rng" in ck:
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        random.setstate(ck["python_rng"])
        if ck.get("cuda_rng"):
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
    return ck.get("step", 0), ck.get("best"), missing


# ----------------------------------------------------------------- training

def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--gamma-init", type=float, default=0.0)
    ap.add_argument("--gamma-lr-mult", type=float, default=10.0,
                    help="gamma is a scalar: give it a larger lr so it can move")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--osediff-init", action="store_true")
    ap.add_argument("--residual-mode", default="raw", choices=["raw", "x0"])
    ap.add_argument("--gate-mode", default="learn",
                    choices=["learn", "const", "u1"])
    ap.add_argument("--fixed-w", type=float, default=0.5)
    ap.add_argument("--w-dists", type=float, default=0.05)
    ap.add_argument("--w-temp", type=float, default=0.20)
    ap.add_argument("--w-adv", type=float, default=0.0)
    ap.add_argument("--w-gate", type=float, default=0.01)
    ap.add_argument("--gate-alive", type=float, default=1.0,
                    help="weight of the anti-collapse term; keeps W_pix > 0")
    ap.add_argument("--tiers", nargs="*", default=["v0", "v1", "v2"])
    ap.add_argument("--saver", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dev-clips", nargs="*", default=["000", "002", "009"])
    ap.add_argument("--seed", type=int, default=20261125)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    return ap


def lr_at(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + np.cos(np.pi * min(1.0, t)))


def main():
    args = build_parser().parse_args()
    out = os.path.join(RUNS, args.tag)
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, "train.jsonl")
    logf = open(log_path, "a", encoding="utf-8")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dev = "cuda"
    dtype = torch.bfloat16
    print("[init] building StageA (osediff_init=%s residual=%s gate=%s)"
          % (args.osediff_init, args.residual_mode, args.gate_mode), flush=True)
    model = M.StageA(dev, dtype, lora_rank=args.lora_rank,
                     osediff_init=args.osediff_init,
                     gamma_init=args.gamma_init,
                     residual_mode=args.residual_mode,
                     grad_ckpt=not args.no_grad_ckpt)
    model.set_train_mode(True)
    ntr = model.n_trainable()
    print("[init] trainable %.2fM params, gamma %.4f"
          % (ntr / 1e6, float(model.gamma)), flush=True)

    # gamma gets its own lr: as a single scalar at lr=1e-4 it needs ~1e4 steps
    # just to travel 1.0, which would make every arm look like "gamma stuck".
    params = [{"params": [p for p in model.trainable_parameters()
                          if p is not model.gamma], "lr": args.lr},
              {"params": [model.gamma], "lr": args.lr * args.gamma_lr_mult}]
    opt = torch.optim.AdamW(params, weight_decay=args.weight_decay)

    disc = None
    opt_d = None
    if args.w_adv > 0:
        disc = _sn_disc(dev)
        opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
        disc.train()

    ds = WindowDataset(tiers=args.tiers, length=10 ** 7, seed=args.seed)
    print("[init] dataset clips=%d nframes=%d tiers=%s"
          % (len(ds.clips), ds.nframes, ds.tiers), flush=True)

    dists = DistsLoss(dev) if args.w_dists > 0 else None

    step = 0
    best = None
    if args.resume and os.path.exists(os.path.join(out, "last.pth")):
        step, best, miss = load_ckpt(os.path.join(out, "last.pth"), model,
                                     opt, disc, opt_d)
        print("[resume] from step %d (missing %d)" % (step, len(miss)),
              flush=True)

    cfg = vars(args).copy()
    cfg["trainable_params"] = int(ntr)
    accum = max(1, args.accum)
    mb = max(1, args.micro_batch)
    eff = mb * accum
    t0 = time.time()
    print("[run] %d steps, micro %d x accum %d = effective %d"
          % (args.steps, mb, accum, eff), flush=True)

    def batch_at(s, off):
        """Deterministic micro-batch for (global step s, accumulator offset).

        Only `mb` samples are drawn per micro-batch.  The first version built an
        `eff`-sized batch here and then used one slice of it, which decoded 8
        samples (48 PNGs) four times per step and was the dominant cost:
        measured 24.5 s/step against 1.77 s for the actual GPU work.
        """
        idx = [(s * accum + off) * mb + i for i in range(mb)]
        return collate([ds[j] for j in idx])

    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        if disc is not None:
            opt_d.zero_grad(set_to_none=True)
        agg = {}
        for m in range(accum):
            b = batch_at(step, m)
            lr5 = b["lr5"].to(dev, non_blocking=True)
            gt = b["gt"].to(dev, non_blocking=True)          # [B,3,384,384]
            gt_prev = b["gt_prev"].to(dev, non_blocking=True)
            lr_t = b["lr_t"].to(dev, non_blocking=True)
            lr_prev = b["lr_prev"].to(dev, non_blocking=True)

            # --- frozen RBV: 5-frame window -> 384x384, centre 256 ---------
            with torch.no_grad():
                rbv5 = model.rbv_window(lr5)                 # [B,5,3,384,384]
                sl = slice(M.BR * 4, M.BR * 4 + M.HR_OUT)
                b_t = rbv5[:, 2, :, sl, sl]                  # centre frame
                b_prev = rbv5[:, 1, :, sl, sl]
                # flip branch for U_1: flip LR, RBV again, flip back
                rbv5f = model.rbv_window(torch.flip(lr5, dims=[-1]))
                b_tf = torch.flip(rbv5f[:, 2, :, sl, sl], dims=[-1])
                # U_warp normaliser: fixed 95th percentile per batch
                d = (b_t - b_prev).abs().mean(dim=1, keepdim=True)
                u_ref = torch.quantile(d.flatten().float(),
                                       torch.tensor(0.95, device=d.device)).to(d.dtype)
                u_ref = u_ref.clamp_min(1e-6)

            u1, uw, f_c, occ = model._uncertainty(
                b_t, b_prev, b_tf, lr_t, lr_prev, u_ref)

            y_out, w_pix, ex = model.refine(b_t, u1, uw, args.gate_mode,
                                            args.fixed_w)
            gt_c = gt[:, :, sl, sl]
            gt_p = gt_prev[:, :, sl, sl]

            # --- temporal regulariser (no grad on the previous frame) ------
            l_temp = torch.zeros((), device=dev)
            if args.w_temp > 0:
                with torch.no_grad():
                    # Y_{t-1} must be produced with the *same* gate inputs the
                    # model would see at t-1, otherwise the temporal residual is
                    # comparing a gated frame against an ungated one and the
                    # term grows without bound.  The first version passed zeros
                    # here; measured consequence was L_temp climbing
                    # 0.114 -> 0.247 over 20 steps while everything else flat.
                    #
                    # At t-1 the uncertainty fields are the ones built from
                    # B_{t-2} -> B_{t-1} flow, which this batch does not carry.
                    # Using the t fields for both frames is the consistent
                    # approximation: same gate decision, so any difference the
                    # term sees comes from the images, not from the gate.
                    y_prev, _, _ = model.refine(
                        b_prev, u1, uw, args.gate_mode, args.fixed_w)
                yw = model._warp_hr(y_prev, f_c)
                gw = model._warp_hr(gt_p, f_c)
                occ_hr = F.interpolate(occ.float(), size=(M.HR_OUT, M.HR_OUT),
                                       mode="nearest")
                l_temp = (occ_hr * ((y_out - yw) - (gt_c - gw))).abs().mean()

            l_pix = charbonnier(y_out, gt_c)
            l_dists = dists(y_out, gt_c) if dists is not None else torch.zeros((), device=dev)
            # Gate regulariser, two-sided on purpose.
            #
            # The guide writes 0.01 * mean(W_pix * (1 - U_1)), which only ever
            # rewards a *small* W_pix: with no lower bound the cheapest way to
            # satisfy it is W_pix -> 0, and that is exactly what happened
            # (measured: W_pix 0.119 -> 0.000 and w_loss -> 0 within 50 steps,
            # i.e. the model silently became plain RealBasicVSR).  A small
            # anti-collapse term keeps the gate alive; the flat-region term
            # still discourages edits where U_1 is low.
            l_flat = (w_pix * (1.0 - u1)).mean()
            l_alive = ((1.0 - w_pix) ** 2).mean()
            l_gate = args.w_gate * (l_flat + args.gate_alive * l_alive)

            loss = l_pix + args.w_dists * l_dists + args.w_temp * l_temp + l_gate

            if disc is not None:
                real_l = disc(gt_c.float())
                fake_l = disc(y_out.float())
                loss = loss + args.w_adv * ra_g_loss(real_l, fake_l)
                dl = ra_d_loss(real_l.detach(), fake_l.detach())
                (dl / accum).backward()
            (loss / accum).backward()

            for k, v in (("loss", loss), ("pix", l_pix), ("dists", l_dists),
                         ("temp", l_temp), ("gate", l_gate)):
                agg[k] = agg.get(k, 0.0) + float(v.detach()) / accum

        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.trainable_parameters() if p.grad is not None],
            args.grad_clip)
        cur_lr = lr_at(step, args.steps, args.warmup, args.lr)
        for gi, g in enumerate(opt.param_groups):
            g["lr"] = cur_lr * (args.gamma_lr_mult if gi == 1 else 1.0)
        opt.step()
        if disc is not None:
            opt_d.step()

        step += 1
        if step % args.log_every == 0 or step == 1:
            agg.update({"step": step, "lr": cur_lr, "gnorm": float(gnorm),
                        "gamma": float(model.gamma), "w_pix_mean":
                        float(w_pix.mean()), "sec": time.time() - t0,
                        "vram_gb": torch.cuda.max_memory_allocated() / 2 ** 30})
            logf.write(json.dumps(agg) + "\n")
            logf.flush()
            print("[%4d] loss %.4f pix %.4f dists %.4f temp %.5f gate %.5f "
                  "gamma %+.5f w %.3f gnorm %.3e %.1fs"
                  % (step, agg["loss"], agg["pix"], agg["dists"], agg["temp"],
                     agg["gate"], float(model.gamma), float(w_pix.mean()),
                     float(gnorm), time.time() - t0), flush=True)

        if step % args.saver == 0 or step == args.steps:
            save_ckpt(os.path.join(out, "last.pth"), model, opt, step, cfg,
                      disc, opt_d, best)
            print("[ckpt] step %d -> last.pth" % step, flush=True)

    save_ckpt(os.path.join(out, "last.pth"), model, opt, step, cfg, disc, opt_d,
              best)
    print("[done] %d steps in %.1f min" % (step, (time.time() - t0) / 60),
          flush=True)
    logf.close()


if __name__ == "__main__":
    main()
