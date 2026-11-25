# -*- coding: utf-8 -*-
"""M2-B trainer: pixel-space uncertainty-weighted multi-frame fine-tuning.

Loss (guide section 2.3), on the middle 3 of 5 frames so neither end of the
bidirectional propagation is used as a target:

    L_G = lp * L_pix + ld * L_DISTS + lt * L_temp + la * L_adv_g

    L_pix   = mean( w_pix(u) * Charbonnier(Y_hat, Y) )
    L_temp  = mean over pairs (2<-1, 3<-2) of  sum(O*|dY_hat - dY_gt|)/sum(O)
    L_adv_g = mean( w_adv(u) * relativistic map(loss) )     # per-pixel map

Why idx = [1,2,3]
-----------------
RealBasicVSR propagates features in BOTH directions. Frame 0 only ever receives
a backward pass and frame 4 only a forward pass, so their outputs are the least
constrained; they are also where a temporal loss has only one neighbour available.
Using the middle three gives every target frame a neighbour on both sides and two
gradient-carrying temporal pairs.

Mean-aligned ablation
---------------------
The `uniform` arm replaces w_pix / w_adv with their own batch spatial MEAN, so
the two arms have identical total loss magnitude and any difference is purely the
spatial allocation. This is what makes B1 vs B2 a causal comparison instead of a
weight-scale comparison.

Checkpoint policy
-----------------
Only the trainable propagation branch + discriminator, ~30M + 4.4M params. The
frozen cleaning/spynet weights and the reference model are never written.
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
M2B = os.path.dirname(HERE)
sys.path.insert(0, M2B)

import warp as W                                       # noqa: E402
from model_rbv_ft import UARBVSR                       # noqa: E402

M2A = "/data/disk1/home/ghe270/p3/m2_stageA"
sys.path.insert(0, os.path.join(M2A, "scripts"))
SRC_GT = "/data/disk1/shared/aiaa3201/project1_video_sr/train_sample_sharp"
LR_ROOT = os.path.join(M2A, "data", "train_lr")
RUNS = os.path.join(M2B, "runs")
MID = [1, 2, 3]
TEMP_PAIRS = [(2, 1), (3, 2)]


# ------------------------------------------------------------------ dataset

class Window5(torch.utils.data.Dataset):
    """5-frame LR window -> 96x96 crop; returns the LR window and the HR GT.

    Reuses the exact geometry of m2_stageA/dataset.py: LR crop 96 -> RBV 384 ->
    centre 256 for training, with the GT crop taken from the same 4x region.
    Sampling is a pure function of (seed, index) so a resume reads the same
    batch that would have been read.
    """

    def __init__(self, tiers=("v0", "v1", "v2"), length=200000, seed=0,
                 crop=96, clips=None, lr_root=LR_ROOT, gt_root=SRC_GT):
        self.tiers = list(tiers)
        self.length = int(length)
        self.crop = int(crop)
        self.lr_root = lr_root
        self.gt_root = gt_root
        avail = sorted(os.listdir(os.path.join(lr_root, self.tiers[0])))
        self.clips = [c for c in (clips or avail)
                      if os.path.isdir(os.path.join(lr_root, self.tiers[0], c))]
        self.nframes = len([f for f in
                            os.listdir(os.path.join(lr_root, self.tiers[0],
                                                    self.clips[0]))
                            if f.lower().endswith(".png")])
        self.seed = int(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        import cv2
        r = np.random.default_rng(self.seed * 1000003 + index)
        clip = self.clips[int(r.integers(len(self.clips)))]
        tier = self.tiers[int(r.integers(len(self.tiers)))]
        # centre index chosen so a 5-frame window fits
        c0 = int(r.integers(2, self.nframes - 2))
        lo = max(0, min(c0 - 2, self.nframes - 5))
        ids = list(range(lo, lo + 5))

        lr_dir = os.path.join(self.lr_root, tier, clip)
        lr = np.stack([cv2.cvtColor(cv2.imread(os.path.join(
            lr_dir, "%08d.png" % i)), cv2.COLOR_BGR2RGB) for i in ids])
        gt_dir = os.path.join(self.gt_root, clip)
        gts = []
        for i in ids:
            g = cv2.cvtColor(cv2.imread(os.path.join(
                gt_dir, "%08d.png" % i)), cv2.COLOR_BGR2RGB)
            gts.append(g)
        gts = np.stack(gts)

        h, w = lr.shape[1:3]
        c = self.crop
        y = int(r.integers(0, h - c + 1))
        x = int(r.integers(0, w - c + 1))
        lr = lr[:, y:y + c, x:x + c]
        gts = gts[:, y * 4:(y + c) * 4, x * 4:(x + c) * 4]

        a = np.random.default_rng(self.seed * 7331 + index)
        if a.random() < 0.5:
            lr = lr[:, :, ::-1]
            gts = gts[:, :, ::-1]
            # NOTE: a horizontal flip is applied to LR and GT together, so the
            # pair stays aligned. Flipping only one of them would silently train
            # the model to mirror the content.

        def to_t(arr):
            return torch.from_numpy(np.ascontiguousarray(arr)) \
                .permute(0, 3, 1, 2).float() / 255.0

        return {"lr5": to_t(lr), "gt5": to_t(gts),
                "meta": {"clip": clip, "tier": tier, "frames": ids}}


def collate(batch):
    return {"lr5": torch.stack([b["lr5"] for b in batch]),
            "gt5": torch.stack([b["gt5"] for b in batch]),
            "meta": [b["meta"] for b in batch]}


# ------------------------------------------------------------------ losses

def charbonnier(pred, target, eps=1e-12):
    return torch.sqrt((pred - target) ** 2 + eps * eps)


class DistsLoss:
    def __init__(self, device="cuda"):
        for p in ("/data/disk1/home/ghe270/p3/quality_metrics/scripts",
                  "/data/disk1/home/ghe270/p3/quality_metrics"):
            if p not in sys.path:
                sys.path.insert(0, p)
        import common as C
        self.net = C.DistsMetric(device=device).net
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()

    def __call__(self, pred, target):
        return self.net(pred.float(), target.float()).mean()


def sn_discriminator(device="cuda"):
    """SN-UNet; outputs a per-pixel [B,1,H,W] map, which is what lets the
    adversarial term be weighted spatially."""
    import types
    import torchvision
    _mod = types.ModuleType("torchvision.transforms.functional_tensor")
    from torchvision.transforms import functional as _F
    for _n in dir(_F):
        setattr(_mod, _n, getattr(_F, _n))
    sys.modules["torchvision.transforms.functional_tensor"] = _mod
    from basicsr.archs.discriminator_arch import UNetDiscriminatorSN
    return UNetDiscriminatorSN(num_in_ch=3, num_feat=64,
                               skip_connection=True).to(device)


def ra_map_g(real_l, fake_l):
    """Per-pixel relativistic generator map (not reduced: caller weights it)."""
    return 0.5 * (
        F.binary_cross_entropy_with_logits(
            fake_l - real_l.mean(), torch.ones_like(fake_l), reduction="none")
        + F.binary_cross_entropy_with_logits(
            real_l - fake_l.mean(), torch.zeros_like(real_l),
            reduction="none"))


def ra_loss_d(real_l, fake_l):
    return 0.5 * (
        F.binary_cross_entropy_with_logits(
            real_l - fake_l.mean(), torch.ones_like(real_l))
        + F.binary_cross_entropy_with_logits(
            fake_l - real_l.mean(), torch.zeros_like(fake_l)))


# --------------------------------------------------------------- checkpoint

def ckpt_state(model, disc=None):
    st = {"net": {k: v.detach().cpu() for k, v in model.net.state_dict().items()}}
    if disc is not None:
        st["disc"] = disc.state_dict()
    return st


def save_ckpt(path, model, opt, step, cfg, disc=None, opt_d=None, best=None):
    tmp = path + ".tmp"
    torch.save({"model": ckpt_state(model, disc), "optimizer": opt.state_dict(),
                "step": step, "config": cfg, "best": best,
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None}, tmp)
    os.replace(tmp, path)


def load_ckpt(path, model, opt=None, disc=None, opt_d=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    own = model.net.state_dict()
    for k, v in ck["model"]["net"].items():
        if k in own:
            own[k] = v
    model.net.load_state_dict(own, strict=False)
    if disc is not None and "disc" in ck["model"]:
        disc.load_state_dict(ck["model"]["disc"])
    if opt is not None and "optimizer" in ck:
        opt.load_state_dict(ck["optimizer"])
    if opt_d is not None and "optimizer_d" in ck:
        opt_d.load_state_dict(ck["optimizer_d"])
    if "torch_rng" in ck:
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        random.setstate(ck["python_rng"])
        if ck.get("cuda_rng"):
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
    return ck.get("step", 0), ck.get("best")


# ------------------------------------------------------------------- train

def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr-d", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lr-floor", type=float, default=5e-6)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--w-pix", type=float, default=0.50)
    ap.add_argument("--w-dists", type=float, default=0.15)
    ap.add_argument("--w-temp", type=float, default=0.30)
    ap.add_argument("--w-adv", type=float, default=0.003)
    ap.add_argument("--mode", default="uncertain",
                    choices=["uncertain", "uniform", "shuffled", "none"])
    ap.add_argument("--shuffle-seed", type=int, default=12345)
    ap.add_argument("--tiers", nargs="*", default=["v0", "v1", "v2"])
    ap.add_argument("--saver", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=20261125)
    return ap


def lr_at(step, total, warmup, base, floor):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return floor + (base - floor) * 0.5 * (1 + np.cos(np.pi * min(1.0, t)))


def main():
    args = parser().parse_args()
    out = os.path.join(RUNS, args.tag)
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "train.jsonl"), "a", encoding="utf-8")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dev = "cuda"
    print("[init] building UARBVSR (mode=%s)" % args.mode, flush=True)
    model = UARBVSR(dev)
    ntr = model.n_trainable()
    print("[init] trainable %.2fM params" % (ntr / 1e6), flush=True)
    grp = model.frozen_report()
    for k in sorted(grp):
        tot, tr = grp[k]
        print("   %-28s %8.2fM  trainable=%s"
              % (k, tot / 1e6, "YES" if tr else "no"), flush=True)

    opt = torch.optim.Adam(model.trainable_parameters(), lr=args.lr,
                           betas=(0.9, 0.99))
    disc = None
    opt_d = None
    if args.w_adv > 0:
        disc = sn_discriminator(dev)
        disc.train()
        opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr_d,
                                 betas=(0.9, 0.99))

    ds = Window5(tiers=args.tiers, length=10 ** 8, seed=args.seed)
    print("[init] dataset clips=%d nframes=%d" % (len(ds.clips), ds.nframes),
          flush=True)
    dists = DistsLoss(dev) if args.w_dists > 0 else None

    step = 0
    best = None
    if args.resume and os.path.exists(os.path.join(out, "last.pth")):
        step, best = load_ckpt(os.path.join(out, "last.pth"), model, opt, disc,
                               opt_d)
        print("[resume] from step %d" % step, flush=True)

    cfg = vars(args).copy()
    cfg["trainable_params"] = int(ntr)
    mb, accum = max(1, args.micro_batch), max(1, args.accum)
    sl = slice(W.BR * 4, W.BR * 4 + W.HR_OUT)
    t0 = time.time()
    print("[run] %d steps, micro %d x accum %d = %d"
          % (args.steps, mb, accum, mb * accum), flush=True)

    def batch_at(s, off):
        idx = [(s * accum + off) * mb + i for i in range(mb)]
        return collate([ds[j] for j in idx])

    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        if disc is not None:
            opt_d.zero_grad(set_to_none=True)
        agg = {}

        for m in range(accum):
            b = batch_at(step, m)
            lr5 = b["lr5"].to(dev)                    # [B,5,3,96,96]
            gt5 = b["gt5"].to(dev)                    # [B,5,3,384,384]

            # fixed uncertainty from the frozen reference
            with torch.no_grad():
                B, N = lr5.shape[0], lr5.shape[1]
                b_ref, b_ref_f = [], []
                for bi in range(B):
                    r, rf = model.reference_forward(lr5[bi])
                    b_ref.append(r[:, :, sl, sl])
                    b_ref_f.append(rf[:, :, sl, sl])
                b_ref = torch.stack(b_ref, 0)          # [B,5,3,256,256]
                b_ref_f = torch.stack(b_ref_f, 0)
                u = W.uncertainty_u1(b_ref.flatten(0, 1),
                                     b_ref_f.flatten(0, 1))   # [B*5,1,256,256]
                u = u.view(B, N, 1, W.HR_OUT, W.HR_OUT)

                if args.mode == "uncertain":
                    w_pix_map = W.pix_weight(u)
                    w_adv_map = W.adv_weight(u)
                elif args.mode == "uniform":
                    # same total magnitude, no spatial structure
                    w_pix_map = W.pix_weight(u).mean(dim=(1, 2, 3, 4),
                                                     keepdim=True).expand_as(u)
                    w_adv_map = W.adv_weight(u).mean(dim=(1, 2, 3, 4),
                                                     keepdim=True).expand_as(u)
                elif args.mode == "shuffled":
                    g = torch.Generator(device=u.device)
                    g.manual_seed(args.shuffle_seed + step)
                    perm = torch.randperm(u.shape[-1] * u.shape[-2],
                                          device=u.device, generator=g)
                    us = u.flatten(3)[..., perm].view_as(u)
                    w_pix_map = W.pix_weight(us)
                    w_adv_map = W.adv_weight(us)
                else:                                      # "none"
                    w_pix_map = torch.ones_like(u)
                    w_adv_map = torch.ones_like(u)

            # differentiable forward through cleaning(no_grad)+prop(trainable)
            pred = []
            for bi in range(B):
                p = model.forward_trainable(lr5[bi])
                pred.append(p[:, :, sl, sl])
            pred = torch.stack(pred, 0)                    # [B,5,3,256,256]
            gt_c = gt5[:, :, :, sl, sl]

            mid = slice(1, 4)
            p_mid = pred[:, mid]
            g_mid = gt_c[:, mid]
            w_pix_mid = w_pix_map[:, mid]
            w_adv_mid = w_adv_map[:, mid]

            l_pix = (w_pix_mid * charbonnier(p_mid, g_mid)).mean()
            l_dists = (dists(p_mid.reshape(-1, 3, W.HR_OUT, W.HR_OUT),
                             g_mid.reshape(-1, 3, W.HR_OUT, W.HR_OUT))
                       if dists is not None else torch.zeros((), device=dev))

            # multi-frame end-to-end temporal term
            l_temp = torch.zeros((), device=dev)
            if args.w_temp > 0:
                acc_t = []
                for (ti, pi) in TEMP_PAIRS:
                    for bi in range(B):
                        flow, occ = W.flow_and_occ(
                            model.net.basicvsr.spynet,
                            lr5[bi, ti:ti + 1], lr5[bi, pi:pi + 1], dev)
                        ph = W.warp_hr(pred[bi, pi:pi + 1], flow)
                        gh = W.warp_hr(gt_c[bi, pi:pi + 1], flow)
                        # flow_and_occ returns a 64x64 mask (the LR region that
                        # maps onto the 256x256 HR crop), so it must be
                        # upsampled before weighting HR tensors.  Without this
                        # the term raises "size of tensor a (64) must match
                        # b (256)".
                        occ_hr = F.interpolate(
                            occ.float(), size=(W.HR_OUT, W.HR_OUT),
                            mode="nearest")
                        num = (occ_hr * ((pred[bi, ti:ti + 1] - ph)
                                         - (gt_c[bi, ti:ti + 1] - gh))).abs().sum()
                        den = occ_hr.sum() + 1e-6
                        acc_t.append(num / den)
                l_temp = torch.stack(acc_t).mean()

            loss = (args.w_pix * l_pix + args.w_dists * l_dists
                    + args.w_temp * l_temp)

            l_adv_g = torch.zeros((), device=dev)
            l_adv_d = torch.zeros((), device=dev)
            if disc is not None:
                for p_ in disc.parameters():
                    p_.requires_grad_(False)
                pm = p_mid.reshape(-1, 3, W.HR_OUT, W.HR_OUT)
                gm = g_mid.reshape(-1, 3, W.HR_OUT, W.HR_OUT)
                real_l = disc(gm).detach()
                fake_l = disc(pm)
                g_map = ra_map_g(real_l, fake_l)          # [K,1,H,W]
                wa = w_adv_mid.reshape(-1, 1, W.HR_OUT, W.HR_OUT)
                l_adv_g = (wa * g_map).mean()
                loss = loss + args.w_adv * l_adv_g

            (loss / accum).backward()

            if disc is not None:
                for p_ in disc.parameters():
                    p_.requires_grad_(True)
                real_d = disc(gm)
                fake_d = disc(pm.detach())
                l_adv_d = ra_loss_d(real_d, fake_d)
                (l_adv_d / accum).backward()

            for k, v in (("loss", loss), ("pix", l_pix), ("dists", l_dists),
                         ("temp", l_temp), ("adv_g", l_adv_g),
                         ("adv_d", l_adv_d)):
                agg[k] = agg.get(k, 0.0) + float(v.detach()) / accum
            agg["u_mean"] = float(u.mean())
            agg["w_pix_mean"] = float(w_pix_map.mean())

        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in model.trainable_parameters() if p.grad is not None],
            args.grad_clip)
        cur = lr_at(step, args.steps, args.warmup, args.lr, args.lr_floor)
        for g in opt.param_groups:
            g["lr"] = cur
        opt.step()
        if disc is not None:
            ratio = cur / max(1e-12, args.lr)
            for g in opt_d.param_groups:
                g["lr"] = args.lr_d * ratio
            opt_d.step()

        step += 1
        if step % args.log_every == 0 or step == 1:
            agg.update({"step": step, "lr": cur, "gnorm": float(gn),
                        "sec": time.time() - t0,
                        "vram_gb": torch.cuda.max_memory_allocated() / 2 ** 30})
            logf.write(json.dumps(agg) + "\n")
            logf.flush()
            print("[%4d] loss %.4f pix %.4f dists %.4f temp %.5f adv_g %.4f "
                  "adv_d %.4f u %.3f w %.3f gnorm %.2e %.1fs"
                  % (step, agg["loss"], agg["pix"], agg["dists"], agg["temp"],
                     agg["adv_g"], agg["adv_d"], agg["u_mean"],
                     agg["w_pix_mean"], float(gn), time.time() - t0),
                  flush=True)

        if step % args.saver == 0 or step == args.steps:
            save_ckpt(os.path.join(out, "last.pth"), model, opt, step, cfg,
                      disc, opt_d, best)
            print("[ckpt] %d -> last.pth" % step, flush=True)

    save_ckpt(os.path.join(out, "last.pth"), model, opt, step, cfg, disc, opt_d,
              best)
    print("[done] %d steps in %.1f min" % (step, (time.time() - t0) / 60),
          flush=True)
    logf.close()


if __name__ == "__main__":
    main()
