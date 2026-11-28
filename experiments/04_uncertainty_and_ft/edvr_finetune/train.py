# -*- coding: utf-8 -*-
"""EDVR-L perceptual/GAN fine-tuning on the supplied REDS training clips.

This is deliberately a small, explicit trainer rather than a BasicSR YAML run:
all four loss terms, optimizer selection, sampling and checkpoint contents are
visible here.  It starts from the official REDS EDVR-L weights and never changes
the inference-time generator used by p3/quality_metrics/scripts/predictions.py.

Protocol
--------
* Input: 5 consecutive LR frames, centre target.  LR comes from
  train_sample_bicubic, not an improvised cv2 downsample.
* Random crop: LR 64x64 -> GT 256x256 (the official EDVR training GT size).
  Random horizontal flip and transpose, the official train transforms.
* Loss: Charbonnier + lpips_weight * LPIPS(Alex) + dists_weight * DISTS
  + adv_weight * relativistic average GAN.  Pixel weight is fixed at 1.
* When adv_weight=0 no discriminator is constructed, so the perceptual scan
  cannot accidentally pay for (or learn from) an adversarial branch.
* D has its own Adam optimizer and trains once per generator step using
  real/fake relativistic average GAN losses.  SN-UNet is BasicSR's official
  discriminator implementation.
* Everything needed to resume exactly is checkpointed: model, both optimizers,
  the step, and the RNG states.  Results never silently overwrite another run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time

ROOT = "/data/disk1/home/ghe270/p3/quality_metrics"
sys.path.insert(0, os.path.join(ROOT, "fine_tune", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, "/data/disk1/home/ghe270/p0")

import common as C

# Keep the BasicSR compatibility shim in one place.
import types
import torchvision
_mod = types.ModuleType("torchvision.transforms.functional_tensor")
from torchvision.transforms import functional as _F  # noqa: E402
for _n in dir(_F):
    setattr(_mod, _n, getattr(_F, _n))
sys.modules["torchvision.transforms.functional_tensor"] = _mod

import cv2
import numpy as np
import torch
import torch.nn.functional as F

DATA = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT_ROOT = DATA + "/train_sample_sharp"
LR_ROOT = DATA + "/train_sample_bicubic"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_generator(device="cuda"):
    from basicsr.archs.edvr_arch import EDVR
    net = EDVR(num_in_ch=3, num_out_ch=3, num_feat=128, num_frame=5,
               deformable_groups=8, num_extract_block=5,
               num_reconstruct_block=40, center_frame_idx=2,
               hr_in=False, with_predeblur=False, with_tsa=True)
    state = torch.load(C.CK_EDVR, map_location="cpu", weights_only=False)["params"]
    net.load_state_dict(state, strict=True)
    return net.to(device)


def to_nchw01(frames, device="cuda"):
    """uint8 RGB HWC -> float NCHW [0,1]; same range as evalkit."""
    x = torch.from_numpy(np.stack(frames)).to(device)
    return x.float().div_(255).permute(0, 3, 1, 2).contiguous()


class WindowDataset(torch.utils.data.Dataset):
    """One random five-frame LR/GT window per call, loaded on demand.

    Frame IDs are verified to be shared by LR and GT, not paired by list position.
    Returned LR has shape [5,3,64,64], GT [3,256,256].
    """

    def __init__(self, crop=64, length=1000, seed=0):
        self.crop = crop
        self.length = length
        self.seed = seed
        self.clips = sorted(os.listdir(GT_ROOT))
        self.meta = []
        for clip in self.clips:
            ids = C.clip_frame_ids(os.path.join(GT_ROOT, clip))
            lr_ids = C.clip_frame_ids(os.path.join(LR_ROOT, clip))
            common_ids = sorted(set(ids) & set(lr_ids))
            if common_ids != ids:
                raise RuntimeError("clip %s: LR/GT frame ids differ" % clip)
            self.meta.append((clip, ids))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Per-worker deterministic RNG; index variation avoids identical windows.
        seed = self.seed + index * 1009 + (torch.utils.data.get_worker_info().id
                                           if torch.utils.data.get_worker_info() else 0)
        rng = random.Random(seed)
        clip, ids = self.meta[index % len(self.meta)]
        start = rng.randrange(0, len(ids) - 4)
        chosen = ids[start:start + 5]
        lr = [C.read_rgb(os.path.join(LR_ROOT, clip, "%08d.png" % i)) for i in chosen]
        gt = C.read_rgb(os.path.join(GT_ROOT, clip, "%08d.png" % chosen[2]))

        h, w = lr[0].shape[:2]
        x = rng.randrange(w - self.crop + 1)
        y = rng.randrange(h - self.crop + 1)
        s = self.crop * 4
        lr = [im[y:y + self.crop, x:x + self.crop] for im in lr]
        gt = gt[y * 4:y * 4 + s, x * 4:x * 4 + s]
        if rng.random() < 0.5:
            lr = [np.ascontiguousarray(im[:, ::-1]) for im in lr]
            gt = np.ascontiguousarray(gt[:, ::-1])
        if rng.random() < 0.5:
            lr = [np.ascontiguousarray(im.transpose(1, 0, 2)) for im in lr]
            gt = np.ascontiguousarray(gt.transpose(1, 0, 2))
        return to_nchw01(lr, "cpu").float(), to_nchw01([gt], "cpu")[0].float(), clip, chosen


class LossBundle:
    def __init__(self, device, lpips_weight, dists_weight):
        self.lpips = None
        self.dists = None
        if lpips_weight:
            from evalkit.metrics import LPIPS
            self.lpips = LPIPS(device=device).net
            for p in self.lpips.parameters():
                p.requires_grad_(False)
        if dists_weight:
            self.dists = C.DistsMetric(device=device).net
            # alpha/beta were constructed trainable in the reference module;
            # metric weights must remain frozen while gradients reach the input.
            for p in self.dists.parameters():
                p.requires_grad_(False)

    def __call__(self, pred, target, lpips_weight, dists_weight):
        values = {}
        if self.lpips is not None:
            values["lpips"] = self.lpips(pred * 2 - 1, target * 2 - 1).mean()
        if self.dists is not None:
            values["dists"] = self.dists(pred, target).mean()
        return values


def relativistic_g_loss(real_logits, fake_logits):
    return (F.binary_cross_entropy_with_logits(fake_logits - real_logits.mean(),
                                               torch.ones_like(fake_logits))
            + F.binary_cross_entropy_with_logits(real_logits - fake_logits.mean(),
                                                 torch.zeros_like(real_logits))) / 2


def relativistic_d_loss(real_logits, fake_logits):
    return (F.binary_cross_entropy_with_logits(real_logits - fake_logits.mean(),
                                               torch.ones_like(real_logits))
            + F.binary_cross_entropy_with_logits(fake_logits - real_logits.mean(),
                                                 torch.zeros_like(fake_logits))) / 2


def discriminator(device):
    from basicsr.archs.discriminator_arch import UNetDiscriminatorSN
    return UNetDiscriminatorSN(num_in_ch=3, num_feat=64, skip_connection=True).to(device)


def save_checkpoint(path, net, opt, step, config, disc=None, opt_d=None):
    tmp = path + ".tmp"
    state = {"model": net.state_dict(), "optimizer": opt.state_dict(),
             "step": step, "config": config,
             "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
             "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    if disc is not None:
        state["discriminator"] = disc.state_dict()
        state["optimizer_d"] = opt_d.state_dict()
    torch.save(state, tmp)
    os.replace(tmp, path)


def train(args):
    if args.lpips_weight and args.dists_weight:
        raise ValueError("perceptual scan changes one perceptual loss at a time")
    if args.lpips_weight < 0 or args.dists_weight < 0 or args.adv_weight < 0:
        raise ValueError("loss weights must be non-negative")
    out = os.path.join(ROOT, "fine_tune", "runs", args.name)
    os.makedirs(out, exist_ok=True)
    config = vars(args).copy()
    config["base_weights"] = C.CK_EDVR
    config["base_weights_sha256"] = sha256(C.CK_EDVR)
    config["data"] = {"gt": GT_ROOT, "lr": LR_ROOT,
                      "crop_lr": args.crop, "crop_gt": args.crop * 4,
                      "frames": 5, "center": 2,
                      "augment": ["hflip", "transpose"]}
    C.write_json(os.path.join(out, "config.json"), config)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    device = args.device
    net = build_generator(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr,
                           betas=(args.beta1, 0.99), weight_decay=0)
    disc = opt_d = None
    if args.adv_weight:
        disc = discriminator(device).train()
        opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr_d,
                                 betas=(args.beta1, 0.99), weight_decay=0)

    dataset = WindowDataset(args.crop, length=args.dataset_length, seed=args.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch,
                                         shuffle=True, num_workers=args.workers,
                                         pin_memory=True, drop_last=True,
                                         persistent_workers=args.workers > 0)
    losses = LossBundle(device, args.lpips_weight, args.dists_weight)

    start = 1
    resume_path = args.resume or os.path.join(out, "last.pth")
    if args.resume or (args.auto_resume and os.path.exists(resume_path)):
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start = ck["step"] + 1
        if disc is not None:
            disc.load_state_dict(ck["discriminator"])
            opt_d.load_state_dict(ck["optimizer_d"])
        random.setstate(ck["python_rng"])
        np.random.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"])
        if ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        print("resumed at step", start, flush=True)

    log_path = os.path.join(out, "train.jsonl")
    logf = open(log_path, "a", encoding="utf-8")
    it = iter(loader)
    t0 = time.time()
    for step in range(start, args.iterations + 1):
        try:
            lr, gt, clips, ids = next(it)
        except StopIteration:
            it = iter(loader)
            lr, gt, clips, ids = next(it)
        lr, gt = lr.to(device, non_blocking=True), gt.to(device, non_blocking=True)
        pred = net(lr)
        pixel = charbonnier(pred, gt)
        perceptual = losses(pred, gt, args.lpips_weight, args.dists_weight)
        total = pixel
        total = total + args.lpips_weight * perceptual["lpips"] if "lpips" in perceptual else total
        total = total + args.dists_weight * perceptual["dists"] if "dists" in perceptual else total
        adv = pred.new_zeros(())
        if disc is not None:
            with torch.no_grad():
                real_logits = disc(gt)
            fake_logits = disc(pred)
            adv = relativistic_g_loss(real_logits, fake_logits)
            total = total + args.adv_weight * adv

        opt.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 100.0)
        opt.step()

        d_loss = pred.new_zeros(())
        if disc is not None:
            opt_d.zero_grad(set_to_none=True)
            real_logits = disc(gt)
            fake_logits = disc(pred.detach())
            d_loss = relativistic_d_loss(real_logits, fake_logits)
            d_loss.backward()
            opt_d.step()

        if step == 1 or step % args.log_every == 0:
            row = {"step": step, "batch_size": lr.shape[0],
                   "pixel": float(pixel.detach()),
                   "lpips": float(perceptual.get("lpips", pred.new_zeros(())).detach()),
                   "dists": float(perceptual.get("dists", pred.new_zeros(())).detach()),
                   "adv_g": float(adv.detach()), "adv_d": float(d_loss.detach()),
                   "total": float(total.detach()), "grad_norm": float(grad_norm),
                   "seconds": time.time() - t0,
                   "max_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "pred_range": [float(pred.detach().min()), float(pred.detach().max())],
                   "example": {"clip": clips[0], "frame_ids": ids[0].tolist()}}
            logf.write(json.dumps(row) + "\n")
            logf.flush()
            print(json.dumps(row), flush=True)
        if step % args.save_every == 0 or step == args.iterations:
            save_checkpoint(os.path.join(out, "last.pth"), net, opt, step, config,
                            disc, opt_d)
            save_checkpoint(os.path.join(out, "step_%06d.pth" % step), net, opt,
                            step, config, disc, opt_d)
    logf.close()


def charbonnier(pred, target, eps=1e-12):
    """Exact BasicSR CharbonnierLoss: eps is added directly, not squared."""
    return torch.sqrt((pred - target) ** 2 + eps).mean()


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--crop", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--dataset-length", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr-d", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--lpips-weight", type=float, default=0.0)
    p.add_argument("--dists-weight", type=float, default=0.0)
    p.add_argument("--adv-weight", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=20261119)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--resume", default=None)
    p.add_argument("--auto-resume", action="store_true")
    return p


if __name__ == "__main__":
    train(parser().parse_args())
