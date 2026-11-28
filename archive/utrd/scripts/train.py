# -*- coding: utf-8 -*-
"""U-TRD training, three-arm prototype stage.

Data comes from the course training clips (train_sample_sharp/bicubic), never
from the 30 validation clips used for the final table.  One random LR 64x64 /
GT 256x256 five-frame window per sample, matching the proven EDVR fine-tune
protocol so the only new variable is the residual predictor.

Arms:
  det       -> direct residual regression (same capacity control)
  uniform   -> 4-step residual diffusion, scalar noise
  uncertain -> same, per-pixel uncertainty noise
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, "/data/disk1/home/ghe270/p0")

import numpy as np
import torch
import torch.nn.functional as F

import model as M

DATA = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT_ROOT = DATA + "/train_sample_sharp"
LR_ROOT = DATA + "/train_sample_bicubic"
OUT = os.path.join(ROOT, "runs")
CACHE = os.path.join(ROOT, "cache")


# ------------------------------------------------------------------ data

def read_rgb(path):
    import cv2
    a = cv2.imread(path)
    if a is None:
        raise RuntimeError("unreadable %s" % path)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def frame_ids(clip_dir):
    import re
    ids = []
    for name in sorted(os.listdir(clip_dir)):
        if name.lower().endswith(".png"):
            ids.append(int(re.findall(r"(\d+)", name)[-1]))
    if not ids:
        raise RuntimeError("no frames in %s" % clip_dir)
    return ids


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, crop=64, length=4000, seed=1):
        self.crop = crop
        self.length = length
        self.seed = seed
        self.clips = sorted(os.listdir(GT_ROOT))
        self.meta = []
        for clip in self.clips:
            ids = frame_ids(os.path.join(GT_ROOT, clip))
            lr_ids = frame_ids(os.path.join(LR_ROOT, clip))
            if ids != lr_ids:
                raise RuntimeError("LR/GT frame ids differ for %s" % clip)
            self.meta.append((clip, ids))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = random.Random(self.seed + index * 7919)
        clip, ids = self.meta[index % len(self.meta)]
        start = rng.randrange(0, len(ids) - 4)
        chosen = ids[start:start + 5]
        lr = [read_rgb(os.path.join(LR_ROOT, clip, "%08d.png" % i)) for i in chosen]
        gt = read_rgb(os.path.join(GT_ROOT, clip, "%08d.png" % chosen[2]))

        h, w = lr[0].shape[:2]
        c = self.crop
        x, y = rng.randrange(w - c + 1), rng.randrange(h - c + 1)
        lr = [im[y:y + c, x:x + c] for im in lr]
        gt = gt[y * 4:y * 4 + c * 4, x * 4:x * 4 + c * 4]
        if rng.random() < 0.5:
            lr = [np.ascontiguousarray(im[:, ::-1]) for im in lr]
            gt = np.ascontiguousarray(gt[:, ::-1])
        if rng.random() < 0.5:
            lr = [np.ascontiguousarray(im.transpose(1, 0, 2)) for im in lr]
            gt = np.ascontiguousarray(gt.transpose(1, 0, 2))
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().div_(255).permute(2, 0, 1)
        return torch.stack([to_t(f) for f in lr]), to_t(gt), clip, str(chosen[2])


def collate(batch):
    lr = torch.stack([b[0] for b in batch])
    gt = torch.stack([b[1] for b in batch])
    return lr, gt, [b[2] for b in batch], [b[3] for b in batch]


# ------------------------------------------------------------------ losses

def charbonnier(pred, target, eps=1e-12):
    return torch.sqrt((pred - target) ** 2 + eps).mean()


class Perceptual:
    """DISTS from the same implementation used for evaluation (frozen)."""

    def __init__(self, device="cuda"):
        from pyiqa.archs.dists_arch import DISTS
        w = ("/data/disk1/home/ghe270/p3/quality_metrics/weights/"
             "DISTS_weights-f5e65c96.pth")
        self.net = DISTS(pretrained_model_path=w).to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    def __call__(self, pred, gt):
        return self.net(pred, gt).mean()


# ------------------------------------------------------------------ train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=M.MODES)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--dists-weight", type=float, default=0.01)
    ap.add_argument("--pix-weight", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    name = args.mode + (("_" + args.tag) if args.tag else "")
    out_dir = os.path.join(OUT, name)
    os.makedirs(out_dir, exist_ok=True)
    if not args.resume and os.path.exists(os.path.join(out_dir, "last.pth")):
        raise SystemExit("%s already exists; pass --resume or use another tag" % out_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    device = "cuda"

    edvr = M.FrozenEDVR(device)
    net = M.ResidualUNet(base=args.base).to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))
    perc = Perceptual(device) if args.dists_weight else None

    ds = WindowDataset(args.crop, length=4000, seed=args.seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                         num_workers=args.workers, pin_memory=True,
                                         drop_last=True,
                                         collate_fn=collate,
                                         persistent_workers=args.workers > 0)

    cfg = {k: v for k, v in vars(args).items()}
    cfg.update({"edvr_weights": M.CK_EDVR, "ab": M.AB.tolist(),
                "steps_diffusion": M.STEPS, "u_floor": M.U_FLOOR,
                "model": "ResidualUNet", "num_params": sum(p.numel() for p in net.parameters())})
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)

    start = 1
    ck_path = os.path.join(out_dir, "last.pth")
    if args.resume and os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start = ck["step"] + 1
        random.setstate(ck["python_rng"])
        np.random.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"])
        print("resumed at %d" % start, flush=True)

    log = open(os.path.join(out_dir, "train.jsonl"), "a", encoding="utf-8")
    it = iter(loader)
    t0 = time.time()
    for step in range(start, args.steps + 1):
        try:
            lr, gt, _, _ = next(it)
        except StopIteration:
            it = iter(loader)
            lr, gt, _, _ = next(it)
        lr = lr.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        base, feat, u = M.edvr_outputs(edvr, lr)
        st = M.make_state(args.mode, base, gt, u)
        pred = net(st["x"], st["t"], base, F.interpolate(
            lr[:, 2], scale_factor=4, mode="bilinear", align_corners=False), feat, u)

        if args.mode == "det":
            r_hat = pred
            y_hat = (base + r_hat).clamp(0, 1)
            loss_diff = pred.new_zeros(())
        else:
            eps_hat = pred
            r_hat = M.x0_from_eps(st["x"], eps_hat, base, st["c"], st["ab"])
            y_hat = (base + r_hat).clamp(0, 1)
            loss_diff = (eps_hat - st["eps"]).abs().mean()

        loss_pix = charbonnier(y_hat, gt)
        loss = args.pix_weight * loss_pix + loss_diff
        loss_per = pred.new_zeros(())
        if perc is not None:
            loss_per = args.dists_weight * perc(y_hat, gt)
            loss = loss + loss_per

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 100.0)
        opt.step()

        if step == 1 or step % 10 == 0:
            row = {"step": step, "time_s": round(time.time() - t0, 1),
                   "loss": float(loss), "pix": float(loss_pix),
                   "diff": float(loss_diff), "perc": float(loss_per),
                   "grad_norm": float(gn),
                   "vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
        if step % 250 == 0 or step == args.steps:
            state = {"model": net.state_dict(), "optimizer": opt.state_dict(),
                     "step": step, "config": cfg,
                     "python_rng": random.getstate(),
                     "numpy_rng": np.random.get_state(),
                     "torch_rng": torch.get_rng_state()}
            tmp = ck_path + ".tmp"
            torch.save(state, tmp)
            os.replace(tmp, ck_path)
            torch.save(state, os.path.join(out_dir, "step_%06d.pth" % step))
            for old in sorted(os.listdir(out_dir)):
                if old.startswith("step_") and old != "step_%06d.pth" % step:
                    os.remove(os.path.join(out_dir, old))
    log.close()
    print("done %s in %.1f min" % (name, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
