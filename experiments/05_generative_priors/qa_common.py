# -*- coding: utf-8 -*-
"""Shared utilities for generative-prior qualification.

Fixed protocol (see docs/P3_generative_prior_qualification_roadmap.md):
  clips 000/002/009, frames 40-49, full 1280x720, RGB uint8 PNG.
  Block diagnostics use 64x64 blocks on a 704x1280 canvas because 720 is not
  divisible by 64; full-frame metrics still use all 720 rows.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("TORCH_HOME",
                      "/data/disk1/home/ghe270/p3/quality_metrics/weights/torch_home")

import numpy as np
import torch

ROOT = "/data/disk1/home/ghe270/p3/prior_qualification"
P3 = "/data/disk1/home/ghe270/p3"
QUAL = os.path.join(P3, "quality_metrics")
sys.path.insert(0, os.path.join(QUAL, "scripts"))
sys.path.insert(0, "/data/disk1/home/ghe270/p0")
sys.path.insert(0, os.path.join(P3, "utrd", "scripts"))

DATA = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT_ROOT = DATA + "/val_sharp"
LR_ROOT = DATA + "/val_sharp_bicubic"
DEV_CLIPS = ("000", "002", "009")
FRAME_START, FRAME_END = 40, 50          # 40..49 inclusive
BLOCK = 64
BLOCK_H = 704                             # 11 * 64
BLOCK_W = 1280                            # 20 * 64
GROUPS = ("texture", "flat", "uncertain")

EDVR_CK = ("/data/disk1/shared/aiaa3201/checkpoints/EDVR_REDS_SR_L/"
           "EDVR_L_x4_SR_REDS_official-9f5f5039.pth")
FT_CK = (QUAL + "/fine_tune/runs/adv_lr3e5_d003_a001/step_000900.pth")


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_rgb(path):
    import cv2
    a = cv2.imread(path)
    if a is None:
        raise RuntimeError("unreadable %s" % path)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def write_rgb(path, arr):
    import cv2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def frame_ids(clip):
    import re
    d = os.path.join(GT_ROOT, clip)
    ids = []
    for name in sorted(os.listdir(d)):
        if name.lower().endswith(".png"):
            ids.append(int(re.findall(r"(\d+)", name)[-1]))
    return ids


def dev_ids(clip):
    ids = frame_ids(clip)
    return [i for i in ids if FRAME_START <= i < FRAME_END]


def to_tensor(frames):
    a = np.stack(frames)
    return torch.from_numpy(a).float().div(255).permute(0, 3, 1, 2)


def build_edvr(finetuned=False):
    """Official REDS EDVR-L, optionally loading the perceptual fine-tune."""
    import model as UM

    net = UM.FrozenEDVR("cuda")
    if finetuned:
        ck = torch.load(FT_CK, map_location="cpu", weights_only=False)
        if "model" not in ck:
            raise RuntimeError("fine-tune checkpoint has no 'model' key")
        net.net.load_state_dict(ck["model"], strict=True)
        net.net.eval()
    return net


@torch.no_grad()
def infer_clip(net, lr_frames, device="cuda"):
    """Centre-frame EDVR for every frame, five-frame replicate-at-edge windows."""
    x = to_tensor(lr_frames).to(device)
    n = x.shape[0]
    out = []
    for c in range(n):
        idx = [max(0, min(n - 1, c + d - 2)) for d in range(5)]
        y, _feat = net(x[idx][None])       # returns (out [1,3,4H,4W], feat)
        y = y[0]                           # [3,4H,4W]
        out.append(y.clamp(0, 1).mul(255).round().byte()
                   .permute(1, 2, 0).cpu().numpy())
    return out


@torch.no_grad()
def infer_flip(net, lr_frames, device="cuda"):
    """EDVR on the horizontally flipped clip, flipped back to the original grid."""
    flipped = [np.ascontiguousarray(f[:, ::-1]) for f in lr_frames]
    pred = infer_clip(net, flipped, device)
    return [np.ascontiguousarray(p[:, ::-1]) for p in pred]


def _lap_score(gt_rgb):
    import cv2
    y = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32)
    lap = np.abs(cv2.Laplacian(y, cv2.CV_32F, ksize=3))
    h, w = lap.shape
    return lap[:BLOCK_H, :BLOCK_W].reshape(BLOCK_H // BLOCK, BLOCK,
                                           BLOCK_W // BLOCK, BLOCK).mean((1, 3))


def _block_mean_u(a, b):
    d = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)
    return d[:BLOCK_H, :BLOCK_W].reshape(BLOCK_H // BLOCK, BLOCK,
                                         BLOCK_W // BLOCK, BLOCK).mean((1, 3))


def make_regions(clip, ids, gt, base, flip):
    """Per-frame texture/flat/EDVR-disagreement block coordinates."""
    out = {}
    for fid, g, b, f in zip(ids, gt, base, flip):
        tex = _lap_score(g)
        unc = _block_mean_u(b, f)
        rows, cols = tex.shape
        coords = [(r, c) for r in range(rows) for c in range(cols)]
        t_flat = tex.reshape(-1)
        t_unc = unc.reshape(-1)
        order_t = np.argsort(t_flat)
        order_u = np.argsort(t_unc)
        n = len(coords)
        k = max(1, int(round(n * 0.30)))
        flat = [coords[i] for i in order_t[:k]]
        texture = [coords[i] for i in order_t[-k:]]
        uncertain = [coords[i] for i in order_u[-k:]]
        out[str(fid)] = {
            "texture": texture, "flat": flat, "uncertain": uncertain,
            "texture_threshold": float(np.quantile(t_flat, 0.70)),
            "flat_threshold": float(np.quantile(t_flat, 0.30)),
            "uncertain_threshold": float(np.quantile(t_unc, 0.70)),
            "n_blocks": n, "blocks_per_group": k,
        }
    return {"clip": clip, "frames": {k: v for k, v in out.items()},
            "block": BLOCK, "canvas": [BLOCK_H, BLOCK_W],
            "note": "last 16 rows excluded from block diagnostics only; "
                    "full-frame metrics use 720 rows"}


def _patches(frames, coords):
    out = []
    for im, cs in zip(frames, coords):
        for r, c in cs:
            out.append(im[r * BLOCK:(r + 1) * BLOCK, c * BLOCK:(c + 1) * BLOCK])
    return out


def score_frames(pred, gt, ids, regions, lp, dists):
    from evalkit import metrics as EM

    if len(pred) != len(gt) or len(pred) != len(ids):
        raise RuntimeError("frame counts differ: %d/%d/%d" % (len(pred), len(gt), len(ids)))
    for p, g in zip(pred, gt):
        if p.shape != g.shape:
            raise RuntimeError("shape mismatch %s vs %s" % (p.shape, g.shape))
        if p.shape[:2] != (720, 1280):
            raise RuntimeError("unexpected size %s" % (p.shape,))

    pY = np.stack([EM.luma(x) for x in pred])
    gY = np.stack([EM.luma(x) for x in gt])
    row = {
        "psnr": float(EM.psnr_mean(pY, gY)),
        "ssim": float(EM.ssim_gpu(pY, gY, device="cuda").mean()),
        "lpips": float(lp.pair(np.stack(pred), np.stack(gt)).mean()),
        "dists": float(dists.per_frame(pred, gt).mean()),
        "n_frames": len(ids),
        "blocks": {},
    }
    per_frame_dists = dists.per_frame(pred, gt).tolist()
    per_frame_lpips = lp.pair(np.stack(pred), np.stack(gt)).tolist()
    row["per_frame"] = {"ids": ids, "dists": per_frame_dists, "lpips": per_frame_lpips}

    for group in GROUPS:
        coords = [regions["frames"][str(i)][group] for i in ids]
        pp, gg = _patches(pred, coords), _patches(gt, coords)
        bl = float(lp.pair(np.stack(pp), np.stack(gg)).mean())
        bd = float(dists.per_frame(pp, gg, bs=16).mean())
        row["blocks"][group] = {"lpips": bl, "dists": bd,
                                "n_patches": len(pp)}
    return row


def summarize(rows):
    keys = ("psnr", "ssim", "lpips", "dists")
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    for g in GROUPS:
        out[g + "_lpips"] = float(np.mean([r["blocks"][g]["lpips"] for r in rows]))
        out[g + "_dists"] = float(np.mean([r["blocks"][g]["dists"] for r in rows]))
    return out


def record_provenance(extra=None):
    import platform
    import subprocess
    prov = {
        "created": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
        "edvr_weights": {"path": EDVR_CK, "sha256": sha256(EDVR_CK)},
        "finetune_weights": {"path": FT_CK, "sha256": sha256(FT_CK)},
        "protocol": {"clips": list(DEV_CLIPS), "frames": [FRAME_START, FRAME_END - 1],
                     "block": BLOCK, "block_canvas": [BLOCK_H, BLOCK_W]},
    }
    if extra:
        prov.update(extra)
    return prov
