# -*- coding: utf-8 -*-
"""M2 Stage A inference on REDS-Real c9 (or any <GT,LQ> clip tree).

Tiled HR inference
------------------
The model is trained on 256x256, but REDS-Real frames are 720p.  Running the
whole frame at once would (a) exceed the tiling behaviour the SD2.1 VAE was
validated with and (b) change the RBV receptive context versus training.  So
inference tiles the *HR* stage with the same geometry the training crop used:

    HR tile 256, LR context 96 (i.e. 16 px of LR context = 64 px of HR),
    HR stride 192  ->  32 px of HR overlap, discarded before stitching.

RealBasicVSR is run once over the whole frame (it is fully convolutional and
cheap in fp32), so the tiles only need to re-encode B_t -> latent -> decode.  The
gate is therefore evaluated with true 256x256 context everywhere.

Stitching
---------
Overlaps are blended with a raised-cosine window, so no hard seam can appear.
The value that is actually written is

    Y = B + W_pix * (decode(z_hat) - decode(z_base))

exactly as in training, which keeps step-0 identity valid on full frames too:
with an untrained model the output is bit-identical to RealBasicVSR.

Usage:
  python eval_stageA.py --run runs/P2_learn_M/last.pth --name p2_dev \
      --clips 000 002 009 --limit 9
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
m2 = os.path.dirname(HERE)
sys.path.insert(0, m2)

import model_stageA as M                       # noqa: E402
from train_stageA import load_ckpt             # noqa: E402

PRIOR = "/data/disk1/home/ghe270/p3/prior_qualification"
DATA = os.environ.get("RBVSR_DATA") or os.path.join(PRIOR, "data", "reds_real_c9")
GT = os.path.join(DATA, "GT")
LQ = os.path.join(DATA, "LQ")
OUT_ROOT = os.path.join(m2, "outputs")


def read_rgb(p):
    a = cv2.imread(p)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def write_rgb(p, a):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cv2.imwrite(p, cv2.cvtColor(np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8),
                                cv2.COLOR_RGB2BGR))


def blend_window(size, overlap):
    """Tile blend weight: 1 in the interior, linear ramp across `overlap`.

    A raised cosine was tried first and is wrong here: it reaches ~0 at the very
    edge (0.0006 for overlap=64), so at a seam where two edge ramps meet the
    summed weight approaches 0 and acc/wsum amplifies quantisation noise.  That
    is exactly what produced max|diff| = 139/255 against pure RealBasicVSR while
    the mean stayed at 0.0014.

    A linear ramp from a small positive floor keeps the sum bounded away from 0
    everywhere: with two contributors the seam weight is 2*floor, and the floor
    is 0.05 rather than 0.0.
    """
    # The floor must be high enough that corner pixels (covered by exactly one
    # tile, weight floor*floor) are not amplified.  0.25^2 = 0.0625 keeps the
    # worst-case normaliser within 1/0.0625 = 16x of the interior, which is
    # still a weighted mean and never a division by ~0.
    floor = 0.25
    w = np.ones(size, dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(floor, 1.0, overlap + 1, dtype=np.float32)[:-1]
        w[:overlap] = ramp
        w[-overlap:] = ramp[::-1]
    return w


@torch.no_grad()
def refine_tiled(model, b_t_hr, u1_hr, uw_hr, tile=256, stride=192,
                 gate_mode="learn", fixed_w=0.5):
    """Y_out for one full HR frame, tiled in the latent stage."""
    _, _, H, W = b_t_hr.shape
    acc = torch.zeros_like(b_t_hr)
    wsum = torch.zeros(1, 1, H, W, device=b_t_hr.device, dtype=b_t_hr.dtype)
    ov = tile - stride
    wy = torch.tensor(blend_window(tile, ov), device=b_t_hr.device,
                      dtype=b_t_hr.dtype)
    wx = wy
    win = (wy[:, None] * wx[None, :])[None, None]
    ys = list(range(0, max(1, H - tile + 1), stride))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if ys[-1] != H - tile:
        ys.append(H - tile)
    if xs[-1] != W - tile:
        xs.append(W - tile)
    for y in ys:
        for x in xs:
            bt = b_t_hr[:, :, y:y + tile, x:x + tile]
            u1 = u1_hr[:, :, y:y + tile, x:x + tile]
            uw = uw_hr[:, :, y:y + tile, x:x + tile]
            y_out, _, _ = model.refine(bt, u1, uw, gate_mode, fixed_w)
            acc[:, :, y:y + tile, x:x + tile] += y_out * win
            wsum[:, :, y:y + tile, x:x + tile] += win
    return acc / wsum.clamp_min(1e-6)


@torch.no_grad()
def run_clip(model, clip, out_dir, tile, stride, gate_mode, fixed_w, limit,
             save_gate=False):
    gt_dir = os.path.join(GT, clip)
    lq_dir = os.path.join(LQ, clip)
    ids = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(lq_dir)
                 if f.lower().endswith(".png"))
    if limit:
        ids = ids[:limit]
    # LQ frames are 320x180 for 1280x720 GT (verified: build_reds_real writes /4)
    lq = np.stack([read_rgb(os.path.join(lq_dir, "%08d.png" % i)) for i in ids])
    h, w = lq.shape[1:3]
    scale = 4
    H, W = h * scale, w * scale

    t_lq = torch.from_numpy(lq).permute(0, 3, 1, 2).float().to("cuda") / 255.0
    # RBV over the whole sequence (fp32 inside rbv_window), then x4 to HR
    full = model.rbvsr(t_lq.float())                # [N,3,4h,4w]
    full = full.to(model.dtype)
    # flip branch for U_1
    full_f = model.rbvsr(torch.flip(t_lq, dims=[-1]).float())
    full_f = torch.flip(full_f, dims=[-1]).to(model.dtype)

    out = {}
    for k, fid in enumerate(ids):
        b_t = full[k:k + 1]
        b_tf = full_f[k:k + 1]
        if k > 0:
            lr_t, lr_prev = t_lq[k:k + 1], t_lq[k - 1:k + 1][:1]
            b_prev = full[k - 1:k]
        elif len(ids) > 1:
            # First frame: use the NEXT frame as the neighbour instead of
            # b_t itself.  With b_prev = b_t the difference is identically 0,
            # u_ref clamps to 1e-6 and u1 saturates to 1.000 across the whole
            # first frame -- a strong, silent gate error on 1 of every 9 frames.
            lr_t, lr_prev = t_lq[0:1], t_lq[1:2]
            b_prev = full[1:2]
        else:
            lr_t, lr_prev, b_prev = t_lq[0:1], t_lq[0:1], b_t
        u_ref = (b_t - b_prev).abs().mean(1, keepdim=True).flatten().float()\
            .quantile(0.95).clamp_min(1e-6).to(b_t.dtype)
        u1, uw, _, _ = model._uncertainty(b_t, b_prev, b_tf, lr_t, lr_prev, u_ref)
        y = refine_tiled(model, b_t, u1, uw, tile, stride, gate_mode, fixed_w)
        write_rgb(os.path.join(out_dir, clip, "%08d.png" % fid),
                  y[0].float().cpu().numpy().transpose(1, 2, 0))
        if save_gate:
            write_rgb(os.path.join(out_dir + "_gate", clip, "%08d_w.png" % fid),
                      torch.cat([u1, uw], 1)[0, :1].float().cpu().numpy()
                      .transpose(1, 2, 0))
        out[fid] = True
    return len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="path to last.pth / best.pth")
    ap.add_argument("--name", required=True)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--gate-mode", default="learn")
    ap.add_argument("--fixed-w", type=float, default=0.5)
    ap.add_argument("--residual-mode", default="raw")
    ap.add_argument("--osediff-init", action="store_true")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--save-gate", action="store_true")
    args = ap.parse_args()

    clips = args.clips or sorted(os.listdir(GT))
    out_dir = os.path.join(OUT_ROOT, args.name)
    torch.manual_seed(0)
    t0 = time.time()
    model = M.StageA("cuda", torch.bfloat16, lora_rank=args.lora_rank,
                     osediff_init=args.osediff_init,
                     residual_mode=args.residual_mode)
    model.set_train_mode(False)
    step, best, miss = load_ckpt(args.run, model)
    print("[load] %s step=%d missing=%d" % (args.run, step, len(miss)),
          flush=True)
    print("[model] gamma=%.5f residual=%s gate=%s"
          % (float(model.gamma), model.residual_mode, args.gate_mode),
          flush=True)

    total = 0
    for clip in clips:
        n = run_clip(model, clip, out_dir, args.tile, args.stride,
                     args.gate_mode, args.fixed_w, args.limit, args.save_gate)
        total += n
        print("[%s] %d frames  %.1f min elapsed"
              % (clip, n, (time.time() - t0) / 60), flush=True)
    print("[done] %d frames -> %s (%.1f min)"
          % (total, out_dir, (time.time() - t0) / 60), flush=True)
    meta = {"run": args.run, "step": step, "clips": clips, "frames": total,
            "gamma": float(model.gamma), "residual_mode": model.residual_mode,
            "gate_mode": args.gate_mode, "tile": args.tile,
            "stride": args.stride, "limit": args.limit,
            "dir": out_dir, "elapsed_min": (time.time() - t0) / 60}
    with open(os.path.join(OUT_ROOT, args.name + "_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
