# -*- coding: utf-8 -*-
"""Score `blend_sqrt` on MVSR4x, the cross-domain set B1 loses LPIPS on.

Why this is worth 5 minutes
---------------------------
`B1_uniform` improves PSNR on MVSR4x (+0.096 dB) but *worsens* LPIPS by 10.7%
because MVSR4x's real degradation is lighter than the training distribution, so
the learned high-frequency compensation overshoots.

`blend_sqrt` pulls the output back toward RealBasicVSR by ~50% on average.  The
question this answers: does that pull recover the LPIPS loss while keeping the
PSNR gain?  Both branches are run in the same process on the same frames, so the
blend cannot drift from its two inputs.

Usage:
  python eval_mvsr_blend.py --out results/cross_mvsr_blend_sqrt/metrics.json
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import warp as W                           # noqa: E402
from models.ua_realbasicvsr import UARBVSR             # noqa: E402
from train_rbv_ft import load_ckpt                     # noqa: E402

DATA = os.environ.get("VSR_REAL_DATA_ROOT", os.path.join(REPO_ROOT, "data"))
RBV_RUN = os.environ.get("RBVSR_STEP0_CKPT", os.path.join(REPO_ROOT, "runs", "rbv_step0.pth"))
B1_RUN = os.environ.get("RBVSR_B1_CKPT", os.path.join(REPO_ROOT, "runs", "realworld_pixel_ft", "B1_uniform", "last.pth"))


def rd(p):
    a = cv2.imread(p)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def list_frames(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(".png"))


def psnr(a, b):
    """Same luma-PSNR convention as eval_cross.py, so rows are comparable."""
    ya = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    yb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    mse = float(np.mean((ya - yb) ** 2))
    return 99.0 if mse <= 1e-9 else float(10 * np.log10(255.0 ** 2 / mse))


def infer(model, t, chunk=9):
    """Whole-clip forward with the same pad/trim convention as eval_cross.py."""
    _, _, h, w = t.shape
    ph, pw = (-h) % 4, (-w) % 4
    if ph or pw:
        t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="replicate")
    y = model.infer_sequence(t, chunk=chunk)
    return y[:, :, :h * 4, :w * 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--chunk", type=int, default=9)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="constant control, written alongside as blend_const")
    args = ap.parse_args()

    src_root = os.path.join(DATA, "mvsr4x", "LQ4x")
    gt_root = os.path.join(DATA, "mvsr4x", "GT")
    clips = args.clips or sorted(d for d in os.listdir(src_root)
                                 if os.path.isdir(os.path.join(src_root, d)))

    qm_scripts = os.path.join(REPO_ROOT, "experiments", "04_uncertainty_and_ft", "quality_metrics")
    if qm_scripts not in sys.path:
        sys.path.insert(0, qm_scripts)
    from evalkit import metrics as EM
    import common as C

    lp = EM.LPIPS(device="cuda", torch_home=os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch")))
    dists = C.DistsMetric(device="cuda")

    torch.manual_seed(0)
    rbv = UARBVSR("cuda")
    load_ckpt(RBV_RUN, rbv)
    rbv.net.eval()
    b1 = UARBVSR("cuda")
    load_ckpt(B1_RUN, b1)
    b1.net.eval()
    print("[load] both arms ready", flush=True)

    rows = []
    t0 = time.time()
    with torch.no_grad():
        for clip in clips:
            sd = os.path.join(src_root, clip)
            files = list_frames(sd)
            if not files:
                continue
            lq = np.stack([rd(os.path.join(sd, f)) for f in files])
            t = torch.from_numpy(lq).permute(0, 3, 1, 2).float().cuda() / 255.
            yr = infer(rbv, t, args.chunk)
            y1 = infer(b1, t, args.chunk)

            # Real sqrt(U1) router: flip disagreement of the frozen reference
            # model, q95-normalised, then sqrt -- exactly as in training.
            with torch.no_grad():
                bb, bf = rbv.reference_forward(t)
                u = W.uncertainty_u1(bb, bf)
            a = torch.sqrt(u.clamp(0, 1))          # [N,1,4H,4W]
            hh, ww = y1.shape[2], y1.shape[3]
            if a.shape[2] != hh or a.shape[3] != ww:
                a = torch.nn.functional.interpolate(a, size=(hh, ww), mode="bilinear",
                                                    align_corners=False)
            blend = (1.0 - a) * yr + a * y1
            # constant control at the same average weight, isolates "how much
            # of B1" from "where B1 goes" (same test as REDS-Real)
            const = (1.0 - args.alpha) * yr + args.alpha * y1
            u_mean = float(a.mean())

            def to_u8(x):
                return [(x[k].clamp(0, 1).float().cpu().numpy()
                         .transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
                        for k in range(x.shape[0])]

            bl = to_u8(blend)
            co = to_u8(const)
            rb = to_u8(yr)

            gfiles = list_frames(os.path.join(gt_root, clip))
            gts = [rd(os.path.join(gt_root, clip, f)) for f in gfiles][:len(bl)]
            if not gts or gts[0].shape != bl[0].shape:
                print("  [skip %s] shape mismatch" % clip, flush=True)
                continue

            row = {"clip": clip, "n_frames": len(bl), "u_sqrt_mean": u_mean}
            for tag, pr in (("rbv", rb), ("blend", bl), ("const", co)):
                pp = np.stack(pr)
                gg = np.stack(gts)
                row[tag] = {
                    "psnr": float(np.mean([psnr(p, g) for p, g in zip(pr, gts)])),
                    "lpips": float(lp.pair(pp, gg).mean()),
                    "dists": float(dists.per_frame(pr, gts).mean()),
                }
            rows.append(row)
            print("[%s] %.1fs  blend PSNR %.4f LPIPS %.5f"
                  % (clip, time.time() - t0, row["blend"]["psnr"],
                     row["blend"]["lpips"]), flush=True)

    out = {"dataset": "mvsr4x", "n_clips": len(rows),
           "alpha_const": args.alpha,
           "sqrt_u1_mean": float(np.mean([r["u_sqrt_mean"] for r in rows])),
           "clips": rows}
    for tag in ("rbv", "blend", "const"):
        out["mean_" + tag] = {
            k: float(np.mean([r[tag][k] for r in rows]))
            for k in ("psnr", "lpips", "dists")}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for tag in ("rbv", "blend", "const"):
        m = out["mean_" + tag]
        print("%-6s PSNR %.4f LPIPS %.5f DISTS %.5f"
              % (tag, m["psnr"], m["lpips"], m["dists"]), flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
