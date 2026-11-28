# -*- coding: utf-8 -*-
"""M2 visualisation: texture comparison strip + gate heat-map + residual map.

Two deliverables (guide section 5, M2-3):

  compare_textures_m2.png
      The SAME three crop coordinates as M1's compare_textures.png, six columns:
      GT | EDVR | RealBasicVSR | OSEDiff | RBV->OSEDiff (zero-shot) | Ours.
      Title columns are drawn once at the top; each row is one crop site.

  gate_vis.png
      Per row: B_t | U_1 | W_pix (learned gate) | |Y_out - B_t| * 5.
      This is the evidence that the gate learned *where* not to trust the
      diffusion step (flat walls / text vs foliage).

Note on cv2.putText: OpenCV 5.0.0 renders LINE_AA with thickness>=3 misaligned
(same string drawn twice for an outline yields doubled tails and missing inner
strokes), so labels are drawn once at thickness=1 (recorded in the workspace
AGENTS.md).

Usage:
  python vis_gate.py --run runs/P2_learn_M/last.pth --name p2 \
      --crops 000:320,180 002:400,120 009:360,240
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
m2 = os.path.dirname(HERE)
sys.path.insert(0, m2)

PRIOR = "/data/disk1/home/ghe270/p3/prior_qualification"
DATA = os.path.join(PRIOR, "data", "reds_real_c9")
OUTS = os.path.join(PRIOR, "outputs", "reds_real_c9")
OUTS_REAL = os.path.join(PRIOR, "outputs", "realvsr")
# EDVR and OSEDiff c9 frames were written under outputs/realvsr/<name>_c9,
# not under outputs/reds_real_c9/<name> (those dirs are empty).
SUB_MAP = {"edvr": ("realvsr", "edvr_c9"),
           "osediff": ("realvsr", "osediff_c9"),
           "gen_thres255_generator": ("reds_real_c9", "gen_thres255_generator"),
           "rbv_osediff": ("reds_real_c9", "rbv_osediff")}
RES = os.path.join(m2, "results")


def rd(p):
    a = cv2.imread(p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB) if a is not None else None


def label(img, text, scale=0.5):
    """Draw one text label on an opaque strip.  thickness=1 (see module doc)."""
    # Force uint8 RGB: vstack with a float64 (0..1) tile promoted the whole
    # canvas to float64, and cv2.imwrite cannot store float64 -- it fell back to
    # CV_8U and truncated 0..1 to 0/1, i.e. a black tile.  That is the
    # "Unsupported depth image ... fallbacked to CV_8U" warning in the log.
    img = to_u8(img)
    h = img.shape[0]
    strip = np.zeros((22, img.shape[1], 3), np.uint8)
    strip[:] = 32
    s = cv2.putText(strip, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([s, img])


def crop_center(img, cx, cy, size=192):
    h, w = img.shape[:2]
    x0 = max(0, min(cx - size // 2, w - size))
    y0 = max(0, min(cy - size // 2, h - size))
    return img[y0:y0 + size, x0:x0 + size]


# Same three crop sites as M1's compare_textures.png, in M1's convention:
#   (clip, frame, x, y) where (x, y) is the TOP-LEFT of a 160px crop, zoomed 2x.
# Mixing conventions was the bug that made GT and RealBasicVSR columns show
# different content in the first render: they were taken as (cx, cy) centre
# points with a 192px size, which lands on a different place in the frame.
M1_PICKS = [
    ("016", 22, 932, 116, "foliage: fine leaf texture"),
    ("000", 24, 196, 20, "brick facade + railing, repeating pattern"),
    ("005", 24, 756, 100, "scaffold grid, thin structures"),
]
CROP = 160
ZOOM = 2


def to_u8(img):
    """float[0,1] or uint8 -> uint8 RGB, without changing already-valid input."""
    if img.dtype == np.uint8:
        return img
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    return np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)


def topleft_crop(img, x, y, crop=CROP, zoom=ZOOM):
    h, w = img.shape[:2]
    x = max(0, min(x, w - crop))
    y = max(0, min(y, h - crop))
    c = img[y:y + crop, x:x + crop]
    return cv2.resize(c, (crop * zoom, crop * zoom),
                      interpolation=cv2.INTER_NEAREST)


def find_frame(clip, prefer=24):
    g = os.path.join(DATA, "GT", clip)
    ids = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(g)
                 if f.endswith(".png"))
    return prefer if prefer in ids else ids[len(ids) // 2]


def build_gate(run, arm_out, specs, out_path, clip_mode="learn"):
    from model_stageA import StageA
    from train_stageA import load_ckpt
    model = StageA("cuda", torch.bfloat16, lora_rank=16)
    model.set_train_mode(False)
    step, _, _ = load_ckpt(run, model)
    print("[gate] %s step %d gamma %.4f" % (run, step, float(model.gamma)))

    rows = []
    for clip, fid, x, y, what in specs:
        fn = "%08d.png" % fid
        lq = rd(os.path.join(DATA, "LQ", clip, fn))
        gt = rd(os.path.join(DATA, "GT", clip, fn))
        if lq is None or gt is None:
            continue
        lq_seq = torch.from_numpy(lq).permute(2, 0, 1)[None].float().cuda() / 255.
        ids = sorted(int(os.path.splitext(f)[0]) for f in
                     os.listdir(os.path.join(DATA, "LQ", clip))
                     if f.endswith(".png"))
        fid_index = ids.index(fid) if fid in ids else 0
        full = [rd(os.path.join(DATA, "LQ", c, "%08d.png" % i))
                for i, c in [(i, clip) for i in ids]]
        t = torch.from_numpy(np.stack(full)).permute(0, 3, 1, 2).float().cuda() / 255.
        with torch.no_grad():
            # Run the whole clip so B_{t-1} exists: U_1 needs a flip pair and
            # U_warp needs the previous frame.  Using (b - b) here made u_ref
            # clamp to 1e-6 and every U_1 saturated to 1.000 (caught in review).
            seq = model.rbvsr(t.float()).to(model.dtype)     # [N,3,720,1280]
            seqf = torch.flip(
                model.rbvsr(torch.flip(t, dims=[-1]).float()),
                dims=[-1]).to(model.dtype)
            k = min(fid_index, seq.shape[0] - 1)
            b = seq[k:k + 1]
            bf = seqf[k:k + 1]
            # neighbour index: previous frame, but for k == 0 use the NEXT one
            # (same fix as eval_stageA.py -- b_prev == b_t gave u_ref = 1e-6
            # and saturated U_1 to 1.000 on the first frame).
            kn = k - 1 if k > 0 else min(1, seq.shape[0] - 1)
            b_prev = seq[kn:kn + 1]
            lr_t = t[k:k + 1]
            lr_prev = t[kn:kn + 1]
            u_ref = (b - b_prev).abs().mean(1, keepdim=True).flatten().float()\
                .quantile(0.95).clamp_min(1e-6).to(b.dtype)
            u1, uw, _, _ = model._uncertainty(b, b_prev, bf, lr_t, lr_prev, u_ref)
            # one 256x256 window for the picture; the gate is fully conv
            y, w, ex = model.refine(b[:, :, :256, :256],
                                    u1[:, :, :256, :256],
                                    uw[:, :, :256, :256], clip_mode, 0.5)
        bt = b[:, :, :256, :256][0].float().cpu().numpy().transpose(1, 2, 0)
        yy = y[0].float().cpu().numpy().transpose(1, 2, 0)
        u1c = u1[:, :, :256, :256][0, 0].float().cpu().numpy()
        wc = w[0, 0].float().cpu().numpy()
        res = np.abs(yy - bt) * 5.0

        def norm01(a):
            a = a.astype(np.float32)
            lo, hi = float(a.min()), float(a.max())
            return (a - lo) / max(1e-6, hi - lo)

        tiles = [
            label(np.clip(bt, 0, 1), "B_t (RealBasicVSR)"),
            label(cv2.applyColorMap((norm01(u1c) * 255).astype(np.uint8),
                                    cv2.COLORMAP_TURBO)[..., ::-1].copy(), "U_1"),
            label(cv2.applyColorMap((norm01(wc) * 255).astype(np.uint8),
                                    cv2.COLORMAP_TURBO)[..., ::-1].copy(),
                  "W_pix (learned gate)"),
            label(np.clip(res, 0, 1), "|Y_out - B_t| x5"),
        ]
        info = np.zeros((256, 256 * 4, 3), np.uint8)
        rows.append(np.hstack(tiles))
        print("  clip %s  W_pix mean %.3f  |Y-B| mean %.5f  U1 mean %.3f"
              % (clip, float(wc.mean()), float(np.abs(yy - bt).mean()),
                 float(u1c.mean())))

    if not rows:
        print("[gate] no rows produced")
        return None
    sheet = np.vstack(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(to_u8(sheet), cv2.COLOR_RGB2BGR))
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="")
    ap.add_argument("--arm", default="", help="outputs/<arm> holding Ours frames")
    ap.add_argument("--name", default="m2")
    ap.add_argument("--crops", nargs="*", default=None,
                    help="clip:frame:x,y  (defaults to M1's three picks)")
    ap.add_argument("--gate-only", action="store_true")
    args = ap.parse_args()

    if args.crops:
        specs = []
        for sp in args.crops:
            parts = sp.split(":")
            clip = parts[0]
            fid = int(parts[1]) if len(parts) > 1 else 24
            xy = parts[2] if len(parts) > 2 else "0,0"
            x, y = [int(v) for v in xy.split(",")]
            specs.append((clip, fid, x, y, ""))
    else:
        specs = M1_PICKS

    if not args.gate_only and args.arm:
        out = os.path.join(RES, "compare_textures_m2.png")
        cols_arm = args.arm
        cols = [("GT", None), ("EDVR", "edvr"),
                ("RealBasicVSR", "gen_thres255_generator"),
                ("OSEDiff", "osediff"), ("RBV->OSEDiff", "rbv_osediff"),
                ("Ours", "__OURS__")]
        rows = []
        for clip, fid, x, y, what in specs:
            fn = "%08d.png" % fid
            cells = []
            for nm, sub in cols:
                if sub is None:
                    p = os.path.join(DATA, "GT", clip, fn)
                elif sub == "__OURS__":
                    p = os.path.join(m2, "outputs", cols_arm, clip, fn)
                else:
                    root, name = SUB_MAP.get(sub, ("reds_real_c9", sub))
                    base = OUTS_REAL if root == "realvsr" else OUTS
                    p = os.path.join(base, name, clip, fn)
                img = rd(p)
                img = (label(np.full((CROP * ZOOM, CROP * ZOOM, 3), 40, np.uint8),
                             nm + " MISSING")
                       if img is None else label(topleft_crop(img, x, y), nm))
                cells.append(img)
            rows.append(np.hstack(cells))
        sheet = np.vstack(rows)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, cv2.cvtColor(to_u8(sheet), cv2.COLOR_RGB2BGR))
        print("wrote", out)

    if args.run:
        out = os.path.join(RES, "gate_vis_%s.png" % args.name)
        p = build_gate(args.run, args.arm, specs, out)
        if p:
            print("wrote", p)


if __name__ == "__main__":
    main()
