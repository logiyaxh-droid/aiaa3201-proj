# -*- coding: utf-8 -*-
"""M2-B cross-domain evaluation: MVSR4x (has GT) and VideoLQ (no GT).

Why this matters more than the 30-clip number
---------------------------------------------
M2-B was fine-tuned on 20 clips from `train_sample_sharp`.  A same-domain win on
REDS-Real c9 could be 20-clip overfitting.  MVSR4x is a real phone dual-camera
set that shares no content with training, so a positive delta there means the
fine-tuning learned something transferable; VideoLQ is real network video with
no reference, so it can only be judged on no-reference quality plus temporal
consistency against the input itself.

Protocols
---------
MVSR4x : GT is the target, LQ4x is the input (256 -> 1024, x4). Standard metrics.
VideoLQ: no GT.  MUSIQ (no-reference, higher better) + E_warp computed on the
         model output sequence AND on the untouched input, so the delta says
         whether the model made the real video *less* temporally stable.
         Per-clip means, unweighted across clips (resolutions differ).

Usage:
  python eval_cross.py --run runs/B1_uniform/last.pth --name cross_b1 \
      --dataset mvsr4x|vlq
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
M2B = os.path.dirname(HERE)
sys.path.insert(0, M2B)

from model_rbv_ft import UARBVSR          # noqa: E402
from train_rbv_ft import load_ckpt        # noqa: E402

PRIOR = "/data/disk1/home/ghe270/p3/prior_qualification"
DATA = os.path.join(PRIOR, "data")
OUT_ROOT = "/tmp/m2b_cross"                # scratch: nothing here is a deliverable


def rd(p):
    a = cv2.imread(p)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def list_frames(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(".png"))


def e_warp_sequence(frames, tol=3.0, margin=8):
    """Occlusion-aware reprojection error, same convention as score_reds_real.py.

    The consistency test is a SUM: |f_fwd(p) + f_bwd(p + f_fwd(p))| <= tol.
    Written as a difference it reports ~14 px on identical flow and the mask
    never fires.
    """
    if len(frames) < 2:
        return None
    vals = []
    for a, b in zip(frames[:-1], frames[1:]):
        ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        f_fwd = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5,
                                             1.2, 0)
        f_bwd = cv2.calcOpticalFlowFarneback(gb, ga, None, 0.5, 3, 15, 3, 5,
                                             1.2, 0)
        h, w = ga.shape
        # remap needs float32 maps; arange gives int64 and cv2 asserts on it
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        bx = np.clip(gx + f_fwd[..., 0], 0, w - 1).astype(np.float32)
        by = np.clip(gy + f_fwd[..., 1], 0, h - 1).astype(np.float32)
        back = cv2.remap(f_bwd, bx, by, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
        fb = np.abs(f_fwd + back).sum(axis=2)
        warped = cv2.remap(b, gx + f_fwd[..., 0], gy + f_fwd[..., 1],
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        valid = fb <= tol
        valid[:margin, :] = False
        valid[-margin:, :] = False
        valid[:, :margin] = False
        valid[:, -margin:] = False
        if not valid.any():
            continue
        diff = np.abs(warped.astype(np.float32) - a.astype(np.float32)).mean(axis=2)
        vals.append(float(diff[valid].mean()))
    return float(np.mean(vals)) if vals else None


def psnr(a, b):
    ya = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    yb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    mse = float(np.mean((ya - yb) ** 2))
    return 99.0 if mse <= 1e-9 else float(10 * np.log10(255.0 ** 2 / mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dataset", required=True, choices=["mvsr4x", "vlq"])
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--limit-clips", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=9)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = UARBVSR("cuda")
    step, _ = load_ckpt(args.run, model)
    model.net.eval()
    print("[load] %s step=%d" % (args.run, step), flush=True)

    if args.dataset == "mvsr4x":
        src_root = os.path.join(DATA, "mvsr4x", "LQ4x")
        gt_root = os.path.join(DATA, "mvsr4x", "GT")
    else:
        src_root = os.path.join(DATA, "vlq_eval")
        gt_root = None
    # vlq_eval also contains meta.json; only directories are clips
    clips = args.clips or sorted(d for d in os.listdir(src_root)
                                 if os.path.isdir(os.path.join(src_root, d)))
    if args.limit_clips:
        clips = clips[:args.limit_clips]

    # MUSIQ for VideoLQ
    musiq = None
    if args.dataset == "vlq":
        sys.path.insert(0, "/data/disk1/home/ghe270/p3/quality_metrics/scripts")
        import common as C
        musiq = C.MusiqMetric(device="cuda") if hasattr(C, "MusiqMetric") else None
        if musiq is None:
            print("[warn] no MusiqMetric in common.py; skipping MUSIQ", flush=True)

    rows = []
    t0 = __import__("time").time()
    with torch.no_grad():
        for clip in clips:
            sd = os.path.join(src_root, clip)
            files = list_frames(sd)
            if not files:
                continue
            # keep native aspect: RBV is x4, VideoLQ inputs are already x4-ready
            lqs = [rd(os.path.join(sd, f)) for f in files]
            lq = np.stack(lqs)
            t = torch.from_numpy(lq).permute(0, 3, 1, 2).float().cuda() / 255.
            # pad to a multiple of 4 so the pixel-shuffle tail divides evenly
            _, _, h, w = t.shape
            ph, pw = (-h) % 4, (-w) % 4
            if ph or pw:
                t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="replicate")
            y = model.infer_sequence(t, chunk=args.chunk)
            # trim the padding using the OUTPUT size (x4), not the input size.
            # Using h, w from the LR tensor cut every prediction back to the LR
            # resolution (256x256 instead of 1024x1024), which made every GT
            # comparison report a shape mismatch and silently produced PSNR=n/a.
            y = y[:, :, :h * 4, :w * 4]
            preds = [(y[k].clamp(0, 1).float().cpu().numpy()
                      .transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
                     for k in range(y.shape[0])]

            row = {"clip": clip, "n_frames": len(preds),
                   "h": int(h), "w": int(w)}
            if gt_root:
                gfiles = list_frames(os.path.join(gt_root, clip))
                gts = [rd(os.path.join(gt_root, clip, f)) for f in gfiles][:len(preds)]
                if gts and gts[0].shape == preds[0].shape:
                    row["psnr"] = float(np.mean([psnr(p, g) for p, g in zip(preds, gts)]))
                else:
                    row["psnr"] = None
                    print("    [shape mismatch] gt %s vs pred %s"
                          % (None if not gts else gts[0].shape, preds[0].shape),
                          flush=True)
            row["e_warp"] = e_warp_sequence(preds)
            row["e_warp_in"] = e_warp_sequence(lqs)
            if musiq is not None:
                # MusiqMetric is not callable: the interface is per_frame(list).
                # Calling it directly raised "'MusiqMetric' object is not
                # callable", which the earlier broad except swallowed, so the
                # first VideoLQ run silently reported MUSIQ as n/a.
                try:
                    row["musiq"] = float(np.mean(musiq.per_frame(preds[:6])))
                    row["musiq_in"] = float(np.mean(musiq.per_frame(lqs[:6])))
                except Exception as e:
                    row["musiq_err"] = "%s: %s" % (type(e).__name__, str(e)[:100])
                    print("    [musiq failed] %s" % row["musiq_err"], flush=True)
            rows.append(row)
            msg = "  %s %df %dx%d" % (clip, len(preds), w, h)
            if row.get("psnr") is not None:
                msg += " PSNR %.3f" % row["psnr"]
            if row.get("e_warp"):
                msg += " E_warp %.3f (in %.3f)" % (row["e_warp"], row["e_warp_in"] or -1)
            if row.get("musiq"):
                msg += " MUSIQ %.2f (in %.2f)" % (row["musiq"], row["musiq_in"])
            print(msg, flush=True)

    def agg(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return float(np.mean(v)) if v else None

    summary = {"run": args.run, "step": step, "dataset": args.dataset,
               "n_clips": len(rows), "elapsed_sec": __import__("time").time() - t0,
               "mean": {k: agg(k) for k in ("psnr", "e_warp", "e_warp_in",
                                            "musiq", "musiq_in")},
               "clips": rows}
    os.makedirs(os.path.join(M2B, "results", args.name), exist_ok=True)
    dst = os.path.join(M2B, "results", args.name, "metrics.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    m = summary["mean"]
    print("\n[%s] %d clips  PSNR %s  E_warp out %s / in %s  MUSIQ out %s / in %s"
          % (args.dataset, len(rows),
             "n/a" if m["psnr"] is None else "%.4f" % m["psnr"],
             "n/a" if m["e_warp"] is None else "%.4f" % m["e_warp"],
             "n/a" if m["e_warp_in"] is None else "%.4f" % m["e_warp_in"],
             "n/a" if m["musiq"] is None else "%.3f" % m["musiq"],
             "n/a" if m["musiq_in"] is None else "%.3f" % m["musiq_in"]), flush=True)
    print("wrote", dst, flush=True)


if __name__ == "__main__":
    main()
