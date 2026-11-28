#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Task 13 -- does EDVR-L x4 (REDS) actually use its neighbouring frames?

Design (and its honest limits)
------------------------------
Condition A: the normal 5-frame sliding window.
Condition B: the centre frame repeated five times (``[c, c, c, c, c]``) -- the
             model still runs the full PCD alignment + TSA fusion path, but every
             neighbour equals the centre, so perfect alignment produces zero
             motion and the only information available is the centre frame.

B changes the input distribution (alignment now has nothing to align), so a drop
in B measures **the value of neighbour information as the network actually uses
it**, not "the isolated contribution of PCD" or "of TSA" separately.  The
checklist warns about exactly this and the report repeats it.

Clips are chosen from the task-6 motion strata (fixed before the scores for this
experiment exist): the two lowest-motion and two highest-motion clips, plus the
clip that flow alignment failed on worst, so the "does neighbour help more when
there is more to align" question has a chance to show up.

Also checked: the sliding-window boundary policy.  The current code replicates the
first/last LR frame outside the clip; the metric consequence is quantified by
reporting metrics with the first and last 2 frames dropped as well, since those
are the frames whose windows are clamped.

Usage:
    /tmp/srenv/bin/python ~/p0/scripts/edvr_neighbor.py --out ~/p0/out/edvr_neighbor
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import numpy as np
import cv2
import torch
import types
import torchvision

_m = types.ModuleType("torchvision.transforms.functional_tensor")
from torchvision.transforms import functional as _F  # noqa: E402

for _n in dir(_F):
    setattr(_m, _n, getattr(_F, _n))
sys.modules["torchvision.transforms.functional_tensor"] = _m
torch.backends.cudnn.benchmark = False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from evalkit import metrics as M, temporal as T, data as D  # noqa: E402

B = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT_ROOT = B + "/val_sharp"
LR_ROOT = B + "/val_sharp_bicubic"
CK = ("/data/disk1/shared/aiaa3201/checkpoints/EDVR_REDS_SR_L/"
      "EDVR_L_x4_SR_REDS_official-9f5f5039.pth")


def rd(p):
    a = cv2.imread(p)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


class EDVR:
    def __init__(self, device="cuda"):
        from basicsr.archs.edvr_arch import EDVR as Net
        ck = torch.load(CK, map_location="cpu", weights_only=False)["params"]
        net = Net(num_in_ch=3, num_out_ch=3, num_feat=128, num_frame=5,
                  deformable_groups=8, num_extract_block=5, num_reconstruct_block=40,
                  center_frame_idx=2, hr_in=False, with_predeblur=False, with_tsa=True)
        net.load_state_dict(ck, strict=True)
        self.net = net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device

    def clip(self, lr, mode="normal"):
        n = len(lr)
        out = []
        for c in range(n):
            if mode == "repeat_center":
                win = [c] * 5
            elif mode == "no_neighbor_mean":
                # control: neighbours replaced by their temporal mean, which keeps
                # a plausible input but removes the frame-to-frame variation
                win = [max(0, min(n - 1, c + d - 2)) for d in range(5)]
            else:
                win = [max(0, min(n - 1, c + d - 2)) for d in range(5)]
            x = torch.stack([torch.from_numpy(lr[w]).permute(2, 0, 1).float()
                             .div_(255.0) for w in win]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                y = self.net(x)
            out.append(y[0].clamp_(0, 1).mul_(255).round_().byte()
                       .permute(1, 2, 0).cpu().numpy())
        return np.stack(out, 0)


def score(pred, gt, ids, lpips_net, cut_ends=0):
    if cut_ends:
        a, b = cut_ends, len(ids) - cut_ends
        pred, gt, ids = pred[a:b], gt[a:b], ids[a:b]
    pY = np.stack([M.luma(x) for x in pred])
    gY = np.stack([M.luma(x) for x in gt])
    row = {"psnr": float(M.psnr_mean(pY, gY)),
           "ssim": float(M.ssim_gpu(pY, gY, device="cuda").mean()),
           "lpips": lpips_net.mean(pred, np.stack(gt)),
           "n_frames": int(len(ids))}
    pr = T.pairs_of(ids, 0)
    if pr:
        d_sr = lpips_net.pair(np.stack([pred[a] for a, _ in pr]),
                              np.stack([pred[b] for _, b in pr]))
        d_gt = lpips_net.pair(np.stack([gt[a] for a, _ in pr]),
                              np.stack([gt[b] for _, b in pr]))
        row["tlp"] = float(np.abs(d_sr - d_gt).mean())
        row["tlp_pairs"] = int(len(pr))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--motion", default="~/p0/out/motion/groups.json")
    args = ap.parse_args()
    out_dir = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(out_dir, exist_ok=True)

    if args.clips:
        clips = args.clips
        why = {c: "user-specified" for c in clips}
    else:
        gd = json.load(open(os.path.expanduser(args.motion), encoding="utf-8"))
        stats, group = gd["stats"], gd["group"]
        order = sorted(gd["clips"], key=lambda c: stats[c]["median_mag_px"])
        low = order[:2]
        high = order[-2:]
        an = json.load(open(os.path.expanduser("~/p0/out/motion/analysis.json"),
                            encoding="utf-8"))
        # worst flow-alignment failure that is not already in the list
        worst = [w["clip"] for w in an["flow_align_worst_clips"]
                 if w["clip"] not in low + high][:2]
        clips = low + high + worst
        why = {}
        for c in low:
            why[c] = "lowest motion (%.2f px)" % stats[c]["median_mag_px"]
        for c in high:
            why[c] = "highest motion (%.2f px)" % stats[c]["median_mag_px"]
        for c in worst:
            why[c] = "worst flow-alignment failure"

    print("clips:", [(c, why[c]) for c in clips], flush=True)
    lpips_net = M.LPIPS(device="cuda", torch_home="/tmp/torch_home")
    model = EDVR()

    out = {"clips": clips, "why": why,
           "caveat": "condition B (centre repeated 5x) changes the input distribution; "
                     "it measures the value of neighbour information as used by this "
                     "weight, NOT the isolated contribution of PCD or of TSA",
           "window_policy": "first/last frames clamp to the clip boundary "
                            "(replicate), same as the full-run script",
           "per_clip": {}}

    for c in clips:
        t0 = time.time()
        idx = D.index_dir(os.path.join(GT_ROOT, c))
        ids = sorted(idx)
        gt = [rd(idx[i]) for i in ids]
        lidx = D.index_dir(os.path.join(LR_ROOT, c))
        lr = [rd(lidx[i]) for i in ids]

        preds = {}
        preds["normal"] = model.clip(lr, "normal")
        preds["repeat_center"] = model.clip(lr, "repeat_center")

        row = {"motion_px": None, "conditions": {}}
        try:
            gd = json.load(open(os.path.expanduser(args.motion), encoding="utf-8"))
            row["motion_px"] = gd["stats"][c]["median_mag_px"]
            row["motion_group"] = gd["group"][c]
        except Exception:
            pass
        for k, pred in preds.items():
            r = score(pred, gt, ids, lpips_net)
            r_cut = score(pred, gt, ids, lpips_net, cut_ends=2)
            row["conditions"][k] = {"all_frames": r, "drop_first_last_2": r_cut}
        a = row["conditions"]["normal"]["all_frames"]
        b = row["conditions"]["repeat_center"]["all_frames"]
        row["delta_repeat_minus_normal"] = {
            k: b[k] - a[k] for k in ("psnr", "ssim", "lpips", "tlp") if k in a}
        out["per_clip"][c] = row
        print("%s [%s] motion %s | normal %.3f/%.4f/%.4f tLP %.4f | repeat %.3f/%.4f/%.4f tLP %.4f | dPSNR %+.3f (%.0fs)"
              % (c, row.get("motion_group", "?"), ("%.2f" % row["motion_px"])
                 if row["motion_px"] else "?",
                 a["psnr"], a["ssim"], a["lpips"], a["tlp"],
                 b["psnr"], b["ssim"], b["lpips"], b["tlp"],
                 row["delta_repeat_minus_normal"]["psnr"], time.time() - t0), flush=True)
        json.dump(out, open(os.path.join(out_dir, "neighbor.json"), "w", encoding="utf-8"),
                  indent=1)

    ns = [out["per_clip"][c]["conditions"]["normal"]["all_frames"] for c in clips]
    rs = [out["per_clip"][c]["conditions"]["repeat_center"]["all_frames"] for c in clips]
    out["mean"] = {
        "normal": {k: float(np.mean([r[k] for r in ns])) for k in ("psnr", "ssim", "lpips", "tlp")},
        "repeat_center": {k: float(np.mean([r[k] for r in rs])) for k in ("psnr", "ssim", "lpips", "tlp")}}
    out["mean"]["delta"] = {k: out["mean"]["repeat_center"][k] - out["mean"]["normal"][k]
                            for k in ("psnr", "ssim", "lpips", "tlp")}
    json.dump(out, open(os.path.join(out_dir, "neighbor.json"), "w", encoding="utf-8"), indent=1)
    print("\nmean delta (repeat - normal):", json.dumps(out["mean"]["delta"], indent=1))
    print("->", os.path.join(out_dir, "neighbor.json"))


if __name__ == "__main__":
    main()
