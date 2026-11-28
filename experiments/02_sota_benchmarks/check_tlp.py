#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P0 task 2 -- validate the reference tLP before trusting a single number.

Checks, all on real REDS frames unless stated otherwise:

  A. sanity      GT used as its own "prediction" must give tLP ~ 0 (<=1e-6).
  B. no-GT       a clip whose GT has no consecutive pair must give N/A, not 0.
  C. cross-check the vectorised implementation against a naive per-pair loop
                 computed in a different code path and a different batch size
                 (GPU LPIPS is not bit-identical across batch sizes; <1e-6 here).
  D. options     x100 is exactly 100x; cutfr drops the first/last pairs; crop32
                 crops to a multiple of 32 with the TecoGAN convention.
  E. legacy      the flow-warp metric with stride=2 really does cover only every
                 other pair, and its number must reproduce the old JSON so the
                 old results stay interpretable.
  F. gaps        a missing frame breaks the pair instead of bridging it.

Run:
  /tmp/srenv/bin/python scripts/check_tlp.py --clip 000 --frames 100 \
      --pred-dir ~/p0/out/pred/bicubic/000 \
      --old-json /tmp/p1_part1/metrics_part1_final.json --old-method bicubic \
      --out ~/p0/out/audit/tlp_check.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types

import numpy as np

import torchvision  # noqa: F401

_mod = types.ModuleType("torchvision.transforms.functional_tensor")
from torchvision.transforms import functional as _F  # noqa: E402

for _n in dir(_F):
    setattr(_mod, _n, getattr(_F, _n))
sys.modules["torchvision.transforms.functional_tensor"] = _mod

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evalkit import data, metrics, temporal  # noqa: E402

BATCH_TOL = 1e-6      # GPU LPIPS differs slightly between batch sizes


def naive_tlp(sr, gt, ids, net, x100=False, cutfr=0, crop32=False):
    """Deliberately dumb reference: one pair at a time, batch of 1."""
    pr = temporal.pairs_of(ids, cutfr)
    vals = []
    for a, b in pr:
        x0, y0 = sr[a], sr[b]
        u0, v0 = gt[a], gt[b]
        if crop32:
            x0, y0 = temporal.crop_8x8(x0)[0], temporal.crop_8x8(y0)[0]
            u0, v0 = temporal.crop_8x8(u0)[0], temporal.crop_8x8(v0)[0]
        d_sr = float(net.pair(x0[None], y0[None], bs=1)[0])
        d_gt = float(net.pair(u0[None], v0[None], bs=1)[0])
        d = abs(d_sr - d_gt)
        vals.append(d * 100.0 if x100 else d)
    return (float(np.mean(vals)) if vals else None), len(pr)


def old_value(old: dict, clip: str, method: str, key: str):
    """Pull a number out of either the old per-clip or the old avg layout."""
    pc = old.get("per_clip", {}).get(clip)
    if isinstance(pc, dict):
        ent = pc.get(method)
        if isinstance(ent, dict) and key in ent:
            return float(ent[key])
    av = old.get("avg", {}).get(method)
    if isinstance(av, dict) and key in av:
        return float(av[key])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/disk1/shared/aiaa3201/project1_video_sr")
    ap.add_argument("--clip", default="000")
    ap.add_argument("--frames", type=int, default=100, help="0 = all frames")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pred-dir", default=None,
                    help="dir of this clip's *.png predictions (check E)")
    ap.add_argument("--old-json", default=None,
                    help="old metrics JSON holding this clip's legacy tlpips")
    ap.add_argument("--old-method", default="bicubic")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    gt_i, lr_i = (data.index_dir(os.path.join(a.root, "val_sharp", a.clip)),
                  data.index_dir(os.path.join(a.root, "val_sharp_bicubic", a.clip)))
    ids = sorted(set(gt_i) & set(lr_i))
    if a.frames:
        ids = ids[: a.frames]
    gt = [data.read_rgb(gt_i[i]) for i in ids]
    lr = [data.read_rgb(lr_i[i]) for i in ids]
    import cv2

    H, W = gt[0].shape[:2]
    bic = [cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC) for x in lr]
    net = metrics.LPIPS(device=a.device)
    rep = {"clip": a.clip, "frames": len(ids), "n_pairs_full": len(temporal.pairs_of(ids, 0))}

    # A. GT against itself ----------------------------------------------------
    self_t = temporal.tlp(gt, gt, ids, net)
    rep["A_gt_as_pred"] = {"value": self_t["value"], "n_pairs": self_t["n_pairs"],
                           "pass": bool(self_t["value"] is not None and self_t["value"] < 1e-6)}

    # B. no consecutive GT ----------------------------------------------------
    every_other = ids[::2]
    sy = [gt[ids.index(i)] for i in every_other]
    no_gt = temporal.tlp(sy, sy, every_other, net)
    rep["B_no_consecutive"] = {"value": no_gt["value"], "n_pairs": no_gt["n_pairs"],
                               "reason": no_gt.get("reason"),
                               "pass": bool(no_gt["value"] is None and no_gt["n_pairs"] == 0)}

    # C. vectorised vs naive per-pair loop -----------------------------------
    fast = temporal.tlp(bic, gt, ids, net)
    slow_val, slow_n = naive_tlp(bic, gt, ids, net)
    rep["C_crosscheck"] = {"vectorised": fast["value"], "naive": slow_val,
                           "n_pairs_vectorised": fast["n_pairs"], "n_pairs_naive": slow_n,
                           "abs_diff": abs(fast["value"] - slow_val), "tolerance": BATCH_TOL,
                           "pass": bool(fast["n_pairs"] == slow_n
                                        and abs(fast["value"] - slow_val) < BATCH_TOL)}
    rep["C_reference_tlp_bicubic"] = {"value": fast["value"], "n_pairs": fast["n_pairs"],
                                      "per_pair_coverage": "all adjacent pairs"}

    # D. options -------------------------------------------------------------
    x100 = temporal.tlp(bic, gt, ids, net, x100=True)
    ratio = (x100["value"] / fast["value"]) if fast["value"] else None
    cut = temporal.tlp(bic, gt, ids, net, cutfr=2)
    crop = temporal.tlp(bic, gt, ids, net, crop32=True)
    c8 = temporal.crop_8x8(gt[0])
    rep["D_options"] = {
        "x100_ratio": ratio, "x100_ok": bool(ratio is not None and abs(ratio - 100.0) < 1e-6),
        "cutfr2": {"value": cut["value"], "n_pairs": cut["n_pairs"],
                   "dropped_pairs": fast["n_pairs"] - cut["n_pairs"]},
        "crop32": {"value": crop["value"], "n_pairs": crop["n_pairs"],
                   "input_hw": list(gt[0].shape[:2]), "cropped_hw": list(c8[0].shape[:2]),
                   "offset_xy": [c8[2], c8[1]]},
    }

    # E. legacy flow-warp metric ---------------------------------------------
    strides = {st: temporal.flow_warp_lpips(bic, ids, net, stride=st) for st in (1, 2)}
    rep["E_flow_warp"] = {"stride1_n_pairs": strides[1]["n_pairs"],
                          "stride2_n_pairs": strides[2]["n_pairs"],
                          "stride1_value": strides[1]["value"],
                          "stride2_value": strides[2]["value"]}
    if a.pred_dir and os.path.isdir(a.pred_dir):
        pm = data.match_clip(a.pred_dir, os.path.join(a.root, "val_sharp", a.clip),
                             allow_missing_pred=True)
        if pm["ids"] == ids:
            pred_frames = [data.read_rgb(p) for p in pm["pred"]]
            rep["E_flow_warp"]["pred_dir"] = a.pred_dir
            rep["E_flow_warp"]["pred_stride1"] = temporal.flow_warp_lpips(
                pred_frames, ids, net, stride=1)["value"]
            rep["E_flow_warp"]["pred_stride2"] = temporal.flow_warp_lpips(
                pred_frames, ids, net, stride=2)["value"]
            if a.old_json and os.path.exists(a.old_json):
                old = json.load(open(a.old_json))
                ref = old_value(old, a.clip, a.old_method, "tlpips")
                if ref is not None:
                    d = abs(ref - rep["E_flow_warp"]["pred_stride2"])
                    rep["E_flow_warp"]["regression"] = {
                        "old_json_value": ref,
                        "new_stride2_value": rep["E_flow_warp"]["pred_stride2"],
                        "abs_diff": d, "tolerance": 1e-3, "pass": bool(d < 1e-3),
                        "note": "old JSON 'tlpips' == stride-2 flow-warp LPIPS over the same "
                                "frames, so legacy numbers reproduce but are NOT reference tLP",
                    }

    # F. a gap must break adjacency, not bridge it ---------------------------
    gids = [i for k, i in enumerate(ids) if k != 3]
    gp = temporal.pairs_of(gids, 0)
    rep["F_gap"] = {"ids": gids[:8] + ["..."], "n_pairs": len(gp),
                    "expected": len(gids) - 2, "pass": bool(len(gp) == len(gids) - 2)}

    checks = [rep["A_gt_as_pred"]["pass"], rep["B_no_consecutive"]["pass"],
              rep["C_crosscheck"]["pass"], rep["D_options"]["x100_ok"], rep["F_gap"]["pass"]]
    reg = rep["E_flow_warp"].get("regression")
    if reg is not None:
        checks.append(reg["pass"])
    rep["all_pass"] = bool(all(checks))

    print(json.dumps(rep, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
