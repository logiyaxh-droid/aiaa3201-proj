#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P0 task 10 -- what grid does TecoGAN's upsampler assume, and is +0.375 right?

The pre-P0 pipeline pre-shifted every REDS LR frame by +0.375 px before feeding
it to TecoGAN, justified by a comment claiming ``bicubic_four`` assumes a
half-pixel-centred grid while the course LR frames use ``align_corners=True`` --
and the 0.375 was originally chosen because it maximised PSNR on the test data.
This script replaces the claim with a measurement that depends on no dataset,
weight or PSNR.

**Exact ramp probe.**  A linear ramp is reproduced exactly by any cubic
interpolator with a unity-sum kernel, so the *content position* of every output
sample reads straight off: feed ``x[i] = i`` (and separately ``y[j] = j``), fit
``out = a + b*j`` on the middle 80 % (away from the padded border), and get::

    content_position(j) = origin + step * j        [in LR pixel units]

``step`` identifies the grid *spacing* (0.25 = each LR pixel covers exactly 4 HR
pixels; (N-1)/(4N-1) = the grid spans corner to corner).  ``origin`` identifies
the *registration*: content at LR coordinate u appears at output index
j = (u - origin)/step, so a smaller origin means the same content shows up at a
larger j, i.e. the image is displaced right/down.  The displacement between two
upsamplers is therefore::

    d_LR = origin_ref - origin_tf
    d_HR = SCALE * d_LR

References: OpenCV ``INTER_CUBIC`` (the course bicubic baseline) and torch
``interpolate(align_corners=False/True)`` (a PyTorch port).

Usage:
  /tmp/tf1env/bin/python  scripts/probe_grid.py --part tf  --out ~/p0/out/audit/grid.json
  /tmp/srenv/bin/python   scripts/probe_grid.py --part ref --out ~/p0/out/audit/grid.json
  /tmp/srenv/bin/python   scripts/probe_grid.py --part compare \
      --tf ~/p0/out/audit/grid.json --ref ~/p0/out/audit/grid.json \
      --out ~/p0/out/audit/grid_convention.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

H, W = 36, 72          # LR size; *4 -> 144 x 288
SCALE = 4
SEED = 7


def synthetic() -> np.ndarray:
    """Unit-range RGB probe: random dots (broadband) + a hard step + thin lines."""
    rs = np.random.RandomState(SEED)
    a = np.zeros((H, W, 3), np.float32)
    a[rs.rand(H, W) > 0.94] = 1.0
    a[:, W // 2:] = 0.35
    a[4:6, :] = 0.9
    a[:, 3:4] = 0.75
    return a


def ramp_h() -> np.ndarray:
    r = np.tile(np.arange(W, dtype=np.float32)[None, :], (H, 1))
    return np.repeat((r / float(W - 1))[..., None], 3, axis=2)


def ramp_v() -> np.ndarray:
    r = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))
    return np.repeat((r / float(H - 1))[..., None], 3, axis=2)


def fit_ramp(img: np.ndarray, horizontal: bool, n_in: int) -> dict:
    """out = a + b*j on the middle 80 %; returns origin/step in input-pixel units."""
    prof = img.mean(axis=2).astype(np.float64)
    prof = prof.mean(axis=0) if horizontal else prof.mean(axis=1)
    j = np.arange(len(prof), dtype=np.float64)
    lo, hi = int(0.1 * len(prof)), int(0.9 * len(prof))
    b, a = np.polyfit(j[lo:hi], prof[lo:hi], 1)
    resid = float(np.abs(np.polyval([b, a], j[lo:hi]) - prof[lo:hi]).max())
    return {"origin_input_px": float(a * (n_in - 1)),
            "step_input_per_output": float(b * (n_in - 1)),
            "max_linear_resid": resid,
            "step_if_corner_grid": (n_in - 1) / float(n_in * SCALE - 1),
            "step_if_pixel_centre": 1.0 / SCALE}


# --------------------------------------------------------------------- impls
def tf_upsample(a: np.ndarray) -> np.ndarray:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import tensorflow as tf

    sys.path.insert(0, "/tmp/TecoGAN-master")
    from lib.ops import bicubic_four

    ph = tf.placeholder(tf.float32, [1, None, None, 3])
    out = bicubic_four(ph)
    with tf.Session() as s:
        return s.run(out, {ph: a[None]})[0]


def cv_upsample(a: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.resize(a, (W * SCALE, H * SCALE), interpolation=cv2.INTER_CUBIC)


def torch_upsample(a: np.ndarray, align_corners: bool) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(a).permute(2, 0, 1)[None]
    y = F.interpolate(x, size=(H * SCALE, W * SCALE), mode="bicubic",
                      align_corners=align_corners)
    return y[0].permute(1, 2, 0).numpy()


def phase_offset(ref: np.ndarray, mov: np.ndarray) -> tuple:
    """Empirical cross-check: phase correlation on the synthetic edge pattern."""
    import cv2

    a = cv2.cvtColor(np.clip(ref * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(np.clip(mov * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(np.float32(a), np.float32(b), win)
    return float(dx), float(dy), float(resp)


def describe(name: str, fn) -> dict:
    rh = fit_ramp(fn(ramp_h()), True, W)
    rv = fit_ramp(fn(ramp_v()), False, H)
    up = fn(synthetic())
    return {"impl": name, "hr_hw": list(up.shape[:2]),
            "value_range": [float(up.min()), float(up.max())],
            "horizontal": rh, "vertical": rv}


def classify(rec: dict) -> str:
    """Name the grid from the measured origin/step (not from a comment)."""
    h, v = rec["horizontal"], rec["vertical"]
    spaced = (abs(h["step_input_per_output"] - 0.25) < 2e-3
              and abs(v["step_input_per_output"] - 0.25) < 2e-3)
    corner = (abs(h["step_input_per_output"] - h["step_if_corner_grid"]) < 2e-3
              and abs(v["step_input_per_output"] - v["step_if_corner_grid"]) < 2e-3)
    if corner:
        return "corner grid: grid spans corner to corner (align_corners=True style)"
    if spaced and abs(h["origin_input_px"] + 0.375) < 5e-3:
        return ("half-pixel-centred: sample i sits at (i+0.5)/4-0.5 in LR units "
                "(origin -0.375, step 0.25)")
    if spaced and abs(h["origin_input_px"]) < 5e-3:
        return ("edge-origin with step 0.25: LR pixel i maps exactly onto HR 4i, so the "
                "first and last LR samples land on the HR border")
    if spaced:
        return ("step 0.25 with origin %.4f LR px: pixel-centred spacing, but a %.4f px "
                "registration offset" % (h["origin_input_px"], h["origin_input_px"]))
    return ("unclassified: step=%r origin=%r"
            % (h["step_input_per_output"], h["origin_input_px"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["tf", "ref", "compare"], required=True)
    ap.add_argument("--tf", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    stem = a.out.replace(".json", "")

    if a.part == "tf":
        rec = describe("TecoGAN lib/ops.py:bicubic_four (TF 1.15)", tf_upsample)
        np.save(stem + "_tf_phase.npy", tf_upsample(synthetic()))
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"part": "tf", "grid": classify(rec), "record": rec}, f, indent=2)

    elif a.part == "ref":
        impls = {
            "cv2_inter_cubic": cv_upsample,
            "torch_align_corners_false": lambda x: torch_upsample(x, False),
            "torch_align_corners_true": lambda x: torch_upsample(x, True),
        }
        recs = {}
        for k, f in impls.items():
            recs[k] = describe(k, f)
            np.save(stem + "_%s_phase.npy" % k, f(synthetic()))
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"part": "ref", "records": recs}, f, indent=2)

    else:
        tfj = json.load(open(a.tf, encoding="utf-8"))
        tf_rec = tfj["record"]
        refj = json.load(open(a.ref, encoding="utf-8"))
        tf_phase = np.load(a.tf.replace(".json", "_tf_phase.npy"))

        rows = {}
        for k, rec in refj["records"].items():
            # content(j) = origin + step*j.  A LARGER origin means the same output
            # index shows content that sits further right, i.e. that image is
            # displaced LEFT.  So image displacement (positive = right) is the
            # negation of the origin difference.
            dh = tf_rec["horizontal"]["origin_input_px"] - rec["horizontal"]["origin_input_px"]
            dv = tf_rec["vertical"]["origin_input_px"] - rec["vertical"]["origin_input_px"]
            ph = np.load(a.ref.replace(".json", "_%s_phase.npy" % k))
            dx, dy, resp = phase_offset(ph, tf_phase)
            rows[k] = {"displacement_lr_px": [dh, dv],
                       "displacement_hr_px": [dh * SCALE, dv * SCALE],
                       "phase_corr_hr_px": [dx, dy],
                       "phase_corr_response": resp,
                       "reference_grid": classify(rec)}
        c = rows["cv2_inter_cubic"]["displacement_lr_px"]
        rep = {
            "part": "compare",
            "tf_impl": tf_rec["impl"],
            "tf_grid": tfj["grid"],
            "tf_geometry": {
                "origin_input_px": [tf_rec["horizontal"]["origin_input_px"],
                                    tf_rec["vertical"]["origin_input_px"]],
                "step_input_per_output": [tf_rec["horizontal"]["step_input_per_output"],
                                          tf_rec["vertical"]["step_input_per_output"]],
                "max_linear_resid": [tf_rec["horizontal"]["max_linear_resid"],
                                     tf_rec["vertical"]["max_linear_resid"]],
            },
            "references": rows,
            "sign_convention": {
                "displacement": "positive = bicubic_four's IMAGE sits right/down of the "
                                "reference (the same content appears at a larger HR index)",
                "corrective_input_shift": "warping the LR input with +s moves the output "
                                          "content right by s, so the input shift that "
                                          "cancels a displacement d is exactly d",
            },
            "corrective_lr_shift_vs_cv2": {
                "dx": c[0], "dy": c[1],
                "magnitude_lr_px": float(np.hypot(c[0], c[1])),
                "note": "warp the LR input by this much (cv2.warpAffine, +shift moves "
                        "content right/down) to register bicubic_four onto the "
                        "cv2/OpenCV half-pixel convention",
                "matches_pre_p0_shift_0p375": bool(abs(c[0] - 0.375) < 0.06
                                                   and abs(c[1] - 0.375) < 0.06),
            },
        }
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)

    print(json.dumps(json.load(open(a.out, encoding="utf-8")), indent=2)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
