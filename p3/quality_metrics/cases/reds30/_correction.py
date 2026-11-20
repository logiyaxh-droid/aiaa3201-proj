# -*- coding: utf-8 -*-
"""Recheck of two claims in the first report that the paired data does not support.

Kept as a script so the numbers in the corrected text can be regenerated.

Claim 1 (WRONG as stated): "EDVR/EDSR/TecoGAN improve with motion while tavg gets
worse, and this is the one directional difference between methods."
    The unpaired group means are confounded: the high-motion group happens to hold
    easier clips.  Paired differences vs bicubic, and vs edvr, are flat.

Claim 2 (OVERSTATED): "the defect is worse in textured regions than in flat ones."
    True for hf_ratio (relative high-frequency recovery), but in ABSOLUTE error the
    textured blocks carry almost all of the error for every method, and absolute
    error is what DISTS/MAE actually penalise.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

ROOT = "/data/disk1/home/ghe270/p3/quality_metrics"
P0 = "/data/disk1/home/ghe270/p0/out/full/results"
MOTION = 9.092, 15.364          # q1, q2 from d2_motion_groups.json


def clip_groups():
    return json.load(open(os.path.join(ROOT, "cases/reds30/d2_motion_groups.json")))[
        "clip_groups"]


def paired_vs_bicubic():
    import csv
    rows = list(csv.DictReader(open(os.path.join(ROOT, "summary/reds30/per_clip.csv"))))
    by = {}
    for r in rows:
        by.setdefault(r["clip"], {})[r["method"]] = r
    g = clip_groups()
    out = {}
    for m in ("edsr", "edvr", "tecogan", "tavg3", "tavg5", "flowtavg3", "flowtavg5"):
        acc = {"low": [], "mid": [], "high": []}
        for c, ms in by.items():
            if m in ms and "bicubic" in ms and c in g:
                acc[g[c]].append(float(ms[m]["dists"]) - float(ms["bicubic"]["dists"]))
        out[m] = {k: float(np.mean(v)) for k, v in acc.items() if v}
    return out


def block_absolute_vs_relative():
    z = np.load(os.path.join(ROOT, "cases/reds30/d1_blocks.npz"), allow_pickle=True)
    meth = z["meth"]
    hfg, hfp, mae = (z[k].astype(float) for k in ("hf_gt", "hf_pred", "mae"))
    thr = 12.082
    out = {}
    for m in ("bicubic", "edsr", "edvr", "tecogan"):
        k = meth == m
        tex = k & (hfg > thr)
        flat = k & (hfg <= thr)
        out[m] = {
            "hf_ratio_textured": float(hfp[tex].mean() / hfg[tex].mean()),
            "hf_ratio_flat": float(hfp[flat].mean() / hfg[flat].mean()),
            "mae_textured": float(mae[tex].mean()),
            "mae_flat": float(mae[flat].mean()),
            "share_of_total_abs_error_in_textured":
                float(mae[tex].sum() / (mae[tex].sum() + mae[flat].sum())),
        }
    return out


if __name__ == "__main__":
    print("== claim 1: paired dDISTS vs bicubic, by motion group ==")
    p = paired_vs_bicubic()
    print("%-12s %9s %9s %9s" % ("method", "low", "mid", "high"))
    for m, v in p.items():
        print("%-12s %+9.4f %+9.4f %+9.4f"
              % (m, v["low"], v["mid"], v["high"]))
    print("\n  -> all three groups agree in sign and are within ~0.01 of each other;")
    print("     the earlier 'improves with motion' reading came from unpaired means.")

    print("\n== claim 2: relative vs absolute error by texture ==")
    b = block_absolute_vs_relative()
    print("%-10s %10s %10s %10s %10s %8s" % ("method", "hf_ratio_T", "hf_ratio_F",
                                             "mae_T", "mae_F", "T share"))
    for m, v in b.items():
        print("%-10s %10.3f %10.3f %10.3f %10.3f %7.0f%%"
              % (m, v["hf_ratio_textured"], v["hf_ratio_flat"],
                 v["mae_textured"], v["mae_flat"],
                 100 * v["share_of_total_abs_error_in_textured"]))
    print("\n  -> textured blocks are 80-85% of all absolute error for every method,")
    print("     so the limitation is 'fails where the content is', not 'fails at edges'.")
