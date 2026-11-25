# -*- coding: utf-8 -*-
"""Pilot Go/No-Go judge (guide section 5, Step M2-2).

Every arm is scored on the SAME three dev clips (000 / 002 / 009, 27 frames) with
the M1 metric stack, then compared against RealBasicVSR measured on those same
clips -- not against the 30-clip table, because the dev subset is easier and the
thresholds are relative.

Thresholds are relative to the dev-clip RealBasicVSR row:
    LPIPS  improvement >= 5%          (perceptual win)
    PSNR   drop      <= 0.40 dB       (fidelity floor)
    E_warp <= baseline * 1.15         (temporal not broken)
    visual check                      (manual, not judgeable here)

Usage:
  python pilot_judge.py runs/                # reads results/pilot_*/metrics.json
  python pilot_judge.py --out results/pilot_judgement.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

PRIOR = "/data/disk1/home/ghe270/p3/prior_qualification"
DEV = ["000", "002", "009"]
M2 = "/data/disk1/home/ghe270/p3/m2_stageA"


def load_rbv_dev():
    p = os.path.join(PRIOR, "results", "reds_real_c9", "rbv", "metrics.json")
    d = json.load(open(p, encoding="utf-8"))
    cm = d["clip_means"]
    out = {}
    for k in ("psnr", "ssim", "lpips", "dists", "tlp"):
        out[k] = float(np.mean([cm[c][k] for c in DEV]))
    out["e_warp"] = float(np.mean([cm[c]["temporal"]["e_warp_sr"] for c in DEV]))
    out["n_frames"] = int(sum(cm[c]["n_frames"] for c in DEV))
    return out


def load_arm(path):
    d = json.load(open(path, encoding="utf-8"))
    cm = d["clip_means"]
    have = [c for c in DEV if c in cm]
    if len(have) != len(DEV):
        return None, "clips %s missing (have %s)" % (
            [c for c in DEV if c not in cm], list(cm))
    out = {}
    for k in ("psnr", "ssim", "lpips", "dists", "tlp"):
        vals = [cm[c][k] for c in have if cm[c].get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    ew = [cm[c]["temporal"]["e_warp_sr"] for c in have
          if cm[c]["temporal"].get("e_warp_sr") is not None]
    out["e_warp"] = float(np.mean(ew)) if ew else None
    out["n_frames"] = int(sum(cm[c]["n_frames"] for c in have))
    out["_path"] = path
    return out, None


def judge(arm, base):
    """Return per-criterion pass flags and the overall verdict."""
    lp_imp = (base["lpips"] - arm["lpips"]) / base["lpips"]
    psnr_drop = base["psnr"] - arm["psnr"]
    ew_ratio = (arm["e_warp"] / base["e_warp"]) if base["e_warp"] else None
    c = {
        "lpips_improve_pct": lp_imp * 100,
        "lpips_pass": lp_imp >= 0.05,
        "psnr_delta_db": -psnr_drop,
        "psnr_pass": psnr_drop <= 0.40,
        "e_warp_ratio": ew_ratio,
        "e_warp_pass": ew_ratio is not None and ew_ratio <= 1.15,
    }
    c["auto_pass"] = bool(c["lpips_pass"] and c["psnr_pass"] and c["e_warp_pass"])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(M2, "results"))
    ap.add_argument("--out", default=os.path.join(M2, "results",
                                                 "pilot_judgement.json"))
    args = ap.parse_args()

    base = load_rbv_dev()
    print("RealBasicVSR on dev clips %s (%d frames):"
          % ("/".join(DEV), base["n_frames"]))
    print("  PSNR %.4f  SSIM %.4f  LPIPS %.5f  DISTS %.5f  tLP %.5f  E_warp %.4f"
          % (base["psnr"], base["ssim"], base["lpips"], base["dists"],
             base["tlp"], base["e_warp"]))
    print("  thresholds: LPIPS <= %.5f (-5%%), PSNR >= %.4f (-0.40 dB), "
          "E_warp <= %.4f (+15%%)"
          % (base["lpips"] * 0.95, base["psnr"] - 0.40, base["e_warp"] * 1.15))

    arms = {}
    for p in sorted(glob.glob(os.path.join(args.results, "pilot_*",
                                           "metrics.json"))):
        name = os.path.basename(os.path.dirname(p))
        arm, err = load_arm(p)
        if arm is None:
            print("\n[%s] SKIP: %s" % (name, err))
            continue
        j = judge(arm, base)
        arms[name] = {"metrics": arm, "judge": j}
        print("\n[%s] n=%d" % (name, arm["n_frames"]))
        print("  PSNR %.4f (%+.4f dB)   LPIPS %.5f (%+.1f%%)   DISTS %.5f"
              % (arm["psnr"], j["psnr_delta_db"], arm["lpips"],
                 j["lpips_improve_pct"], arm["dists"]))
        print("  tLP %.5f   E_warp %.4f (ratio %.3f)"
              % (arm["tlp"], arm["e_warp"], j["e_warp_ratio"] or -1))
        print("  PASS lpips=%s psnr=%s e_warp=%s  -> %s"
              % (j["lpips_pass"], j["psnr_pass"], j["e_warp_pass"],
                 "GO" if j["auto_pass"] else "NO-GO"))

    out = {"dev_clips": DEV, "baseline": base, "arms": arms,
           "all_auto_pass": [k for k, v in arms.items()
                             if v["judge"]["auto_pass"]]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("\nwrote", args.out)
    print("auto GO: %s" % (out["all_auto_pass"] or "none"))


if __name__ == "__main__":
    main()
