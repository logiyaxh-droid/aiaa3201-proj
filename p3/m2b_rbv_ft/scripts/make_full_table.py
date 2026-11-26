# -*- coding: utf-8 -*-
"""Build the full 30-clip table: all baselines vs the three M2-B arms.

Baselines are read from the provenance-trusted `results/reds_real_c9/*/metrics.json`
(270-frame, same scorer as M2-B).  Anything not present there is left blank
rather than copied from a different protocol.
"""
import json
import os

PRIOR = "/data/disk1/home/ghe270/p3/prior_qualification"
M2B = "/data/disk1/home/ghe270/p3/m2b_rbv_ft"

# name -> (path, label)
BASE = [
    ("bicubic", os.path.join(PRIOR, "results/reds_real_c9/bicubic_c9/metrics.json")),
    ("EDVR", os.path.join(PRIOR, "results/reds_real_c9/edvr_c9/metrics.json")),
    ("FT-EDVR", os.path.join(PRIOR, "results/reds_real_c9/ft_c9/metrics.json")),
    ("RealBasicVSR", os.path.join(PRIOR, "results/reds_real_c9/rbv/metrics.json")),
    ("DOVE 5B", os.path.join(PRIOR, "results/reds_real_c9/dove_c9/metrics.json")),
    ("OSEDiff", os.path.join(PRIOR, "results/reds_real_c9/osediff_c9/metrics.json")),
    ("RBV->OSEDiff", os.path.join(PRIOR, "results/reds_real_c9/rbv_osediff/metrics.json")),
]
OURS = [
    ("B1_uniform", os.path.join(M2B, "results/full_b1/metrics.json")),
    ("B2_uncertain", os.path.join(M2B, "results/full_b2/metrics.json")),
    ("B2b_perc_boost", os.path.join(M2B, "results/full_b2b/metrics.json")),
]


def load(p):
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    m = d["mean"]
    return {"psnr": m.get("psnr"), "ssim": m.get("ssim"),
            "lpips": m.get("lpips"), "dists": m.get("dists"),
            "tlp": m.get("tlp"), "e_warp": m.get("e_warp_sr"),
            "n_clips": m.get("n_clips"), "n_frames": m.get("n_frames")}


def main():
    rows = []
    for name, p in BASE + OURS:
        r = load(p)
        if r is None:
            rows.append((name, None))
            continue
        rows.append((name, r))

    print("REDS-Real c9, 30 clips x 9 frames = 270 frames, one scorer\n")
    hdr = ("%-16s %8s %8s %8s %8s %8s %8s" %
           ("method", "PSNR", "SSIM", "LPIPS", "DISTS", "tLP", "E_warp"))
    print(hdr)
    print("-" * len(hdr))
    for name, r in rows:
        if r is None:
            print("%-16s %8s" % (name, "n/a"))
            continue
        def f(v, w, prec):
            return ("n/a".rjust(w) if v is None else ("%*.4f" % (w, v)))
        print("%-16s %8s %8s %8s %8s %8s %8s"
              % (name, f(r["psnr"], 8, 4), f(r["ssim"], 8, 4),
                 f(r["lpips"], 8, 4), f(r["dists"], 8, 4),
                 f(r["tlp"], 8, 4), f(r["e_warp"], 8, 4)))

    base = dict(rows)["RealBasicVSR"]
    print("\nDelta vs RealBasicVSR (the model M2-B fine-tunes):")
    print("%-16s %10s %10s %10s %10s" % ("method", "dPSNR", "dLPIPS%", "dDISTS%", "dEwarp%"))
    for name, r in rows:
        if r is None or name == "RealBasicVSR":
            continue
        def df(v, b):
            if v is None or b in (None, 0):
                return "n/a".rjust(10)
            return "%+10.2f" % ((v - b) / b * 100)
        print("%-16s %10s %10s %10s %10s"
              % (name,
                 ("n/a".rjust(10) if r["psnr"] is None
                  else "%+10.4f" % (r["psnr"] - base["psnr"])),
                 df(r["lpips"], base["lpips"]),
                 df(r["dists"], base["dists"]),
                 df(r["e_warp"], base["e_warp"])))

    out = {"rows": [{"method": n, **r} for n, r in rows if r],
           "baseline": "RealBasicVSR", "n_frames": 270}
    dst = os.path.join(M2B, "results", "full_table.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("\nwrote", dst)


if __name__ == "__main__":
    main()
