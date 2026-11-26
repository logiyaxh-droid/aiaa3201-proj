# -*- coding: utf-8 -*-
"""Collect the cross-domain + region results into one printable report."""
import json
import os

M2B = "/data/disk1/home/ghe270/p3/m2b_rbv_ft/results"
PRIOR = "/data/disk1/home/ghe270/p3/prior_qualification"


def load(p):
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def f(v, w=9, p=3):
    return "n/a".rjust(w) if v is None else ("%*.4f" % (w, v))


def main():
    print("=" * 72)
    print("MVSR4x (15 clips x 5 frames, real phone dual-camera, NOT in training)")
    print("=" * 72)
    rows = [("RealBasicVSR (step-0)", "cross_mvsr_rbv"),
            ("B1_uniform", "cross_mvsr_b1"),
            ("B2_uncertain", "cross_mvsr_b2"),
            ("B2b_perc_boost", "cross_mvsr_b2b")]
    base = None
    print("%-24s %9s %9s %9s" % ("method", "PSNR", "E_warp", "dPSNR"))
    for label, name in rows:
        d = load(os.path.join(M2B, name, "metrics.json"))
        if not d:
            print("%-24s %9s" % (label, "missing"))
            continue
        m = d["mean"]
        if base is None:
            base = m
        dd = (m["psnr"] - base["psnr"]) if (m["psnr"] and base["psnr"]) else None
        print("%-24s %9s %9s %9s"
              % (label, f(m["psnr"]), f(m["e_warp"]),
                 "n/a" if dd is None else "%+9.4f" % dd))

    print()
    print("=" * 72)
    print("VideoLQ (50 clips, real network video, no GT) -- MUSIQ higher better")
    print("=" * 72)
    rows = [("RealBasicVSR (step-0)", "cross_vlq_rbv"),
            ("B1_uniform", "cross_vlq_b1"),
            ("B2_uncertain", "cross_vlq_b2"),
            ("B2b_perc_boost", "cross_vlq_b2b")]
    inp = None
    print("%-24s %9s %9s %11s %9s" % ("method", "MUSIQ", "E_warp",
                                      "dMUSIQ/input", "dEwarp/input"))
    for label, name in rows:
        d = load(os.path.join(M2B, name, "metrics.json"))
        if not d:
            print("%-24s %9s" % (label, "missing"))
            continue
        m = d["mean"]
        if inp is None:
            inp = m
        dm = (m["musiq"] - inp["musiq_in"]) if (m["musiq"] and inp["musiq_in"]) else None
        de = ((m["e_warp"] - inp["e_warp_in"]) / inp["e_warp_in"] * 100
              if (m["e_warp"] and inp["e_warp_in"]) else None)
        print("%-24s %9s %9s %11s %9s"
              % (label, f(m["musiq"]), f(m["e_warp"]),
                 "n/a" if dm is None else "%+9.3f" % dm,
                 "n/a" if de is None else "%+8.2f%%" % de))
    print("%-24s %9s %9s" % ("(input, untouched)", f(inp["musiq_in"]),
                             f(inp["e_warp_in"])) if inp else "")

    print()
    print("=" * 72)
    print("Region split on 30 clips (u1 <= 0.25 flat / >= 0.75 textured)")
    print("=" * 72)
    r = load(os.path.join(M2B, "regions.json"))
    if r:
        print("%-12s %-6s %9s %9s %9s %9s"
              % ("method", "region", "PSNR", "E_warp", "HF", "HF_gt"))
        for name in ("baseline", "b1", "b2", "b2b"):
            for tag in ("flat", "tex"):
                x = r["results"][name][tag]
                print("%-12s %-6s %9s %9s %9s %9s"
                      % (name, tag, f(x["psnr"], 9, 4), f(x["e_warp"], 9, 4),
                         f(x["hf"], 9, 4), f(x["hf_gt"], 9, 4)))

    assert os.path.exists(os.path.join(PRIOR, "results")), "sanity"


if __name__ == "__main__":
    main()
