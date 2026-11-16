#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Part 2 step 4 -- assemble the final Part 1 / Part 2 comparison table.

Merges metrics_part1_final.json (Part 1, reused as-is), metrics_edvr.json and
metrics_tecogan.json into one table plus a per-clip JSON, and prints the deltas
that the report argues about. Read-only w.r.t. the server, writes only into
/tmp/p2.
"""
import os
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import json
import numpy as np

P1 = "/tmp/p1_part1/metrics_part1_final.json"
ED = "/tmp/p2/metrics_edvr.json"
TE = "/tmp/p2/metrics_tecogan.json"
OUT = "/tmp/p2/part2_table.json"

KEYS = ["psnr", "ssim", "lpips", "tlpips"]
ORDER = [
    ("bicubic", "Bicubic (Part 1)"),
    ("lanczos", "Lanczos (Part 1)"),
    ("tavg5", "5-frame avg, no align (Part 1)"),
    ("flowtavg5", "5-frame avg + flow align (Part 1)"),
    ("edsr", "EDSR x4 (Part 1)"),
    ("edvr", "EDVR-L x4 (Part 2)"),
    ("tecogan", "TecoGAN (Part 2)"),
]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    p1, ed, te = load(P1), load(ED), load(TE)
    rows = {}

    if p1:
        for k, v in p1["avg"].items():
            rows[k] = {kk: float(v[kk]) for kk in KEYS if kk in v}
        p1_per = p1.get("per_clip", {})
    else:
        p1_per = {}
    if ed:
        rows["edvr"] = {kk: float(ed["avg"][kk]) for kk in KEYS if kk in ed["avg"]}
    if te:
        rows["tecogan"] = {kk: float(te["avg"][kk]) for kk in KEYS if kk in te["avg"]}

    print("%-32s %8s %8s %8s %8s" % ("method", "PSNR", "SSIM", "LPIPS", "tLPIPS"))
    print("-" * 68)
    for key, label in ORDER:
        if key not in rows:
            continue
        r = rows[key]
        print("%-32s %8.3f %8.4f %8.4f %8.4f" % (
            label, r["psnr"], r["ssim"], r["lpips"], r["tlpips"]))

    # the two headline claims the report needs
    d = {}
    if "edvr" in rows and "edsr" in rows:
        d["edvr_vs_edsr"] = {k: rows["edvr"][k] - rows["edsr"][k] for k in KEYS}
    if "tecogan" in rows and "edvr" in rows:
        d["tecogan_vs_edvr"] = {k: rows["tecogan"][k] - rows["edvr"][k] for k in KEYS}
    if "tecogan" in rows and "bicubic" in rows:
        d["tecogan_vs_bicubic"] = {k: rows["tecogan"][k] - rows["bicubic"][k] for k in KEYS}

    print("\ndeltas (positive = higher):")
    for k, v in d.items():
        print("  %-22s " % k + " ".join("%s %+.3f" % (kk, v[kk]) for kk in KEYS))

    # per-clip win rates: how often does EDVR beat EDSR, TecoGAN beat EDVR
    def wins(a, b, key, higher_better):
        ca, cb = {}, {}
        for src, dst in ((p1_per, ca),):
            for c, r in src.items():
                if key in r and isinstance(r[key], dict):
                    dst[c] = r[key]
                elif key in r:
                    dst[c] = r
        if ed and os.path.exists(ED):
            for c, r in json.load(open(ED))["per_clip"].items():
                cb[c] = r
        common = sorted(set(ca) & set(cb))
        if not common:
            return None
        if higher_better:
            n = sum(1 for c in common if cb[c][key] > ca[c][key])
        else:
            n = sum(1 for c in common if cb[c][key] < ca[c][key])
        return n, len(common)

    for key, hb in (("psnr", True), ("tlpips", False)):
        w = wins(None, None, key, hb)
        if w:
            print("  EDVR beats EDSR on %-6s in %d/%d clips" % (key, w[0], w[1]))

    out = {"table": {k: rows[k] for k, _ in ORDER if k in rows}, "deltas": d}
    json.dump(out, open(OUT, "w"), indent=2)
    print("\nsaved %s" % OUT)


if __name__ == "__main__":
    main()
