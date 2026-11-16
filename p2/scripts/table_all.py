#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merge the A/B results with the C/D/E run and rebuild the final wild table.

Two different JSON layouts have to be reconciled:
  table_wild_final.json : per_method[method][clip][metric]
  metrics_*_cde.json    : per_clip[clip][method][metric]   (part1)
                          per_clip[clip][metric]           (part2, single method)
Everything is normalised to per[method][clip][metric].
"""
import os, json
import numpy as np

W = "/tmp/wild"
KEYS = ["psnr", "ssim", "lpips", "tlpips"]
ORDER = [
    ("bicubic", "Bicubic"), ("lanczos", "Lanczos"), ("bicubic_unsharp", "Bicubic+USM"),
    ("edsr", "EDSR x4"), ("tavg3", "3-frame avg"), ("tavg5", "5-frame avg"),
    ("tavg5_unsharp", "5-frame avg+USM"), ("flowtavg3", "Flow-aligned 3"),
    ("flowtavg5", "Flow-aligned 5"), ("flowtavg5_unsharp", "Flow-aligned 5+USM"),
    ("edvr", "EDVR-L x4"), ("tecogan", "TecoGAN"),
]
SRC = {
    "A": ("outdoor telephoto, pedestrian + car", "GT 1072x720 / LR 268x180"),
    "B": ("indoor low-light, handheld", "GT 1280x720 / LR 320x180"),
    "C": ("dark corridor, walking forward", "GT 1280x720 / LR 320x180"),
    "D": ("night shrubs, handheld", "GT 1280x720 / LR 320x180"),
    "E": ("night bikes + signage", "GT 1280x720 / LR 320x180"),
}


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def merge():
    per = {}

    def put(method, clip, row):
        per.setdefault(method, {})[clip] = {k: float(row[k]) for k in KEYS}

    old = load(W + "/table_wild_final.json")
    if old:
        for m, clips in old.get("per_method", {}).items():
            for c, r in clips.items():
                if c.startswith(("A", "B")):
                    put(m, c, r)

    p1 = load(W + "/part1/metrics_cde.json")
    if p1:
        for c, methods in p1["per_clip"].items():
            for m, r in methods.items():
                put(m, c, r)

    ed = load(W + "/part2/metrics_edvr_cde.json")
    if ed:
        for c, r in ed["per_clip"].items():
            put("edvr", c, r)

    te = load(W + "/part2/metrics_tecogan_wild.json")
    if te:
        for c, r in te["per_clip"].items():
            if c.startswith(("C", "D", "E")):
                put("tecogan", c, r)
    return per


def mmean(clips):
    return {k: float(np.mean([clips[c][k] for c in clips])) for k in KEYS}


def main():
    per = merge()
    allc = sorted({c for m in per.values() for c in m})
    print("clips per method:")
    for m in sorted(per):
        print("  %-20s %2d  %s" % (m, len(per[m]), ",".join(sorted(per[m]))))

    print("\n" + "=" * 66)
    print("WILD, all sources (%d clips x 100 frames)" % len(allc))
    print("=" * 66)
    print("%-22s %8s %8s %8s %8s" % ("method", "PSNR", "SSIM", "LPIPS", "tLPIPS"))
    print("-" * 66)
    avgs = {}
    for k, label in ORDER:
        if k not in per or not per[k]:
            continue
        v = mmean(per[k])
        avgs[k] = v
        print("%-22s %8.3f %8.4f %8.4f %8.4f"
              % (label, v["psnr"], v["ssim"], v["lpips"], v["tlpips"]))

    for src in "ABCDE":
        note, geo = SRC[src]
        print("\n--- %s: %s (%s) ---" % (src, note, geo))
        print("%-22s %8s %8s %8s %8s" % ("method", "PSNR", "SSIM", "LPIPS", "tLPIPS"))
        for k, label in ORDER:
            if k not in per:
                continue
            cs = [c for c in per[k] if c.startswith(src)]
            if not cs:
                continue
            v = mmean({c: per[k][c] for c in cs})
            print("%-22s %8.3f %8.4f %8.4f %8.4f"
                  % (label, v["psnr"], v["ssim"], v["lpips"], v["tlpips"]))

    print("\n" + "=" * 66)
    print("difficulty ladder (bicubic PSNR = how hard the source is)")
    print("=" * 66)
    rows = []
    for src in "ABCDE":
        cs = [c for c in per["bicubic"] if c.startswith(src)]
        if cs:
            rows.append((np.mean([per["bicubic"][c]["psnr"] for c in cs]), src))
    for v, s in sorted(rows):
        print("  %s  bicubic %6.2f dB   %s" % (s, v, SRC[s][0]))

    print("\nEDVR - EDSR (PSNR gain from multi-frame alignment), per source:")
    for src in "ABCDE":
        a = [c for c in per["edvr"] if c.startswith(src)]
        if not a:
            continue
        d = np.mean([per["edvr"][c]["psnr"] - per["edsr"][c]["psnr"] for c in a])
        wins = sum(1 for c in a if per["edvr"][c]["psnr"] > per["edsr"][c]["psnr"])
        print("  %s  %+6.2f dB   %d/%d clips positive" % (src, d, wins, len(a)))

    wins = sum(1 for c in per["tecogan"]
               if c in per["edvr"] and per["tecogan"][c]["lpips"] < per["edvr"][c]["lpips"])
    tot = sum(1 for c in per["tecogan"] if c in per["edvr"])
    print("\nTecoGAN LPIPS beats EDVR in %d/%d clips" % (wins, tot))

    json.dump({"per_method": per, "avg": avgs,
               "sources": {s: {"note": SRC[s][0], "geometry": SRC[s][1]} for s in SRC}},
              open(W + "/table_wild_all.json", "w"), indent=2, default=float)
    print("\nsaved " + W + "/table_wild_all.json")


if __name__ == "__main__":
    main()
