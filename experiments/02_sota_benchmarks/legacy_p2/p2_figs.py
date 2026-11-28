#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Part 2 step 5 -- the two figures the report actually argues with.

1. scatter.png  : PSNR (x) vs tLPIPS (y) for all seven methods, i.e. the
                  distortion / temporal-consistency plane. One glance shows
                  that no method wins both axes.
2. perclip.png  : per-clip PSNR for EDSR vs EDVR across the 30 REDS val clips,
                  which is the evidence that EDVR's win is systematic and not
                  an average-of-outliers artifact.

Pure matplotlib, no seaborn, writes into /tmp/p2/out/figures.
"""
import os
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P1 = "/tmp/p1_part1/metrics_part1_final.json"
ED = "/tmp/p2/metrics_edvr.json"
TE = "/tmp/p2/metrics_tecogan.json"
FIG = "/tmp/p2/out/figures"
os.makedirs(FIG, exist_ok=True)

# method -> (display name, marker, colour)
STYLE = [
    ("bicubic", "Bicubic", "o", "#888888"),
    ("lanczos", "Lanczos", "D", "#aaaaaa"),
    ("tavg5", "5-frame avg (no align)", "v", "#c44e52"),
    ("flowtavg5", "5-frame avg + flow", "s", "#4c72b0"),
    ("edsr", "EDSR x4", "^", "#dd8452"),
    ("edvr", "EDVR-L x4", "*", "#55a868"),
    ("tecogan", "TecoGAN", "P", "#8172b3"),
]


def main():
    p1 = json.load(open(P1))
    ed = json.load(open(ED))
    te = json.load(open(TE))
    avg = dict(p1["avg"])
    avg["edvr"] = ed["avg"]
    avg["tecogan"] = te["avg"]

    # ---------------- figure 1: PSNR vs tLPIPS
    # explicit per-point label offsets: the seven points cluster into three
    # groups, so auto-placement collides. (dx, dy, ha) in data/axes coords.
    OFF = {
        "bicubic":   (-0.05,  0.0045, "right"),
        "lanczos":   ( 0.05, -0.0055, "left"),
        "tavg5":     ( 0.00,  0.0040, "center"),
        "flowtavg5": ( 0.00, -0.0045, "center"),
        "edsr":      ( 0.00, -0.0042, "center"),
        "edvr":      ( 0.00,  0.0048, "center"),
        "tecogan":   ( 0.00, -0.0040, "center"),
    }
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for key, label, mk, col in STYLE:
        if key not in avg:
            continue
        r = avg[key]
        big = key in ("edvr", "tecogan")
        ax.scatter(r["psnr"], r["tlpips"], s=230 if big else 120,
                   marker=mk, color=col, edgecolor="k", linewidth=0.6, zorder=3)
        dx, dy, ha = OFF.get(key, (0, 0.004, "center"))
        ax.annotate(label, (r["psnr"] + dx, r["tlpips"] + dy), ha=ha, va="center",
                    fontsize=8.5, color=col, fontweight="bold", zorder=4)
    ax.set_xlabel("PSNR (dB)  $\\uparrow$  better fidelity", fontsize=10)
    ax.set_ylabel("tLPIPS  $\\downarrow$  better temporal consistency", fontsize=10)
    ax.set_title("Part 1 vs Part 2 on REDS val (30 clips x 100 frames)", fontsize=11)
    ax.grid(alpha=0.3, linestyle=":")
    ax.invert_yaxis()          # lower tLPIPS is better -> "better" ends up at the top
    ax.margins(x=0.14, y=0.16)
    ax.text(0.015, 0.985, "no method wins both axes",
            transform=ax.transAxes, fontsize=8, va="top", style="italic", color="#555")
    fig.tight_layout()
    p = os.path.join(FIG, "scatter_psnr_tlpips.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print("wrote", p)

    # ---------------- figure 2: per-clip PSNR, EDSR vs EDVR
    per_ed = ed["per_clip"]
    per_p1 = p1["per_clip"]
    clips = sorted(per_ed)
    e_ed = [per_ed[c]["psnr"] for c in clips]
    e_es = [per_p1[c]["edsr"]["psnr"] for c in clips]
    x = np.arange(len(clips))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    w = 0.4
    ax.bar(x - w / 2, e_es, w, label="EDSR x4 (frame-wise)", color="#dd8452")
    ax.bar(x + w / 2, e_ed, w, label="EDVR-L x4 (aligned, 5-frame)", color="#55a868")
    ax.set_xticks(x)
    ax.set_xticklabels(clips, rotation=90, fontsize=7)
    ax.set_xlabel("REDS val clip", fontsize=9)
    ax.set_ylabel("PSNR (dB)", fontsize=9)
    ax.set_title("EDVR beats EDSR on all 30/30 clips (mean +2.62 dB)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    fig.tight_layout()
    p = os.path.join(FIG, "perclip_psnr_edsr_vs_edvr.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print("wrote", p)

    # ---------------- figure 3: per-clip tLPIPS, the temporal story
    t_ed = [per_ed[c]["tlpips"] for c in clips]
    t_es = [per_p1[c]["edsr"]["tlpips"] for c in clips]
    t_fl = [per_p1[c]["flowtavg5"]["tlpips"] for c in clips]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(x, t_es, "-o", ms=3, lw=1, label="EDSR x4", color="#dd8452")
    ax.plot(x, t_ed, "-s", ms=3, lw=1, label="EDVR-L x4", color="#55a868")
    ax.plot(x, t_fl, "-^", ms=3, lw=1, label="flow-aligned 5-frame", color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels(clips, rotation=90, fontsize=7)
    ax.set_ylabel("tLPIPS $\\downarrow$", fontsize=9)
    ax.set_xlabel("REDS val clip", fontsize=9)
    ax.set_title("Temporal consistency per clip: explicit flow alignment still wins", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    p = os.path.join(FIG, "perclip_tlpips.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
