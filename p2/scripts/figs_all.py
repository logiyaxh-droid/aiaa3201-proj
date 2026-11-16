#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figures for all five wild sources.

patch figures  : texture-selected crop, one row per method (detail reconstruction)
subject figures: motion-peak crop (A only, where a discrete mover exists)
scatter        : PSNR vs tLPIPS for every method, per source
ladder         : EDVR-EDSR gain vs source difficulty, the headline result
"""
import os
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import sys, glob, json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/tmp/wild"
GT, LR = ROOT + "/gt", ROOT + "/lr"
P1S, EDS, TES = ROOT + "/part1/samples", ROOT + "/part2/samples", ROOT + "/part2"
HOLD = ROOT + "/_ab_hold"
OUT = ROOT + "/out"
FIG = OUT + "/figures"
os.makedirs(FIG, exist_ok=True)
SHOW = ["gt", "bicubic", "edsr", "flowtavg5", "edvr", "tecogan"]
ALLM = ["bicubic", "lanczos", "bicubic_unsharp", "edsr", "tavg5",
        "flowtavg5", "flowtavg5_unsharp", "edvr", "tecogan"]


def gt_path(clip, idx=2):
    p = "%s/%s/%08d.png" % (GT, clip, idx)
    if os.path.exists(p):
        return p
    return "%s/gt_%s/%08d.png" % (HOLD, clip, idx)


def sample_path(clip, m, idx=2):
    if m == "gt":
        return gt_path(clip, idx)
    if m == "edvr":
        return "%s/%s/edvr.png" % (EDS, clip)
    if m == "tecogan":
        return "%s/%s/%08d.png" % (TES, clip, idx)
    return "%s/%s/%s.png" % (P1S, clip, m)


def load(p, size=None):
    a = cv2.imread(p)
    if a is None:
        return None
    if size and (a.shape[1], a.shape[0]) != size:
        a = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
    return a


def pick_texture(gt, pw, ph):
    g = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = np.abs(g - cv2.GaussianBlur(g, (0, 0), 2.0))
    H, W = hp.shape
    if ph >= H or pw >= W:
        return 0, 0
    best, bxy = -1.0, (0, 0)
    for y in range(0, H - ph, 8):
        for x in range(0, W - pw, 8):
            s = float(hp[y:y + ph, x:x + pw].std())
            if s > best:
                best, bxy = s, (x, y)
    return bxy


def make_patch(clip):
    gt = load(gt_path(clip))
    if gt is None:
        return None
    H, W = gt.shape[:2]
    pw, ph = min(224, W), min(224, H)
    x, y = pick_texture(gt, pw, ph)
    tiles = []
    for m in SHOW:
        img = load(sample_path(clip, m))
        if img is None:
            continue
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        c = img[y:y + ph, x:x + pw]
        c = cv2.resize(c, (pw * 3, ph * 3), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(c, (0, 0), (c.shape[1] - 1, 30), (0, 0, 0), -1)
        cv2.putText(c, m.upper(), (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        tiles.append(c)
    if not tiles:
        return None
    p = os.path.join(FIG, "%s_patch.png" % clip)
    cv2.imwrite(p, np.hstack(tiles))
    return p


def make_scatter():
    tab = json.load(open(ROOT + "/table_wild_all.json"))
    per = tab["per_method"]
    style = [
        ("bicubic", "Bicubic", "o", "#888888"), ("lanczos", "Lanczos", "D", "#aaaaaa"),
        ("tavg5", "5-frame avg", "v", "#c44e52"), ("flowtavg5", "5-frame avg + flow", "s", "#4c72b0"),
        ("edsr", "EDSR x4", "^", "#dd8452"), ("edvr", "EDVR-L x4", "*", "#55a868"),
        ("tecogan", "TecoGAN", "P", "#8172b3"),
    ]
    for src in "ABCDE":
        plt.figure(figsize=(7.2, 5.2))
        for k, label, marker, col in style:
            cs = [c for c in per.get(k, {}) if c.startswith(src)]
            if not cs:
                continue
            x = np.mean([per[k][c]["psnr"] for c in cs])
            y = np.mean([per[k][c]["tlpips"] for c in cs])
            plt.scatter(x, y, marker=marker, s=170, color=col, edgecolor="k", linewidth=0.5, zorder=3)
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(9, 5), fontsize=8)
        plt.xlabel("PSNR (dB)  higher is better")
        plt.ylabel("tLPIPS  lower is better")
        plt.title("wild %s  (no method wins both axes)" % src)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, "scatter_%s.png" % src), dpi=150)
        plt.close()
    print("scatter per source done")


def make_ladder():
    """The headline: how much multi-frame alignment buys, as a function of difficulty."""
    tab = json.load(open(ROOT + "/table_wild_all.json"))
    per = tab["per_method"]
    xs, ys, labels = [], [], []
    for src in "ABCDE":
        a = [c for c in per["edvr"] if c.startswith(src)]
        if not a:
            continue
        diff = np.mean([per["bicubic"][c]["psnr"] for c in per["edvr"] if c in per["bicubic"]])
        gain = np.mean([per["edvr"][c]["psnr"] - per["edsr"][c]["psnr"] for c in a])
        xs.append(diff)
        ys.append(gain)
        labels.append(src)
    order = np.argsort(xs)
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]
    labels = [labels[i] for i in order]

    plt.figure(figsize=(7.0, 4.8))
    plt.plot(xs, ys, "o-", color="#55a868", markersize=9)
    for x, y, l in zip(xs, ys, labels):
        plt.annotate(l, (x, y), textcoords="offset points", xytext=(8, -12), fontsize=10)
    plt.axhline(0, color="k", linewidth=0.8, linestyle="--")
    plt.xlabel("source difficulty  (bicubic PSNR, dB — lower = harder)")
    plt.ylabel("EDVR − EDSR  (PSNR gain, dB)")
    plt.title("multi-frame alignment pays off exactly when the source is hard")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "ladder_gain_vs_difficulty.png"), dpi=150)
    plt.close()
    print("ladder done")


if __name__ == "__main__":
    clips = sorted(d for d in os.listdir(GT) if os.path.isdir(os.path.join(GT, d)))
    for c in (sys.argv[1:] or clips):
        if c in clips:
            print("patch", c, make_patch(c), flush=True)
    if not sys.argv[1:]:
        make_scatter()
        make_ladder()
