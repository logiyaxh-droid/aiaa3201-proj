#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figures for the full-run tables: quality trade-offs, per-clip evidence.

Reads ``out/full/summary/table.json`` + ``fid.json`` (written by
``scripts/aggregate_full.py``) and draws, into ``out/full/figures/``:

  fig_tradeoff_psnr_tlp.png   PSNR vs reference tLP, all methods, labelled
  fig_tradeoff_psnr_fid.png   PSNR vs FID
  fig_perclip_tlp.png         per-clip tLP of the main methods (30 groups)
  fig_perclip_gain.png        per-clip paired gains the report claims
                              (edsr-bicubic, flowtavg5-tavg5, edvr-edsr)

No GPU, no model, no frames -- it only reads the JSON.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "bicubic": "#7f7f7f", "lanczos": "#a6a6a6", "bicubic_unsharp": "#bcbcbc",
    "edsr": "#1f77b4", "tavg3": "#ff9896", "tavg5": "#d62728",
    "tavg5_unsharp": "#e7969c", "flowtavg3": "#98df8a", "flowtavg5": "#2ca02c",
    "flowtavg5_unsharp": "#74c476", "edvr": "#9467bd", "tecogan": "#ff7f0e",
}


def load(root):
    s = os.path.join(root, "summary")
    table = json.load(open(os.path.join(s, "table.json"), encoding="utf-8"))
    fid_path = os.path.join(s, "fid.json")
    fid = json.load(open(fid_path, encoding="utf-8")) if os.path.exists(fid_path) else {}
    return table, fid


def scatter(ax, xs, ys, labels, colors, xlabel, ylabel, title, logy=False,
            offsets=None):
    """Labelled scatter with a fixed margin so no label is clipped."""
    offsets = offsets or {}
    for x, y, lab, c in zip(xs, ys, labels, colors):
        ax.scatter(x, y, s=70, color=c, edgecolor="white", linewidth=0.8, zorder=3)
        dx, dy = offsets.get(lab, (6, 5))
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=8, color=c, zorder=4)
    xs = np.asarray(xs, float)
    pad = 0.06 * (xs.max() - xs.min() or 1.0)
    ax.set_xlim(xs.min() - pad, xs.max() + pad * 3.2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle=":")


def main():
    root = os.path.abspath(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1
                                             else "~/p0/out/full"))
    fig_dir = os.path.join(root, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    table, fid = load(root)
    per_method = table["per_method"]
    methods = [m for m in PALETTE if m in per_method]
    labels = [m for m in methods]
    colors = [PALETTE[m] for m in methods]

    # --- PSNR vs tLP
    pts = [(m, per_method[m]["psnr"], per_method[m]["tlp"]) for m in methods
           if per_method[m].get("psnr") is not None and per_method[m].get("tlp") is not None]
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    scatter(ax, [p[1] for p in pts], [p[2] for p in pts], [p[0] for p in pts],
            [PALETTE[p[0]] for p in pts], "PSNR (dB) *", "reference tLP (lower is better)",
            "Spatial fidelity vs reference temporal consistency\n(REDS val, 30 clips x 100 frames)",
            logy=True,
            offsets={"bicubic": (6, -12), "lanczos": (6, 4), "edvr": (-14, 10)})
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_tradeoff_psnr_tlp.png"), dpi=150)
    plt.close(fig)

    # --- PSNR vs FID
    have = [(m, per_method[m]["psnr"], fid[m]["value"]) for m in methods
            if m in fid and fid[m].get("value") is not None
            and per_method[m].get("psnr") is not None]
    if have:
        fig, ax = plt.subplots(figsize=(7.6, 5.4))
        scatter(ax, [p[1] for p in have], [p[2] for p in have], [p[0] for p in have],
                [PALETTE[p[0]] for p in have], "PSNR (dB) *", "FID (lower is better)",
                "PSNR vs FID (3000 frames, InceptionV3 2048-d)",
                offsets={"tavg5": (6, 4), "edvr": (-14, 8)})
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig_tradeoff_psnr_fid.png"), dpi=150)
        plt.close(fig)

    # --- per-clip tLP
    clips = table["clips"]
    main_methods = [m for m in ["bicubic", "edsr", "tavg5", "flowtavg5", "edvr", "tecogan"]
                    if m in methods]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    width = 0.8 / max(1, len(main_methods))
    xs = np.arange(len(clips))
    for k, m in enumerate(main_methods):
        ys = [per_method[m].get("tlp") for _ in clips]
        ys = [table["per_clip"].get(m, {}).get(c, {}).get("tlp") for c in clips]
        ys = [np.nan if y is None else y for y in ys]
        ax.bar(xs + k * width - 0.4, ys, width, label=m, color=PALETTE[m])
    ax.set_xticks(xs)
    ax.set_xticklabels(clips, fontsize=7, rotation=90)
    ax.set_ylabel("reference tLP ↓")
    ax.set_title("Reference tLP per clip (30 clips x 99 pairs)")
    ax.legend(fontsize=8, ncol=len(main_methods))
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_perclip_tlp.png"), dpi=150)
    plt.close(fig)

    # --- paired per-clip gains
    pairs = [("edsr", "bicubic", "psnr", "dB"), ("flowtavg5", "tavg5", "psnr", "dB"),
             ("edvr", "edsr", "psnr", "dB"), ("flowtavg5", "tavg5", "tlp", "")]
    pairs = [(a, b, k, u) for a, b, k, u in pairs
             if a in methods and b in methods]
    if pairs:
        fig, axes = plt.subplots(len(pairs), 1, figsize=(10, 2.4 * len(pairs)),
                                 sharex=True)
        if len(pairs) == 1:
            axes = [axes]
        for ax, (a, b, k, unit) in zip(axes, pairs):
            g = []
            for c in clips:
                ca = table["per_clip"].get(a, {}).get(c, {}).get(k)
                cb = table["per_clip"].get(b, {}).get(c, {}).get(k)
                g.append(np.nan if (ca is None or cb is None) else ca - cb)
            ax.bar(np.arange(len(clips)), g,
                   color=["#2ca02c" if (v is not None and not np.isnan(v)
                                        and ((v > 0) if k != "tlp" else (v < 0))) else "#d62728"
                          for v in g])
            ax.axhline(0, color="k", linewidth=0.8)
            ax.set_ylabel("%s − %s%s" % (a, b, (" (%s)" % unit) if unit else ""), fontsize=8)
            ax.grid(axis="y", alpha=0.25, linestyle=":")
        axes[-1].set_xticks(np.arange(len(clips)))
        axes[-1].set_xticklabels(clips, fontsize=7, rotation=90)
        axes[0].set_title("Per-clip paired gain (green = better direction)")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig_perclip_gain.png"), dpi=150)
        plt.close(fig)

    print("figures ->", fig_dir)
    for f in sorted(os.listdir(fig_dir)):
        print("  ", f)


if __name__ == "__main__":
    main()
