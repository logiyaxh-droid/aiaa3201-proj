#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rebuild the difficulty ladder figure.

BUG FIXED: the first version computed `diff` by iterating over per["edvr"] clips while
indexing per["bicubic"], which averages the *global* bicubic PSNR instead of each
source's own. That collapsed every point onto x=27.0 and the curve became a vertical
line. Now the x coordinate is each source's own bicubic mean.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/tmp/wild"
tab = json.load(open(ROOT + "/table_wild_all.json"))
per = tab["per_method"]

NOTE = {
    "A": "outdoor telephoto",
    "B": "indoor low-light",
    "C": "dark corridor",
    "D": "night shrubs",
    "E": "night bikes",
}

xs, ys, labels, notes = [], [], [], []
for src in "ABCDE":
    ed = [c for c in per["edvr"] if c.startswith(src)]
    if not ed:
        continue
    xs.append(np.mean([per["bicubic"][c]["psnr"] for c in ed if c in per["bicubic"]]))
    ys.append(np.mean([per["edvr"][c]["psnr"] - per["edsr"][c]["psnr"] for c in ed]))
    labels.append(src)
    notes.append(NOTE[src])

order = np.argsort(xs)
xs = [xs[i] for i in order]
ys = [ys[i] for i in order]
labels = [labels[i] for i in order]
notes = [notes[i] for i in order]

print("ladder points:")
for x, y, l, n in zip(xs, ys, labels, notes):
    print("  %s  bicubic %.2f dB  EDVR-EDSR %+.2f dB  %s" % (l, x, y, n))

fig, ax = plt.subplots(figsize=(7.6, 5.0))
ax.plot(xs, ys, "o-", color="#55a868", markersize=10, linewidth=1.8, zorder=3)
for x, y, l, n in zip(xs, ys, labels, notes):
    ax.annotate("%s  (%s)" % (l, n), (x, y), textcoords="offset points",
                xytext=(10, 8), fontsize=9)
ax.axhline(0, color="k", linewidth=0.9, linestyle="--")
ax.set_xlabel("source difficulty   (bicubic PSNR, dB — lower is harder)")
ax.set_ylabel("EDVR − EDSR   (PSNR gain, dB)")
ax.set_title("multi-frame alignment pays off exactly when the source is hard")
ax.grid(alpha=0.3)
ax.set_ylim(min(ys) - 0.5, max(ys) + 0.7)
ax.set_xlim(min(xs) - 1.0, max(xs) + 2.4)
fig.tight_layout()
out = ROOT + "/out/figures/ladder_gain_vs_difficulty.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print("saved " + out)
