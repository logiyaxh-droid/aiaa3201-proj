"""Region analysis: does the uncertainty map actually point at *semantic* targets?

The decile curve says the map is calibrated. That is necessary but not
interesting on its own -- the Laplacian baseline is calibrated too, and it knows
nothing about the model. What a reviewer wants to know is whether the map
localises the failure modes the paper cares about:

  high-frequency texture (brick / foliage / fabric)  -> hallucination risk
  text / signage                                     -> the money shot
  flat sky / wall                                    -> should be LOW, the
                                                        model is reliable there

If the map fires everywhere the Laplacian fires, it is just an edge detector.
So: correlate the map with a texture-energy proxy AND check that flat regions
stay quiet. That distinction is what justifies calling it a *model* uncertainty
rather than a filter.
"""
import os, glob, json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
# 帧放在本脚本上一级的 frames/ 里；本地开发时也可能用同级的 data/
_HERE = os.path.dirname(ROOT)
DATA = os.path.join(_HERE, "frames")
if not os.path.isdir(DATA):
    DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e4e4e4"
C_TEX, C_FLAT, C_TEXT = "#c1666b", "#2a6f97", "#e09f3e"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight",
})

CLIP = "000"


def ly(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)


def local_var(y, k=15):
    m = cv2.blur(y, (k, k))
    m2 = cv2.blur(y * y, (k, k))
    return np.maximum(m2 - m * m, 0)


def analyse(clip=CLIP, frames=100):
    gtf = sorted(glob.glob(os.path.join(DATA, "val_sharp", clip, "*.png")))[:frames]
    bvf = sorted(glob.glob(os.path.join(DATA, "bvsrpp_frame", clip, "*.png")))[:frames]
    edf = sorted(glob.glob(os.path.join(DATA, "edvr_frame", clip, "*.png")))[:frames]
    n = min(len(gtf), len(bvf), len(edf))
    if n == 0:
        return None

    # accumulate per-region error for 3 texture bands defined per frame
    buckets = {"flat": [[], []], "mid": [[], []], "textured": [[], []],
               "text_like": [[], []]}
    agree = []
    for i in range(n):
        gt = ly(cv2.imread(gtf[i]))
        bv = ly(cv2.imread(bvf[i]))
        ed = ly(cv2.imread(edf[i]))
        U = np.abs(bv - ed)                       # model disagreement
        E = np.abs(bv - gt)                       # true error
        V = local_var(gt)

        q1, q2 = np.quantile(V, 0.33), np.quantile(V, 0.66)
        for name, m in (("flat", V <= q1), ("mid", (V > q1) & (V <= q2)),
                        ("textured", V > q2)):
            buckets[name][0].append(U[m].mean())
            buckets[name][1].append(E[m].mean())

        # "text-like": very high local contrast in a thin structure = edges of glyphs
        # cheap proxy: strong gradient AND low local variance (thin strokes on flat bg)
        gx = cv2.Sobel(gt, cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(gt, cv2.CV_32F, 0, 1, 3)
        gmag = np.hypot(gx, gy)
        m_text = (gmag > np.quantile(gmag, 0.97)) & (V < np.quantile(V, 0.5))
        if m_text.sum() > 500:
            buckets["text_like"][0].append(U[m_text].mean())
            buckets["text_like"][1].append(E[m_text].mean())

        # ranking agreement inside each frame
        us = np.argsort(U.ravel())
        es = np.argsort(E.ravel())
        k = int(U.size * 0.1)
        top_u, top_e = set(us[-k:].tolist()), set(es[-k:].tolist())
        agree.append(len(top_u & top_e) / k)

    rows = []
    for name, (u, e) in buckets.items():
        if u:
            rows.append(dict(region=name, mean_u=float(np.mean(u)),
                             mean_err=float(np.mean(e)), n_frames=len(u)))
    return dict(clip=clip, rows=rows, top10_overlap=float(np.mean(agree)))


def fig_regions(res):
    if not res:
        print("skip: no data")
        return
    rows = res["rows"]
    labels = [r["region"] for r in rows]
    u = [r["mean_u"] for r in rows]
    e = [r["mean_err"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))

    x = np.arange(len(labels))
    w = 0.36
    axes[0].bar(x - w / 2, u, w, color=C_TEX, label="Disagreement U")
    axes[0].bar(x + w / 2, e, w, color=C_FLAT, label="True error")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8.5)
    axes[0].set_ylabel("Mean |·| (Y channel)")
    axes[0].set_title("Does uncertainty track texture difficulty?")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left",
                   bbox_to_anchor=(0, 1.14), ncol=2, borderaxespad=0)

    # normalised profile, so the *shape* is readable regardless of absolute scale
    un = np.array(u) / max(u)
    en = np.array(e) / max(e)
    axes[1].plot(x, un, "o-", color=C_TEX, lw=2, ms=5, label="Disagreement U (norm.)")
    axes[1].plot(x, en, "s--", color=C_FLAT, lw=2, ms=5, label="True error (norm.)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8.5)
    axes[1].set_ylim(0, 1.15)
    axes[1].set_ylabel("Normalised to max")
    axes[1].set_title("Profile match: r = %.3f" % np.corrcoef(un, en)[0, 1])
    axes[1].legend(frameon=False, fontsize=8, loc="lower left",
                   bbox_to_anchor=(0, 1.14), ncol=2, borderaxespad=0)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_region_profile.png"))
    plt.close(fig)
    print("fig5 ok")
    for r in rows:
        print("  %-10s U %.3f  err %.3f" % (r["region"], r["mean_u"], r["mean_err"]))
    print("  top-10%% overlap: %.3f" % res["top10_overlap"])


if __name__ == "__main__":
    res = analyse()
    if res:
        json.dump(res, open(os.path.join(OUT, "region_profile.json"), "w"), indent=2)
        fig_regions(res)

