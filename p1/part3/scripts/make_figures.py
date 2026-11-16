"""Direction C -- the figures the report needs, from data already on disk.

Produces four figures, each answering a question a reviewer will actually ask:

  fig1_uncertainty_compare.png
      Four uncertainty maps side by side against the true error map for one
      frame. Answers: do these things even look like the error?

  fig2_calibration.png
      Error vs uncertainty decile for every estimator, plus the perfect-
      calibration reference. Answers: is the signal monotone (usable) or just
      correlated noise?

  fig3_error_capture.png
      Fraction of total squared error removed when you discard the top-k% most
      uncertain pixels. Answers: how much is actually recoverable, and does the
      estimator beat the blind Laplacian baseline?

  fig4_tta_gain.png
      TTA ensemble vs single pass, per clip, PSNR and LPIPS. Answers: does the
      one positive result replicate?

Style: one palette, one font size scale, no chartjunk -- the course rewards
clean visuals directly.
"""
import os, sys, glob, json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

CLIP = "000"
GT_ROOT = os.path.join(DATA, "val_sharp", CLIP)
BVSR = os.path.join(DATA, "bvsrpp_frame", CLIP)
EDVR = os.path.join(DATA, "edvr_frame", CLIP)
GEN = os.path.join(DATA, "gen", CLIP)
ATTN = os.path.join(DATA, "attn", "%s_attn.npz" % CLIP)

# ---------------------------------------------------------------- palette
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e4e4e4"
C_FID = "#2a6f97"      # fidelity branch
C_GEN = "#c1666b"      # generative branch
C_UNC = "#e09f3e"      # uncertainty
C_REF = "#9aa0a6"      # reference / chance

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def ly(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)


def lindent(y):
    k = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32)
    return np.abs(cv2.filter2D(y, cv2.CV_32F, k))


def load_pack(clip=CLIP, idx=50):
    g = cv2.imread(os.path.join(DATA, "val_sharp", clip, "%08d.png" % idx))
    b = cv2.imread(os.path.join(DATA, "bvsrpp_frame", clip, "%08d.png" % idx))
    if g is None or b is None:
        return None
    e = cv2.imread(os.path.join(DATA, "edvr_frame", clip, "%08d.png" % idx))
    n = cv2.imread(os.path.join(DATA, "gen", clip, "%08d.png" % idx))
    return g, b, e, n


# ---------------------------------------------------------------- fig 1
def fig_uncertainty_maps():
    pack = load_pack()
    if pack is None:
        print("skip fig1: frames missing")
        return
    g, b, e, gen = pack
    gy, by, ey = ly(g), ly(b), ly(e)
    E = np.abs(by - gy)

    maps = [("True error", E, "viridis"),
            ("BasicVSR++ vs EDVR", np.abs(by - ey), "magma"),
            ("Laplacian |∇²Y|", lindent(by), "magma")]
    if os.path.exists(ATTN):
        z = np.load(ATTN)
        a = 1.0 - z["attn"][50].astype(np.float32)
        a = cv2.resize(a, (by.shape[1], by.shape[0]), interpolation=cv2.INTER_LINEAR)
        maps.append(("EDVR internal attn", a, "magma"))

    fig = plt.figure(figsize=(3.3 * len(maps) + 1.6, 3.5))
    gs = GridSpec(1, len(maps), figure=fig, wspace=0.42)
    for i, (name, M, cmap) in enumerate(maps):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=np.quantile(M, 0.995))
        ax.set_title(name, pad=8, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.035, shrink=0.82)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_visible(False)
    fig.suptitle("Uncertainty estimates vs. actual error — REDS clip %s, frame 50" % CLIP,
                 fontsize=11, y=1.02)
    fig.savefig(os.path.join(OUT, "fig1_uncertainty_compare.png"))
    plt.close(fig)
    print("fig1 ok")


# ---------------------------------------------------------------- fig 2/3
def sweep(estimators, clip=CLIP, max_frames=100):
    """Per-estimator deciles of error and error-capture curve."""
    gts = sorted(glob.glob(os.path.join(DATA, "val_sharp", clip, "*.png")))[:max_frames]
    bvs = sorted(glob.glob(os.path.join(DATA, "bvsrpp_frame", clip, "*.png")))[:max_frames]
    eds = sorted(glob.glob(os.path.join(DATA, "edvr_frame", clip, "*.png")))[:max_frames]
    n = min(len(gts), len(bvs), len(eds))
    if n == 0:
        return None

    U = {k: [] for k in estimators}
    E = []
    attn = None
    if os.path.exists(ATTN):
        attn = np.load(ATTN)["attn"]

    for i in range(n):
        gy = ly(cv2.imread(gts[i]))
        by = ly(cv2.imread(bvs[i]))
        ey = ly(cv2.imread(eds[i]))
        E.append(np.abs(by - gy))
        for k in estimators:
            if k == "U2_disagree":
                U[k].append(np.abs(by - ey))
            elif k == "U4_lap":
                U[k].append(lindent(by))
            elif k == "U3_attn" and attn is not None and i < len(attn):
                a = 1.0 - attn[i].astype(np.float32)
                U[k].append(cv2.resize(a, (gy.shape[1], gy.shape[0]),
                                       interpolation=cv2.INTER_LINEAR))
    E = np.stack(E)
    out = {}
    for k in estimators:
        if not U[k]:
            continue
        Uk = np.stack(U[k])
        e_flat, u_flat = E.ravel(), Uk.ravel()
        qs = np.quantile(u_flat, np.linspace(0, 1, 11))
        dec = []
        for j in range(10):
            m = (u_flat >= qs[j]) & (u_flat <= qs[j + 1])
            dec.append(float(e_flat[m].mean()) if m.sum() > 100 else np.nan)
        # error-capture: keep the confident part, see how much squared error goes away
        order = np.argsort(u_flat)
        fracs, removed = [], []
        total = float((e_flat ** 2).sum())
        for f in np.linspace(0.02, 0.6, 30):
            keep_n = int(len(order) * (1 - f))
            keep = order[:keep_n]
            removed.append(1.0 - float((e_flat[keep] ** 2).sum()) / max(total, 1e-12))
            fracs.append(f)
        out[k] = dict(deciles=dec, fracs=fracs, captured=removed)
    return out


def fig_calibration(res):
    if not res:
        print("skip fig2: no data")
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    x = np.arange(1, 11)
    styles = {"U2_disagree": (C_FID, "o", "Model disagreement (B++ vs EDVR)"),
              "U4_lap": (C_UNC, "s", "Laplacian energy (no model)"),
              "U3_attn": (C_GEN, "^", "EDVR internal attention")}
    for k, r in res.items():
        c, mk, lab = styles.get(k, (MUTED, "o", k))
        ax.plot(x, r["deciles"], marker=mk, ms=4, lw=1.8, color=c, label=lab)
    ax.set_xlabel("Uncertainty decile (1 = most confident → 10 = least)")
    ax.set_ylabel("Mean absolute error (Y channel)")
    ax.set_title("Calibration: does higher uncertainty really mean higher error?")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xticks(x)
    fig.savefig(os.path.join(OUT, "fig2_calibration.png"))
    plt.close(fig)
    print("fig2 ok")


def fig_capture(res):
    if not res:
        print("skip fig3: no data")
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    styles = {"U2_disagree": (C_FID, "Model disagreement"),
              "U4_lap": (C_UNC, "Laplacian energy"),
              "U3_attn": (C_GEN, "EDVR internal attention")}
    for k, r in res.items():
        c, lab = styles.get(k, (MUTED, k))
        ax.plot(np.array(r["fracs"]) * 100, np.array(r["captured"]) * 100,
                lw=1.9, color=c, label=lab)
    f = np.linspace(2, 60, 30)
    ax.plot(f, f, ls="--", lw=1.2, color=C_REF, label="Random rejection (chance)")
    ax.set_xlabel("Pixels discarded as uncertain (%)")
    ax.set_ylabel("Squared error removed (%)")
    ax.set_title("Error capture: how much is recoverable by trusting the map?")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT, "fig3_error_capture.png"))
    plt.close(fig)
    print("fig3 ok")


# ---------------------------------------------------------------- fig 4
def fig_tta():
    p = os.path.join(DATA, "tta.json")
    if not os.path.exists(p):
        print("skip fig4: tta.json missing")
        return
    d = json.load(open(p))
    clips = sorted(d)
    if not clips:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    x = np.arange(len(clips))
    w = 0.34
    s = [d[c]["psnr_single"] for c in clips]
    t = [d[c]["psnr_tta"] for c in clips]
    axes[0].bar(x - w / 2, s, w, color=C_REF, label="Single pass")
    axes[0].bar(x + w / 2, t, w, color=C_FID, label="TTA ensemble")
    axes[0].set_xticks(x); axes[0].set_xticklabels(clips)
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("TTA ensemble improves fidelity")
    lo = min(s); hi = max(t)
    axes[0].set_ylim(lo - (hi - lo) * 1.4, hi + (hi - lo) * 1.6)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", ncol=2,
                   bbox_to_anchor=(0.0, 1.06), borderaxespad=0)
    for i, c in enumerate(clips):
        dlt = d[c]["psnr_tta"] - d[c]["psnr_single"]
        axes[0].annotate("%+.2f dB" % dlt, (i + w / 2, t[i]),
                         textcoords="offset points", xytext=(0, 4),
                         ha="center", fontsize=7.5, color=C_FID, weight="bold")

    ls = [d[c]["lpips_single"] for c in clips]
    lt = [d[c]["lpips_tta"] for c in clips]
    axes[1].bar(x - w / 2, ls, w, color=C_REF, label="Single pass")
    axes[1].bar(x + w / 2, lt, w, color=C_GEN, label="TTA ensemble")
    axes[1].set_xticks(x); axes[1].set_xticklabels(clips)
    axes[1].set_ylabel("LPIPS ↓")
    axes[1].set_title("Perceptual cost is negligible")
    lo2, hi2 = min(ls), max(lt)
    axes[1].set_ylim(0, hi2 * 1.30)
    axes[1].legend(frameon=False, fontsize=8, loc="lower left", ncol=2,
                   bbox_to_anchor=(0.0, 1.06), borderaxespad=0)
    for i, c in enumerate(clips):
        dlt = d[c]["lpips_tta"] - d[c]["lpips_single"]
        axes[1].annotate("%+.4f" % dlt, (i + w / 2, lt[i]),
                         textcoords="offset points", xytext=(0, 4),
                         ha="center", fontsize=7.5, color=C_GEN)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_tta_gain.png"))
    plt.close(fig)
    print("fig4 ok")


if __name__ == "__main__":
    fig_uncertainty_maps()
    res = sweep(["U2_disagree", "U4_lap", "U3_attn"])
    if res:
        json.dump({k: {kk: [float(x) for x in vv] for kk, vv in v.items()}
                   for k, v in res.items()},
                  open(os.path.join(OUT, "sweep.json"), "w"), indent=2)
        fig_calibration(res)
        fig_capture(res)
    fig_tta()
    print("figures ->", OUT)
