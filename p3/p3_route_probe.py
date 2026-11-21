"""Part 3 / Direction C -- the decisive feasibility experiment.

Direction C says: compute a pixel-wise uncertainty, trust a fidelity model
(BasicVSR++) where it is confident, and a generative model where the fidelity
model is not. Before building any of that, three questions have to be answered
with numbers, and all three are cheap because the frames are already on disk:

  Q1  Is "the two models disagree" a usable proxy for "this pixel is wrong"?
      -> Spearman(disagreement, |fidelity - GT|)

  Q2  How much is there to win at all?
      -> oracle PSNR of a per-pixel min-error pick (upper bound, unattainable)
         versus the best single model (what we ship today)

  Q3  Does a trivial router capture a real fraction of that bound?
      -> route to the generative branch where disagreement > quantile q,
         measure PSNR/LPIPS of the fused frame across q, sweep q.

Run: python p3_route_probe.py 000 002 009 010 019 022 026
"""
import os, sys, glob, json
import numpy as np
import cv2
from scipy.stats import spearmanr

GT_ROOT = "/data/disk1/shared/aiaa3201/project1_video_sr/val_sharp"
EDVR = "/tmp/p2/edvr_frame"
BVSR = "/data/disk2/home/ghe270/p3/bvsrpp_frame"
OUT = "/data/disk2/home/ghe270/p3"


def gray_y(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)


def psnr(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return 10.0 * np.log10(255.0 ** 2 / max((d * d).mean(), 1e-12))


def load(clip):
    gts = sorted(glob.glob(os.path.join(GT_ROOT, clip, "*.png")))
    eds = sorted(glob.glob(os.path.join(EDVR, clip, "*.png")))
    bvs = sorted(glob.glob(os.path.join(BVSR, clip, "*.png")))
    n = min(len(gts), len(eds), len(bvs))
    if n == 0:
        return None
    G = [cv2.imread(gts[i]) for i in range(n)]
    E = [cv2.imread(eds[i]) for i in range(n)]
    B = [cv2.imread(bvs[i]) for i in range(n)]
    return G, E, B, n


if __name__ == "__main__":
    clips = sys.argv[1:] or ["000"]
    report = {}
    for clip in clips:
        r = load(clip)
        if r is None:
            print("skip", clip, flush=True)
            continue
        G, E, B, n = r
        psnr_b = np.mean([psnr(gray_y(B[i]), gray_y(G[i])) for i in range(n)])
        psnr_e = np.mean([psnr(gray_y(E[i]), gray_y(G[i])) for i in range(n)])

        # --- Q1 + Q2 on Y channel, all pixels of all frames pooled
        eB, eE, dis, orc = [], [], [], []
        for i in range(n):
            gy, by, ey = gray_y(G[i]), gray_y(B[i]), gray_y(E[i])
            eB.append(np.abs(by - gy))
            eE.append(np.abs(ey - gy))
            dis.append(np.abs(by - ey))
            orc.append(np.minimum(np.abs(by - gy), np.abs(ey - gy)))
        eB, eE, dis, orc = map(np.stack, (eB, eE, dis, orc))
        sub = np.random.RandomState(0).choice(eB.size, min(400000, eB.size), replace=False)
        rho = float(spearmanr(eB.ravel()[sub], dis.ravel()[sub]).correlation)

        def p_of(err):
            return float(10.0 * np.log10(255.0 ** 2 / max((err.astype(np.float64) ** 2).mean(), 1e-12)))

        # --- Q3 router sweep (Y only; colour channels keep the fidelity branch)
        rows = []
        for q in (0.99, 0.95, 0.90, 0.80, 0.70, 0.5):
            fused_err, mask_frac = [], []
            for i in range(n):
                m = dis[i] > np.quantile(dis[i], q)
                fused_err.append(np.where(m, np.abs(gray_y(E[i]) - gray_y(G[i])),
                                          np.abs(gray_y(B[i]) - gray_y(G[i]))))
                mask_frac.append(m.mean())
            rows.append(dict(q=q, psnr=p_of(np.stack(fused_err)), routed=float(np.mean(mask_frac))))

        report[clip] = dict(frames=int(n),
                            psnr_bvsrpp=round(float(psnr_b), 3),
                            psnr_edvr=round(float(psnr_e), 3),
                            spearman_disagree_err=round(rho, 4),
                            oracle_upper_bound=round(p_of(orc), 3),
                            router=[{k: round(v, 4) for k, v in row.items()} for row in rows])
        print(clip, json.dumps(report[clip]), flush=True)

    json.dump(report, open(os.path.join(OUT, "route_probe.json"), "w"), indent=2)
    print("OK ->", os.path.join(OUT, "route_probe.json"))
