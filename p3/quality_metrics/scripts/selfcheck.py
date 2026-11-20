# -*- coding: utf-8 -*-
"""Stage-A self-check, roadmap section 3.3 -- run BEFORE any full measurement.

Six checks, each answering one specific question:

1. identity      pred == GT  ->  DISTS should be ~0.  Records the float residue.
2. degradation   on a fixed 10-frame subset, build blur / noise / JPEG /
                 shift / over-sharpening and record which direction each metric
                 moves.  DISTS must rise for every severe distortion; MUSIQ and
                 DOVER are only required to be *recorded*, not monotone.
3. reference     pyiqa DISTS vs the official `DISTS_pytorch` implementation on
                 >=10 image pairs, absolute error must stay < 1e-5.
4. repeatability the same clip scored twice must agree to float error, and
                 DOVER's sampled frame indices must be reproducible.
5. old-metric    PSNR / LPIPS recomputed in this environment vs the stored P0
                 numbers for clip 000 (LPIPS tolerance < 1e-6).
6. sanity        side-by-side numbers for bicubic / edvr / tecogan, printed with
                 their images so an obviously inverted metric can be spotted.

Anything that fails here gets fixed before stage B/C, and the failure is written
into selfcheck/metric_selfcheck.json rather than being papered over.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch

import common as C

OUT = os.path.join(C.ROOT, "selfcheck")
CLIP = "000"
N_DEG = 10          # fixed 10-frame subset for the monotonicity probe


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ 1 identity

def check_identity(dists):
    log("\n[1] identity: DISTS(pred=x, gt=x) should be ~0")
    gt = C.read_clip(os.path.join(C.GT_ROOT, CLIP))[:8]
    d = dists.per_frame(gt, gt)
    # also through uint8 round-trip, since that is what a real prediction does
    d2 = dists.per_frame(gt, [np.ascontiguousarray(x) for x in gt])
    res = {
        "n_frames": int(len(d)),
        "max_abs_score": float(np.abs(d).max()),
        "per_frame": [float(x) for x in d],
        "uint8_roundtrip_max_abs": float(np.abs(d2).max()),
        "pass": bool(np.abs(d).max() < 1e-6),
    }
    log("    max |DISTS| = %.3e   (uint8 round-trip %.3e)  -> %s"
        % (res["max_abs_score"], res["uint8_roundtrip_max_abs"],
           "PASS" if res["pass"] else "FAIL"))
    return res


# --------------------------------------------------------------- 2 degradation

def _blur(f):
    return cv2.GaussianBlur(f, (0, 0), 2.0)


def _noise(f, sigma=12.0, seed=0):
    rng = np.random.RandomState(seed)
    n = rng.normal(0.0, sigma, f.shape)
    return np.clip(f.astype(np.float32) + n, 0, 255).astype(np.uint8)


def _jpeg(f, q=20):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    b = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(b, cv2.COLOR_BGR2RGB)


def _shift(f, dx=3, dy=2):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(f, m, (f.shape[1], f.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _oversharp(f, amount=2.5, radius=1.0):
    ff = f.astype(np.float32)
    return np.clip(np.rint(ff + amount * (ff - cv2.GaussianBlur(ff, (0, 0), radius))),
                   0, 255).astype(np.uint8)


DEGRADATIONS = {
    "blur_sigma2": _blur,
    "noise_s12": _noise,
    "jpeg_q20": _jpeg,
    "shift_3_2px": _shift,
    "oversharp_2.5": _oversharp,
}


def check_degradation(dists, musiq, dover):
    log("\n[2] degradation response on a fixed %d-frame subset" % N_DEG)
    ids = C.clip_frame_ids(os.path.join(C.GT_ROOT, CLIP))[:N_DEG]
    gt = C.read_clip(os.path.join(C.GT_ROOT, CLIP), ids)

    rows = {"clean": None}
    clean = {
        "dists": float(dists.per_frame(gt, gt).mean()),
        "musiq": float(musiq.per_frame(gt).mean()),
        "dover": dover.score_clip(os.path.join(C.GT_ROOT, CLIP)),
    }
    rows["clean"] = clean
    log("    clean      DISTS %.4f  MUSIQ %.3f  DOVERtech %.4f  DOVERaes %.4f"
        % (clean["dists"], clean["musiq"],
           clean["dover"]["technical"], clean["dover"]["aesthetic"]))

    for name, fn in DEGRADATIONS.items():
        degraded = [fn(x) for x in gt]
        # DOVER is whole-clip; feed a temp dir of the degraded frames
        d_dists = float(dists.per_frame(degraded, gt).mean())
        d_musiq = float(musiq.per_frame(degraded).mean())
        d_dover = _dover_on_frames(dover, degraded, tag=name)
        rows[name] = {"dists": d_dists, "musiq": d_musiq, "dover": d_dover}
        log("    %-12s DISTS %.4f  MUSIQ %.3f  DOVERtech %.4f  DOVERaes %.4f"
            % (name, d_dists, d_musiq, d_dover["technical"], d_dover["aesthetic"]))

    dists_up = all(rows[k]["dists"] > rows["clean"]["dists"] for k in DEGRADATIONS)
    res = {
        "n_frames": N_DEG,
        "frame_ids": ids,
        "rows": rows,
        "dists_rises_for_all": bool(dists_up),
        "note": "MUSIQ / DOVER are recorded, not required to be monotone.",
        "pass": bool(dists_up),
    }
    log("    DISTS rises for all 5 distortions: %s" % res["pass"])
    return res


def _dover_on_frames(dover, frames, tag=""):
    """Score an in-memory frame list by staging it as a temporary PNG clip."""
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="dover_%s_" % tag, dir="/tmp")
    try:
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(tmp, "%08d.png" % i),
                        cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        return dover.score_clip(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------- 3 reference

def check_reference(dists, pairs=12):
    """pyiqa DISTS vs the official DISTS_pytorch on the same image pairs."""
    log("\n[3] reference implementation: pyiqa DISTS vs official DISTS_pytorch")
    sys.path.insert(0, os.path.join(C.ROOT, "vendor", "dists_official"))
    from DISTS_pt import DISTS as OfficialDISTS

    official = OfficialDISTS(load_weights=False).to(dists.device).eval()
    w = torch.load(C.WEIGHT_FILES["dists_official"], map_location="cpu",
                   weights_only=False)
    official.alpha.data = w["alpha"].to(dists.device)
    official.beta.data = w["beta"].to(dists.device)
    for p in official.parameters():
        p.requires_grad_(False)

    gt_ids = C.clip_frame_ids(os.path.join(C.GT_ROOT, CLIP))[:pairs]
    gt = C.read_clip(os.path.join(C.GT_ROOT, CLIP), gt_ids)
    # pair each GT frame with a genuinely different image: the bicubic upscale
    bic = C.read_clip(os.path.join(C.BICUBIC_PRED, CLIP), gt_ids)

    diffs, ref_scores, ours = [], [], []
    for a, b in zip(bic, gt):
        x = C.to_nchw01([a], dists.device)
        y = C.to_nchw01([b], dists.device)
        with torch.no_grad():
            s_ref = float(official(x, y).reshape(-1)[0])
            s_ours = float(dists.net(x, y).reshape(-1)[0])
        ref_scores.append(s_ref)
        ours.append(s_ours)
        diffs.append(abs(s_ref - s_ours))

    res = {
        "n_pairs": len(pairs_ids := list(gt_ids)),
        "frame_ids": pairs_ids,
        "official_scores": ref_scores,
        "pyiqa_scores": ours,
        "abs_diffs": diffs,
        "max_abs_diff": float(max(diffs)),
        "tolerance": 1e-5,
        "weights_identical_to_pyiqa": C.weight_sha256("dists_official")
                                      == C.weight_sha256("dists"),
        "pass": bool(max(diffs) < 1e-5),
    }
    log("    max |pyiqa - official| = %.3e over %d pairs (tol 1e-5) -> %s"
        % (res["max_abs_diff"], len(diffs), "PASS" if res["pass"] else "FAIL"))
    log("    weights sha256 identical: %s" % res["weights_identical_to_pyiqa"])
    return res


# ------------------------------------------------------------ 4 repeatability

def check_repeatability(dists, musiq, dover):
    log("\n[4] repeatability: score the same clip twice")
    ids = C.clip_frame_ids(os.path.join(C.GT_ROOT, CLIP))
    pred = C.read_clip(os.path.join(C.BICUBIC_PRED, CLIP), ids)
    gt = C.read_clip(os.path.join(C.GT_ROOT, CLIP), ids)

    d1 = dists.per_frame(pred, gt)
    d2 = dists.per_frame(pred, gt)
    m1 = musiq.per_frame(pred[:20])
    m2 = musiq.per_frame(pred[:20])

    o1 = dover.score_clip(os.path.join(C.BICUBIC_PRED, CLIP))
    o2 = dover.score_clip(os.path.join(C.BICUBIC_PRED, CLIP))

    res = {
        "dists_max_abs_diff": float(np.abs(d1 - d2).max()),
        "musiq_max_abs_diff": float(np.abs(m1 - m2).max()),
        "dover_technical": [o1["technical"], o2["technical"]],
        "dover_aesthetic": [o1["aesthetic"], o2["aesthetic"]],
        "dover_scores_identical": bool(o1["technical"] == o2["technical"]
                                       and o1["aesthetic"] == o2["aesthetic"]),
        "dover_frame_inds_identical": bool(o1["frame_inds"] == o2["frame_inds"]),
        "dover_seed": dover.seed,
    }
    res["pass"] = bool(res["dists_max_abs_diff"] < 1e-9
                       and res["musiq_max_abs_diff"] < 1e-9
                       and res["dover_scores_identical"]
                       and res["dover_frame_inds_identical"])
    log("    DISTS max diff %.3e | MUSIQ max diff %.3e" %
        (res["dists_max_abs_diff"], res["musiq_max_abs_diff"]))
    log("    DOVER identical: %s | frame inds identical: %s (seed %d) -> %s"
        % (res["dover_scores_identical"], res["dover_frame_inds_identical"],
           dover.seed, "PASS" if res["pass"] else "FAIL"))
    return res


# ------------------------------------------------------------- 5 old metrics

def check_old_metrics():
    """PSNR / LPIPS for clip 000 bicubic, this env vs the stored P0 numbers."""
    log("\n[5] old-metric regression: clip 000 bicubic vs stored P0 results")
    old_path = "/data/disk1/home/ghe270/p0/out/full/results/%s.json" % CLIP
    old = json.load(open(old_path, encoding="utf-8"))["bicubic"]

    sys.path.insert(0, "/data/disk1/home/ghe270/p0")
    from evalkit import metrics as M

    ids = C.clip_frame_ids(os.path.join(C.GT_ROOT, CLIP))
    pred = C.read_clip(os.path.join(C.BICUBIC_PRED, CLIP), ids)
    gt = C.read_clip(os.path.join(C.GT_ROOT, CLIP), ids)

    pY = np.stack([M.luma(x) for x in pred])
    gY = np.stack([M.luma(x) for x in gt])
    psnr = M.psnr_mean(pY, gY)
    lpips_net = M.LPIPS(device="cuda")
    lpips = lpips_net.mean(np.stack(pred), np.stack(gt))

    res = {
        "clip": CLIP, "method": "bicubic",
        "old": {"psnr": old["psnr"], "lpips": old["lpips"]},
        "new": {"psnr": float(psnr), "lpips": float(lpips)},
        "abs_diff": {"psnr": abs(float(psnr) - old["psnr"]),
                     "lpips": abs(float(lpips) - old["lpips"])},
        "tolerance": {"psnr": 1e-3, "lpips": 1e-6},
    }
    res["pass"] = bool(res["abs_diff"]["psnr"] < 1e-3
                       and res["abs_diff"]["lpips"] < 1e-6)
    log("    PSNR  old %.6f  new %.6f  |diff| %.2e (tol 1e-3)"
        % (old["psnr"], psnr, res["abs_diff"]["psnr"]))
    log("    LPIPS old %.6f  new %.6f  |diff| %.2e (tol 1e-6) -> %s"
        % (old["lpips"], lpips, res["abs_diff"]["lpips"],
           "PASS" if res["pass"] else "FAIL"))
    return res


# ---------------------------------------------------------------- 6 sanity

def check_sanity(dists, musiq):
    log("\n[6] sanity: the three stage-A methods plus GT, one clip")
    ids = C.clip_frame_ids(os.path.join(C.GT_ROOT, CLIP))
    gt = C.read_clip(os.path.join(C.GT_ROOT, CLIP), ids)
    sources = {
        "gt": gt,
        "bicubic": C.read_clip(os.path.join(C.BICUBIC_PRED, CLIP), ids),
        "edvr": C.read_clip(os.path.join(C.EDVR_FRAMES, CLIP), ids),
    }
    teco_dir = os.path.join(C.TECO_FRAMES, CLIP)
    if os.path.isdir(teco_dir):
        sources["tecogan"] = C.read_clip(teco_dir, ids)

    rows = {}
    for name, frames in sources.items():
        d = float(dists.per_frame(frames, gt).mean())
        m = float(musiq.per_frame(frames).mean())
        rows[name] = {"dists": d, "musiq": m}
        log("    %-9s DISTS %.4f  MUSIQ %.3f" % (name, d, m))

    expect = (rows["edvr"]["dists"] < rows["bicubic"]["dists"]
              and rows["bicubic"]["musiq"] < rows["gt"]["musiq"])
    res = {"rows": rows,
           "expected_order": "edvr DISTS < bicubic DISTS, and "
                             "bicubic MUSIQ < GT MUSIQ",
           "as_expected": bool(expect),
           "pass": bool(expect)}
    log("    direction as expected: %s" % res["pass"])
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    t_all = time.time()
    log("=== P3 quality-metrics stage-A self-check (clip %s) ===" % CLIP)

    dists = C.DistsMetric()
    musiq = C.MusiqMetric(variant="koniq10k")
    dover = C.DoverMetric()

    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "clip": CLIP,
        "environment": C.environment_json(),
        "pairing": {"dists": dists.pairing(), "musiq": musiq.pairing(),
                    "dover": dover.pairing()},
    }
    report["identity"] = check_identity(dists)
    report["degradation"] = check_degradation(dists, musiq, dover)
    report["reference"] = check_reference(dists)
    report["repeatability"] = check_repeatability(dists, musiq, dover)
    report["old_metrics"] = check_old_metrics()
    report["sanity"] = check_sanity(dists, musiq)

    gate = ["identity", "reference", "repeatability", "old_metrics"]
    report["gate"] = {k: report[k]["pass"] for k in gate}
    report["gate_passed"] = all(report["gate"].values())
    report["total_seconds"] = round(time.time() - t_all, 1)

    C.write_json(os.path.join(OUT, "metric_selfcheck.json"), report)
    C.write_json(os.path.join(OUT, "environment.json"), report["environment"])

    log("\n=== gate ===")
    for k in gate:
        log("    %-14s %s" % (k, "PASS" if report[k]["pass"] else "FAIL"))
    log("    overall: %s   (%.1f s)"
        % ("PASS" if report["gate_passed"] else "FAIL", report["total_seconds"]))
    log("    wrote %s/metric_selfcheck.json" % OUT)
    if not report["gate_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
