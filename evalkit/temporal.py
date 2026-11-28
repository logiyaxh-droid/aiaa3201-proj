# -*- coding: utf-8 -*-
"""Temporal metrics -- the reference tLP, plus the legacy diagnostic kept apart.

Two different things used to share the name "tLPIPS" before P0.  They answer
different questions and are now computed, named and stored separately.

1. ``tlp`` -- the reference definition, re-implemented per
   TecoGAN ``metrics.py`` (thunil/TecoGAN, ``metrics.py`` lines ~194-210)::

       d_SR(t) = LPIPS(SR[t-1], SR[t])
       d_GT(t) = LPIPS(GT[t-1], GT[t])
       tLP(t)  = |d_SR(t) - d_GT(t)|

   averaged over *all* valid consecutive frame pairs of a clip.  Requires GT.
   If GT has no consecutive pair (every-other-frame datasets, single-GT-frame
   datasets such as Vimeo-90K im4 only), the value is **N/A (None)**, never 0.

   Alignment switches, all recorded in the run config and result JSON:

   ``x100``        multiply the per-pair absolute difference by 100 (TecoGAN's
                   own ``tLP100``).  Off by default, so the raw LPIPS-unit
                   number is what gets stored.
   ``cutfr``       drop that many frames at BOTH ends of the clip before pairing
                   (TecoGAN uses ``cutfr = 2``; default 0 here so the number
                   covers every pair unless a config asks otherwise).
   ``crop32``      apply TecoGAN's ``crop_8x8`` centre crop (to a multiple of
                   32, keeping >= 16 px of margin) to every frame of BOTH
                   sequences before LPIPS (default False).
   ``lpips_model`` must match between the two sequences; the same ``LPIPS``
                   instance is used for both, so this holds by construction.

2. ``flow_warp_lpips`` -- the legacy diagnostic that was previously labelled
   "tLPIPS": ``LPIPS(warp(prev_output, flow), cur_output)``.  It measures how
   well an *output* sequence is registered/stable against itself and has no GT
   in it, so it can be computed on wild video.  Kept for continuity with the old
   JSONs, clearly renamed, and its ``stride`` is stored because ``stride=2``
   only covers every other adjacent pair.
"""
from __future__ import annotations

import numpy as np

TLP_DEFINITION = {
    "formula": "tLP(t) = |LPIPS(SR[t-1],SR[t]) - LPIPS(GT[t-1],GT[t])|, mean over pairs",
    "source": "TecoGAN metrics.py (thunil/TecoGAN), re-implemented",
    "requires_gt": True,
    "no_gt_value": "N/A (null), never 0",
    "pairs": "consecutive frames inside one clip only; never across clips or gaps",
}


def pairs_of(ids: list, cutfr: int) -> list:
    """Valid (prev, cur) index pairs inside a clip's frame list.

    ``ids`` is the ordered list of frame ids actually present.  Consecutive means
    ``ids[k+1] == ids[k] + 1``; a gap breaks the adjacency rather than being
    silently bridged, and pairs touching one of the first/last ``cutfr`` frames
    are dropped (TecoGAN's ``cutfr = 2`` excludes the two frames at each end).
    """
    lo = int(cutfr)
    last = len(ids) - 1 - int(cutfr)       # last index still allowed as a pair member
    out = []
    for k in range(lo, last):
        if ids[k + 1] == ids[k] + 1:
            out.append((k, k + 1))
    return out


def crop_8x8(img: np.ndarray):
    """Literal port of TecoGAN's ``crop_8x8`` (centre crop to a multiple of 32)."""
    ori_h, ori_w = img.shape[0], img.shape[1]
    h = (ori_h // 32) * 32
    w = (ori_w // 32) * 32
    while h > ori_h - 16:
        h -= 32
    while w > ori_w - 16:
        w -= 32
    y = (ori_h - h) // 2
    x = (ori_w - w) // 2
    return img[y:y + h, x:x + w], y, x


def tlp(pred: list, gt: list, ids: list, lpips_net, x100: bool = False,
        cutfr: int = 0, crop32: bool = False) -> dict:
    """Reference tLP for one clip.

    ``pred``/``gt`` are equal-length lists of uint8 RGB frames, ``ids`` the
    matching integer frame ids.  Returns a dict with the mean and the pair count;
    ``value`` is None when there is no valid pair (no GT, or GT not consecutive).

    Batch-size note: LPIPS on GPU is not bit-identical across batch sizes
    (cudnn picks different kernels), so a per-pair loop and this batched call
    agree to ~1e-7 but not to 0.  The check script asserts < 1e-6.
    """
    if len(pred) != len(gt) or len(pred) != len(ids):
        raise ValueError("tlp: pred/gt/ids length mismatch %d/%d/%d"
                         % (len(pred), len(gt), len(ids)))
    pr = pairs_of(ids, cutfr)
    if not pr:
        return {"value": None, "n_pairs": 0,
                "reason": "no consecutive frame pair inside the clip"
                          + (" (cutfr=%d)" % cutfr if cutfr else "")}

    def prep(x):
        return crop_8x8(x)[0] if crop32 else x

    sr_prev = np.stack([prep(pred[a]) for a, _ in pr])
    sr_cur = np.stack([prep(pred[b]) for _, b in pr])
    gt_prev = np.stack([prep(gt[a]) for a, _ in pr])
    gt_cur = np.stack([prep(gt[b]) for _, b in pr])

    d_sr = lpips_net.pair(sr_prev, sr_cur)
    d_gt = lpips_net.pair(gt_prev, gt_cur)
    d = np.abs(d_sr - d_gt)
    if x100:
        d = d * 100.0
    return {"value": float(d.mean()), "n_pairs": int(len(pr)),
            "d_sr_mean": float(d_sr.mean()), "d_gt_mean": float(d_gt.mean()),
            "x100": bool(x100), "cutfr": int(cutfr), "crop32": bool(crop32)}


# --------------------------------------------------------------------------- legacy
def fb_target_to_source(target_gray, source_gray, scale=4):
    """Flow that pulls ``source`` onto ``target`` (direction calibrated in flow_selftest).

    ``calcOpticalFlowFarneback(target, source)`` then ``remap(source, xy + f)``.
    The opposite order raises the warp error by ~30 %; the self-test proves which
    is which from first principles instead of trusting the docs.
    """
    import cv2

    h, w = target_gray.shape
    if scale == 1:
        return cv2.calcOpticalFlowFarneback(target_gray, source_gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
    ts = cv2.resize(target_gray, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    ss = cv2.resize(source_gray, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    f = cv2.calcOpticalFlowFarneback(ts, ss, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) * float(scale)


def warp_onto(src, target_gray, src_gray, gx, gy, scale=4):
    import cv2

    f = fb_target_to_source(target_gray, src_gray, scale=scale)
    return cv2.remap(src, gx + f[..., 0], gy + f[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def flow_warp_lpips(pred: list, ids: list, lpips_net, stride: int = 2, scale: int = 4) -> dict:
    """Legacy diagnostic: LPIPS(warp(prev), cur) on the OUTPUT sequence (no GT)."""
    import cv2

    if len(ids) != len(pred):
        raise ValueError("flow_warp_lpips: pred/ids length mismatch")
    H, W = pred[0].shape[:2]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    grays = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in pred]
    al, bl, used = [], [], []
    for i in range(1, len(pred), stride):
        if ids[i] != ids[i - 1] + 1:
            continue
        al.append(warp_onto(pred[i - 1], grays[i], grays[i - 1], gx, gy, scale=scale))
        bl.append(pred[i])
        used.append((ids[i - 1], ids[i]))
    if not al:
        return {"value": None, "n_pairs": 0, "stride": stride, "flow_scale": scale,
                "reason": "no consecutive pair at this stride"}
    v = lpips_net.pair(np.stack(al), np.stack(bl))
    return {"value": float(v.mean()), "n_pairs": int(len(used)), "stride": int(stride),
            "flow_scale": int(scale),
            "note": "adjacent-pair coverage: every %d-th pair, NOT all adjacent pairs"
                    % stride}
