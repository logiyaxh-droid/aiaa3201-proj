#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FID over the cached per-clip InceptionV3 features.

``scipy.linalg.sqrtm`` on the 2048x2048 product of two non-symmetric matrices is
generically slow (it iterates Sylvester solves when the eigenvalues are complex);
on this shared machine it exceeded 25 min for one method.  Both covariances are
PSD, so the matrix square root of ``s1 @ s2`` is computed from the eigendecomposition
instead::

    s1^1/2 = V diag(sqrt(eig)) V^T            (eigh, s1 PSD)
    M      = s1^1/2 @ s2 @ s1^1/2             (PSD)
    tr(sqrt(s1 s2)) = tr(sqrt(M)) = sum(sqrt(eig(M)))

``--check`` verifies this against scipy's sqrtm on a small synthetic PSD pair, so
the number is not just "some FID-like quantity".
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np


def psd_sqrt(s):
    ev, V = np.linalg.eigh(s)
    ev = np.maximum(ev, 0.0)
    return (V * np.sqrt(ev)) @ V.T


def fid(f1, f2):
    """Frechet distance between two Gaussians fitted to the feature sets."""
    m1, m2 = f1.mean(0), f2.mean(0)
    s1 = np.cov(f1, rowvar=False)
    s2 = np.cov(f2, rowvar=False)
    s1h = psd_sqrt(s1)
    M = s1h @ s2 @ s1h
    ev = np.linalg.eigvalsh((M + M.T) / 2.0)
    tr = float(np.sqrt(np.maximum(ev, 0.0)).sum())
    diff = m1 - m2
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2.0 * tr)


def check():
    rng = np.random.default_rng(0)
    n, d = 400, 32
    a = rng.normal(size=(n, d)) * rng.uniform(0.5, 2.0, size=d)
    b = rng.normal(loc=0.3, size=(n, d)) * rng.uniform(0.5, 2.0, size=d)
    from scipy import linalg
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cm, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(cm):
        cm = cm.real
    ref = float((a.mean(0) - b.mean(0)).dot(a.mean(0) - b.mean(0))
                + np.trace(s1) + np.trace(s2) - 2.0 * np.trace(cm))
    mine = fid(a, b)
    print("scipy sqrtm  FID = %.10f" % ref)
    print("eigh  sqrt   FID = %.10f" % mine)
    print("abs diff         = %.3e  (pass=%s)" % (abs(ref - mine), abs(ref - mine) < 1e-6))
    return 0 if abs(ref - mine) < 1e-6 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--methods", nargs="*", default=None)
    args = ap.parse_args()
    if args.check:
        sys.exit(check())

    root = os.path.abspath(os.path.expanduser(args.root))
    feat = os.path.join(root, "feat")
    clips = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(feat, "gt", "*.npy")))
    methods = args.methods or sorted(os.path.basename(d) for d in glob.glob(os.path.join(feat, "*"))
                                     if os.path.isdir(d))
    gt = np.concatenate([np.load(os.path.join(feat, "gt", c + ".npy")) for c in clips])
    print("GT features %s from %d clips" % (gt.shape, len(clips)), flush=True)
    out = {}
    for m in methods:
        if m == "gt":
            continue
        files = [os.path.join(feat, m, c + ".npy") for c in clips]
        have = [p for p in files if os.path.exists(p)]
        if len(have) != len(clips):
            out[m] = {"value": None, "n_frames": 0,
                      "reason": "features incomplete (%d/%d clips)" % (len(have), len(clips))}
            print("%-20s incomplete %d/%d" % (m, len(have), len(clips)), flush=True)
            continue
        t = time.time()
        f = np.concatenate([np.load(p) for p in files])
        v = fid(gt, f)
        out[m] = {"value": v, "n_frames": int(f.shape[0]), "dims": int(f.shape[1])}
        print("%-20s FID %8.3f   (%d frames, %.1fs)" % (m, v, f.shape[0], time.time() - t),
              flush=True)
    path = os.path.join(root, "summary", "fid.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
    print("->", path, flush=True)


if __name__ == "__main__":
    main()
