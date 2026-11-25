# -*- coding: utf-8 -*-
"""M2-1: build 3-tier degraded LR training set from the read-only sharp clips.

Why three tiers instead of one
------------------------------
M1 showed that REDS-Real c9 is *harder* than the clean bicubic domain: RealBasicVSR
sits at 20.822 dB there while EDVR (trained on clean bicubic) cannot beat plain
bicubic.  A Stage A model that only ever sees the REDS-Real intensity would be
fitted to one operating point and could not tell us whether the gate/latent
residual mechanism generalises.  So training uses three intensities and the
Pilot/final evaluation stays pinned to REDS-Real c9 (v2 aligned).

Determinism contract
--------------------
Every clip x tier draws from its own seed.  `numpy.random.default_rng(seed)` is
used for *all* noise; the legacy global `np.random` is never touched, because the
M1 pipeline did use it and that made reruns depend on call order.  A per-file
SHA256 manifest is written so the LR set can be re-verified byte for byte.

Disk contract
-------------
Only 320x180 LR PNG is written (~500 MB for 20 clips x 100 frames x 3 tiers).
GT is NOT copied: training reads it directly from the read-only shared mount.

Usage:
  python build_train_lr.py                       # build everything
  python build_train_lr.py --clips 000 --frames 8 --out /tmp/lr_probe  # smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

SRC_DEFAULT = "/data/disk1/shared/aiaa3201/project1_video_sr/train_sample_sharp"

# sigma is in 1/255 units; bitrate in bits/s.  v2 mirrors REDS-Real exactly so
# the training distribution contains the evaluation operating point.
TIERS = {
    "v0": {"sigma": (1.0, 6.0), "bitrate": (2.0e5, 5.0e5),
           "jpeg": (60, 95), "resize": ((0.4, 1.5), (0.5, 1.2))},
    "v1": {"sigma": (1.0, 10.0), "bitrate": (5.0e4, 2.0e5),
           "jpeg": (45, 90), "resize": ((0.25, 1.4), (0.4, 1.2))},
    "v2": {"sigma": (1.0, 12.0), "bitrate": (1.0e4, 1.0e5),
           "jpeg": (30, 95), "resize": ((0.15, 1.5), (0.3, 1.2))},
}


def _rand_kernel(rng, rrng):
    """Random blur kernel, mirroring RealBasicVSR's kernel_list."""
    kind = rrng.choices(
        ["iso", "aniso", "generalized_iso", "generalized_aniso",
         "plateau_iso", "plateau_aniso", "sinc"],
        weights=[0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1])[0]
    ks = rrng.choice([7, 9, 11, 13, 15, 17, 19, 21])
    if kind == "sinc":
        omega = rrng.uniform(0.1, 0.5)
        k = np.arange(ks) - ks // 2
        xx, yy = np.meshgrid(k, k)
        r = np.sqrt(xx ** 2 + yy ** 2)
        kern = np.sinc(r * omega)
        s = kern.sum()
        return kern / s if s > 0 else np.ones((ks, ks)) / (ks * ks)

    def gauss(sx, sy, beta=2.0, plateau=False):
        k = np.arange(ks) - ks // 2
        xx, yy = np.meshgrid(k, k)
        if plateau:
            t = (np.abs(xx) / sx) ** beta + (np.abs(yy) / sy) ** beta
            kern = np.clip(1 - t, 0, None)
        else:
            kern = np.exp(-0.5 * ((xx / sx) ** 2 + (yy / sy) ** 2))
        s = kern.sum()
        return kern / s if s > 0 else np.ones((ks, ks)) / (ks * ks)

    if kind == "iso":
        s = rrng.uniform(0.2, 3.0)
        return gauss(s, s)
    if kind == "aniso":
        return gauss(rrng.uniform(0.2, 3.0), rrng.uniform(0.2, 3.0))
    if kind == "generalized_iso":
        s = rrng.uniform(0.2, 3.0)
        return gauss(s, s, rrng.uniform(0.5, 4.0))
    if kind == "generalized_aniso":
        return gauss(rrng.uniform(0.2, 3.0), rrng.uniform(0.2, 3.0),
                     rrng.uniform(0.5, 4.0))
    if kind == "plateau_iso":
        s = rrng.uniform(0.2, 3.0)
        return gauss(s, s, rrng.uniform(1, 2), plateau=True)
    s = rrng.uniform(0.2, 3.0)
    return gauss(s, s * 0.5, rrng.uniform(1, 2), plateau=True)


def _add_noise(img, nprng, rrng, sigma_hi):
    """Gaussian or poisson noise.  `img` is float32 in [0,1].

    All draws come from the explicit `nprng`, never from the global np.random.
    """
    if rrng.random() < 0.5:
        sigma = rrng.uniform(1.0, sigma_hi) / 255.0
        noise = nprng.normal(0.0, sigma, img.shape).astype(np.float32)
        if rrng.random() < 0.4:                      # gray noise
            noise = np.repeat(noise[..., :1], img.shape[2], axis=2)
        return np.clip(img + noise, 0.0, 1.0)
    scale = rrng.uniform(0.5, 3.0)
    clean = np.clip(img, 0.0, 1.0)
    vals = len(np.unique(clean))
    vals = 2 ** int(np.ceil(np.log2(max(vals, 2))))
    out = nprng.poisson(clean * vals / scale) * scale / vals
    return np.clip(out.astype(np.float32), 0.0, 1.0)


def _video_compress(frame_dir, rrng, ffmpeg, bitrate_lo, bitrate_hi):
    """Encode the sequence to a lossy video and read it back."""
    codec = rrng.choice(["libx264", "mpeg4"])
    bitrate = int(rrng.uniform(bitrate_lo, bitrate_hi))
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "c.mp4" if codec == "libx264" else "c.avi")
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", "10",
               "-pattern_type", "glob", "-i", os.path.join(frame_dir, "*.png"),
               "-c:v", codec, "-b:v", str(bitrate), out]
        if subprocess.run(cmd, capture_output=True).returncode != 0:
            return None
        back = tempfile.mkdtemp()
        try:
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", out,
                            os.path.join(back, "%08d.png")], capture_output=True)
            files = sorted(os.listdir(back))
            if not files:
                return None
            return [cv2.cvtColor(cv2.imread(os.path.join(back, f)),
                                 cv2.COLOR_BGR2RGB) for f in files]
        finally:
            shutil.rmtree(back, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def degrade(frames, rrng, nprng, ffmpeg, tier):
    """Apply the RealBasicVSR second-order degradation to one frame sequence."""
    cfg = TIERS[tier]
    cur = [f.astype(np.float32) for f in frames]
    for stage in range(2):
        k = _rand_kernel(nprng, rrng)
        cur = [np.clip(cv2.filter2D(f, -1, k), 0, 255) for f in cur]
        h, w = cur[0].shape[:2]
        lo, hi = cfg["resize"][stage]
        s = rrng.uniform(lo, hi)
        interp = rrng.choice([cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC])
        nw, nh = max(8, int(w * s)), max(8, int(h * s))
        cur = [cv2.resize(f, (nw, nh), interpolation=interp) for f in cur]
        cur = [cv2.resize(f, (w, h), interpolation=interp) for f in cur]
        cur = [np.clip(_add_noise(f / 255.0, nprng, rrng, cfg["sigma"][1])
                       * 255.0, 0, 255) for f in cur]
        q = rrng.randint(*cfg["jpeg"])
        cur = [cv2.imdecode(cv2.imencode(".jpg", f.astype(np.uint8),
                                         [cv2.IMWRITE_JPEG_QUALITY, q])[1],
                            cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32)
               for f in cur]
        tmpd = tempfile.mkdtemp()
        try:
            for i, f in enumerate(cur):
                cv2.imwrite(os.path.join(tmpd, "%08d.png" % i),
                            cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8),
                                         cv2.COLOR_RGB2BGR))
            got = _video_compress(tmpd, rrng, ffmpeg,
                                  *cfg["bitrate"])
            if got and len(got) == len(cur) and got[0].shape == cur[0].shape:
                cur = [g.astype(np.float32) for g in got]
        finally:
            shutil.rmtree(tmpd, ignore_errors=True)
    return cur


def seed_of(clip, tier, base):
    """Deterministic, order-independent seed for one (clip, tier) pair."""
    h = hashlib.sha256(("%s|%s|%d" % (clip, tier, base)).encode()).digest()
    return int.from_bytes(h[:8], "big")


def build_clip(job):
    """Worker: degrade one (clip, tier) and write 320x180 LR PNG."""
    clip, tier, src, out, seed, ffmpeg = job
    ids = sorted(os.path.splitext(f)[0]
                 for f in os.listdir(os.path.join(src, clip))
                 if f.lower().endswith(".png"))
    frames = [cv2.cvtColor(cv2.imread(os.path.join(src, clip, i + ".png")),
                           cv2.COLOR_BGR2RGB) for i in ids]
    if not frames or frames[0] is None:
        return "%s/%s: unreadable" % (clip, tier)
    h, w = frames[0].shape[:2]
    s = seed_of(clip, tier, seed)
    rrng = __import__("random").Random(s)
    nprng = np.random.default_rng(s)
    t0 = time.time()
    deg = degrade(frames, rrng, nprng, ffmpeg, tier)
    dst = os.path.join(out, tier, clip)
    os.makedirs(dst, exist_ok=True)
    for fid, f in zip(ids, deg):
        lr = cv2.resize(f, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(dst, fid + ".png"),
                    cv2.cvtColor(np.clip(lr, 0, 255).astype(np.uint8),
                                 cv2.COLOR_RGB2BGR))
    return "%s/%s %d frames lr=%dx%d %.1fs" % (
        clip, tier, len(ids), w // 4, h // 4, time.time() - t0)


def sha256_tree(root):
    h = hashlib.sha256()
    n = 0
    for dirpath, _, files in sorted(os.walk(root)):
        for f in sorted(files):
            p = os.path.join(dirpath, f)
            h.update(os.path.relpath(p, root).encode())
            with open(p, "rb") as fh:
                h.update(fh.read())
            n += 1
    return h.hexdigest(), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--out", default="/data/disk1/home/ghe270/p3/m2_stageA/data/train_lr")
    ap.add_argument("--clips", nargs="*", default=None,
                    help="subset of clip names, e.g. 000 001; default all 20")
    ap.add_argument("--frames", type=int, default=0, help="0 = all 100")
    ap.add_argument("--tiers", nargs="*", default=["v0", "v1", "v2"])
    ap.add_argument("--seed", type=int, default=20261125)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    import random as _random
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    clips = args.clips or sorted(d for d in os.listdir(args.src)
                                 if os.path.isdir(os.path.join(args.src, d)))
    if args.frames:
        # keep a contiguous prefix so the 5-frame windows stay adjacent
        pass

    jobs = []
    for clip in clips:
        for tier in args.tiers:
            jobs.append((clip, tier, args.src, args.out, args.seed, ffmpeg))

    print("jobs=%d workers=%d out=%s" % (len(jobs), args.workers, args.out),
          flush=True)
    if args.workers <= 1:
        for j in jobs:
            print(build_clip(j), flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for msg in ex.map(build_clip, jobs):
                print(msg, flush=True)
        _ = _random

    digest, nfiles = sha256_tree(args.out)
    man = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_gt": os.path.abspath(args.src),
        "seed": args.seed, "clips": clips, "tiers": args.tiers,
        "frames_per_clip": args.frames or None,
        "tier_params": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                            for kk, vv in v.items()} for k, v in TIERS.items()},
        "degradation": "RealBasicVSR second-order (blur/resize/noise/jpeg/"
                       "video-compression, applied twice) then /4 INTER_AREA",
        "rng": "numpy.random.default_rng(sha256(clip|tier|seed)) + random.Random",
        "n_files": nfiles, "sha256": digest,
    }
    mp = os.path.join(args.out, "manifest.json")
    os.makedirs(args.out, exist_ok=True)
    with open(mp, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, ensure_ascii=False)
    print("wrote %d files, sha256 %s" % (nfiles, digest), flush=True)
    print("manifest", mp, flush=True)


if __name__ == "__main__":
    main()
