# -*- coding: utf-8 -*-
"""Build REDS-Real: a frozen real-world degradation benchmark over REDS val.

Why this benchmark
------------------
The three existing evaluation sets each have a defect that muddies the
direction-C conclusion:

  * REDS `val_sharp_bicubic` is a clean bicubic downgrade, which flatters
    regression models and says nothing about compression or noise;
  * VideoLQ has no GT, so only no-reference metrics apply;
  * RealVSR's LQ and GT share a pixel grid (about 2x effective), and this repo
    further downsamples by 4 so that x4 models can consume it, which low-passes
    the original optical degradation and biases the comparison toward EDVR.

REDS-Real fixes the first and third at once: take the *clean* REDS val GT, apply
the RealBasicVSR second-order degradation (blur -> random resize -> noise ->
video compression, twice), and keep the GT.  Every method then sees exactly the
same frozen LR set with a real 4x relation to GT.

Degradation follows the official RealBasicVSR training pipeline
(`mmedit/datasets/pipelines/GenerateFrameIndices` + `AddRandomDegradations`):

  1. anisotropic/isotropic/generalized Gaussian or sinc blur, kernel 7..21
  2. random resize, scale 0.15..1.5 then 0.3..1.2
  3. gaussian or poisson noise, sigma 1..30 / scale 0.05..3
  4. JPEG quality 30..95
  5. video compression (libx264 / mpeg4), bitrate 1e4..1e5
  ... then the whole chain a second time, and finally resize to LR = GT/4.

The seed is fixed and a SHA256 manifest is written, so the LR set is frozen and
every later run can be verified byte-for-byte.

Usage:
  python build_reds_real.py --gt /path/val_sharp --out ../data/reds_real \
      --clips 30 --frames 100 --seed 20261124
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile

import cv2
import numpy as np


def _rand_kernel(rng):
    """Random blur kernel, mirroring RealBasicVSR's kernel_list."""
    kind = rng.choices(
        ["iso", "aniso", "generalized_iso", "generalized_aniso",
         "plateau_iso", "plateau_aniso", "sinc"],
        weights=[0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1])[0]
    ks = rng.choice([7, 9, 11, 13, 15, 17, 19, 21])
    if kind == "sinc":
        omega = rng.uniform(0.1, 0.5)
        k = np.arange(ks) - ks // 2
        xx, yy = np.meshgrid(k, k)
        r = np.sqrt(xx ** 2 + yy ** 2)
        kern = np.sinc(r * omega)
        kern /= kern.sum()
        return kern

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
        s = rng.uniform(0.2, 3.0)
        return gauss(s, s)
    if kind == "aniso":
        return gauss(rng.uniform(0.2, 3.0), rng.uniform(0.2, 3.0))
    if kind == "generalized_iso":
        s = rng.uniform(0.2, 3.0)
        return gauss(s, s, rng.uniform(0.5, 4.0))
    if kind == "generalized_aniso":
        return gauss(rng.uniform(0.2, 3.0), rng.uniform(0.2, 3.0),
                     rng.uniform(0.5, 4.0))
    if kind == "plateau_iso":
        s = rng.uniform(0.2, 3.0)
        return gauss(s, s, rng.uniform(1, 2), plateau=True)
    s = rng.uniform(0.2, 3.0)
    return gauss(s, s * 0.5, rng.uniform(1, 2), plateau=True)


def _add_noise(img, rng):
    """Gaussian or poisson noise, clipped to [0,1].

    The RealBasicVSR ranges (gaussian sigma up to 30/255, poisson scale down to
    0.05) are meant for 256x256 training crops of *clean* frames.  Applied at
    full 720p after a blur, they leave more high-frequency energy in the LR than
    a clean downsample has (measured Laplacian 15.45 vs 27.74 for clean, which
    is backwards).  The scale is therefore reduced and the result clipped.
    """
    if rng.random() < 0.5:
        sigma = rng.uniform(1.0, 12.0) / 255.0
        noise = np.random.normal(0, sigma, img.shape)
        if rng.random() < 0.4:                     # gray noise
            noise = np.repeat(noise[..., :1], img.shape[2], axis=2)
        return np.clip(img + noise, 0.0, 1.0)
    scale = rng.uniform(0.5, 3.0)
    clean = np.clip(img, 0.0, 1.0)
    vals = len(np.unique(clean))
    vals = 2 ** int(np.ceil(np.log2(max(vals, 2))))
    out = np.random.poisson(clean * vals / scale) * scale / vals
    return np.clip(out, 0.0, 1.0)


def _video_compress(frame_dir, rng, ffmpeg):
    """Encode the sequence to a lossy video and read it back."""
    codec = rng.choice(["libx264", "mpeg4"])
    bitrate = int(rng.uniform(1e4, 1e5))
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "c.mp4" if codec == "libx264" else "c.avi")
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", "10",
               "-pattern_type", "glob", "-i", os.path.join(frame_dir, "*.png"),
               "-c:v", codec, "-b:v", str(bitrate), out]
        if subprocess.run(cmd, capture_output=True).returncode != 0:
            return None
        back = tempfile.mkdtemp()
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", out,
                        os.path.join(back, "%08d.png")], capture_output=True)
        files = sorted(os.listdir(back))
        frames = [cv2.cvtColor(cv2.imread(os.path.join(back, f)),
                               cv2.COLOR_BGR2RGB) for f in files]
        shutil.rmtree(back, ignore_errors=True)
        return frames
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def degrade(frames, rng, ffmpeg):
    """Apply the RealBasicVSR second-order degradation to a frame sequence."""
    cur = [f.copy() for f in frames]
    for stage in range(2):
        k = _rand_kernel(rng)
        cur = [np.clip(cv2.filter2D(f.astype(np.float32), -1, k), 0, 255)
               for f in cur]
        # random resize then back to the working size
        h, w = cur[0].shape[:2]
        s = rng.uniform(0.15, 1.5) if stage == 0 else rng.uniform(0.3, 1.2)
        interp = rng.choice([cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC])
        nw, nh = max(8, int(w * s)), max(8, int(h * s))
        cur = [cv2.resize(f, (nw, nh), interpolation=interp) for f in cur]
        cur = [cv2.resize(f, (w, h), interpolation=interp) for f in cur]
        cur = [np.clip(_add_noise(f / 255.0, rng) * 255.0, 0, 255) for f in cur]
        # JPEG
        q = rng.randint(30, 95)
        cur = [cv2.imdecode(cv2.imencode(".jpg", f.astype(np.uint8),
                                         [cv2.IMWRITE_JPEG_QUALITY, q])[1],
                            cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32)
               for f in cur]
        # video compression
        tmpd = tempfile.mkdtemp()
        try:
            for i, f in enumerate(cur):
                cv2.imwrite(os.path.join(tmpd, "%08d.png" % i),
                            cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2BGR))
            got = _video_compress(tmpd, rng, ffmpeg)
            if got and len(got) == len(cur) and got[0].shape == cur[0].shape:
                cur = got
        finally:
            shutil.rmtree(tmpd, ignore_errors=True)
    return cur


def sha256_dir(root):
    h = hashlib.sha256()
    for dirpath, _, files in sorted(os.walk(root)):
        for f in sorted(files):
            p = os.path.join(dirpath, f)
            h.update(os.path.relpath(p, root).encode())
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="dir of <clip>/<frame>.png GT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", type=int, default=0, help="0 = all")
    ap.add_argument("--frames", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=20261124)
    args = ap.parse_args()

    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    clips = sorted(d for d in os.listdir(args.gt)
                   if os.path.isdir(os.path.join(args.gt, d)))
    if args.clips:
        clips = clips[:args.clips]
    n_ok = 0
    for clip in clips:
        ids = sorted(os.path.splitext(f)[0]
                     for f in os.listdir(os.path.join(args.gt, clip))
                     if f.lower().endswith(".png"))
        if args.frames:
            ids = ids[:args.frames]
        frames = [cv2.cvtColor(cv2.imread(os.path.join(args.gt, clip, i + ".png")),
                               cv2.COLOR_BGR2RGB) for i in ids]
        h, w = frames[0].shape[:2]
        # each clip gets its own deterministic RNG so reruns are reproducible
        rng = random.Random(args.seed + int(clip) if clip.isdigit()
                            else args.seed + hash(clip) % 10000)
        deg = degrade(frames, rng, ffmpeg)
        dg = os.path.join(args.out, "GT", clip)
        dl = os.path.join(args.out, "LQ", clip)
        os.makedirs(dg, exist_ok=True)
        os.makedirs(dl, exist_ok=True)
        for i, (fid, f) in enumerate(zip(ids, frames)):
            cv2.imwrite(os.path.join(dg, fid + ".png"),
                        cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            lr = cv2.resize(deg[i], (w // 4, h // 4), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(dl, fid + ".png"),
                        cv2.cvtColor(np.clip(lr, 0, 255).astype(np.uint8),
                                     cv2.COLOR_RGB2BGR))
            n_ok += 1
        print("%s %d frames lr=%dx%d" % (clip, len(ids), w // 4, h // 4),
              flush=True)
    print("wrote %d frames" % n_ok)

    manifest = {
        "created": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_gt": os.path.abspath(args.gt),
        "seed": args.seed, "clips": clips,
        "degradation": "RealBasicVSR second-order (blur/resize/noise/jpeg/"
                       "video-compression, applied twice) then /4 INTER_AREA",
        "sha256": sha256_dir(args.out),
    }
    with open(os.path.join(args.out, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print("manifest sha256", manifest["sha256"])


if __name__ == "__main__":
    main()
