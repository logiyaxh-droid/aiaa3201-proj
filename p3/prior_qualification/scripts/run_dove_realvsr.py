# -*- coding: utf-8 -*-
"""Run DOVE on the RealVSR subset and save PNG sequences for scoring.

DOVE is a video model: it needs consecutive frames, not single images.  Each
RealVSR clip contributes 5 consecutive frames (20-24), which does not satisfy
DOVE's (F-1) % 8 == 0 requirement, so the clip is padded to 9 frames by
repeating the last frame; only the first 5 outputs are kept.

Configuration matches the earlier qualification run (is_vae_st, tile 128,
overlap_hw 16, chunk_len 9, overlap_t 0).  overlap_t 1 previously produced
"Write count > 1 in region" during VAE stitching, so it stays 0.

Usage:
  python run_dove_realvsr.py --limit 1          # smoke
  python run_dove_realvsr.py                    # full 50 clips
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, "vendor", "DOVE-main")
WEIGHTS = os.path.join(ROOT, "weights_dove")
LQ4X = os.path.join(ROOT, "data", "realvsr", "LQ4x")
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name", default="dove")
    args = ap.parse_args()

    clips = sorted(d for d in os.listdir(LQ4X)
                   if os.path.isdir(os.path.join(LQ4X, d)))
    if args.limit:
        clips = clips[:args.limit]

    work = os.path.join(ROOT, "outputs", "realvsr", "_dove_in")
    out_root = os.path.join(ROOT, "outputs", "realvsr", args.name)
    os.makedirs(work, exist_ok=True)
    os.makedirs(out_root, exist_ok=True)

    for clip in clips:
        dst = os.path.join(out_root, clip)
        if os.path.isdir(dst) and len(glob.glob(os.path.join(dst, "*.png"))) >= 5:
            print("skip %s (already done)" % clip, flush=True)
            continue

        src = os.path.join(work, clip)
        shutil.rmtree(src, ignore_errors=True)
        os.makedirs(src)
        ids = sorted(os.path.splitext(f)[0]
                     for f in os.listdir(os.path.join(LQ4X, clip))
                     if f.lower().endswith(".png"))
        for f in ids:
            shutil.copy(os.path.join(LQ4X, clip, f + ".png"),
                        os.path.join(src, f + ".png"))
        # pad to (F-1) % 8 == 0 by repeating the last frame
        n = len(ids)
        pad = 0
        while (n + pad - 1) % 8 != 0:
            pad += 1
        for k in range(pad):
            shutil.copy(os.path.join(src, ids[-1] + ".png"),
                        os.path.join(src, "pad%03d.png" % k))

        # DOVE only globs *video* extensions, so the padded sequence has to be
        # muxed into an mp4 first.  system ffmpeg is absent on this host, so use
        # the binary bundled with imageio-ffmpeg.  Lossless-ish encode keeps the
        # 4x comparison meaningful.
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        vid = os.path.join(work, clip + ".mp4")
        enc = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error",
             "-framerate", "8", "-pattern_type", "glob",
             "-i", os.path.join(src, "*.png"),
             "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", vid],
            capture_output=True, text=True)
        if enc.returncode != 0 or not os.path.exists(vid):
            print("encode failed %s: %s" % (clip, enc.stderr[-300:]), flush=True)
            continue

        cmd = [PY, os.path.join(VENDOR, "inference_script.py"),
               "--input_dir", work,
               "--model_path", WEIGHTS,
               "--output_path", dst,
               "--png_save",
               "--is_vae_st",
               "--is_cpu_offload",
               "--tile_size_hw", "128", "128",
               "--overlap_hw", "16", "16",
               "--chunk_len", "9",
               "--overlap_t", "0",
               "--upscale", "4",
               "--dtype", "bfloat16"]
        print("RUN", clip, flush=True)
        r = subprocess.run(cmd, cwd=VENDOR, capture_output=True, text=True)
        tail = (r.stdout or "")[-600:] + (r.stderr or "")[-600:]
        print("%s rc=%d\n%s" % (clip, r.returncode, tail), flush=True)
        if os.path.exists(vid):
            os.remove(vid)

    # DOVE writes <output_path>/<clip>/<clip>/NNN.png numbered from 000, so remap
    # them onto the RealVSR frame ids (the first 5 of the padded 9) and flatten
    # the extra directory level.
    for clip in clips:
        d = os.path.join(out_root, clip)
        inner = os.path.join(d, clip)
        src_dir = inner if os.path.isdir(inner) else d
        if not os.path.isdir(src_dir):
            print("no output dir for %s" % clip, flush=True)
            continue
        ids = sorted(os.path.splitext(f)[0]
                     for f in os.listdir(os.path.join(LQ4X, clip))
                     if f.lower().endswith(".png"))
        outs = sorted(glob.glob(os.path.join(src_dir, "*.png")))
        print("%s remap %d outs -> %d ids (src=%s)"
              % (clip, len(outs), len(ids), src_dir), flush=True)
        for k, f in enumerate(outs[:len(ids)]):
            shutil.move(f, os.path.join(d, ids[k] + ".png"))
        if os.path.isdir(inner):
            shutil.rmtree(inner, ignore_errors=True)
        kept = sorted(os.path.basename(f) for f in glob.glob(os.path.join(d, "*.png")))
        print("%s kept %s" % (clip, kept), flush=True)
    shutil.rmtree(work, ignore_errors=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
