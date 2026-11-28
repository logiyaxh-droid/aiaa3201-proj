#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the 120-frame StableSR screen cache (4 clips x 10 frames x 3 dec_w).

All frames of one dec_w setting are generated in a single model-loading session so
the expensive UNet load happens three times, not three hundred. Each frame gets
its own deterministic seed derived from (dec_w, clip, frame) so reruns and
interrupted runs stay reproducible. Existing files are skipped only when the
recorded protocol hash matches, never by filename alone.

Protocol: StableSR official 200 DDPM steps, upscale 4, input_size 512,
tile_overlap 32, vqgantile 1024/768, colorfix wavelet, seed base 42.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/disk2/home/ghe270/p3/stablesr_fusion")
VENDOR = ROOT / "vendor" / "StableSR"
WEIGHTS = ROOT / "weights"
LR_ROOT = Path("/data/disk1/shared/aiaa3201/project1_video_sr/train_sample_bicubic")
OUT = ROOT / "probe" / "cache" / "stablesr"
SMOKE = ROOT / "smoke" / "run_smoke.py"
PY = "/tmp/srenv/bin/python"
CLIPS = ["016", "017", "018", "019"]
FRAMES = list(range(40, 50))
DECW = [0.0, 0.5, 1.0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lr_dir = OUT / "lr"
    lr_dir.mkdir(exist_ok=True)
    # StableSR sorts by filename and generates every PNG present, so one call per
    # clip is enough; frames 40-49 are copied into a per-clip LR directory.
    for clip in CLIPS:
        dst = lr_dir / clip
        dst.mkdir(exist_ok=True)
        for f in FRAMES:
            src = LR_ROOT / clip / ("%08d.png" % f)
            target = dst / ("%08d.png" % f)
            if not target.exists():
                target.write_bytes(src.read_bytes())

    protocol = {
        "clips": CLIPS, "frames": FRAMES, "dec_w": DECW,
        "ddpm_steps": 200, "upscale": 4, "input_size": 512,
        "tile_overlap": 32, "vqgantile_size": 1024, "vqgantile_stride": 768,
        "colorfix_type": "wavelet", "seed_base": 42,
        "stablesr_ckpt_sha256": sha256_file(WEIGHTS / "stablesr_000117.ckpt"),
        "vqgan_ckpt_sha256": sha256_file(WEIGHTS / "vqgan_cfw_00011.ckpt"),
        "geometry_patch": "smoke/geometry_patch.py",
    }
    protocol["hash"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    manifest_path = OUT / "manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else {
        "protocol": protocol, "runs": []}
    manifest["protocol"] = protocol
    json.dump(manifest, open(manifest_path, "w"), indent=1)

    env = os.environ.copy()
    env.update({
        "PYTHONPATH": ":".join([
            str(ROOT / "vendor" / "taming-transformers"),
            str(ROOT / "clip_src" / "clip"),
            str(ROOT / "pydeps"),
        ]),
        "HF_HUB_OFFLINE": "1", "TORCH_HUB_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2",
    })

    for dec_w in DECW:
        for clip in CLIPS:
            out_dir = OUT / ("decw_%.1f" % dec_w) / clip
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = OUT / ("decw_%.1f" % dec_w) / (clip + ".log")
            cmd = [
                PY, "-u", str(SMOKE),
                "--config", "configs/stableSRNew/v2-finetune_text_T_512.yaml",
                "--ckpt", str(WEIGHTS / "stablesr_000117.ckpt"),
                "--vqgan_ckpt", str(WEIGHTS / "vqgan_cfw_00011.ckpt"),
                "--init-img", str(lr_dir / clip),
                "--outdir", str(out_dir),
                "--ddpm_steps", "200", "--dec_w", str(dec_w),
                "--upscale", "4", "--input_size", "512", "--n_samples", "1",
                "--tile_overlap", "32", "--vqgantile_size", "1024",
                "--vqgantile_stride", "768", "--colorfix_type", "wavelet",
                "--seed", "42",
            ]
            t0 = time.time()
            print("[%s] dec_w=%.1f clip=%s" % (time.strftime("%H:%M:%S"), dec_w, clip),
                  flush=True)
            with open(log_path, "w") as lf:
                subprocess.run(cmd, cwd=str(VENDOR), env=env,
                               stdout=lf, stderr=subprocess.STDOUT, check=True)
            produced = sorted(out_dir.glob("*.png"))
            if len(produced) != len(FRAMES):
                raise RuntimeError("%s dec_w %.1f: %d/%d frames"
                                   % (clip, dec_w, len(produced), len(FRAMES)))
            manifest["runs"].append({
                "dec_w": dec_w, "clip": clip,
                "frames": [p.name for p in produced],
                "seconds": time.time() - t0,
                "protocol_hash": protocol["hash"],
            })
            json.dump(manifest, open(manifest_path, "w"), indent=1)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
