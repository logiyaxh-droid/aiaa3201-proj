# -*- coding: utf-8 -*-
"""Provenance: what weights, what code, what environment produced a number.

Part of the P0 fix is that a result row must be able to answer three questions
without asking the author: which weight file (SHA256, not just a filename),
which version of the metric code (hash of every evalkit module), and which
runtime.  Nothing here needs network access.
"""
from __future__ import annotations

import hashlib
import os
import platform
import sys

WEIGHT_NOTES = {
    "edsr": {
        "file": "EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth",
        "paper_claim": "EDSR x4",
        "train_data": "DIV2K (800 training images) -- NOT REDS",
        "source": "BasicSR official pretrain (basicsr/experiments/pretrained_models), "
                  "mirrored on the course shared disk",
        "uploaded": "2026-11-05 on the shared disk; upstream file name carries the "
                    "BasicSR release hash 0c287733",
    },
    "edvr_reds": {
        "file": "EDVR_L_x4_SR_REDS_official-9f5f5039.pth",
        "paper_claim": "EDVR-L x4 trained on REDS",
        "train_data": "REDS train_sharp, 240 clips 000-239, val_partition=REDS4 "
                      "(clips 000/011/015/020 held out of training)",
        "source": "BasicSR official pretrain, mirrored on the course shared disk",
        "config": "options/train/EDVR/train_EDVR_L_x4_SR_REDS.yml",
        "test_config": "options/test/EDVR/test_EDVR_L_x4_SR_REDS.yml (REDS4 clips)",
    },
    "edvr_vimeo": {
        "file": "EDVR_L_x4_SR_Vimeo90K_official-162b54e4.pth",
        "paper_claim": "EDVR-L x4 trained on Vimeo-90K septuplets",
        "train_data": "Vimeo-90K septuplet training split (sep_trainlist.txt)",
        "source": "BasicSR official pretrain, mirrored on the course shared disk",
    },
    "tecogan": {
        "file": "TecoGAN{,.index,.data-00000-of-00001,.meta}",
        "paper_claim": "TecoGAN",
        "train_data": "Vimeo-90K septuplet training split (TecoGAN ships "
                      "Vimeo-90K septuplets as its training data), degraded with the "
                      "official Gaussian pipeline (tf_data_gaussDownby4, sigma=1.5), "
                      "NOT with bicubic degradation and NOT with real-world video",
        "source": "official TF1 release (TecoGAN_v1 / TecoGAN.pb checkpoint), mirrored "
                  "on the course shared disk",
    },
}


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def file_digest(path: str) -> dict:
    if not os.path.exists(path):
        return {"path": path, "exists": False}
    st = os.stat(path)
    return {"path": path, "exists": True, "bytes": st.st_size, "sha256": sha256(path)}


def weights_manifest(weight_paths: dict) -> dict:
    """SHA256 + documented provenance for each configured weight."""
    man = {}
    for key, path in weight_paths.items():
        entry = {"configured_path": path}
        if key == "tecogan":
            base = path
            parts = {}
            for suf in ("", ".index", ".data-00000-of-00001", ".meta"):
                d = file_digest(base + suf)
                if d["exists"]:
                    parts[os.path.basename(base + suf)] = d["sha256"]
            entry["files"] = parts
        else:
            entry.update(file_digest(path))
        entry.update(WEIGHT_NOTES.get(key, {}))
        man[key] = entry
    return man


def code_manifest(pkg_dir: str) -> dict:
    out = {}
    for f in sorted(os.listdir(pkg_dir)):
        if f.endswith(".py"):
            out[f] = sha256(os.path.join(pkg_dir, f))
    return out


def environment() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_device"] = torch.cuda.get_device_name(0)
            env["cuda_version"] = torch.version.cuda
    except Exception as e:                     # pragma: no cover
        env["torch_error"] = repr(e)
    for mod in ("cv2", "numpy", "lpips", "scipy", "skimage"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import basicsr

        env["basicsr"] = getattr(basicsr, "__version__", "unknown")
        env["basicsr_path"] = os.path.dirname(basicsr.__file__)
    except Exception:
        pass
    return env
