# -*- coding: utf-8 -*-
"""Run configuration: explicit paths, no hidden ``/tmp`` dependencies.

Every run writes a frozen copy of the resolved config next to its results, so a
number in the report can be traced to the exact inputs that produced it.  The
defaults below are the *measured* defaults of the pre-P0 scripts; overriding a
field is how the P0 re-checks keep old and new results comparable instead of
silently changing what a metric means.
"""
from __future__ import annotations

import copy
import json
import os

DEFAULTS = {
    "data": {
        "root": "/data/disk1/shared/aiaa3201/project1_video_sr",
        "gt_sub": "val_sharp",
        "lr_sub": "val_sharp_bicubic",
        "frame_pattern": "*.png",
    },
    "weights": {
        "edsr": "/data/disk1/shared/aiaa3201/checkpoints/EDSR_baseline_x4/"
                "EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth",
        "edvr_reds": "/data/disk1/shared/aiaa3201/checkpoints/EDVR_REDS_SR_L/"
                     "EDVR_L_x4_SR_REDS_official-9f5f5039.pth",
        "edvr_vimeo": "/data/disk1/shared/aiaa3201/checkpoints/EDVR_Vimeo90K_SR_L/"
                      "EDVR_L_x4_SR_Vimeo90K_official-162b54e4.pth",
        "tecogan": "/data/disk1/shared/aiaa3201/checkpoints/TecoGAN/TecoGAN",
    },
    "out": {
        "root": None,          # required: no more implicit /tmp/... fallbacks
    },
    "runtime": {
        "device": "cuda",
        "torch_home": "/tmp/torch_home",
        "batch": 4,
    },
    "clips": {
        "only": None,          # explicit clip list; overrides limit/start
        "start": 0,
        "limit": None,
    },
    "metrics": {
        "need": ["psnr", "ssim", "lpips"],
        "tlp": {"enabled": True, "x100": False, "cutfr": 0, "crop32": False},
        "flow_warp": {"enabled": True, "stride": 2, "scale": 4},
    },
    "predictions": {
        "root": None,          # <root>/<method>/<clip>/<frame>.png  OR  <root>/<clip>
        "layout": "method_first",
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load(path: str | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        if path.endswith((".yaml", ".yml")):
            import yaml

            cfg = _merge(cfg, yaml.safe_load(txt) or {})
        else:
            cfg = _merge(cfg, json.loads(txt))
    if overrides:
        cfg = _merge(cfg, overrides)
    if not cfg["out"]["root"]:
        raise ValueError("config.out.root must be set explicitly "
                         "(the old scripts' hard-coded /tmp outputs are gone)")
    return cfg


def freeze(cfg: dict, out_dir: str, name: str = "run_config.json") -> str:
    """Write the resolved config into the run directory (created if needed)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True, ensure_ascii=False)
    return path
