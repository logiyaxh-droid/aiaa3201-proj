# -*- coding: utf-8 -*-
"""Memory-lean SUPIR model creation for a 31 GB / shared-GPU host.

The stock create_SUPIR_model() does:
    model = instantiate(...).cpu()          # fp32 skeleton, ~5 GB
    model.load_state_dict(SDXL)             # +6 GB fp32, transient copy
    model.load_state_dict(SUPIR_F)          # +2.5 GB
and only afterwards the caller does .half().to("cuda"), so the peak resident
set is skeleton + both fp32 state dicts at once (~20 GB).  On this host only
~16 GB is available and the process is killed by the OOM reaper.

This variant casts every tensor to fp16 **as it is read** and frees each state
dict before the next one loads, so the peak stays near skeleton + one file.
"""
from __future__ import annotations

import gc
import os

import torch


def _load_fp16(path, device="cpu"):
    from safetensors.torch import load_file as sf_load

    if path.lower().endswith(".safetensors"):
        sd = sf_load(path, device=device)
    else:
        from SUPIR.util import get_state_dict
        sd = get_state_dict(torch.load(path, map_location=device))
    out = {}
    for k, v in sd.items():
        out[k] = v.half() if v.is_floating_point() else v
        del v
    del sd
    gc.collect()
    return out


def create_supir_lean(config_path, SUPIR_sign="F"):
    from omegaconf import OmegaConf
    from sgm.util import instantiate_from_config

    # Build the skeleton directly in fp16.  instantiate_from_config() otherwise
    # allocates every parameter in fp32 first: the SDXL UNet alone is ~2.6B
    # params, i.e. ~10 GB of RAM, which is what the OOM reaper was killing.
    torch.set_default_dtype(torch.float16)
    try:
        config = OmegaConf.load(config_path)
        model = instantiate_from_config(config.model)
    finally:
        torch.set_default_dtype(torch.float32)
    print("skeleton built (fp16)", flush=True)

    for label, path in (("SDXL", config.SDXL_CKPT),
                        ("SUPIR_" + SUPIR_sign,
                         config.SUPIR_CKPT_F if SUPIR_sign == "F"
                         else config.SUPIR_CKPT_Q)):
        if not path:
            continue
        sd = _load_fp16(path)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("loaded %s tensors=%d missing=%d unexpected=%d"
              % (label, len(sd), len(missing), len(unexpected)), flush=True)
        del sd
        gc.collect()

    return model
