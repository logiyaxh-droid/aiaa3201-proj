# -*- coding: utf-8 -*-
"""Shared, configuration-driven evaluation kit for AIAA3201 Project 1 (parts 1 & 2).

Everything here is deliberately explicit about its conventions so a second person
can reproduce a number without reading three different copies of the same code.

Modules
-------
config   : load a YAML/JSON run config, apply CLI overrides, freeze it into the run dir.
data     : clip/frame discovery with strict ``clip + frame_id`` matching and hard errors.
metrics  : the single spatial-metric implementation (PSNR / SSIM / LPIPS).
temporal : the single temporal-metric implementation (tLP by the TecoGAN definition,
           plus the legacy flow-warp LPIPS kept as a separately named diagnostic).
provenance : SHA256 of weights, git/file hashes of code, environment capture.
"""
__version__ = "1.0.0"
