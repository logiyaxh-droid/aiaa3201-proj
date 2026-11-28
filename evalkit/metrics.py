# -*- coding: utf-8 -*-
"""Spatial metrics -- ONE implementation shared by Part 1 and Part 2.

Conventions are fixed here once and recorded in every result JSON:

PSNR
    computed on the Y channel of ``cv2.COLOR_RGB2YCrCb`` after clipping/rounding
    the prediction to uint8.  data range 255, no border crop, per-frame PSNR then
    arithmetic mean over frames.

SSIM
    exact replica of
    ``skimage.metrics.structural_similarity(win_size=7, gaussian_weights=False,
    use_sample_covariance=True, data_range=255)`` on the same Y channel, run on the
    GPU in float64.  Verified against skimage to 1e-13 elsewhere; a re-check is
    available through ``--selfcheck``.

LPIPS
    ``lpips`` 0.1.4, ``net="alex"``, weights = torchvision AlexNet IMAGENET1K_V1
    (``lpips.LPIPS`` downloads ``alexnet-owt-7be5be79.pth``), inputs RGB in [-1, 1],
    default per-layer weights (``version="0.1"``), no spatial averaging of the
    distance map -- ``.mean()`` over the batch, exactly as ``lpips_mean`` before.

Why this file exists
--------------------
Before P0 the same ~40 lines were copy-pasted into ``p1_part1_final.py``,
``p1_vimeo.py``, ``p2_edvr.py``, ``p2_teco_metrics.py`` and the wild-video scripts.
Any drift between copies would silently break the Part 1 <-> Part 2 comparison.
"""
from __future__ import annotations

import numpy as np
import torch

CONVENTIONS = {
    "psnr": {
        "color": "Y of cv2.COLOR_RGB2YCrCb",
        "data_range": 255.0,
        "prediction_precision": "uint8 (clipped + rounded)",
        "reduction": "per-frame PSNR, then mean over frames",
        "border_crop": "none",
    },
    "ssim": {
        "color": "Y of cv2.COLOR_RGB2YCrCb",
        "win_size": 7,
        "gaussian_weights": False,
        "use_sample_covariance": True,
        "data_range": 255.0,
        "border_crop": "none",
        "reference_impl": "skimage.metrics.structural_similarity",
    },
    "lpips": {
        "package": "lpips 0.1.4",
        "net": "alex",
        "linears": "default (v0.1)",
        "input": "RGB, [-1,1]",
        "reduction": "mean over frames",
    },
}

_K = 7
_KERNEL = None
_COV = float(_K * _K) / float(_K * _K - 1)


def _kernel(device: str) -> torch.Tensor:
    global _KERNEL
    if _KERNEL is None or _KERNEL.device.type != torch.device(device).type:
        _KERNEL = torch.ones(1, 1, _K, _K, dtype=torch.float64, device=device) / float(_K * _K)
    return _KERNEL


def luma(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB (...,3) -> Y plane, identical to ``cv2.cvtColor(RGB2YCrCb)[...,0]``."""
    import cv2

    a = np.asarray(rgb)
    if a.ndim == 3:
        return cv2.cvtColor(a, cv2.COLOR_RGB2YCrCb)[..., 0]
    return np.stack([cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb)[..., 0] for x in a])


def ssim_gpu(a_u8: np.ndarray, b_u8: np.ndarray, device: str = "cuda",
             data_range: float = 255.0) -> np.ndarray:
    """Per-frame SSIM for a stack of equal-size single-channel uint8 images."""
    k = _kernel(device)
    a = torch.from_numpy(np.ascontiguousarray(a_u8)).to(device).double().unsqueeze(1)
    b = torch.from_numpy(np.ascontiguousarray(b_u8)).to(device).double().unsqueeze(1)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    def f(x):
        return torch.nn.functional.conv2d(x, k)

    ux, uy = f(a), f(b)
    uxx = _COV * (f(a * a) - ux * ux)
    uyy = _COV * (f(b * b) - uy * uy)
    uxy = _COV * (f(a * b) - ux * uy)
    m = ((2 * ux * uy + C1) * (2 * uxy + C2)) / ((ux ** 2 + uy ** 2 + C1) * (uxx + uyy + C2))
    return m.mean(dim=(1, 2, 3)).cpu().numpy()


def psnr_frames(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-frame PSNR for a stack (N,H,W,C) or (N,H,W) of uint8 images."""
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = (d * d).mean(axis=tuple(range(1, d.ndim)))
    return 10.0 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-12))


def psnr_mean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(psnr_frames(a, b)))


class LPIPS:
    """Thin wrapper so every caller uses one torchvision-AlexNet weight file."""

    def __init__(self, device: str = "cuda", torch_home: str | None = None):
        import os

        if torch_home:
            os.environ.setdefault("TORCH_HOME", torch_home)
        import lpips as _lpips

        self.device = device
        self.net = _lpips.LPIPS(net="alex").to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _to_tensor(x: np.ndarray, device: str) -> torch.Tensor:
        return (torch.from_numpy(np.ascontiguousarray(x)).to(device).float()
                .div_(127.5).sub_(1.0).permute(0, 3, 1, 2))

    def pair(self, a: np.ndarray, b: np.ndarray, bs: int = 25) -> np.ndarray:
        """Per-pair LPIPS for equal-length stacks of uint8 RGB frames."""
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape:
            raise ValueError("LPIPS pair shape mismatch: %s vs %s" % (a.shape, b.shape))
        out = []
        for i in range(0, len(a), bs):
            with torch.no_grad():
                d = self.net(self._to_tensor(a[i:i + bs], self.device),
                             self._to_tensor(b[i:i + bs], self.device))
            out.append(d.reshape(-1).cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    def mean(self, a: np.ndarray, b: np.ndarray, bs: int = 25) -> float:
        return float(self.pair(a, b, bs=bs).mean())


def evaluate_stack(pred: list, gt: list, device: str = "cuda", lpips_net: LPIPS | None = None,
                   need=("psnr", "ssim", "lpips")) -> dict:
    """Spatial metrics for one clip, on a strictly matched (pred, gt) frame list."""
    if len(pred) != len(gt) or not pred:
        raise ValueError("evaluate_stack needs equal, non-empty lists (%d vs %d)"
                         % (len(pred), len(gt)))
    shapes = {tuple(x.shape[:2]) for x in pred} | {tuple(x.shape[:2]) for x in gt}
    if len(shapes) != 1:
        raise ValueError("frame size mismatch inside clip: %s" % sorted(shapes))
    out = {}
    pY = np.stack([luma(x) for x in pred])
    gY = np.stack([luma(x) for x in gt])
    if "psnr" in need:
        out["psnr"] = psnr_mean(pY, gY)
    if "ssim" in need:
        out["ssim"] = float(ssim_gpu(pY, gY, device=device).mean())
    if "lpips" in need:
        if lpips_net is None:
            lpips_net = LPIPS(device=device)
        out["lpips"] = lpips_net.mean(np.stack(pred), np.stack(gt))
    out["n_frames"] = len(pred)
    return out
