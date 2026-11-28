# -*- coding: utf-8 -*-
"""Shared plumbing for the P3 perceptual-quality metrics.

Everything that could change what a number means is fixed here once and then
recorded in every result JSON: frame pairing by (clip, frame_id), the colour
order and value range handed to each metric, the exact weight files (SHA256, not
just file names), the DOVER sampling scheme, and the runtime.

Design notes that matter for comparability
------------------------------------------
* Frames are read from PNG.  We never decode the deliverable H.264 videos: video
  compression changes both texture and temporal statistics, which is exactly what
  DISTS / MUSIQ / DOVER are being asked to measure.
* DOVER's official loader is `decord.VideoReader(mp4)`.  Rather than re-implement
  its sampling, we substitute a VideoReader-compatible reader over a PNG
  directory, so `ViewDecompositionDataset`, `UnifiedFrameSampler`,
  `get_spatial_fragments` and the ImageNet mean/std normalisation are all the
  official code paths, unchanged.
* DOVER is NOT deterministic in eval mode: `UnifiedFrameSampler.get_frame_indices`
  draws `np.random.randint` offsets and `get_spatial_fragments` draws
  `torch.randint` crop offsets inside each fragment.  Both are seeded explicitly
  here, and the seed is written into the result.  Repeating with the same seed
  must reproduce bit-identical scores; that is one of the self-checks.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time

os.environ.setdefault("TORCH_HOME",
                      "/data/disk1/home/ghe270/p3/quality_metrics/weights/torch_home")
# Do not let a stray HF call hang a 30-clip run behind a blocked host.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch

ROOT = "/data/disk1/home/ghe270/p3/quality_metrics"
SCRIPTS = os.path.join(ROOT, "scripts")
WEIGHTS = os.path.join(ROOT, "weights")
DOVER_SRC = os.path.join(ROOT, "vendor", "DOVER")

GT_ROOT = "/data/disk1/shared/aiaa3201/project1_video_sr/val_sharp"
LR_ROOT = "/data/disk1/shared/aiaa3201/project1_video_sr/val_sharp_bicubic"
TECO_FRAMES = "/tmp/p2/tecogan"
EDVR_FRAMES = "/tmp/p2/edvr_frame"
BICUBIC_PRED = "/data/disk1/home/ghe270/p0/out/pred/bicubic"

CK_EDSR = ("/data/disk1/shared/aiaa3201/checkpoints/EDSR_baseline_x4/"
           "EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth")
CK_EDVR = ("/data/disk1/shared/aiaa3201/checkpoints/EDVR_REDS_SR_L/"
           "EDVR_L_x4_SR_REDS_official-9f5f5039.pth")

WEIGHT_FILES = {
    "dists": os.path.join(WEIGHTS, "DISTS_weights-f5e65c96.pth"),
    "dists_official": os.path.join(WEIGHTS, "official_DISTS_weights.pt"),
    "musiq_koniq10k": os.path.join(WEIGHTS, "musiq_koniq_ckpt-e95806b9.pth"),
    "musiq_paq2piq": os.path.join(WEIGHTS, "musiq_paq2piq_ckpt-364c0c84.pth"),
    "musiq_spaq": os.path.join(WEIGHTS, "musiq_spaq_ckpt-358bb6af.pth"),
    "dover": os.path.join(WEIGHTS, "DOVER.pth"),
    "vgg16": os.path.join(WEIGHTS, "torch_home", "hub", "checkpoints",
                          "vgg16-397923af.pth"),
    "convnext_tiny_1k": os.path.join(WEIGHTS, "torch_home", "hub", "checkpoints",
                                     "convnext_tiny_1k_224_ema.pth"),
    "edsr": CK_EDSR,
    "edvr_reds": CK_EDVR,
}

DOVER_SEED = 20261119     # fixed sampling seed; recorded in every DOVER result


# ------------------------------------------------------------------ provenance

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_WEIGHT_HASH_CACHE: dict = {}


def weight_sha256(key: str) -> str:
    if key not in _WEIGHT_HASH_CACHE:
        _WEIGHT_HASH_CACHE[key] = sha256(WEIGHT_FILES[key])
    return _WEIGHT_HASH_CACHE[key]


def runtime_info() -> dict:
    import platform

    import cv2

    info = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cudnn"] = torch.backends.cudnn.version()
    for mod in ("pyiqa", "timm", "einops"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception as exc:                      # never fail a run for this
            info[mod] = "unavailable: %s" % exc
    return info


def environment_json() -> dict:
    """Everything needed to rebuild the numbers: versions + weight hashes."""
    env = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime": runtime_info(),
        "weights": {},
        "protocol": json.load(open(os.path.join(ROOT, "configs", "protocol.json"),
                                   encoding="utf-8")),
    }
    for key, path in WEIGHT_FILES.items():
        if not os.path.exists(path):
            env["weights"][key] = {"path": path, "sha256": None, "missing": True}
            continue
        env["weights"][key] = {
            "path": path,
            "bytes": os.path.getsize(path),
            "sha256": weight_sha256(key),
        }
    return env


# ------------------------------------------------------------------ frame io

def clip_frame_ids(clip_dir: str) -> list:
    """Sorted frame ids for one clip directory. Duplicate ids are fatal."""
    import re

    ids, owner = [], {}
    for name in sorted(os.listdir(clip_dir)):
        if not name.lower().endswith(".png"):
            continue
        digits = re.findall(r"(\d+)", name)
        if not digits:
            raise RuntimeError("cannot parse a frame id from %r" % name)
        fid = int(digits[-1])
        if fid in owner:
            raise RuntimeError("%s: frames %r and %r share id %d"
                               % (clip_dir, owner[fid], name, fid))
        owner[fid] = name
        ids.append(fid)
    if not ids:
        raise RuntimeError("no PNG frames in %s" % clip_dir)
    return ids


def read_rgb(path: str) -> np.ndarray:
    import cv2

    a = cv2.imread(path)
    if a is None:
        raise RuntimeError("unreadable frame: %s" % path)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def read_clip(clip_dir: str, ids: list | None = None) -> list:
    """RGB uint8 frames for a clip, ordered by frame id (never by glob order)."""
    if ids is None:
        ids = clip_frame_ids(clip_dir)
    return [read_rgb(os.path.join(clip_dir, "%08d.png" % i)) for i in ids]


# ------------------------------------------------------------------ metrics

def to_nchw01(frames_u8: list, device: str = "cuda") -> torch.Tensor:
    """uint8 RGB HWC list -> float32 NCHW in [0,1] (the range DISTS/MUSIQ expect)."""
    a = np.stack(frames_u8)
    return (torch.from_numpy(a).to(device).float().div_(255.0).permute(0, 3, 1, 2))


class DistsMetric:
    """pyiqa DISTS, RGB [0,1], full frame, no resize, no crop. Lower is better."""

    name = "dists"
    lower_better = True

    def __init__(self, device: str = "cuda", weights_key: str = "dists"):
        from pyiqa.archs.dists_arch import DISTS

        self.device = device
        self.net = DISTS(pretrained_model_path=WEIGHT_FILES[weights_key]).to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    def pairing(self) -> dict:
        return {
            "impl": "pyiqa 0.1.13 pyiqa.archs.dists_arch.DISTS",
            "weight_file": os.path.basename(WEIGHT_FILES["dists"]),
            "weights_sha256": weight_sha256("dists"),
            "input": "RGB uint8 -> float32 [0,1], NCHW",
            "resize": "none (native 1280x720)",
            "crop": "none",
            "per_frame": True,
            "lower_better": True,
        }

    def per_frame(self, pred: list, gt: list, bs: int = 8) -> np.ndarray:
        if len(pred) != len(gt):
            raise RuntimeError("DISTS needs equal frame lists (%d vs %d)"
                               % (len(pred), len(gt)))
        out = []
        for i in range(0, len(pred), bs):
            x = to_nchw01(pred[i:i + bs], self.device)
            y = to_nchw01(gt[i:i + bs], self.device)
            if x.shape != y.shape:
                raise RuntimeError("DISTS shape mismatch %s vs %s"
                                   % (tuple(x.shape), tuple(y.shape)))
            with torch.no_grad():
                d = self.net(x, y)
            out.append(d.reshape(-1).cpu().numpy().astype(np.float64))
        return np.concatenate(out)


class MusiqMetric:
    """pyiqa MUSIQ, RGB [0,1], the library's own multi-scale preprocessing.

    Scores are MOS-like in ~[0,100]; higher is better.  No GT is involved, so the
    GT frames are scored separately as a domain reference -- never as "100".
    """

    name = "musiq"
    lower_better = False

    def __init__(self, device: str = "cuda", variant: str = "koniq10k"):
        from pyiqa.archs.musiq_arch import MUSIQ

        self.variant = variant
        self.device = device
        self.net = MUSIQ(pretrained_model_path=WEIGHT_FILES["musiq_" + variant])
        self.net = self.net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    def pairing(self) -> dict:
        return {
            "impl": "pyiqa 0.1.13 pyiqa.archs.musiq_arch.MUSIQ",
            "variant": self.variant,
            "weight_file": os.path.basename(WEIGHT_FILES["musiq_" + self.variant]),
            "weights_sha256": weight_sha256("musiq_" + self.variant),
            "input": "RGB uint8 -> float32 [0,1], NCHW; library multi-scale patches",
            "longer_side_lengths": [224, 384],
            "resize": "library default (multi-scale, aspect preserved)",
            "crop": "none",
            "per_frame": True,
            "needs_gt": False,
            "lower_better": False,
        }

    def per_frame(self, frames: list, bs: int = 4) -> np.ndarray:
        out = []
        for i in range(0, len(frames), bs):
            x = to_nchw01(frames[i:i + bs], self.device)
            with torch.no_grad():
                s = self.net(x)
            out.append(s.reshape(-1).cpu().numpy().astype(np.float64))
        return np.concatenate(out)


# --------------------------------------------------------------- DOVER

def _install_png_video_reader():
    """Make DOVER's `VideoReader(path)` accept a directory of PNG frames.

    Returns the module whose `VideoReader` was patched.  Frames come back as
    uint8 HWC torch tensors, which is what decord's torch bridge yields, so
    `spatial_temporal_view_decomposition` below is untouched.
    """
    if DOVER_SRC not in sys.path:
        sys.path.insert(0, DOVER_SRC)
    import dover.datasets.dover_datasets as dd

    class PngDirectoryVideoReader:
        def __init__(self, path, *args, **kwargs):
            self.path = path
            self.ids = clip_frame_ids(path)
            self._cache = {}

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            if isinstance(idx, (list, tuple, np.ndarray)):
                return [self[int(i)] for i in idx]
            idx = int(idx)
            if idx < 0:
                idx += len(self.ids)
            if idx not in self._cache:
                img = read_rgb(os.path.join(self.path, "%08d.png" % self.ids[idx]))
                self._cache[idx] = torch.from_numpy(np.ascontiguousarray(img))
            return self._cache[idx]

    dd.VideoReader = PngDirectoryVideoReader
    return dd


class DoverMetric:
    """Official DOVER (full model), whole-clip, one value per branch.

    The sampling follows the official `dover.yml` defaults: technical
    clip_len=32 / frame_interval=2 / num_clips=3 over 7x7 fragments of 32x32;
    aesthetic clip_len=32 / frame_interval=2 / t_frag=32 / num_clips=1 resized to
    224x224.  Because both branches consume the SAME sampled frame indices, a
    100-frame REDS clip gives every method the same input length.
    """

    name = "dover"
    lower_better = False

    # verbatim from DOVER-master/dover.yml (val-* entries all agree)
    SAMPLE_TYPES = {
        "technical": {
            "fragments_h": 7, "fragments_w": 7,
            "fsize_h": 32, "fsize_w": 32, "aligned": 32,
            "clip_len": 32, "frame_interval": 2, "num_clips": 3,
        },
        "aesthetic": {
            "size_h": 224, "size_w": 224,
            "clip_len": 32, "frame_interval": 2, "t_frag": 32, "num_clips": 1,
        },
    }
    MEAN = (123.675, 116.28, 103.53)
    STD = (58.395, 57.12, 57.375)

    def __init__(self, device: str = "cuda", seed: int = DOVER_SEED):
        dd = _install_png_video_reader()
        from dover.models import DOVER as DoverNet

        self.dd = dd
        self.device = device
        self.seed = seed

        self.net = DoverNet(
            backbone={
                "technical": {"type": "swin_tiny_grpb", "checkpoint": True,
                              "pretrained": None},
                "aesthetic": {"type": "conv_tiny"},
            },
            backbone_preserve_keys="technical,aesthetic",
            divide_head=True,
            vqa_head={"in_channels": 768, "hidden_channels": 64},
        )
        state = torch.load(WEIGHT_FILES["dover"], map_location="cpu",
                           weights_only=False)
        self.net.load_state_dict(state, strict=True)
        self.net = self.net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.mean = torch.FloatTensor(self.MEAN)
        self.std = torch.FloatTensor(self.STD)

    def pairing(self) -> dict:
        return {
            "impl": "official DOVER-master (VQAssessment/DOVER), full model",
            "weight_file": os.path.basename(WEIGHT_FILES["dover"]),
            "weights_sha256": weight_sha256("dover"),
            "config": "dover.yml defaults, verbatim",
            "sample_types": self.SAMPLE_TYPES,
            "normalisation": {"mean": self.MEAN, "std": self.STD,
                              "applied_to": "uint8 RGB frames"},
            "frame_source": "PNG directory via a VideoReader-compatible reader",
            "seed": self.seed,
            "one_value_per_clip": True,
            "branches_saved_separately": ["technical", "aesthetic"],
            "fused": "mean of the two branch scores; recorded, never substituted",
            "lower_better": False,
        }

    def _samplers(self):
        samplers = {}
        for stype, sopt in self.SAMPLE_TYPES.items():
            if "t_frag" not in sopt:
                samplers[stype] = self.dd.UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"])
            else:
                samplers[stype] = self.dd.UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"], sopt["t_frag"],
                    sopt["frame_interval"], sopt["num_clips"])
        return samplers

    def score_clip(self, clip_dir: str) -> dict:
        """-> {'technical': float, 'aesthetic': float, 'fused': float, ...}."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        data, frame_inds = self.dd.spatial_temporal_view_decomposition(
            clip_dir, self.SAMPLE_TYPES, self._samplers(), is_train=False)

        video = {}
        num_clips = {}
        for key, v in data.items():
            v = ((v.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)
            c, t, h, w = v.shape
            n = self.SAMPLE_TYPES[key]["num_clips"]
            num_clips[key] = n
            video[key] = (v.reshape(1, c, n, t // n, h, w)
                          .permute(0, 2, 1, 3, 4, 5)
                          .reshape(n, c, t // n, h, w)
                          .to(self.device))

        with torch.no_grad():
            labels = self.net(video, reduce_scores=False)
        scores = [float(np.mean(l.cpu().numpy())) for l in labels]

        out = {"technical": scores[0], "aesthetic": scores[1]}
        out["fused"] = float(np.mean(scores))
        out["branch_order"] = ["technical", "aesthetic"]
        out["frame_inds"] = {k: [int(x) for x in v]
                             for k, v in frame_inds.items()}
        out["n_source_frames"] = len(clip_frame_ids(clip_dir))
        return out


# ------------------------------------------------------------------ stats

def summarize(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}


def write_json(path: str, obj) -> None:
    """Atomic write: a killed run never leaves a half-written result file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
