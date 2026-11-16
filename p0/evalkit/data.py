# -*- coding: utf-8 -*-
"""Strict clip / frame discovery.

The old scripts paired predictions with GT by *position in a sorted glob*, which
silently produces a number whenever one side is missing or misnamed a frame.
Here everything is keyed by ``(clip, frame_id)`` and any disagreement is a hard
error::

    FrameSetError: clip 007: 99 prediction frames vs 100 GT frames
        missing=100  unexpected=['00000042.png']  (not skipped silently)

Frame ids are parsed from the basename (first run of digits), so
``00000042.png``, ``output_0043.png`` and ``im4.png`` all survive; two files that
map to the same id is an error, not a coin flip.
"""
from __future__ import annotations

import os
import re

_DIGITS = re.compile(r"(\d+)")


class FrameSetError(RuntimeError):
    pass


def frame_id(name: str) -> int:
    m = _DIGITS.findall(os.path.basename(name))
    if not m:
        raise FrameSetError("cannot parse a frame id from %r" % name)
    return int(m[-1])


def list_clips(root: str, pattern: str = "*.png") -> list:
    import glob

    if not os.path.isdir(root):
        raise FrameSetError("clip root does not exist: %s" % root)
    clips = sorted(d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d)) and glob.glob(os.path.join(root, d, pattern)))
    if not clips:
        raise FrameSetError("no clips with %r under %s" % (pattern, root))
    return clips


def index_dir(path: str, pattern: str = "*.png") -> dict:
    """{frame_id: path} for one clip directory; duplicate ids are fatal."""
    import glob

    out, dup = {}, {}
    for f in sorted(glob.glob(os.path.join(path, pattern))):
        i = frame_id(f)
        if i in out:
            dup.setdefault(i, [out[i]]).append(f)
        out[i] = f
    if dup:
        raise FrameSetError("%s: frame ids collide for %s" % (path, sorted(dup)[:5]))
    if not out:
        raise FrameSetError("no frames matching %r in %s" % (pattern, path))
    return out


def match_clip(pred_dir: str, gt_dir: str, pattern: str = "*.png",
               allow_missing_pred: bool = False) -> dict:
    """Strictly matched ``(clip, frame_id)`` lists for one clip.

    Raises unless the two sides agree on the frame id set, unless
    ``allow_missing_pred`` is set -- and even then the dropped ids are returned so
    they can be recorded instead of vanishing.
    """
    gt = index_dir(gt_dir, pattern)
    pred = index_dir(pred_dir, pattern)
    missing = sorted(set(gt) - set(pred))
    extra = sorted(set(pred) - set(gt))
    if extra:
        raise FrameSetError("%s: %d prediction frames have no GT counterpart: %s"
                            % (os.path.basename(gt_dir), len(extra),
                               [str(i) for i in extra[:5]]))
    if missing and not allow_missing_pred:
        raise FrameSetError("%s: %d prediction frames missing (%d GT frames): %s"
                            % (os.path.basename(gt_dir), len(missing), len(gt),
                               [str(i) for i in missing[:5]]))
    ids = sorted(set(gt) & set(pred))
    return {"ids": ids,
            "pred": [pred[i] for i in ids],
            "gt": [gt[i] for i in ids],
            "missing": missing, "extra": extra}


def read_rgb(path: str):
    import cv2

    a = cv2.imread(path)
    if a is None:
        raise FrameSetError("unreadable frame: %s" % path)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def read_rgb_list(paths: list) -> list:
    return [read_rgb(p) for p in paths]


def check_size(frames: list, expect=None, label: str = "pred") -> tuple:
    """All frames in a clip must share one size; returns it."""
    sizes = {tuple(f.shape[:2]) for f in frames}
    if len(sizes) != 1:
        raise FrameSetError("%s: frames differ in size: %s" % (label, sorted(sizes)))
    hw = sizes.pop()
    if expect is not None and hw != tuple(expect):
        raise FrameSetError("%s: frame size %s != expected %s" % (label, hw, tuple(expect)))
    return hw
