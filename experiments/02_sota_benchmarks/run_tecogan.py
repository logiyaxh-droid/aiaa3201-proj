#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P0 tasks 10/11/17 -- TecoGAN inference with verified outputs and a real cache key.

Replaces ``p2_teco.sh``.  Everything that was implicit there is now an argument,
and a clip is only marked done when its output has been checked frame by frame.

What changed, and why
---------------------
* **Exit code is checked.**  The old loop ran TecoGAN and then copied whatever
  ``results/tc/output_*.png`` happened to exist, touching ``.done`` either way.
  A crashed or truncated run therefore entered the averages silently.
* **The output is validated, not assumed.**  Expected frame ids come from the LR
  input; the script requires exactly that set, all readable, all at 4x the LR
  size.  Anything else is a failure, not a warning.
* **The cache key is a manifest hash.**  The old ``.done`` marker was keyed on
  nothing, so changing the weight, the shift or the frame list kept reusing the
  stale output.  ``manifest.json`` now records weight SHA256, shift, frame ids,
  LR geometry, code hash and device; ``.done`` only counts when the recorded
  hash equals the current one, and a cached directory is re-validated before use.
* **No shared scratch directory.**  Each run gets its own ``LR/<tag>`` and
  ``results/<tag>`` inside the TecoGAN tree, so two runs cannot clobber each
  other's frames (the old fixed ``LR/tc`` + ``results/tc`` did exactly that).
* **Frame ids are mapped explicitly.**  TecoGAN names its output after the LR
  file it consumed (``output_%04d`` from the 1-based position), after skipping 5
  mirrored pad frames, so the mapping is asserted against the input ids rather
  than assumed.

Usage
-----
    /tmp/srenv/bin/python scripts/run_tecogan.py \
        --lru <clip LR dir> --out <clip out dir> --shift 0.375 --mode cpu
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

TG_DEFAULT = "/tmp/TecoGAN-master"
CK_DEFAULT = "/data/disk1/shared/aiaa3201/checkpoints/TecoGAN/TecoGAN"
PY_TF1 = "/tmp/tf1env/bin/python"


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def list_ids(d: str) -> list:
    return sorted(int(os.path.basename(f)[:-4]) for f in glob.glob(os.path.join(d, "*.png")))


def write_shifted_lr(src: str, dst: str, shift: float, limit: int = 0) -> list:
    """Copy (optionally sub-pixel shifted) LR frames under TecoGAN's 0001.. naming."""
    import cv2
    import numpy as np

    ids = list_ids(src)
    if limit:
        ids = ids[:limit]
    if not ids:
        raise SystemExit("no *.png LR frames in %s" % src)
    os.makedirs(dst, exist_ok=True)
    for old in os.listdir(dst):
        os.remove(os.path.join(dst, old))
    M = np.float32([[1, 0, shift], [0, 1, shift]])
    for n, i in enumerate(ids, 1):
        a = cv2.imread(os.path.join(src, "%08d.png" % i))
        if a is None:
            raise SystemExit("unreadable LR frame %08d in %s" % (i, src))
        if shift:
            # same operation and border mode as the pre-P0 p2_teco.sh
            a = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(os.path.join(dst, "%04d.png" % n), a)
    return ids


def manifest(args, ids: list, lr_hw) -> dict:
    ck = args.checkpoint
    w = {os.path.basename(ck + s): sha256_file(ck + s)
         for s in ("", ".index", ".data-00000-of-00001", ".meta")
         if os.path.exists(ck + s)}
    code = {}
    for f in ("main.py", "lib/ops.py", "lib/frvsr.py", "lib/dataloader.py"):
        p = os.path.join(args.tecogan, f)
        if os.path.exists(p):
            code[f] = sha256_file(p)
    return {"weights": w, "shift": float(args.shift), "frame_ids": ids,
            "lr_hw": list(int(x) for x in lr_hw), "num_resblock": args.num_resblock,
            "mode": args.mode, "tecogan_code": code,
            "driver_sha256": sha256_file(os.path.abspath(__file__))}


def manifest_hash(m: dict) -> str:
    blob = json.dumps({k: m[k] for k in ("weights", "shift", "frame_ids", "lr_hw",
                                         "num_resblock", "mode", "tecogan_code")},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def validate(out_dir: str, expect_ids: list, lr_hw, expect_sha: dict | None = None) -> dict:
    """Hard check of a finished run; raises on any defect.

    ``expect_sha`` (frame id -> sha256, recorded at write time) catches a frame
    that was overwritten by something the same size -- count and pixel size alone
    cannot see that, and a silent wrong frame is exactly the failure mode the
    old ``.done`` marker allowed.
    """
    import cv2

    got = list_ids(out_dir)
    problems = []
    if got != expect_ids:
        missing = sorted(set(expect_ids) - set(got))
        extra = sorted(set(got) - set(expect_ids))
        problems.append("frame id set mismatch: missing=%s extra=%s"
                        % (missing[:5], extra[:5]))
    sizes = {}
    unreadable = []
    for f in glob.glob(os.path.join(out_dir, "*.png")):
        a = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if a is None:
            unreadable.append(os.path.basename(f))
            continue
        sizes[tuple(a.shape[:2])] = sizes.get(tuple(a.shape[:2]), 0) + 1
    if unreadable:
        problems.append("unreadable frames: %s" % unreadable[:5])
    expect_hw = (int(lr_hw[0]) * 4, int(lr_hw[1]) * 4)
    bad = [s for s in sizes if s != expect_hw]
    if bad:
        problems.append("frame size %s, expected %s" % (sorted(sizes), expect_hw))
    if expect_sha:
        badsha = []
        for i in expect_ids:
            fp = os.path.join(out_dir, "%08d.png" % i)
            if not os.path.exists(fp):
                continue
            if sha256_file(fp) != expect_sha.get(str(i), expect_sha.get(i)):
                badsha.append(i)
        if badsha:
            problems.append("frame content changed since it was written: %s" % badsha[:5])
    report = {"n_frames": len(got), "sizes": {str(k): v for k, v in sizes.items()},
              "id_range": [got[0], got[-1]] if got else None,
              "content_verified": bool(expect_sha), "problems": problems}
    if problems:
        raise RuntimeError("output validation failed: " + " | ".join(problems))
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lru", required=True, help="clip LR directory (*.png, 08d names)")
    ap.add_argument("--out", required=True, help="clip output directory")
    ap.add_argument("--shift", type=float, default=0.0, help="LR pre-shift in LR px")
    ap.add_argument("--limit", type=int, default=0, help="frames to use, 0 = all")
    ap.add_argument("--mode", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--tecogan", default=TG_DEFAULT)
    ap.add_argument("--checkpoint", default=CK_DEFAULT)
    ap.add_argument("--num_resblock", type=int, default=16)
    ap.add_argument("--tag", default=None, help="scratch tag (default: derived)")
    ap.add_argument("--force", action="store_true", help="ignore a matching .done")
    a = ap.parse_args()

    import cv2

    os.makedirs(a.out, exist_ok=True)
    probe = sorted(glob.glob(os.path.join(a.lru, "*.png")))
    if not probe:
        raise SystemExit("no *.png LR frames in %s" % a.lru)
    first = cv2.imread(probe[0])
    if first is None:
        raise SystemExit("cannot read %s" % probe[0])
    lr_hw = first.shape[:2]

    probe_ids = list_ids(a.lru)[: a.limit or None]
    mh = manifest_hash(manifest(a, probe_ids, lr_hw))
    done, man_path = os.path.join(a.out, ".done"), os.path.join(a.out, "manifest.json")

    if os.path.exists(done) and os.path.exists(man_path) and not a.force:
        old = json.load(open(man_path, encoding="utf-8"))
        if old.get("manifest_hash") == mh:
            try:
                rep = validate(a.out, probe_ids, lr_hw, old.get("output_sha256"))
                print("cache hit %s (%d frames, manifest %s)"
                      % (a.out, rep["n_frames"], mh[:12]), flush=True)
                return 0
            except RuntimeError as e:
                print("stale cache discarded: %s" % e, flush=True)
        else:
            print("cache key changed (%s -> %s), re-running"
                  % (str(old.get("manifest_hash"))[:12], mh[:12]), flush=True)

    for p in glob.glob(os.path.join(a.out, "*.png")) + [done, os.path.join(a.out, ".failed")]:
        if os.path.exists(p):
            os.remove(p)

    tag = a.tag or ("s%0.3f_%s" % (a.shift, os.path.basename(os.path.abspath(a.out))))
    tag = "run_" + "".join(c if (c.isalnum() or c in "-_.") else "_" for c in tag)
    lr_tmp = os.path.join(a.tecogan, "LR", tag)
    res_root = os.path.join(a.tecogan, "results")
    res_tmp = os.path.join(res_root, tag)

    ids = write_shifted_lr(a.lru, lr_tmp, a.shift, a.limit)
    if os.path.isdir(res_tmp):
        shutil.rmtree(res_tmp)
    os.makedirs(res_tmp, exist_ok=True)

    log = os.path.join(a.out, "run.log")
    cmd = [PY_TF1, "main.py", "--cudaID", "0" if a.mode == "gpu" else "-1",
           "--output_dir", res_root + "/", "--summary_dir", res_tmp + "/",
           "--mode", "inference", "--input_dir_LR", lr_tmp,
           "--output_pre", tag, "--num_resblock", str(a.num_resblock),
           "--checkpoint", a.checkpoint, "--output_ext", "png"]
    print("running: %s" % " ".join(cmd), flush=True)
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, cwd=a.tecogan, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        with open(os.path.join(a.out, ".failed"), "w", encoding="utf-8") as f:
            f.write("inference exit code %d\n" % rc)
        print("FAILED: TecoGAN exited %d, log kept at %s" % (rc, log), flush=True)
        return rc

    # TecoGAN names each output after the LR file it consumed, and skips the 5
    # mirrored pad frames, so output_000k belongs to the k-th real input frame.
    by_idx = {}
    for f in os.listdir(res_tmp):
        if f.startswith("output_") and f.endswith(".png"):
            by_idx[int(f[len("output_"):-4])] = f
    missing = []
    for pos, fid in enumerate(ids, 1):
        cand = by_idx.get(pos)
        if cand is None:
            missing.append(fid)
            continue
        shutil.copyfile(os.path.join(res_tmp, cand), os.path.join(a.out, "%08d.png" % fid))
    if missing:
        with open(os.path.join(a.out, ".failed"), "w", encoding="utf-8") as f:
            f.write("missing outputs for frame ids %s\n" % missing[:20])
        print("FAILED: %d/%d frames were not produced (%s)"
              % (len(missing), len(ids), missing[:5]), flush=True)
        return 2

    out_sha = {str(i): sha256_file(os.path.join(a.out, "%08d.png" % i)) for i in ids}
    try:
        rep = validate(a.out, ids, lr_hw)
    except RuntimeError as e:
        with open(os.path.join(a.out, ".failed"), "w", encoding="utf-8") as f:
            f.write(str(e) + "\n")
        print("FAILED: %s" % e, flush=True)
        return 3

    man = manifest(a, ids, lr_hw)
    man["manifest_hash"] = mh
    man["validation"] = rep
    man["output_sha256"] = out_sha
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)
    with open(done, "w", encoding="utf-8") as f:
        f.write(mh + "\n")
    shutil.rmtree(lr_tmp, ignore_errors=True)
    shutil.rmtree(res_tmp, ignore_errors=True)
    print("OK %s: %d frames, shift=%.3f, manifest %s"
          % (a.out, rep["n_frames"], a.shift, mh[:12]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
