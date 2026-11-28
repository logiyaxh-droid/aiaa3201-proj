#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Is the course `val_sharp` the OFFICIAL REDS validation split, or its training set?

BasicSR's docs say REDS comes in two flavours:

  * REDSOfficial = original validation clips (renamed 240-269 after regrouping)
  * REDS4        = clips 000/011/015/020 of the ORIGINAL TRAINING set

The course shared disk ships `val_sharp/<000..029>` and the clip numbers say
nothing on their own -- 000 exists in both splits.  So the only honest test is
content: pull a few frames straight out of the upstream release
(`snah/REDS` on HuggingFace, `val_sharp.zip` and `train_sharp.zip`) and compare
bytes against the course copy.

The archives are 4.3 GB and 34 GB, so nothing is downloaded in full: a range
request reads the zip central directory, then only the handful of entries whose
CRC/SHA256 we need.  Bounded, resumable, and it never writes a temp mirror.

Usage:
  python reds_audit.py --frames-per-clip 2 --clips 000 011 015 020
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile

HF = "https://huggingface.co/datasets/snah/REDS/resolve/main/"
UA = {"User-Agent": "reds-audit/1.0 (byte-range verification)"}


class HTTPRangeFile(io.RawIOBase):
    """Minimal seekable read-only file backed by HTTP range requests."""

    def __init__(self, url: str, block: int = 1 << 20):
        self.url = url
        self.pos = 0
        self.block = block
        self.fetched_bytes = 0
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            self.etag = r.headers.get("ETag", "")
            self.accept_ranges = r.headers.get("Accept-Ranges", "")
        if self.accept_ranges.lower() != "bytes":
            raise RuntimeError("server does not offer byte ranges: %r" % self.accept_ranges)
        self._cache = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        elif whence == 2:
            self.pos = self.size + off
        else:
            raise ValueError(whence)
        return self.pos

    def _range(self, start: int, end: int) -> bytes:
        key = (start, end)
        if key in self._cache:
            return self._cache[key]
        req = urllib.request.Request(self.url, headers={**UA, "Range": "bytes=%d-%d" % (start, end)})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        if len(data) != end - start + 1:
            raise RuntimeError("short read %d-%d: got %d" % (start, end, len(data)))
        self.fetched_bytes += len(data)
        if len(self._cache) > 256:
            self._cache.clear()
        self._cache[key] = data
        return data

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        data = self._range(self.pos, end)
        self.pos = end + 1
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def local_hashes(root: str, clip: str, frame_ids: list) -> dict:
    out = {}
    for i in frame_ids:
        p = os.path.join(root, "val_sharp", clip, "%08d.png" % i)
        if not os.path.exists(p):
            out[i] = None
            continue
        with open(p, "rb") as f:
            out[i] = {"sha256": sha256(f.read()), "bytes": os.path.getsize(p)}
    return out


def local_hashes_from_json(path: str) -> dict:
    """Read ``{clip: {frame_id: sha256}}`` produced elsewhere (e.g. `sha256sum` on the
    server, since the GPU box cannot reach HuggingFace but this host can)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for clip, per in raw.items():
        out[clip] = {}
        for i_str, h in per.items():
            if h is None:
                out[clip][int(i_str)] = None
            else:
                out[clip][int(i_str)] = {"sha256": h if isinstance(h, str) else h["sha256"],
                                         "bytes": (h.get("bytes") if isinstance(h, dict) else None)}
    return out


def probe_archive(url: str, root: str, clips: list, frame_ids: list) -> dict:
    print("  HEAD %s" % url, flush=True)
    fh = HTTPRangeFile(url)
    print("  size=%.2f GB  etag=%s" % (fh.size / 1e9, fh.etag[:16]), flush=True)
    zf = zipfile.ZipFile(fh)
    names = zf.namelist()
    print("  %d entries, first=%s last=%s" % (len(names), names[0], names[-1]), flush=True)
    res = {"url": url, "size_bytes": fh.size, "etag": fh.etag,
           "n_entries": len(names), "sample_names": names[:3], "clips": {}}
    by_suffix = {}
    for n in names:
        if n.endswith(".png"):
            by_suffix.setdefault("/".join(n.split("/")[-2:]), n)
    for clip in clips:
        per = {}
        for i in frame_ids:
            key = "%s/%08d.png" % (clip, i)
            entry = by_suffix.get(key)
            if entry is None:
                per[i] = {"found": False}
                continue
            data = zf.read(entry)
            per[i] = {"found": True, "entry": entry, "sha256": sha256(data), "bytes": len(data)}
        res["clips"][clip] = per
        got = sum(1 for v in per.values() if v["found"])
        print("  clip %s: %d/%d entries read, %.1f MB fetched so far"
              % (clip, got, len(frame_ids), fh.fetched_bytes / 1e6), flush=True)
    res["fetched_bytes"] = fh.fetched_bytes
    return res


def compare(local: dict, upstream: dict) -> dict:
    out = {}
    for clip, per in upstream["clips"].items():
        rows = {}
        for i_str, up in per.items():
            i = int(i_str)
            lo = local.get(clip, {}).get(i)
            if not up.get("found"):
                rows[i] = {"verdict": "absent_upstream"}
            elif lo is None:
                rows[i] = {"verdict": "absent_locally"}
            else:
                same = lo["sha256"] == up["sha256"]
                rows[i] = {"verdict": "identical" if same else "DIFFERENT",
                           "local_sha256": lo["sha256"][:16],
                           "upstream_sha256": up["sha256"][:16]}
        verdicts = {r["verdict"] for r in rows.values()}
        if verdicts == {"identical"}:
            out[clip] = {"verdict": "identical", "frames": rows}
        elif "DIFFERENT" in verdicts:
            out[clip] = {"verdict": "DIFFERENT", "frames": rows}
        else:
            out[clip] = {"verdict": "inconclusive", "frames": rows}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="course data root (val_sharp lives here)")
    ap.add_argument("--local-hashes", default=None,
                    help="JSON {clip: {frame_id: sha256}} measured on the GPU box; "
                         "overrides --root")
    ap.add_argument("--clips", nargs="*", default=["000", "011", "015", "020"])
    ap.add_argument("--frames-per-clip", type=int, default=2)
    ap.add_argument("--frame-stride", type=int, default=49,
                    help="pick frames 0, stride, 2*stride ... so they are not neighbours")
    ap.add_argument("--out", default="reds_audit_report.json")
    ap.add_argument("--skip-train", action="store_true",
                    help="only test the official validation archive (default tests both)")
    a = ap.parse_args()

    frame_ids = [k * a.frame_stride for k in range(a.frames_per_clip)]
    if a.local_hashes:
        local = local_hashes_from_json(a.local_hashes)
        local = {c: local[c] for c in a.clips}
        missing = [c for c in a.clips if c not in local]
        if missing:
            raise SystemExit("clips missing from %s: %s" % (a.local_hashes, missing))
        src = a.local_hashes
    else:
        if not a.root:
            raise SystemExit("give either --root or --local-hashes")
        local = {c: local_hashes(a.root, c, frame_ids) for c in a.clips}
        src = a.root
    rep = {"course_source": src, "clips": a.clips, "frame_ids": frame_ids,
           "local": local, "archives": {}}

    targets = [("val_sharp.zip", "REDSOfficial validation split")]
    if not a.skip_train:
        targets.append(("train_sharp.zip", "REDS original TRAINING set"))

    for name, label in targets:
        print("== %s (%s) ==" % (name, label), flush=True)
        try:
            up = probe_archive(HF + name, a.root, a.clips, frame_ids)
        except Exception as e:                                   # network/shape problems
            rep["archives"][name] = {"label": label, "error": repr(e)}
            print("  ERROR %r" % (e,), flush=True)
            continue
        up["label"] = label
        up["comparison"] = compare(local, up)
        rep["archives"][name] = up
        for clip, v in up["comparison"].items():
            print("  -> clip %s: %s" % (clip, v["verdict"]), flush=True)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print("wrote %s" % a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
