#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sequential-slice downloader that appends into the final file.

The parallel fetcher keeps one part file per slice and concatenates at the end,
so its peak footprint is ~2x the file.  On this host the remaining free space is
barely larger than the CLIP2 file, so instead download the slices in order and
append each one straight into the output file: peak footprint stays at ~1x.

Concurrency is kept (4 streams) but only the slice at the write head is
appended at a time; the next few slices are prefetched into memory-sized
temporaries, which are removed right after being appended.
"""
from __future__ import annotations

import os
import sys
import threading

import requests

URL = sys.argv[1]
OUT = sys.argv[2]
SLICE = 32 * 1024 * 1024
AHEAD = int(sys.argv[3]) if len(sys.argv) > 3 else 4
RETRY = 60

head = requests.head(URL, allow_redirects=True, timeout=60)
total = int(head.headers.get("x-linked-size") or head.headers.get("Content-Length"))
print("total", total, flush=True)

n = (total + SLICE - 1) // SLICE
tmp_dir = OUT + ".tmpdl"
os.makedirs(tmp_dir, exist_ok=True)

cond = threading.Condition()
next_to_fetch = 0
next_to_write = 0
written_bytes = 0


def fetch_worker():
    global next_to_fetch
    while True:
        with cond:
            while next_to_fetch >= n or next_to_fetch - next_to_write >= AHEAD:
                if next_to_write >= n:
                    return
                cond.wait(timeout=5)
            i = next_to_fetch
            next_to_fetch += 1
        start = i * SLICE
        end = min(total, start + SLICE) - 1
        want = end - start + 1
        pp = os.path.join(tmp_dir, "%05d" % i)
        for attempt in range(RETRY):
            have = os.path.getsize(pp) if os.path.exists(pp) else 0
            if have == want:
                break
            if have > want:
                os.remove(pp)
                have = 0
            try:
                with requests.get(URL, headers={
                        "Range": "bytes=%d-%d" % (start + have, end),
                        "Accept-Encoding": "identity"},
                        stream=True, timeout=(30, 120)) as r:
                    if r.status_code not in (200, 206):
                        raise RuntimeError("status %s" % r.status_code)
                    with open(pp, "ab" if have else "wb") as f:
                        for c in r.iter_content(1 << 18):
                            if c:
                                f.write(c)
            except Exception as e:  # noqa: BLE001
                if attempt % 8 == 0:
                    print("slice %d retry %d %s" % (i, attempt, e), flush=True)
                threading.Event().wait(2)


workers = [threading.Thread(target=fetch_worker, daemon=True) for _ in range(AHEAD)]
for w in workers:
    w.start()

with open(OUT, "wb") as out:
    while next_to_write < n:
        with cond:
            pp = os.path.join(tmp_dir, "%05d" % next_to_write)
            want = min(total, (next_to_write + 1) * SLICE) - next_to_write * SLICE
            while not (os.path.exists(pp) and os.path.getsize(pp) == want):
                cond.wait(timeout=2)
        with open(pp, "rb") as f:
            out.write(f.read())
        out.flush()
        os.remove(pp)
        written_bytes += want
        with cond:
            next_to_write += 1
            cond.notify_all()
        if next_to_write % 20 == 0:
            print("written %d/%d MB" % (written_bytes // 2**20, total // 2**20),
                  flush=True)

try:
    os.rmdir(tmp_dir)
except OSError:
    pass

size = os.path.getsize(OUT)
print("DONE size=%d expected=%d match=%s" % (size, total, size == total), flush=True)
if size != total:
    sys.exit(1)
