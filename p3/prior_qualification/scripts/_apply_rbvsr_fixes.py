# -*- coding: utf-8 -*-
"""Apply the three verified fixes to rbvsr_net.py.

Anchors are literal and must match exactly once; the script aborts instead of
guessing.  Run with --check to only report whether the anchors are present.
"""
from __future__ import annotations

import sys

SRC = "/data/disk1/home/ghe270/p3/prior_qualification/scripts/rbvsr_net.py"

EDITS = [
    # bug 2: grid + flow, not flow.flip(1)
    ("    grid = grid + flow.flip(1)\n", "    grid = grid + flow\n"),
    # bug 3 (not a numeric bug, but align the coarsest level with upstream)
    ("        flow = ref[0].new_zeros(ref[0].size(0), 2,\n"
     "                                int(math.floor(ref[0].size(2) / 2.0)),\n"
     "                                int(math.floor(ref[0].size(3) / 2.0)))\n"
     "        for level in range(len(ref)):\n"
     "            up = F.interpolate(flow, scale_factor=2, mode=\"bilinear\",\n"
     "                               align_corners=True) * 2.0\n"
     "            if up.size(2) != ref[level].size(2):\n"
     "                up = F.pad(up, [0, 0, 0, 1], mode=\"replicate\")\n"
     "            if up.size(3) != ref[level].size(3):\n"
     "                up = F.pad(up, [0, 1, 0, 0], mode=\"replicate\")\n"
     "            flow = self.basic_module[level](torch.cat([\n"
     "                ref[level], _warp(supp[level], up), up], 1)) + up\n",
     "        flow = ref[0].new_zeros(ref[0].size(0), 2,\n"
     "                                ref[0].size(2), ref[0].size(3))\n"
     "        for level in range(len(ref)):\n"
     "            if level == 0:\n"
     "                up = flow\n"
     "            else:\n"
     "                up = F.interpolate(flow, scale_factor=2, mode=\"bilinear\",\n"
     "                                   align_corners=True) * 2.0\n"
     "            flow = self.basic_module[level](torch.cat([\n"
     "                ref[level],\n"
     "                _warp(supp[level], up, padding_mode=\"border\"), up], 1)\n"
     "                ) + up\n"),
    # bug 4: _warp takes a padding_mode, propagation uses the upstream default
    ("def _warp(x, flow):\n"
     '    """Backward warp: sample x at (grid + flow)."""\n',
     "def _warp(x, flow, padding_mode=\"zeros\"):\n"
     '    """Backward warp: sample x at (grid + flow).\n'
     "\n"
     "    `padding_mode` follows mmedit.models.common.flow_warp: the default is\n"
     '    "zeros" for the feature-propagation branches, while SPyNet internally\n'
     '    warps the supporting *image* with "border".\n'
     '    """\n'),
    ('    return F.grid_sample(x, grid.permute(0, 2, 3, 1), align_corners=True,\n'
     '                         mode="bilinear", padding_mode="border")\n',
     '    return F.grid_sample(x, grid.permute(0, 2, 3, 1), align_corners=True,\n'
     '                         mode="bilinear", padding_mode=padding_mode)\n'),
    # bug 1: leaky ReLU after both pixel-shuffle stages
    ("            x = self.basicvsr.upsample1(x)\n"
     "            x = self.basicvsr.upsample2(x)\n",
     "            x = F.leaky_relu(self.basicvsr.upsample1(x),\n"
     "                             negative_slope=0.1, inplace=True)\n"
     "            x = F.leaky_relu(self.basicvsr.upsample2(x),\n"
     "                             negative_slope=0.1, inplace=True)\n"),
]


def main():
    src = open(SRC, encoding="utf-8").read()
    for old, new in EDITS:
        n = src.count(old)
        if n != 1:
            raise SystemExit("anchor count %d (expected 1): %r" % (n, old[:90]))
        src = src.replace(old, new)
    if "--check" in sys.argv:
        print("all %d anchors present exactly once" % len(EDITS))
        return
    open(SRC, "w", encoding="utf-8").write(src)
    print("applied %d edits to %s" % (len(EDITS), SRC))


if __name__ == "__main__":
    main()
