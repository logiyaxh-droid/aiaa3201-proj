"""Direct numerical identification of bicubic_four's sample mapping.

Two probes that need no interpretation of the measured constants:

  impulse  put a single unit pixel at input (i,j) and see which output pixel
           peaks.  If out[k] = orig[k/4] then input i maps to output 4i.
  ramp     out[j] should equal the content at input coordinate j/4.

Printed so the numbers can be pasted into the report verbatim.
"""
import os
import sys

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/tmp/TecoGAN-master")
import tensorflow as tf                      # noqa: E402
from lib.ops import bicubic_four             # noqa: E402

H, W = 12, 20
ph = tf.placeholder(tf.float32, [1, None, None, 3])
out = bicubic_four(ph)

with tf.Session() as s:
    x = np.zeros((H, W, 3), np.float32)
    x[4, 10] = 1.0
    y = s.run(out, {ph: x[None]})[0]
    yh = y.mean(axis=2)
    peak = np.unravel_index(np.argmax(yh), yh.shape)
    print("impulse: input (row=4, col=10) -> output peak", peak,
          "value %.4f" % yh[peak], " out shape", yh.shape)
    print("         row profile 14..20:", np.round(yh[14:21, 40], 3))
    print("         col profile 36..45:", np.round(yh[16, 36:46], 3))

    ramp = np.tile(np.arange(W, dtype=np.float32)[None, :], (H, 1))
    ramp = np.repeat((ramp / (W - 1))[..., None], 3, axis=2)
    y2 = s.run(out, {ph: ramp[None]})[0]
    prof = y2.mean(axis=2).mean(axis=0)
    print("ramp: out[j] for j=0..7 :", np.round(prof[:8], 4))
    print("      implied input coord :", np.round(prof[:8] * (W - 1), 3))
    print("      j/4 would be        :", np.round(np.arange(8) / 4.0, 3))
    print("      out[j] for j=76..79 :", np.round(prof[76:80], 4),
          " implied:", np.round(prof[76:80] * (W - 1), 3),
          " j/4:", np.round(np.arange(76, 80) / 4.0, 3))
