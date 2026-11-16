#!/bin/bash
# TecoGAN inference over the WILD clips, CPU only, with the +0.375 LR pre-shift.
#
# The shift is the same alignment fix used for REDS: TecoGAN's bicubic_four assumes a
# half-pixel-centred source grid while our LR is pixel-centred, which is a constant
# 1.5 HR px (= 0.375 LR px) offset and costs ~3.9 dB if left uncorrected.
#
# usage: bash p2_teco_wild.sh GT_ROOT LR_ROOT OUT
set -u
GT=${1:-/tmp/wild/gt}
LRROOT=${2:-/tmp/wild/lr}
OUT=${3:-/tmp/wild/part2}
TG=/tmp/TecoGAN-master
CK=/data/disk1/shared/aiaa3201/checkpoints/TecoGAN/TecoGAN
SHIFT=0.375
mkdir -p "$OUT"
cd "$TG"

for clip in $(ls "$LRROOT" | sort); do
  d="$OUT/$clip"
  if [ -f "$d/.done" ]; then echo "skip $clip"; continue; fi
  rm -rf "$d"; mkdir -p "$d"
  rm -rf LR/tc; mkdir -p LR/tc
  /tmp/srenv/bin/python - "$LRROOT/$clip" "$TG/LR/tc" "$SHIFT" <<'PY'
import sys, os, glob
import cv2, numpy as np
src, dst, sh = sys.argv[1], sys.argv[2], float(sys.argv[3])
M = np.float32([[1, 0, sh], [0, 1, sh]])
files = sorted(glob.glob(os.path.join(src, "*.png")))
for i, f in enumerate(files, 1):
    a = cv2.imread(f)
    b = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]),
                       flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(os.path.join(dst, "%04d.png" % i), b)
PY
  rm -rf results/tc
  /tmp/tf1env/bin/python main.py --cudaID -1 \
    --output_dir "$TG/results/" --summary_dir "$TG/results/log/" \
    --mode inference --input_dir_LR "$TG/LR/tc" --output_pre tc \
    --num_resblock 16 --checkpoint "$CK" --output_ext png > "$d/run.log" 2>&1
  n=0
  for f in $(ls "$TG"/results/tc/output_*.png 2>/dev/null | sort); do
    b=$(basename "$f" .png); num=${b#output_}
    idx=$((10#$num - 1))
    cp "$f" "$(printf '%s/%08d.png' "$d" $idx)"
    n=$((n+1))
  done
  echo "clip $clip frames=$n" && touch "$d/.done"
done
echo ALLDONE > "$OUT/tecogan.done"
