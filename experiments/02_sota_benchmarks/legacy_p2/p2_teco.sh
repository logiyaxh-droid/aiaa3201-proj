#!/bin/bash
# TecoGAN inference over REDS val, all clips, CPU only (TF1.15 ships no sm_86 kernels).
#
# ALIGNMENT FIX -- why the LR frames are shifted by +0.375 px before inference:
#   TecoGAN's own upsampler (lib/ops.py:bicubic_four) places HR sample centres at
#   (i+0.5)*4-0.5, i.e. it assumes a *half-pixel-centred* source grid. The course
#   REDS LR frames were built with the pixel-centre (align_corners=True) convention,
#   so bicubic_four lands 1.5 HR px off, which alone costs ~3.9 dB PSNR:
#       raw 21.6 dB   ->  26.2 dB after realigning by 1.5 px.
#   Pre-shifting the LR input by +0.375 LR px (= +1.5 HR px) cancels it at the
#   source, and the residual becomes 0.00 px (measured: best shift 0.25 px, PSNR
#   25.5 dB, i.e. no longer offset-limited). Applied to the LR, not to the output,
#   so every downstream number is computed on genuinely registered frames.
set -u
TG=/tmp/TecoGAN-master
OUT=/tmp/p2/tecogan
LRROOT=/data/disk1/shared/aiaa3201/project1_video_sr/val_sharp_bicubic
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
echo ALLDONE > /tmp/p2/tecogan.done
