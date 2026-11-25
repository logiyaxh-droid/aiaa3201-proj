#!/usr/bin/env bash
# Shared-A6000 queue: wait until enough VRAM is free, then run the command.
# Usage: bash wait_gpu.sh "cd /path && python train.py ..."
#        NEED_MB=14000 bash wait_gpu.sh "..."
NEED_MB=${NEED_MB:-14000}
SLEEP=${SLEEP:-120}
while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    if [ -n "$FREE" ] && [ "$FREE" -ge "$NEED_MB" ]; then
        echo "[$(date '+%F %T')] free VRAM=${FREE} MiB >= ${NEED_MB} MiB, launching"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 eval "$1"
        break
    fi
    echo "[$(date '+%F %T')] free VRAM=${FREE:-?} MiB < ${NEED_MB} MiB, sleep ${SLEEP}s"
    sleep "$SLEEP"
done
