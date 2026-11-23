#!/bin/bash
# Download DOVE 5B weights: 5 transformer shards + 4 text_encoder shards,
# plus the CogVideoX VAE / scheduler / tokenizer configs.
#
# Uses the parallel ranged fetcher (scripts/pdl.py) because hf-mirror throttles a
# single connection to a few hundred KB/s while 16 ranged streams reached
# 2.6-37 MB/s in earlier runs.  Each file is verified against x-linked-size.
set -u
cd /data/disk1/home/ghe270/p3/prior_qualification || exit 1
mkdir -p weights_dove/transformer weights_dove/text_encoder

B=https://hf-mirror.com/zhengchen1999/DOVE/resolve/main
V=/data/disk1/home/ghe270/p3/quality_metrics/venv/bin/python

get() {
  local url="$1" out="$2"
  local want
  want=$(curl -sIL "$url" | grep -i "^x-linked-size:" | tail -1 | tr -dc '0-9')
  [ -z "$want" ] && want=$(curl -sIL "$url" | grep -i "^content-length:" | tail -1 | tr -dc '0-9')
  local have
  have=$(stat -c%s "$out" 2>/dev/null || echo 0)
  if [ "$have" = "$want" ] && [ -n "$want" ]; then
    echo "SKIP $out ($have)"; return 0
  fi
  echo "GET $out want=$want"
  python3 scripts/pdl.py "$url" "$out" 16
}

for i in 1 2 3 4 5; do
  f="transformer/diffusion_pytorch_model-0000${i}-of-00005.safetensors"
  get "$B/$f" "weights_dove/$f"
done
for i in 1 2 3 4; do
  f="text_encoder/model-0000${i}-of-00004.safetensors"
  get "$B/$f" "weights_dove/$f"
done

# Small config files (json / txt) fetched directly: no range needed.
for f in \
  "transformer/config.json" "text_encoder/config.json" \
  "scheduler/scheduler_config.json" "tokenizer/tokenizer_config.json" \
  "tokenizer/special_tokens_map.json" "tokenizer/spiece.model" \
  "model_index.json" ; do
  mkdir -p "weights_dove/$(dirname "$f")"
  [ -s "weights_dove/$f" ] || curl -sSL -o "weights_dove/$f" "$B/$f"
done
echo ALLDONE
