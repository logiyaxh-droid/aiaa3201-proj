#!/bin/bash
# Parallel ranged download with retry. Chunks are independent; interrupted
# chunks resume from what is already in their .part file.
URL="$1"; OUT="$2"; TOTAL="$3"; NCHUNK="${4:-10}"
tmpdir="${OUT}.chunks"
mkdir -p "$tmpdir"
CH=$(( (TOTAL + NCHUNK - 1) / NCHUNK ))

fetch() {
  i=$1
  s=$(( i * CH )); e=$(( s + CH - 1 ))
  [ "$e" -ge "$TOTAL" ] && e=$(( TOTAL - 1 ))
  want=$(( e - s + 1 ))
  pf="$tmpdir/c$i"
  for a in $(seq 1 30); do
    have=$(stat -c %s "$pf" 2>/dev/null || echo 0)
    [ "$have" -ge "$want" ] && return 0
    curl -sL --max-time 120 -r $(( s + have ))-$e "$URL" >> "$pf" 2>/dev/null
  done
  have=$(stat -c %s "$pf" 2>/dev/null || echo 0)
  [ "$have" -ge "$want" ]
}

export -f fetch
export URL tmpdir CH TOTAL
seq 0 $(( NCHUNK - 1 )) | xargs -P "$NCHUNK" -I{} bash -c 'fetch {}'

# verify every chunk before assembling
ok=1
for i in $(seq 0 $(( NCHUNK - 1 ))); do
  s=$(( i * CH )); e=$(( s + CH - 1 ))
  [ "$e" -ge "$TOTAL" ] && e=$(( TOTAL - 1 ))
  want=$(( e - s + 1 ))
  have=$(stat -c %s "$tmpdir/c$i" 2>/dev/null || echo 0)
  [ "$have" -lt "$want" ] && { echo "chunk $i short: $have/$want"; ok=0; }
done

if [ "$ok" = "1" ]; then
  : > "$OUT"
  for i in $(seq 0 $(( NCHUNK - 1 ))); do cat "$tmpdir/c$i" >> "$OUT"; done
  got=$(stat -c %s "$OUT")
  echo "assembled $got / $TOTAL"
  [ "$got" = "$TOTAL" ] && { rm -rf "$tmpdir"; echo "OK $OUT"; exit 0; }
fi
echo "INCOMPLETE $OUT"
exit 1
