#!/bin/bash
# Fetch MVSR4x (19-part split zip) and extract the test set.
#
# The archive is a standard split zip: z01..z18 are the data volumes (each
# exactly 2^31 bytes) and the trailing .zip holds the central directory, so the
# only reliable way to read an entry is to concatenate every part in order.
#
# Every curl needs -L: HF answers 302 to the Xet CAS bridge, and a HEAD without
# -L returns 302 with no size (which made every part after z01 look size-less).
set -u
cd /mnt/sda1/shidanhe/utrd_qual || exit 1
mkdir -p mvsr_raw
cd mvsr_raw || exit 1

BASE="https://huggingface.co/datasets/RuixuanJiang/Video_Deblurring_Datasets/resolve/main/MVSR"
FAIL=0

want_of() {
  local url="$1"
  local w
  w=$(curl -sIL "$url" | grep -i "^x-linked-size:" | tail -1 | tr -dc '0-9')
  [ -z "$w" ] && w=$(curl -sIL "$url" | grep -i "^content-length:" | tail -1 | tr -dc '0-9')
  echo "$w"
}

fetch() {
  local name="$1"
  local url="$BASE/MVSR4x%20%282%29.${name}"
  local want have
  want=$(want_of "$url")
  have=$(stat -c%s "$name" 2>/dev/null || echo 0)
  if [ -n "$want" ] && [ "$have" = "$want" ]; then
    echo "SKIP $name $have"; return 0
  fi
  echo "GET $name want=$want have=$have"
  for i in 1 2 3 4 5 6 7 8; do
    curl -sSL --retry 5 --retry-delay 5 -C - -o "$name" "$url"
    have=$(stat -c%s "$name" 2>/dev/null || echo 0)
    echo "  attempt $i -> $have / $want"
    if [ -n "$want" ] && [ "$have" = "$want" ]; then return 0; fi
    if [ -n "$want" ] && [ "$have" -gt "$want" ]; then rm -f "$name"; fi
    sleep 3
  done
  echo "FAILED $name (have=$have want=$want)"
  return 1
}

for i in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18; do
  fetch "z$i" || FAIL=1
done
fetch "zip" || FAIL=1

if [ "$FAIL" -ne 0 ]; then
  echo "INCOMPLETE — rerun this script to resume"
  exit 2
fi

echo "--- all parts present, sizes: ---"
ls -l z01 z02 z03 z04 z05 z06 z07 z08 z09 z10 z11 z12 z13 z14 z15 z16 z17 z18 zip | awk '{print $9, $5}'

echo "--- concatenating ---"
cat z01 z02 z03 z04 z05 z06 z07 z08 z09 z10 z11 z12 z13 z14 z15 z16 z17 z18 zip > mvsr4x_joined.zip
stat -c%s mvsr4x_joined.zip

echo "--- extracting test sets ---"
python3 - <<'PY'
import zipfile
z = zipfile.ZipFile("mvsr4x_joined.zip")
names = [n for n in z.namelist() if "_test/" in n and n.lower().endswith(".png")]
print("test png entries", len(names))
clips = sorted({"/".join(n.split("/")[:3]) for n in names})
print("clip dirs", len(clips), clips[:6])
for n in names:
    z.extract(n, ".")
print("extracted")
PY
echo ALLDONE
