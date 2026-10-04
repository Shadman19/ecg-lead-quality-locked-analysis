#!/bin/bash
# Round 6: fetch BUT QDB v1.0.0 files from PhysioNet's public AWS S3 mirror (same files; verified against the PhysioNet
# SHA256SUMS.txt already downloaded from physionet.org). Accelerometer files are skipped. Usage: bash butqdb_get_s3.sh
set -u
B=https://physionet-open.s3.amazonaws.com/butqdb/1.0.0
D=$HOME/data/butqdb; mkdir -p "$D"; cd "$D" || exit 1
pkill -u "$USER" -f "wget -q -nc" 2>/dev/null; sleep 1
RECS="100001 100002 103001 103002 103003 104001 105001 111001 113001 114001 115001 118001 121001 122001 123001 124001 125001 126001"
get() { curl -s --retry 5 --retry-delay 3 -o "$1" "$B/$1"; }
for f in RECORDS SHA256SUMS.txt subject-info.csv ANNOTATORS LICENSE.txt ann_reader.m; do [ -s "$f" ] || get "$f"; done
n=0
for r in $RECS; do
  mkdir -p "$r"
  for f in ${r}_ECG.hea ${r}_ANN.csv ${r}_ECG.dat; do
    get "$r/$f" & n=$(( n + 1 ))
    if [ $(( n % 6 )) -eq 0 ]; then wait; fi
  done
done
wait
grep -E "_ECG\.(dat|hea)|_ANN\.csv|RECORDS|subject-info|ANNOTATORS|LICENSE|ann_reader" SHA256SUMS.txt > SUMS_needed.txt
sha256sum -c --quiet SUMS_needed.txt && echo BUTQDB_VERIFIED_OK || echo BUTQDB_VERIFY_FAILED
du -sh "$D"
