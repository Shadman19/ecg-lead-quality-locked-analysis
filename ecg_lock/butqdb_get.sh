#!/bin/bash
# Round 6: download BUT QDB v1.0.0 (ECG signals, headers, annotations, metadata; accelerometer files skipped) into ~/data/butqdb
# and verify against the PhysioNet SHA256SUMS.txt. Usage: bash butqdb_get.sh   (runs 6 parallel wget lists, logs to wget_*.log)
set -u
B=https://physionet.org/files/butqdb/1.0.0
D=$HOME/data/butqdb; mkdir -p "$D"; cd "$D" || exit 1
RECS="100001 100002 103001 103002 103003 104001 105001 111001 113001 114001 115001 118001 121001 122001 123001 124001 125001 126001"
W="wget -q -nc --timeout=30 --tries=100 --waitretry=3 -x -nH --cut-dirs=3"
$W $B/RECORDS $B/SHA256SUMS.txt $B/subject-info.csv $B/ANNOTATORS $B/LICENSE.txt $B/ann_reader.m
rm -f list_*.txt; i=0
for r in $RECS; do
  n=$(( i % 6 )); i=$(( i + 1 ))
  for f in ${r}_ECG.hea ${r}_ANN.csv ${r}_ECG.dat; do echo "$B/$r/$f" >> list_$n.txt; done
done
for n in 0 1 2 3 4 5; do nohup $W -i list_$n.txt > wget_$n.log 2>&1 & done
wait
grep -E "_ECG\.(dat|hea)|_ANN\.csv|RECORDS|subject-info|ANNOTATORS|LICENSE|ann_reader" SHA256SUMS.txt > SUMS_needed.txt
sha256sum -c --quiet SUMS_needed.txt && echo BUTQDB_VERIFIED_OK || echo BUTQDB_VERIFY_FAILED
du -sh "$D"
