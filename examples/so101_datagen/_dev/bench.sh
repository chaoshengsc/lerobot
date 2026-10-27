#!/usr/bin/env bash
# Usage: bench.sh <out_dir> <workers> <shards> <episodes_per_shard> [extra generate args...]
# Runs the generator and reports its summary plus the peak proportional memory (PSS) of its workers.
out=$1; workers=$2; shards=$3; eps=$4; shift 4
rm -rf "$out"
.venv/bin/python -m examples.so101_datagen.generate --output_dir="$out" --num_shards="$shards" \
  --episodes_per_shard="$eps" --num_workers="$workers" --merge=false "$@" > "$out.log" 2>&1 &
gen=$!
peak=0
while kill -0 $gen 2>/dev/null; do
  sleep 2
  pss=0
  for p in $(pgrep -P $gen); do
    v=$(awk '/^Pss:/ {print $2}' /proc/$p/smaps_rollup 2>/dev/null)
    pss=$((pss + ${v:-0}))
  done
  [ $pss -gt $peak ] && peak=$pss
done
echo "workers=$workers peak_worker_pss=$((peak / 1024))MB"
grep -A20 "'episodes'" "$out.log" | grep -E "frames_per_second|episodes_per_hour|success_rate|add_frame|annotate|encoder_wait|prepare|render|save|simulate|wall"
grep -m3 -E "Error|Traceback" "$out.log"
