#!/usr/bin/env bash
# Background progress tracker for the shape / head runs: one snapshot every 15 min into artifacts/logs/progress_shape.log
# (plus an immediate line on a crash, a non-finite loss or a kill switch). Started with nohup; stop with pkill -f track_shape.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
OUT=artifacts/logs/progress_shape.log
PAT='tx_shV4*|tx_head*'
b0=$(cat artifacts/logs/train_*.log 2>/dev/null | grep -c "non-finite\|exited with [1-9]\|out of memory"); b0=${b0:-0}
while true; do
  b=$(cat artifacts/logs/train_*.log 2>/dev/null | grep -c "non-finite\|exited with [1-9]\|out of memory"); b=${b:-0}
  { echo "=== $(date -Is)"; $PY monitor_shape.py; } >> $OUT
  if [ "$b" -gt "$b0" ]; then  # event: note it and keep going (supervise.sh auto-resumes)
    grep "non-finite\|exited with [1-9]\|out of memory" artifacts/logs/train_*.log | tail -2 >> $OUT
    b0=$b
  fi
  sleep 900
done
