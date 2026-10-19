#!/usr/bin/env bash
# Run one obs_train.py job with auto-resume on crash (max 5 retries). Usage: supervise.sh <name> <obs_train args...>
cd "$(dirname "$0")"
name=$1; shift
mkdir -p artifacts/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for try in 1 2 3 4 5 6; do
  /home/munchic/miniconda3/envs/ml/bin/python obs_train.py --name "$name" --resume "$@" >> "artifacts/logs/train_$name.log" 2>&1
  code=$?
  echo "[supervise] $(date -Is) $name exited with $code (try $try)" >> "artifacts/logs/train_$name.log"
  [ $code -eq 0 ] && exit 0
  sleep 20
done
exit 1
