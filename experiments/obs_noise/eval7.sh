#!/usr/bin/env bash
# Exp 7 evaluation of finished runs: 1 pass and 8-member ensembles on main, ext, ext2, ext3.
# Usage: eval7.sh <run name>...   (p* -> classification, artifacts/exp7_eval.jsonl; r* -> regression, exp7_eval_reg.jsonl)
cd "$(dirname "$0")"
PY=/home/munchic/miniconda3/envs/ml/bin/python
for name in "$@"; do
  for n in 1 8; do
    if [[ $name == r* ]]; then
      OBS_TAG=_reg $PY obs_eval.py ${name}_e$n=artifacts/pretrain/$name/last.ckpt --out artifacts/exp7_eval_reg.jsonl --sources main,ext,ext2,ext3 --n_ens $n
    else
      $PY obs_eval.py ${name}_e$n=artifacts/pretrain/$name/last.ckpt --out artifacts/exp7_eval.jsonl --sources main,ext,ext2,ext3 --n_ens $n
    fi
  done
done 2>&1 | grep -v -i -E "nvml|raw_cnt" | cut -c1-200
