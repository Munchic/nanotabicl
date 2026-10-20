#!/usr/bin/env bash
# waits for both stage-A variants per task, picks the better one (pick_stageA.py), runs stage B: classification, then regression
cd "$(dirname "$0")"
for task in classification regression; do
  s=; [ $task = regression ] && s=_reg
  until grep -q "tx_obs_col_A$s done" artifacts/logs/phase3.log && grep -q "tx_obs_colmed_A$s done" artifacts/logs/phase3.log; do sleep 60; done
  best=$(/home/munchic/miniconda3/envs/ml/bin/python pick_stageA.py $task 2>> artifacts/logs/phase3.log)
  echo "[queue_txB] $(date -Is) $task stage B from $best" >> artifacts/logs/phase3.log
  ./phase_txB.sh $task $best
done
