#!/usr/bin/env bash
# CENSORING.md, frozen-backbone spec maps (after the unfrozen runs R1 / R1b / R1c failed the TabArena IID guard:
# -4 to -11% RMSE vs stock on wine / airfoil). Backbone frozen => no spec == TabICLv2 exactly, at any weights.
#   F2 tx_censFr_A_reg: linear flag maps + gated row adapter (256), now
#   F1 tx_censF_A_reg : linear flag maps only, once lane7.sh (the other lane) has exited (<= 2 training jobs on the GPU)
# Flags shown on every censored table (spec dropout is moot when no spec == stock), --max_tail 3 --lat_clip 10.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
COMMON="--freeze_backbone --spec_drop 0 --steps 3000 --val_every 250 --val_blocks 4 --max_tail 3 --lat_clip 10"
log() { echo "[lane_censF] $(date -Is) $*" >> artifacts/logs/cens.log; }
sup() {  # sup <name> <args...>: auto exact-resume on crash, max 6 tries
  local name=$1; shift
  for try in 1 2 3 4 5 6; do
    $PY cens_train.py --name "$name" --resume "$@" >> "artifacts/logs/train_$name.log" 2>&1 && { log "$name done"; return 0; }
    log "$name exited with $? (try $try)"; sleep 20
  done
  return 1
}
log "start F2"
sup tx_censFr_A_reg --row_adapter 256 $COMMON &
while pgrep -f "^bash ./lane7.sh" > /dev/null; do sleep 60; done
log "lane7 finished; starting F1"
sup tx_censF_A_reg $COMMON
wait
log "all done"
