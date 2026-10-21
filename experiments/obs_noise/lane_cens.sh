#!/usr/bin/env bash
# CENSORING.md day 1. R1 tx_cens_B_reg (2-state flags) was stopped at step 1000: the spec works but clean CRPS rose
# +0.007, because "spec hidden" and "spec given, nothing censored" were the same input (cens_model docstring).
# Now: R1b tx_censk_B_reg (4-state flags, --known; spec dropout 0.3) immediately; R0 tx_cens0_B_reg (the identical data
# stream, flags never shown: matched control) once lane7.sh has exited, so that at most two training jobs share the GPU.
# Both: TabICLv2 regression, backbone unfrozen (Muon 1.5e-5), flag maps AdamW 3e-4, 4000 steps.
# Update 13:5x: R1b (--known) did not fix clean either; cause = latent-target scale (censored heavy tails). R1c
# tx_censk2_B_reg and its matched control R0c tx_cens02_B_reg use --max_tail 3 --lat_clip 10 (cens_train.prepare).
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
COMMON="--steps 4000 --val_every 250 --val_blocks 4 --max_tail 3 --lat_clip 10"
log() { echo "[lane_cens] $(date -Is) $*" >> artifacts/logs/cens.log; }
sup() {  # sup <name> <args...>: auto exact-resume on crash, max 6 tries (as supervise.sh)
  local name=$1; shift
  for try in 1 2 3 4 5 6; do
    $PY cens_train.py --name "$name" --resume "$@" >> "artifacts/logs/train_$name.log" 2>&1 && { log "$name done"; return 0; }
    log "$name exited with $? (try $try)"; sleep 20
  done
  return 1
}
log "start R1c"
sup tx_censk2_B_reg --known --spec_drop 0.3 $COMMON &
while pgrep -f "^bash ./lane7.sh" > /dev/null; do sleep 60; done
log "lane7 finished; starting R0"
sup tx_cens02_B_reg --known --spec_drop 1.0 $COMMON
wait
log "all done"
