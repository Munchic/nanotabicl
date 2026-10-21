#!/usr/bin/env bash
# Exp 8 after porting the TabICL wrapper preprocessing (model.tabicl_preprocess; every new TabICLv2-arch run uses it via
# tabicl_port.nano_config). Two lanes, each starting when one of the running stage-A runs frees the GPU:
#   lane 1: reference (cls) -> median-start stage A (cls) -> stage B branch (cls) -> stage B branch (reg)
#   lane 2: reference (reg) -> tx_median_B (cls) -> median-start stage A (reg) -> tx_median_B (reg)
# Usage: lanes_preproc.sh <1|2>
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
VN=gauss,cauchy,cauchy_hetero,censored_left_heavy,fail_floor,lod_logistic,plate_effect
CFG='{"center": "attn_med", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true}'
log() { echo "[lanes_preproc] $(date -Is) $*" >> artifacts/logs/phase3.log; }
task_args() { if [ "$1" = regression ]; then echo "--task regression --bank artifacts/bank_reg --y_clip 30"; else echo "--task classification"; fi; }
sfx() { [ "$1" = regression ] && echo _reg; }

ref() {  # step-0 TabICLv2 + median imputation (with the preprocessing): validation + 27-dataset eval, 1 pass
  s=$(sfx $1)
  $PY obs_train.py --name tx_median0p$s --arch tabicl $(task_args $1) --mode median --steps 0 --noise_set v2 \
    --val_blocks 4 --val_noises $VN --micro_batch 2 >> artifacts/logs/train_tx_median0p$s.log 2>&1
  OBS_TAG=$s $PY obs_eval.py tx_median0p${s}_e1=artifacts/pretrain/tx_median0p$s/last.ckpt --out artifacts/tx_eval$s.jsonl \
    --sources main,ext,ext3 --n_ens 1 >> artifacts/logs/train_tx_median0p$s.log 2>&1
  log "tx_median0p$s reference done"
}
stageA() {  # median-start stage A (frozen backbone, branch only), same budget as tx_obs_col_A
  s=$(sfx $1)
  ./supervise.sh tx_obs_colmed_A$s --mode obs --freeze_backbone --rep_cfg "$CFG" --steps 6000 --lr 3e-4 --arch tabicl \
    $(task_args $1) --batch_size 32 --micro_batch 2 --warmup 200 --noise_set v2 --regimes [0.4,0.2,0.2,0.2] \
    --k50_prob 0.2 --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25
  log "tx_obs_colmed_A$s done ($?)"
}

if [ "$1" = 1 ]; then
  until grep -q "tx_obs_col_A done" artifacts/logs/phase3.log; do sleep 60; done
  ref classification
  stageA classification
  ./phase_txB.sh classification tx_obs_colmed_A obs_only
  until grep -q "tx_obs_colmed_A_reg done" artifacts/logs/phase3.log; do sleep 60; done
  ./phase_txB.sh regression tx_obs_colmed_A_reg obs_only
else
  until grep -q "tx_obs_col_A_reg done" artifacts/logs/phase3.log; do sleep 60; done
  ref regression
  ./phase_txB.sh classification median
  stageA regression
  ./phase_txB.sh regression median
fi
