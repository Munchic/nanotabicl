#!/usr/bin/env bash
# Exp 8 lanes with the per-cell gate (center 'attn_medg'), fp32 + wrapper preprocessing, micro-batch 2.
#   lane 1: stage A cls (tx_obs_colmedg_A, 6000) -> stage B cls branch -> (after stage A reg) stage B reg branch
#   lane 2: (after tx_median_B) screens from screen_cfgs.json (3000 each) -> stage A reg (tx_obs_colmedg_A_reg) -> tx_median_B_reg
# Usage: lanes_g.sh <1|2>
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VN=gauss,cauchy,cauchy_hetero,censored_left_heavy,fail_floor,lod_logistic,plate_effect
CFG='{"center": "attn_medg", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true}'
COMMON="--mode obs --freeze_backbone --lr 3e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set v2 --regimes [0.4,0.2,0.2,0.2] --k50_prob 0.2 --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25"
REG="--task regression --bank artifacts/bank_reg --y_clip 30"
log() { echo "[lanes_g] $(date -Is) $*" >> artifacts/logs/phase3.log; }
if [ "$1" = 1 ]; then
  ./supervise.sh tx_obs_colmedg_A --task classification --rep_cfg "$CFG" --steps 6000 $COMMON
  log "tx_obs_colmedg_A done ($?)"
  ./phase_txB.sh classification tx_obs_colmedg_A obs_only
  until grep -q "tx_obs_colmedg_A_reg done" artifacts/logs/phase3.log; do sleep 60; done
  ./phase_txB.sh regression tx_obs_colmedg_A_reg obs_only
else
  until grep -q "tx_median_B done" artifacts/logs/phase3.log; do sleep 60; done
  for name in $(python3 -c "import json; print(' '.join(json.load(open('screen_cfgs.json'))))"); do
    SCFG=$(python3 -c "import json; print(json.dumps(json.load(open('screen_cfgs.json'))['$name']))")
    ./supervise.sh $name --task classification --rep_cfg "$SCFG" --steps 3000 $COMMON
    log "$name done ($?)"
  done
  ./supervise.sh tx_obs_colmedg_A_reg $REG --rep_cfg "$CFG" --steps 6000 $COMMON
  log "tx_obs_colmedg_A_reg done ($?)"
  ./phase_txB.sh regression median
fi
