#!/usr/bin/env bash
# Lane 2, second half (takes over from lanes_preproc.sh lane 2 once tx_median_B finishes): branch screens in stage A
# (classification, 3000 steps each, compared with tx_obs_colmed_A at the same steps), then the regression median-start
# stage A and tx_median_B_reg. Screen configs are read from screen_cfgs.json when each starts (editable meanwhile).
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VN=gauss,cauchy,cauchy_hetero,censored_left_heavy,fail_floor,lod_logistic,plate_effect
log() { echo "[lane2b] $(date -Is) $*" >> artifacts/logs/phase3.log; }
COMMON="--mode obs --freeze_backbone --lr 3e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set v2 --regimes [0.4,0.2,0.2,0.2] --k50_prob 0.2 --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25"
until grep -q "tx_median_B done" artifacts/logs/phase3.log; do sleep 60; done
for name in $(python3 -c "import json; print(' '.join(json.load(open('screen_cfgs.json'))))"); do
  CFG=$(python3 -c "import json; print(json.dumps(json.load(open('screen_cfgs.json'))['$name']))")
  ./supervise.sh $name --task classification --rep_cfg "$CFG" --steps 3000 $COMMON
  log "$name done ($?)"
done
CFG='{"center": "attn_med", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true}'
./supervise.sh tx_obs_colmed_A_reg --task regression --bank artifacts/bank_reg --y_clip 30 --rep_cfg "$CFG" --steps 6000 $COMMON
echo "[lanes_preproc] $(date -Is) tx_obs_colmed_A_reg done ($?)" >> artifacts/logs/phase3.log
./phase_txB.sh regression median
