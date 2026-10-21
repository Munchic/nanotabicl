#!/usr/bin/env bash
# Exp 8: median-start stage A (center 'attn_med': step 0 == TabICLv2 + median imputation exactly), same budget and
# settings as tx_obs_col_A[_reg]; starts when that run's lane frees. Usage: phase_tx2.sh <classification|regression>
cd "$(dirname "$0")"
VN=gauss,cauchy,cauchy_hetero,censored_left_heavy,fail_floor,lod_logistic,plate_effect
CFG='{"center": "attn_med", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true}'
if [ "$1" = regression ]; then T="--task regression --bank artifacts/bank_reg --y_clip 30"; s=_reg; else T="--task classification"; s=; fi
until grep -q "tx_obs_col_A$s done" artifacts/logs/phase3.log; do sleep 60; done
COMMON="--arch tabicl $T --batch_size 32 --micro_batch 4 --warmup 200 --noise_set v2 --regimes [0.4,0.2,0.2,0.2] --k50_prob 0.2 --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25"
./supervise.sh tx_obs_colmed_A$s --mode obs --freeze_backbone --rep_cfg "$CFG" --steps 6000 --lr 3e-4 $COMMON
echo "[phase_tx2] $(date -Is) tx_obs_colmed_A$s done ($?)" >> artifacts/logs/phase3.log
