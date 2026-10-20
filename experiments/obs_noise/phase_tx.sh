#!/usr/bin/env bash
# Exp 8: replicate branch on the released TabICLv2 (27.55M) backbone. Stage A = frozen backbone, branch only
# (p7 column-context config). Matched reference: TabICLv2 + median imputation, no training (tx_median0[_reg]).
# Usage: phase_tx.sh A <classification|regression>
cd "$(dirname "$0")"
VN=gauss,cauchy,cauchy_hetero,censored_left_heavy,fail_floor,lod_logistic,plate_effect
CFG='{"center": "attn", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true}'
if [ "$2" = regression ]; then T="--task regression --bank artifacts/bank_reg --y_clip 30"; s=_reg; else T="--task classification"; s=; fi
COMMON="--arch tabicl $T --batch_size 32 --micro_batch 4 --warmup 200 --noise_set v2 --regimes [0.4,0.2,0.2,0.2] --k50_prob 0.2 --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25"
if [ "$1" = A ]; then
  ./supervise.sh tx_obs_col_A$s --mode obs --freeze_backbone --rep_cfg "$CFG" --steps 6000 --lr 3e-4 $COMMON
  echo "[phase_tx] $(date -Is) tx_obs_col_A$s done ($?)" >> artifacts/logs/phase3.log
fi
