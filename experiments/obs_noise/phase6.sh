#!/usr/bin/env bash
# Exp 7 T2: longer stock pretraining (30k steps, 2.5x) on the 4x bank (8000 blocks), both tasks, plus the het-trained
# regression encoder. Usage: phase6.sh stock   (waits for r7L_median; stock2 then stock2_reg)
#                            phase6.sh r7L     (waits for r7_obs_dist_temp_col; then r7L_obs_dist_temp)
cd "$(dirname "$0")"
if [ "$1" = stock ]; then
  until grep -q "r7L_median exited with 0" artifacts/logs/train_r7L_median.log 2>/dev/null; do sleep 60; done
  ./supervise.sh stock2 --mode clean --steps 30000 --batch_size 32 --micro_batch 16 --lr 1e-3 --warmup 500 --val_every 1000 --n_blocks 8000
  echo "[phase6] $(date -Is) stock2 done ($?)" >> artifacts/logs/phase3.log
  ./supervise.sh stock2_reg --mode clean --task regression --bank artifacts/bank_reg --y_clip 30 --steps 30000 --batch_size 32 --micro_batch 16 --lr 1e-3 --warmup 500 --val_every 1000 --n_blocks 8000
  echo "[phase6] $(date -Is) stock2_reg done ($?)" >> artifacts/logs/phase3.log
else
  until grep -q "r7_obs_dist_temp_col done" artifacts/logs/phase3.log; do sleep 60; done
  STEPS=8000 NOISESET=v2 BEST='{"center": "attn", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true}' \
    TASK=regression INIT=artifacts/pretrain/stock_reg/last.ckpt P=r7L_ LAYOUTS='[0.25,0.25,0.25,0.25]' ./phase3.sh obs_dist_temp
fi
