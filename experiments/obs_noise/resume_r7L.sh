#!/usr/bin/env bash
# resume the paused r7L_obs_dist_temp (exact resume from its last checkpoint) after the first col run finishes
cd "$(dirname "$0")"
until grep -qE "(p7|r7)_obs_dist_temp_col done" artifacts/logs/phase3.log; do sleep 60; done
./supervise.sh r7L_obs_dist_temp --mode obs --init artifacts/pretrain/stock_reg/last.ckpt --steps 8000 --batch_size 32 --lr 0.0003 --warmup 200 --p_clean 0.25 --val_every 1000 --seed 0 --rep_cfg '{"center": "attn", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true}' --rep_lr_mult 1.0 --bank artifacts/bank_reg --micro_batch 4 --n_blocks 2000 --stop_at 0 --task regression --regimes '[0.4,0.2,0.2,0.2]' --k50_prob 0.2 --noise_set v2 --y_clip 30.0 --layouts '[0.25,0.25,0.25,0.25]' --group_vis_prob 0.5 --aux_w 0.0
echo "[resume] $(date -Is) r7L_obs_dist_temp done ($?)" >> artifacts/logs/phase3.log
