#!/usr/bin/env bash
# SHAPE.md screen, pair 1: V0 (backbone sees the mean; marginal ISAB encoder, zero-init embedding) vs V1 (+ per-cell
# replicate-sd columns: step 0 = the mean + sd wrapper input). Shape prior (observe_shape), frozen TabICLv2, d_s 64,
# 2 ISAB blocks (16 inducing points), 6000-step cosine schedule, stopped after step 1000 (resumable for the winner).
# Kill switches: real dev (HAR + LINCS dose, accuracy, > 0.5 below step 0 at 500 / 1000); learning (shape-only
# settings must close >= 10% more of the mean -> oracle loss gap than step 0 by step 1000).
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VN=sh,sh_rep,sh_pop,sh_loc,sh_scale,sh_mix,sh_rare,sh_modesep,sh_tail,sh_state,sh_xor
base='"center": "mean", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true, "d_s": 64, "n_sab": 2, "isab": 16'
COMMON="--task classification --steps 6000 --stop_at 1000 --mode obs --freeze_backbone --lr 3e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set shape --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25 --dev_real HAR,L1000_dose,CellPainting_dose --dev_every 500 --dev_kill_margin 0.5 --dev_kill_until 1000 --learn_kill_at 1000 --learn_kill_min 0.10"
./supervise.sh tx_shV0_A --rep_cfg "{$base}" $COMMON &
sleep 180
./supervise.sh tx_shV1_A --rep_cfg "{$base, \"sd_cols\": true}" $COMMON
wait
echo "[shape] $(date -Is) screen pair 1 (V0, V1) done" >> artifacts/logs/phase3.log
