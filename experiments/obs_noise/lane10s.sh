#!/usr/bin/env bash
# Exp 10 screen after both v6 runs hit the LINCS kill switch at step 500: which part of the v6 prior costs LINCS?
# A: less tail mass (p_tail 0.5 -> 0.2); B: no structured effects (paired well, plate, value-dependent noise).
# Base branch size, same schedule as lane10 (6000-step cosine), stop after the step-500 check (resumable).
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VN=v6,v6_gauss,v6_heavy,v6_paired,v6_fail,v6_mnar,v6_lod_flag,v6_lod_noflag,gauss,contam,cauchy,censored_left_heavy,fail_floor
CFG='{"center": "attn", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true, "flag_emb": true, "miss_emb": true}'
COMMON="--task classification --steps 6000 --stop_at 500 --mode obs --freeze_backbone --lr 3e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set v6 --k50_prob 0.1 --val_every 500 --val_blocks 4 --val_noises $VN --log_every 25 --dev_real L1000_dose,CellPainting_dose --dev_every 500 --dev_kill_margin 0.5 --dev_kill_until 1000"
./supervise.sh tx_v6s_tail_A --rep_cfg "$CFG" --p6 '{"p_tail": 0.2}' $COMMON &
sleep 120
./supervise.sh tx_v6s_nostruct_A --rep_cfg "$CFG" --p6 '{"p_well": 0, "p_plate": 0, "p_valhet": 0}' $COMMON
wait
echo "[lane10s] $(date -Is) screen done" >> artifacts/logs/phase3.log
