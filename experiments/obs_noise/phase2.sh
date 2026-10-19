#!/usr/bin/env bash
# Phase 2: continue from the stock checkpoint with identical budget/noise for every arm. Usage: phase2.sh <arm>...
cd "$(dirname "$0")"
INIT=${INIT:-artifacts/pretrain/stock/last.ckpt}
P=${PREFIX:-}  # run-name prefix, e.g. x_ for exploration runs
COMMON="--init $INIT --steps ${STEPS:-4000} --batch_size 32 --lr 3e-4 --warmup 200 --val_every 1000"
OBS="--mode obs $COMMON --micro_batch 8"
for arm in "$@"; do
  case $arm in
    p2_clean)              ./supervise.sh $P$arm --mode clean $COMMON --micro_batch 16 ;;
    p2_mean)               ./supervise.sh $P$arm --mode mean  $COMMON --micro_batch 16 ;;
    p2_median)             ./supervise.sh $P$arm --mode median $COMMON --micro_batch 16 ;;
    p2_stats)              ./supervise.sh $P$arm --mode stats $COMMON --micro_batch 4 ;;
    p2_obs)                ./supervise.sh $P$arm $OBS ;;                                   # the NOTES.md spec
    p2_obs_nosab)          ./supervise.sh $P$arm $OBS --rep_cfg '{"use_sab": false}' ;;
    p2_obs_ds16)           ./supervise.sh $P$arm $OBS --rep_cfg '{"d_s": 16}' ;;
    p2_obs_nonull)         ./supervise.sh $P$arm $OBS --rep_cfg '{"use_null": false}' ;;
    p2_obs_med)            ./supervise.sh $P$arm $OBS --rep_cfg '{"center": "median"}' ;;
    p2_obs_med_lr10)       ./supervise.sh $P$arm $OBS --rep_cfg '{"center": "median"}' --rep_lr_mult 10 ;;
    p2_obs_med_mf)         ./supervise.sh $P$arm $OBS --rep_cfg '{"center": "median", "med_feat": true}' ;;
    p2_obs_attn)           ./supervise.sh $P$arm $OBS --rep_cfg '{"center": "attn"}' ;;
    p2_obs_attn_mf)        ./supervise.sh $P$arm $OBS --rep_cfg '{"center": "attn", "med_feat": true}' ;;
    p2_obs_attn_mf_lr10)   ./supervise.sh $P$arm $OBS --rep_cfg '{"center": "attn", "med_feat": true}' --rep_lr_mult 10 ;;
    p2_obs_attn_mf_nosab)  ./supervise.sh $P$arm $OBS --rep_cfg '{"center": "attn", "med_feat": true, "use_sab": false}' ;;
    p2_obs_attn_mf_rstd)    ./supervise.sh $P$arm $OBS --rep_cfg '{"center": "attn", "med_feat": true, "std": "medmad", "zclip": 10.0}' ;;
    p2_obs_attn_mf_rstd_rf) ./supervise.sh $P$arm $OBS --rep_cfg '{"center": "attn", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true}' ;;
    p2_obs_frozen)         ./supervise.sh $P$arm $OBS --freeze_backbone ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  echo "[phase2] $(date -Is) $P$arm done ($?)" >> artifacts/logs/phase2.log
done
