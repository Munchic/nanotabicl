#!/usr/bin/env bash
# Exp 7 T2 follow-up: matched median baseline + column-context encoder continued from stock2 (30k-step stock on the
# 8000-block bank), same recipe as p5 / p7 (8000 steps, v2 modes, count regimes, 20% K <= 50) but on the 8000-block bank.
# Waits for stock2 and for the p7 col run (frees its lane).
cd "$(dirname "$0")"
until grep -q "stock2 done" artifacts/logs/phase3.log && grep -q "p7_obs_dist_temp_col done" artifacts/logs/phase3.log; do sleep 60; done
export STEPS=8000 NOISESET=v2 BEST='{"center": "attn", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true}'
# phase3.sh has no bank-size option; append it to COMMON through the TASK-specific BANK variable
TASK=classification INIT=artifacts/pretrain/stock2/last.ckpt P=p8_ EXTRA_N_BLOCKS=8000 ./phase3.sh median obs_dist_temp_col
