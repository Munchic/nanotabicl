#!/usr/bin/env bash
# Exp 7 follow-ups (queued behind the running aux runs): aux without the center term (auxq), column-context encoder
# (col), then the het-trained aux run. Same budget / init / mixture as p5 / r5. Usage: phase5.sh <classification|regression>
cd "$(dirname "$0")"
export STEPS=8000 NOISESET=v2 BEST='{"center": "attn", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true}'
if [ "$1" = regression ]; then T=regression; I=artifacts/pretrain/stock_reg/last.ckpt; c=r; else T=classification; I=artifacts/pretrain/stock/last.ckpt; c=p; fi
until grep -q "${c}7_obs_dist_temp_aux done" artifacts/logs/phase3.log; do sleep 60; done
TASK=$T INIT=$I P=${c}7_ ./phase3.sh obs_dist_temp_auxq obs_dist_temp_col
TASK=$T INIT=$I P=${c}7L_ LAYOUTS='[0.25,0.25,0.25,0.25]' ./phase3.sh obs_dist_temp_aux
