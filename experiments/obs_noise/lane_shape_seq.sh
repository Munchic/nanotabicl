#!/usr/bin/env bash
# SHAPE.md screen, sequential (two shape-prior runs OOM together on the 16 GB GPU): after V0 (lane_shape1) finishes,
# V1 (sd columns), V2 (+ joint path), V3 (learned sd column) one at a time. Same schedule and kill switches as lane_shape1.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VN=sh,sh_rep,sh_pop,sh_loc,sh_scale,sh_mix,sh_rare,sh_modesep,sh_tail,sh_state,sh_xor
base='"center": "mean", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true, "d_s": 64, "n_sab": 2, "isab": 16, "sd_cols": true'
COMMON="--task classification --steps 6000 --stop_at 1000 --mode obs --freeze_backbone --lr 3e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set shape --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25 --dev_real HAR,L1000_dose,CellPainting_dose --dev_every 500 --dev_kill_margin 0.5 --dev_kill_until 1000 --learn_kill_at 1000 --learn_kill_min 0.10"
while pgrep -f "[s]upervise.sh tx_shV0_A" > /dev/null; do sleep 60; done
for spec in "tx_shV1_A|" "tx_shV2_A|, \"joint\": 16" "tx_shV3_A|, \"sd_learn\": true"; do
  name=${spec%%|*}; extra=${spec#*|}
  ./supervise.sh $name --rep_cfg "{$base$extra}" $COMMON
  echo "[shape] $(date -Is) $name screen done" >> artifacts/logs/phase3.log
done
