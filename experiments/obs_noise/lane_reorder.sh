#!/usr/bin/env bash
# Reordered lane (2026-10-19): when stage A regression frees its lane, run the classification matched baseline
# tx_median_B (first GPU test of Muon), then the regression median-start stage A (moved later).
cd "$(dirname "$0")"
until grep -q "tx_obs_col_A_reg done" artifacts/logs/phase3.log; do sleep 60; done
./phase_txB.sh classification median
VN=gauss,cauchy,cauchy_hetero,censored_left_heavy,fail_floor,lod_logistic,plate_effect
CFG='{"center": "attn_med", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true}'
./supervise.sh tx_obs_colmed_A_reg --mode obs --freeze_backbone --rep_cfg "$CFG" --steps 6000 --lr 3e-4 --arch tabicl --task regression --bank artifacts/bank_reg --y_clip 30 --batch_size 32 --micro_batch 4 --warmup 200 --noise_set v2 --regimes [0.4,0.2,0.2,0.2] --k50_prob 0.2 --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25
echo "[phase_tx2] $(date -Is) tx_obs_colmed_A_reg done ($?)" >> artifacts/logs/phase3.log
