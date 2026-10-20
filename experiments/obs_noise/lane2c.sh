#!/usr/bin/env bash
# Lane 2 after the screens: regression stage A with the best branch config (reg_cfg.json, read at launch; default =
# the capacity screen's d_s 64 + 2 SAB), named tx_obs_colmedg_A_reg so that lane 1 (lanes_g.sh 1) picks it up for
# regression stage B; then the regression matched baseline tx_median_B_reg.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VN=gauss,cauchy,cauchy_hetero,censored_left_heavy,fail_floor,lod_logistic,plate_effect
COMMON="--mode obs --freeze_backbone --lr 3e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set v2 --regimes [0.4,0.2,0.2,0.2] --k50_prob 0.2 --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25"
until grep -q "tx_scr_tok_A exited with" artifacts/logs/train_tx_scr_tok_A.log 2>/dev/null; do sleep 60; done
echo "[lane2c] $(date -Is) tx_scr_tok_A done" >> artifacts/logs/phase3.log
CFG=$(cat reg_cfg.json)
echo "[lane2c] $(date -Is) regression stage A config: $CFG" >> artifacts/logs/phase3.log
./supervise.sh tx_obs_colmedg_A_reg --task regression --bank artifacts/bank_reg --y_clip 30 --rep_cfg "$CFG" --steps 6000 $COMMON
echo "[lanes_g] $(date -Is) tx_obs_colmedg_A_reg done ($?)" >> artifacts/logs/phase3.log
./phase_txB.sh regression median
