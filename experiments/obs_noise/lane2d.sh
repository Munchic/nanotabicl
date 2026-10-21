#!/usr/bin/env bash
# Lane 2 (restart): the regression stage A ran out of memory next to the unfrozen stage B (cached memory), so it waits
# for tx_obs_B to finish, then runs (fresh supervise retries; resumes its step-0 checkpoint) with the capacity config.
# The regression matched baseline tx_median_B_reg waits for the classification capacity stage A (lane3) to finish, so
# that at most two jobs share the GPU (lane 1 starts the regression stage B branch run when this stage A is done).
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VN=gauss,cauchy,cauchy_hetero,censored_left_heavy,fail_floor,lod_logistic,plate_effect
COMMON="--mode obs --freeze_backbone --lr 3e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set v2 --regimes [0.4,0.2,0.2,0.2] --k50_prob 0.2 --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25"
until grep -q "tx_obs_B done" artifacts/logs/phase3.log; do sleep 60; done
CFG=$(cat reg_cfg.json)
echo "[lane2d] $(date -Is) regression stage A (re)start, config: $CFG" >> artifacts/logs/phase3.log
./supervise.sh tx_obs_colmedg_A_reg --task regression --bank artifacts/bank_reg --y_clip 30 --rep_cfg "$CFG" --steps 6000 $COMMON
echo "[lanes_g] $(date -Is) tx_obs_colmedg_A_reg done ($?)" >> artifacts/logs/phase3.log
until grep -q "tx_obs_cap_A done" artifacts/logs/phase3.log; do sleep 60; done
./phase_txB.sh regression median
