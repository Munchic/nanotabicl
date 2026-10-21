#!/usr/bin/env bash
# Serialized tail of Exp 8 (two capacity-sized stage-A jobs do not fit on the GPU together):
#  1. when the regression stage A (already running under supervise) exits cleanly: mark it done -> lane 1
#     (lanes_g.sh 1) starts the regression stage B branch run; here the regression matched baseline runs in parallel;
#  2. when both regression stage-B runs are done: evaluate the regression stage A + B and tx_median_B_reg (27 datasets);
#  3. then the classification capacity stage A (6000 steps) and a second classification stage B from it (tx_obs_capB).
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
VN=gauss,cauchy,cauchy_hetero,censored_left_heavy,fail_floor,lod_logistic,plate_effect
log() { echo "[lane4] $(date -Is) $*" >> artifacts/logs/phase3.log; }
until grep -q "tx_obs_colmedg_A_reg exited with 0" artifacts/logs/train_tx_obs_colmedg_A_reg.log; do sleep 60; done
echo "[lanes_g] $(date -Is) tx_obs_colmedg_A_reg done (0)" >> artifacts/logs/phase3.log
./phase_txB.sh regression median
until grep -q "tx_obs_B_reg done" artifacts/logs/phase3.log; do sleep 60; done
for spec in tx_obs_colmedg_A_reg tx_median_B_reg tx_obs_B_reg; do
  OBS_TAG=_reg $PY obs_eval.py ${spec}_e1=artifacts/pretrain/$spec/last.ckpt --out artifacts/tx_eval_reg.jsonl --sources main,ext,ext2,ext3 --n_ens 1 >> artifacts/logs/eval_tx_reg.log 2>&1
done
log "regression stage A / B evaluated"
CFG=$(cat reg_cfg.json)  # the capacity config
./supervise.sh tx_obs_cap_A --task classification --rep_cfg "$CFG" --steps 6000 --mode obs --freeze_backbone --lr 3e-4 \
  --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set v2 --regimes [0.4,0.2,0.2,0.2] --k50_prob 0.2 \
  --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25
log "tx_obs_cap_A done ($?)"
OBS_NAME=tx_obs_capB ./phase_txB.sh classification tx_obs_cap_A obs_only
for n in 1 8; do
  $PY obs_eval.py tx_obs_capB_e$n=artifacts/pretrain/tx_obs_capB/last.ckpt --out artifacts/tx_eval.jsonl --sources main,ext,ext2,ext3 --n_ens $n >> artifacts/logs/eval_tx_capB.log 2>&1
done
log "tx_obs_capB evaluated"
