#!/usr/bin/env bash
# Exp 11 step 1 (training-free): existing TabICLv2 regression checkpoints on the real regression tasks.
cd "$(dirname "$0")"
PY=/home/munchic/miniconda3/envs/ml/bin/python
S=/tmp/claude-1002/-home-munchic-nanotabicl/b53eb5b4-8334-4c18-b331-49e90dd4cfcf/scratchpad
DS=L1000_dose_reg,CellPainting_dose_reg,GasConc,Tox21Pot
P=artifacts/pretrain
for e in 1 8; do
  $PY real_eval_reg.py tx_mean0p_reg_e$e=$P/tx_mean0p_reg/last.ckpt tx_median0p_reg_e$e=$P/tx_median0p_reg/last.ckpt \
    tx_huber0p_reg_e$e=$P/tx_huber0p_reg/last.ckpt tx_obs_col_A_reg_e$e=$P/tx_obs_col_A_reg/last.ckpt \
    tx_obs_colmedg_A_reg_e$e=$P/tx_obs_colmedg_A_reg/last.ckpt tx_obs_B_reg_e$e=$P/tx_obs_B_reg/last.ckpt \
    --datasets $DS --ks 1,all --n_ens $e
  $S/venv/bin/python real_eval_reg.py --wrapper --datasets $DS --ks 1,all --n_ens $e
done
echo "[reg_real] $(date -Is) done" >> artifacts/logs/phase3.log
