#!/usr/bin/env bash
# Big-picture evaluation of tx_v5gen_A (Exp 9) on the held-out real sets (GasDrift, Tox21, UPS1_low, CPTAC) + LINCS dose
# dev, against the off-the-shelf wrapper (mean input), our-pipeline mean / Huber imputation and stage B; 1 and 8 passes;
# k = 1 (one replicate) and all replicates. Fresh records -> artifacts/real_heldout.jsonl (task sets changed since some
# older real_eval.jsonl records).
cd "$(dirname "$0")"
PY=/home/munchic/miniconda3/envs/ml/bin/python
S=/tmp/claude-1002/-home-munchic-nanotabicl/b53eb5b4-8334-4c18-b331-49e90dd4cfcf/scratchpad
DS=GasDrift,Tox21,UPS1_low,CPTAC,L1000_dose,CellPainting_dose
OUT=artifacts/real_heldout.jsonl
for e in 1 8; do
  $PY real_eval.py tx_mean0p_e$e=artifacts/pretrain/tx_mean0p/last.ckpt tx_huber0p_e$e=artifacts/pretrain/tx_huber0p/last.ckpt \
    tx_v5gen_A_e$e=artifacts/pretrain/tx_v5gen_A/last.ckpt tx_obs_B_e$e=artifacts/pretrain/tx_obs_B/last.ckpt \
    --datasets $DS --ks 1,all --n_ens $e --out $OUT
done
REAL_OUT=$OUT $S/venv/bin/python real_tabicl.py $DS
echo "[heldout] $(date -Is) done" >> artifacts/logs/phase3.log
