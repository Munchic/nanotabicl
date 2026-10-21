#!/usr/bin/env bash
# SHAPE.md section 7: the one-time final evaluation (datasets, arms, metrics and success criterion fixed in SHAPE.md
# before any model saw these data). Runs after the full run's dev evaluation.
#   wrapper arms (8 and 1 est.): mean, mean + sd (the reference), Huber + sd (secondary bar, added before any result), sketch, cluster shares (paired sets)
#   ours: tx_shV2b_A final (1 and 8 passes) and its joint-path-off ablation (8 passes)
# Records: artifacts/final_shape_wrapper.jsonl, artifacts/final_shape_ours.jsonl (with reliability calibration).
cd "$(dirname "$0")"
PY=/home/munchic/miniconda3/envs/ml/bin/python
S=/tmp/claude-1002/-home-munchic-nanotabicl/b53eb5b4-8334-4c18-b331-49e90dd4cfcf/scratchpad
DS=NK,HVTN,PreE,Musk,PhysioNet
until grep -q "tx_shV2b_A dev-evaluated" artifacts/logs/phase3.log; do sleep 60; done
CK=artifacts/pretrain/tx_shV2b_A/last.ckpt
for e in 8 1; do
  $PY real_eval.py tx_shV2b_A_e$e=$CK --datasets $DS --ks all --n_ens $e --out artifacts/final_shape_ours.jsonl >> artifacts/logs/final_shape.log 2>&1
done
$PY real_eval.py tx_shV2b_A_nojoint_e8=$CK --datasets $DS --ks all --n_ens 8 --no_paired --out artifacts/final_shape_ours.jsonl >> artifacts/logs/final_shape.log 2>&1
$S/venv/bin/python shape_features.py --datasets $DS --n_est 8,1 --arms mean,mean_sd,huber_sd,sketch --out artifacts/final_shape_wrapper.jsonl >> artifacts/logs/final_shape.log 2>&1
echo "[shape] $(date -Is) final evaluation done" >> artifacts/logs/phase3.log
