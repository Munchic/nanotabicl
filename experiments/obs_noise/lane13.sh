#!/usr/bin/env bash
# Exp 11 step 2: regression stage A on the v5 prior (the classification prior that passed the LINCS kill switch),
# generic branch (center attn, no gates), large branch (d_s 64, 2 SAB), frozen TabICLv2 regression backbone, 4000 steps.
# Two runs: with the learned context-row attention bias (row_attn) and without it (control).
# Kill switch: CellPainting_dose_reg (the only dev set with replicate headroom), -100 x CRPS, stop if > 0.5 below step 0
# (= 0.005 CRPS) at steps 500 / 1000. Then: real regression sets (dev + held-out GasConc, Tox21Pot) and the synthetic
# regression eval, 1 and 8 passes.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
VN=gauss,contam,gauss_rowhet,cauchy,censored_left_heavy,fail_floor,plate_effect
W='{"contam": 6, "gauss": 4, "gauss_rowhet": 3, "plate_effect": 3, "gauss_valhet": 2, "cv_lognormal": 2, "student_t3": 1, "censored": 0.3, "cauchy": 0.3, "cauchy_hetero": 0.3, "cauchy_valhet": 0.3, "censored_left": 0.3, "censored_left_heavy": 0.3, "fail_floor": 0.3, "fail_shift": 0.3, "lod_logistic": 0.3, "saturation": 0.3}'
base='"center": "attn", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true, "d_s": 64, "n_sab": 2'
COMMON="--task regression --bank artifacts/bank_reg --y_clip 30 --steps 4000 --mode obs --freeze_backbone --lr 3e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set v5 --scale_min 0.2 --scale_max 1.5 --small_k_prob 0.8 --regimes [0.4,0.2,0.2,0.2] --k50_prob 0.2 --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25 --dev_real CellPainting_dose_reg --dev_every 500 --dev_kill_margin 0.5 --dev_kill_until 1000"
run() {  # run <name> <extra rep_cfg json>
  ./supervise.sh $1 --rep_cfg "{$base$2}" --noise_weights "$W" $COMMON
  echo "[lane13] $(date -Is) $1 done ($?)" >> artifacts/logs/phase3.log
  for e in 1 8; do
    $PY real_eval_reg.py ${1}_e$e=artifacts/pretrain/$1/last.ckpt --datasets L1000_dose_reg,CellPainting_dose_reg,GasConc,Tox21Pot --ks 1,all --n_ens $e >> artifacts/logs/eval_reg13.log 2>&1
    $PY obs_eval.py ${1}_e$e=artifacts/pretrain/$1/last.ckpt --out artifacts/tx_eval_reg.jsonl --sources main,ext,ext3 --n_ens $e >> artifacts/logs/eval_reg13.log 2>&1
  done
  echo "[lane13] $(date -Is) $1 evaluated" >> artifacts/logs/phase3.log
}
run tx_v5reg_ra_A ', "row_attn": true' &
sleep 150
run tx_v5reg_A ''
wait
