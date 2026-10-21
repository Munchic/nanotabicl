#!/usr/bin/env bash
# Exp 9, v5: real-shaped synthetic prior (mostly Gaussian / contaminated Gaussian, row heteroscedasticity, plate effects;
# large noise logU(0.2, 1.5) x sd; 80% of tables with 2-5 replicates), Huber start with the fixed estimator, frozen
# backbone, base branch size. Two runs in parallel: column-statistics gates (tx_v5g_A) vs plain gates (tx_v5_A).
# LINCS dev is a KILL SWITCH only: checked at steps 0 / 500 / 1000 (and logged every 500 after), stop if more than 0.5
# points below the run's own step 0 (= TabICLv2 + Huber imputation); the final checkpoint is the one reported.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
VN=gauss,contam,gauss_rowhet,cauchy,censored_left_heavy,fail_floor,plate_effect
W='{"contam": 6, "gauss": 4, "gauss_rowhet": 3, "plate_effect": 3, "gauss_valhet": 2, "cv_lognormal": 2, "student_t3": 1, "censored": 0.3, "cauchy": 0.3, "cauchy_hetero": 0.3, "cauchy_valhet": 0.3, "censored_left": 0.3, "censored_left_heavy": 0.3, "fail_floor": 0.3, "fail_shift": 0.3, "lod_logistic": 0.3, "saturation": 0.3}'
base='"center": "attn_huberg", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true'
COMMON="--task classification --steps 4000 --mode obs --freeze_backbone --lr 3e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set v5 --scale_min 0.2 --scale_max 1.5 --small_k_prob 0.8 --regimes [0.4,0.2,0.2,0.2] --k50_prob 0.2 --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25 --dev_real L1000_dose,CellPainting_dose --dev_every 500 --dev_kill_margin 0.5 --dev_kill_until 1000"
run() {  # run <name> <extra rep_cfg json>
  ./supervise.sh $1 --rep_cfg "{$base$2}" --noise_weights "$W" $COMMON
  echo "[lane8] $(date -Is) $1 done ($?)" >> artifacts/logs/phase3.log
  $PY obs_eval.py ${1}_e1=artifacts/pretrain/$1/last.ckpt --out artifacts/tx_eval.jsonl --sources main,ext,ext3 --n_ens 1 >> artifacts/logs/eval_v5.log 2>&1
  $PY real_eval.py $1=artifacts/pretrain/$1/last.ckpt --datasets L1000_dose,CellPainting_dose >> artifacts/logs/eval_v5.log 2>&1
  echo "[lane8] $(date -Is) $1 evaluated (synthetic + LINCS dev; held-out sets not run)" >> artifacts/logs/phase3.log
}
run tx_v5g_A ', "gate_colstats": true' &
sleep 120
run tx_v5_A ''
wait
