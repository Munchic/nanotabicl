#!/usr/bin/env bash
# Exp 9 v4 screens (synthetic-only training): the v3 mixture (contam 6, Gaussian 3, t3 / plate / constant CV 2, rest 1)
# plus efficiency pressure: noise scale ~ logU(0.1, 2) x sd and half of the tables with U{2..5} replicates per cell.
# Two starts in parallel (base branch size d_s 32 so both fit): median start (attn_medg) vs mean start (attn_meang).
# 3000 steps each, then the LINCS dev eval (dose + compound splits) and the 27-dataset synthetic eval.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
VN=gauss,contam,cauchy,cauchy_hetero,censored_left_heavy,fail_floor,lod_logistic,plate_effect
W='{"contam": 6, "gauss": 3, "student_t3": 2, "plate_effect": 2, "cv_lognormal": 2}'
base='"med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true'
COMMON="--task classification --steps 3000 --mode obs --freeze_backbone --lr 3e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set v3 --scale_max 2.0 --small_k_prob 0.5 --regimes [0.4,0.2,0.2,0.2] --k50_prob 0.2 --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25"
run() {  # run <name> <center>
  ./supervise.sh $1 --rep_cfg "{\"center\": \"$2\", $base}" --noise_weights "$W" $COMMON
  echo "[lane6] $(date -Is) $1 done ($?)" >> artifacts/logs/phase3.log
  $PY real_eval.py $1=artifacts/pretrain/$1/last.ckpt --datasets L1000_dose,CellPainting_dose,L1000,CellPainting >> artifacts/logs/eval_v4.log 2>&1
  $PY obs_eval.py ${1}_e1=artifacts/pretrain/$1/last.ckpt --out artifacts/tx_eval.jsonl --sources main,ext,ext3 --n_ens 1 >> artifacts/logs/eval_v4.log 2>&1
  echo "[lane6] $(date -Is) $1 evaluated" >> artifacts/logs/phase3.log
}
run tx_v4med_A attn_medg &
sleep 120  # stagger the start-up (validation memory peaks)
run tx_v4mean_A attn_meang
wait
