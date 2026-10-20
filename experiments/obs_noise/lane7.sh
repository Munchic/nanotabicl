#!/usr/bin/env bash
# Exp 9: Huber start (center 'attn_huberg': step 0 == TabICLv2 + Huber imputation with a column-pooled scale) on the v4
# prior (synthetic only), base branch size, 3000 steps; plus the TabICLv2 + Huber imputation reference (step 0).
# Evaluations: LINCS dev (dose + compound splits) and the 27 synthetic datasets. Gas drift / Tox21 stay untouched.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
VN=gauss,contam,cauchy,cauchy_hetero,censored_left_heavy,fail_floor,lod_logistic,plate_effect
W='{"contam": 6, "gauss": 3, "student_t3": 2, "plate_effect": 2, "cv_lognormal": 2}'
CFG='{"center": "attn_huberg", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true}'
log() { echo "[lane7] $(date -Is) $*" >> artifacts/logs/phase3.log; }
$PY obs_train.py --name tx_huber0p --arch tabicl --mode huber --steps 0 --noise_set v3 --val_blocks 4 --val_noises $VN --micro_batch 2 >> artifacts/logs/train_tx_huber0p.log 2>&1
$PY obs_eval.py tx_huber0p_e1=artifacts/pretrain/tx_huber0p/last.ckpt --out artifacts/tx_eval.jsonl --sources main,ext,ext3 --n_ens 1 >> artifacts/logs/eval_huber.log 2>&1
$PY real_eval.py tx_huber0p=artifacts/pretrain/tx_huber0p/last.ckpt --datasets L1000_dose,CellPainting_dose,L1000,CellPainting >> artifacts/logs/eval_huber.log 2>&1
log "tx_huber0p reference evaluated"
./supervise.sh tx_v4huber_A --task classification --rep_cfg "$CFG" --noise_weights "$W" --steps 3000 --mode obs --freeze_backbone \
  --lr 3e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set v3 --scale_max 2.0 --small_k_prob 0.5 \
  --regimes [0.4,0.2,0.2,0.2] --k50_prob 0.2 --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25
log "tx_v4huber_A done ($?)"
$PY real_eval.py tx_v4huber_A=artifacts/pretrain/tx_v4huber_A/last.ckpt --datasets L1000_dose,CellPainting_dose,L1000,CellPainting >> artifacts/logs/eval_huber.log 2>&1
$PY obs_eval.py tx_v4huber_A_e1=artifacts/pretrain/tx_v4huber_A/last.ckpt --out artifacts/tx_eval.jsonl --sources main,ext,ext3 --n_ens 1 >> artifacts/logs/eval_huber.log 2>&1
log "tx_v4huber_A evaluated"
