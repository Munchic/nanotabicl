#!/usr/bin/env bash
# Exp 9: training prior tilted toward the real LINCS noise shape (v3 + weights: contaminated Gaussian 6, Gaussian 3,
# Student-t3 / plate effect / constant CV 2, all other modes 1). Classification: stage A (capacity config, per-cell
# median start; frozen backbone) -> real (LINCS dose split) + synthetic eval -> stage B from it -> evals.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
VN=gauss,contam,cauchy,cauchy_hetero,censored_left_heavy,fail_floor,lod_logistic,plate_effect
W='{"contam": 6, "gauss": 3, "student_t3": 2, "plate_effect": 2, "cv_lognormal": 2}'
CFG='{"center": "attn_medg", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true, "d_s": 64, "n_sab": 2, "_args": {"noise_set": "v3", "noise_weights": {"contam": 6, "gauss": 3, "student_t3": 2, "plate_effect": 2, "cv_lognormal": 2}, "val_noises": "'$VN'"}}'
log() { echo "[lane5] $(date -Is) $*" >> artifacts/logs/phase3.log; }
# reference for the new validation set (contam included): TabICLv2 + median, step 0
$PY obs_train.py --name tx_median0p_v3 --arch tabicl --mode median --steps 0 --noise_set v3 --val_blocks 4 --val_noises $VN \
  --micro_batch 2 >> artifacts/logs/train_tx_median0p_v3.log 2>&1
./supervise.sh tx_obs_real_A --task classification --rep_cfg "$CFG" --steps 6000 --mode obs --freeze_backbone --lr 3e-4 \
  --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set v3 --regimes [0.4,0.2,0.2,0.2] --k50_prob 0.2 \
  --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25
log "tx_obs_real_A done ($?)"
$PY real_eval.py tx_obs_real_A=artifacts/pretrain/tx_obs_real_A/last.ckpt --datasets L1000_dose,CellPainting_dose,L1000,CellPainting >> artifacts/logs/eval_real.log 2>&1
$PY obs_eval.py tx_obs_real_A_e1=artifacts/pretrain/tx_obs_real_A/last.ckpt --out artifacts/tx_eval.jsonl --sources main,ext,ext3 --n_ens 1 >> artifacts/logs/eval_real.log 2>&1
log "tx_obs_real_A evaluated"
OBS_NAME=tx_obs_realB ./phase_txB.sh classification tx_obs_real_A obs_only
$PY real_eval.py tx_obs_realB=artifacts/pretrain/tx_obs_realB/last.ckpt --datasets L1000_dose,CellPainting_dose,L1000,CellPainting >> artifacts/logs/eval_real.log 2>&1
$PY obs_eval.py tx_obs_realB_e1=artifacts/pretrain/tx_obs_realB/last.ckpt --out artifacts/tx_eval.jsonl --sources main,ext,ext3 --n_ens 1 >> artifacts/logs/eval_real.log 2>&1
log "tx_obs_realB evaluated"
