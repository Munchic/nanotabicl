#!/usr/bin/env bash
# Exp 9, v5 prior with a GENERIC branch (user direction: less hand-built bias in the model, let the prior do the work):
# plain learned center (center 'attn': attention over the replicates, zero-init scores = the mean), no Huber start, no
# gates, no column statistics. Prior: mostly Gaussian / contaminated Gaussian, row heteroscedasticity, plate effects;
# noise logU(0.2, 1.5) x sd; 80% of tables with 2-5 replicates; all other modes kept at low weight. Frozen backbone,
# base branch size, 4000 steps. LINCS dev = kill switch only (steps 500 / 1000, > 0.5 points below step 0 -> stop).
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
VN=gauss,contam,gauss_rowhet,cauchy,censored_left_heavy,fail_floor,plate_effect
W='{"contam": 6, "gauss": 4, "gauss_rowhet": 3, "plate_effect": 3, "gauss_valhet": 2, "cv_lognormal": 2, "student_t3": 1, "censored": 0.3, "cauchy": 0.3, "cauchy_hetero": 0.3, "cauchy_valhet": 0.3, "censored_left": 0.3, "censored_left_heavy": 0.3, "fail_floor": 0.3, "fail_shift": 0.3, "lod_logistic": 0.3, "saturation": 0.3}'
CFG='{"center": "attn", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true}'
./supervise.sh tx_v5gen_A --rep_cfg "$CFG" --noise_weights "$W" --task classification --steps 4000 --mode obs --freeze_backbone \
  --lr 3e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set v5 --scale_min 0.2 --scale_max 1.5 \
  --small_k_prob 0.8 --regimes [0.4,0.2,0.2,0.2] --k50_prob 0.2 --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25 \
  --dev_real L1000_dose,CellPainting_dose --dev_every 500 --dev_kill_margin 0.5 --dev_kill_until 1000
echo "[lane9] $(date -Is) tx_v5gen_A done ($?)" >> artifacts/logs/phase3.log
$PY obs_eval.py tx_v5gen_A_e1=artifacts/pretrain/tx_v5gen_A/last.ckpt --out artifacts/tx_eval.jsonl --sources main,ext,ext3 --n_ens 1 >> artifacts/logs/eval_v5.log 2>&1
$PY real_eval.py tx_v5gen_A=artifacts/pretrain/tx_v5gen_A/last.ckpt --datasets L1000_dose,CellPainting_dose >> artifacts/logs/eval_v5.log 2>&1
echo "[lane9] $(date -Is) tx_v5gen_A evaluated (synthetic + LINCS dev; held-out sets not run)" >> artifacts/logs/phase3.log
