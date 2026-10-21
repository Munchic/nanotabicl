#!/usr/bin/env bash
# Exp 10b: every v6 run (attn center) hit the LINCS kill switch. Decomposition (decomp_lincs.py): the LEARNED CENTER
# carries the damage (center only ~= full: 64.1 / 62.7), while the embedding alone beats mean imputation in every v6 run
# (65.7-65.9 vs 65.4). So: center 'mean' (backbone sees the plain cell mean, no learned location), everything learned
# goes through the additive embedding. Default v6 prior (p_tail 0.5). Large branch + base-size control. Kill switch kept.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
VN=v6,v6_gauss,v6_heavy,v6_paired,v6_fail,v6_mnar,v6_lod_flag,v6_lod_noflag,gauss,contam,cauchy,censored_left_heavy,fail_floor
base='"center": "mean", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true, "flag_emb": true, "miss_emb": true'
COMMON="--task classification --steps 6000 --mode obs --freeze_backbone --lr 3e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set v6 --k50_prob 0.1 --val_every 500 --val_blocks 4 --val_noises $VN --log_every 25 --dev_real L1000_dose,CellPainting_dose --dev_every 500 --dev_kill_margin 0.5 --dev_kill_until 1000"
run() {  # run <name> <extra rep_cfg json>
  ./supervise.sh $1 --rep_cfg "{$base$2}" $COMMON
  echo "[lane12] $(date -Is) $1 done ($?)" >> artifacts/logs/phase3.log
  for e in 1 8; do
    $PY obs_eval.py ${1}_e$e=artifacts/pretrain/$1/last.ckpt --out artifacts/tx_eval.jsonl --sources main,ext,ext3 --n_ens $e >> artifacts/logs/eval_v6.log 2>&1
    $PY real_eval.py ${1}_e$e=artifacts/pretrain/$1/last.ckpt --datasets L1000_dose,CellPainting_dose --ks all --n_ens $e >> artifacts/logs/eval_v6.log 2>&1
  done
  echo "[lane12] $(date -Is) $1 evaluated (synthetic + LINCS dev, 1 and 8 passes; held-out sets not run)" >> artifacts/logs/phase3.log
}
run tx_v6m_A '' &
sleep 120
run tx_v6mcap_A ', "d_s": 64, "n_sab": 2'
wait
