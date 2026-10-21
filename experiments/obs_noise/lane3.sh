#!/usr/bin/env bash
# After classification stage B finishes: evaluate it (27 datasets, fp32, 1 and 8 passes), then train the full-length
# capacity stage A for classification (d_s 64, 2 SAB, per-cell gate; 6000 steps) as the base for a second stage B.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
VN=gauss,cauchy,cauchy_hetero,censored_left_heavy,fail_floor,lod_logistic,plate_effect
until grep -q "tx_obs_B done" artifacts/logs/phase3.log; do sleep 60; done
for n in 1 8; do
  $PY obs_eval.py tx_obs_B_e$n=artifacts/pretrain/tx_obs_B/last.ckpt --out artifacts/tx_eval.jsonl --sources main,ext,ext2,ext3 --n_ens $n >> artifacts/logs/eval_tx_obs_B.log 2>&1
done
echo "[lane3] $(date -Is) tx_obs_B evaluated" >> artifacts/logs/phase3.log
CFG=$(cat reg_cfg.json)  # the capacity config
./supervise.sh tx_obs_cap_A --task classification --rep_cfg "$CFG" --steps 6000 --mode obs --freeze_backbone --lr 3e-4 \
  --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set v2 --regimes [0.4,0.2,0.2,0.2] --k50_prob 0.2 \
  --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25
echo "[lane3] $(date -Is) tx_obs_cap_A done ($?)" >> artifacts/logs/phase3.log
