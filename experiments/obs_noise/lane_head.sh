#!/usr/bin/env bash
# Bitter-lesson head (v7): continuous observation prior + a general set-encoder head. No hand-crafted summary columns,
# no Huber/sd start, no per-cell gates: the model sees raw replicates + mask + user-declared semantics (flags, n = 0,
# paired) and learns location / scale / shape / measurement from the prior. The only learned channel is the general
# additive embedding + the learned attention location + the output-side temperature head.
# This is the screen (1000 steps); the winner's 6000-step run is lane_head_full.sh (gated on artifacts/PROMOTE).
# V4a / V4b / V4c remain the hand-crafted baseline this head must beat. VAL sets only.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
PSV='{"p_lod":0.6,"p_cens_r":0.4,"p_mcar":0.3,"p_well":0.4,"p_plate":0.3,"p_fail":0.25,"sub":1,"p_spec":0.6,"xor":0.3}'
VN=v7,v7_clean,v7_tail,v7_xorp,v7_cens,v7_mnar,v7_well,v7_meas
DEV=HAR,L1000_dose,CellPainting_dose,HVTN
base='"center": "attn", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true, "d_s": 64, "n_sab": 2, "isab": 16, "joint": 16, "flag_emb": true, "miss_emb": true'
COMMON="--task classification --steps 6000 --stop_at 1000 --mode obs --freeze_backbone --lr 3e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set v7 --ps $PSV --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25 --dev_real $DEV --dev_every 500 --dev_kill_margin 0.5 --dev_kill_until 1000"

# V4b / V4c were killed (Huber+sd start is flat: V4a/V4b val Deltaacc ~0). The bar is the locked val_ref.jsonl.
eval -u ${PY} devsplit.py --check "$DEV"
./supervise.sh tx_headM_A --rep_cfg "{$base}" $COMMON --shape_k 128
echo "[head] $(date -Is) tx_headM_A screen done" >> artifacts/logs/phase3.log
for args in "" "--subsample 256 --n_sub 4"; do
  $PY real_eval.py tx_headM_A_e8$( [ -n "$args" ] && echo _sub )=artifacts/pretrain/tx_headM_A/last.ckpt \
      --datasets $DEV --ks all --n_ens 8 $args --out artifacts/head_eval.jsonl >> artifacts/logs/eval_head.log 2>&1
done
$PY calib_summary.py artifacts/head_eval.jsonl >> artifacts/logs/eval_head.log 2>&1
echo "[head] $(date -Is) tx_headM_A val-evaluated (test sets untouched)" >> artifacts/logs/phase3.log
