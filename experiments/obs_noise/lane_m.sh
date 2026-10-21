#!/usr/bin/env bash
# PLAN 2 step 1: learned column-aware estimator (backbone_loc='learned_m'), explicit location + scale columns.
# M1: learned_m columns only (branch off, rep_emb=false). M2: M1 + the V2b branch (joint path, additive embedding),
# warm-started from tx_shV2b_A. V2b settings: shape prior with xor_pairs, frozen backbone, 1000-step screens,
# dev kill on VAL. Serialized on the GPU after the v7 head queue (one shape/head run fits at a time).
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
PSJ='{"xor_pairs":true}'  # passed as its own argument; inner quotes would close the COM string
VN=sh,sh_rep,sh_pop,sh_loc,sh_scale,sh_mix,sh_rare,sh_modesep,sh_tail,sh_state,sh_xor,sh_xorp
DEV=HAR,L1000_dose,CellPainting_dose,HVTN
base='"center": "mean", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true, "d_s": 64, "n_sab": 2, "isab": 16, "sd_cols": true, "backbone_loc": "learned_m"'
COM="--task classification --steps 6000 --stop_at 1000 --mode obs --freeze_backbone --lr 3e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set shape --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25 --dev_real $DEV --dev_every 500 --dev_kill_margin 0.5 --dev_kill_until 1000 --learn_kill_at 1000 --learn_kill_min -1 --learn_kill_settings sh_scale,sh_mix,sh_rare,sh_modesep,sh_tail,sh_xorp"

# headM_reg_A was killed (launched with center=attn, uninterpretable). Gate on headM's completed eval instead.
until grep -q "tx_headM_A val-evaluated" artifacts/logs/phase3.log; do sleep 60; done

# M1: learned_m columns only, no additive embedding
./supervise.sh tx_m1_A --rep_cfg "{$base, \"rep_emb\": false}" $COM --ps "$PSJ" --shape_k 64
echo "[plan2] $(date -Is) tx_m1_A screen done" >> artifacts/logs/phase3.log
$PY real_eval.py tx_m1_A_e8=artifacts/pretrain/tx_m1_A/last.ckpt --datasets $DEV --ks all --n_ens 8 \
    --out artifacts/plan2_eval.jsonl >> artifacts/logs/eval_plan2.log 2>&1
echo "[plan2] $(date -Is) tx_m1_A val-evaluated" >> artifacts/logs/phase3.log

# M2: M1 + V2b branch, warm-started from V2b
./supervise.sh tx_m2_A --rep_cfg "{$base, \"joint\": 16}" --init artifacts/pretrain/tx_shV2b_A/last.ckpt $COM --ps "$PSJ" --shape_k 64
echo "[plan2] $(date -Is) tx_m2_A screen done" >> artifacts/logs/phase3.log
$PY real_eval.py tx_m2_A_e8=artifacts/pretrain/tx_m2_A/last.ckpt --datasets $DEV --ks all --n_ens 8 \
    --out artifacts/plan2_eval.jsonl >> artifacts/logs/eval_plan2.log 2>&1
echo "[plan2] $(date -Is) tx_m2_A val-evaluated" >> artifacts/logs/phase3.log

$PY calib_summary.py artifacts/plan2_eval.jsonl >> artifacts/logs/eval_plan2.log 2>&1
