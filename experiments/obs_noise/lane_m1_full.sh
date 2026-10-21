#!/usr/bin/env bash
# Full 6000-step run of the PLAN 2 step-1 winner (M1: backbone_loc='learned_m', rep_emb=False), resumed from its
# 1000-step screen checkpoint (same RNG stream, no --stop_at). Then the 8-pass VAL eval and artifacts/FULL_DONE,
# which releases the single final --allow_test read in lane_head2.sh.
# Why M1: 8-pass VAL mean 0.753 (= Huber+sd) with wins on CellPainting +0.8 / HVTN +0.3 / L1000 +0.5 vs the best hand
# arm, and dev_real still rising at its screen end (72.77 -> 74.70 -> 78.07). M2 (same columns + the learned branch)
# went 72.36 -> 72.08 at step 500, i.e. the additive branch adds nothing on top of the explicit columns.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
VN=sh,sh_rep,sh_pop,sh_loc,sh_scale,sh_mix,sh_rare,sh_modesep,sh_tail,sh_state,sh_xor,sh_xorp
VAL=HAR,L1000_dose,CellPainting_dose,HVTN
PSJ='{"xor_pairs":true}'
base='"center": "mean", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true, "d_s": 64, "n_sab": 2, "isab": 16, "sd_cols": true, "backbone_loc": "learned_m", "rep_emb": false'
COM="--task classification --steps 6000 --mode obs --freeze_backbone --lr 3e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set shape --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25 --dev_real $VAL --dev_every 500 --dev_kill_margin 0.5 --dev_kill_until 1000 --learn_kill_at 1000 --learn_kill_min -1 --learn_kill_settings sh_scale,sh_mix,sh_rare,sh_modesep,sh_tail,sh_xorp"

./supervise.sh tx_m1_A --rep_cfg "{$base}" $COM --ps "$PSJ" --shape_k 64
echo "[m1_full] $(date -Is) tx_m1_A full run done" >> artifacts/logs/phase3.log
for args in "" "--subsample 256 --n_sub 4"; do
  $PY real_eval.py tx_m1_A_e8$( [ -n "$args" ] && echo _sub )=artifacts/pretrain/tx_m1_A/last.ckpt \
      --datasets $VAL --ks all --n_ens 8 $args --out artifacts/plan2_eval.jsonl >> artifacts/logs/eval_plan2.log 2>&1
done
$PY calib_summary.py artifacts/plan2_eval.jsonl >> artifacts/logs/eval_plan2.log 2>&1
touch artifacts/FULL_DONE
echo "[m1_full] $(date -Is) tx_m1_A val-evaluated; FULL_DONE written" >> artifacts/logs/phase3.log
