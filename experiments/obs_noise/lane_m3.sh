#!/usr/bin/env bash
# M3 = M1 + sd_learn=2: the explicit block becomes [learned location, weighted scale, L1, L2] per replicated feature
# (4 columns instead of 2). Tests the hypothesis behind M1's one real loss (HAR -1.6 vs Huber+sd, -4.5 vs sketch: the
# sketch's advantage is exactly "more slots"). Step 0 == M1's step 0 (learned columns zero-init).
# Screen only: 1000 steps + 8-pass VAL eval, compared at matched steps with M1's screen (mean 0.753; HVTN +0.3,
# L1000 +0.5, CellPainting +0.8, HAR -1.6 vs the best hand arm). The winner then gets the 6000-step run.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
VN=sh,sh_rep,sh_pop,sh_loc,sh_scale,sh_mix,sh_rare,sh_modesep,sh_tail,sh_state,sh_xor,sh_xorp
VAL=HAR,L1000_dose,CellPainting_dose,HVTN
PSJ='{"xor_pairs":true}'
base='"center": "mean", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true, "d_s": 64, "n_sab": 2, "isab": 16, "sd_cols": true, "backbone_loc": "learned_m", "rep_emb": false, "sd_learn": 2'
COM="--task classification --steps 6000 --stop_at 1000 --mode obs --freeze_backbone --lr 3e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set shape --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25 --dev_real $VAL --dev_every 500 --dev_kill_margin 0.5 --dev_kill_until 1000 --learn_kill_at 1000 --learn_kill_min -1 --learn_kill_settings sh_scale,sh_mix,sh_rare,sh_modesep,sh_tail,sh_xorp"

eval -u ${PY} devsplit.py --check "$VAL"
./supervise.sh tx_m3_A --rep_cfg "{$base}" $COM --ps "$PSJ" --shape_k 64
echo "[plan2] $(date -Is) tx_m3_A screen done" >> artifacts/logs/phase3.log
$PY real_eval.py tx_m3_A_e8=artifacts/pretrain/tx_m3_A/last.ckpt --datasets $VAL --ks all --n_ens 8 \
    --out artifacts/plan2_eval.jsonl >> artifacts/logs/eval_plan2.log 2>&1
$PY calib_summary.py artifacts/plan2_eval.jsonl >> artifacts/logs/eval_plan2.log 2>&1
echo "[plan2] $(date -Is) tx_m3_A val-evaluated" >> artifacts/logs/phase3.log
