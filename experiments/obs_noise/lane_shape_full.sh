#!/usr/bin/env bash
# SHAPE.md full run: the screen winner V2b (sd columns + joint path, xor_pairs prior), resumed at step 1000 and
# continued on the same 6000-step schedule. LINCS dev tasks are now marked paired (wells measured on every feature).
# Dev is logged every 500 steps (kill window already passed; monitored by hand). Then: dev eval with calibration
# (1 and 8 passes) and the synthetic eval; the final held-out evaluation is run separately, once.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
VN=sh,sh_rep,sh_pop,sh_loc,sh_scale,sh_mix,sh_rare,sh_modesep,sh_tail,sh_state,sh_xor,sh_xorp
base='"center": "mean", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true, "d_s": 64, "n_sab": 2, "isab": 16, "sd_cols": true, "joint": 16'
./supervise.sh tx_shV2b_A --rep_cfg "{$base}" --task classification --steps 6000 --mode obs --freeze_backbone --lr 3e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set shape --ps '{"xor_pairs":true}' --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25 --dev_real HAR,L1000_dose,CellPainting_dose --dev_every 500 --dev_kill_margin 0.5 --dev_kill_until 1000
echo "[shape] $(date -Is) tx_shV2b_A full run done" >> artifacts/logs/phase3.log
for e in 1 8; do
  $PY real_eval.py tx_shV2b_A_e$e=artifacts/pretrain/tx_shV2b_A/last.ckpt --datasets HAR,L1000_dose,CellPainting_dose --ks all --n_ens $e --out artifacts/shape_dev_eval.jsonl >> artifacts/logs/eval_shape.log 2>&1
done
echo "[shape] $(date -Is) tx_shV2b_A dev-evaluated" >> artifacts/logs/phase3.log
