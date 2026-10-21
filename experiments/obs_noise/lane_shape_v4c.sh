#!/usr/bin/env bash
# V4c: V4's configuration (backbone sees Huber + sd; joint path) warm-started from V2b's trained branch (6000 steps:
# joint structure learned, the HAR gain), 1000 steps to adapt to the Huber input. Queued after V4b's evaluation.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
VN=sh,sh_rep,sh_pop,sh_loc,sh_scale,sh_mix,sh_rare,sh_modesep,sh_tail,sh_state,sh_xor,sh_xorp
base='"center": "mean", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true, "d_s": 64, "n_sab": 2, "isab": 16, "sd_cols": true, "joint": 16, "backbone_loc": "huber"'
until grep -q "tx_shV4b_A evaluated" artifacts/logs/phase3.log; do sleep 60; done
./supervise.sh tx_shV4c_A --rep_cfg "{$base}" --init artifacts/pretrain/tx_shV2b_A/last.ckpt --task classification --steps 1000 --mode obs --freeze_backbone --lr 1e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 100 --noise_set shape --ps '{"xor_pairs":true}' --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25 --dev_real HAR,L1000_dose,CellPainting_dose --dev_every 500 --dev_kill_margin 0.5 --dev_kill_until 1000 --learn_kill_at 1000 --learn_kill_min -1 --learn_kill_settings sh_scale,sh_mix,sh_rare,sh_modesep,sh_tail,sh_xorp
echo "[shape] $(date -Is) tx_shV4c_A done" >> artifacts/logs/phase3.log
$PY real_eval.py tx_shV4c_A_e8=artifacts/pretrain/tx_shV4c_A/last.ckpt --datasets HAR,L1000_dose,CellPainting_dose,HVTN --ks all --n_ens 8 --out artifacts/shape_v4_eval.jsonl >> artifacts/logs/eval_shape_v4.log 2>&1
echo "[shape] $(date -Is) tx_shV4c_A evaluated" >> artifacts/logs/phase3.log
