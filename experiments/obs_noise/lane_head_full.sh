#!/usr/bin/env bash
# Full 6000-step run of the promoted v7 head (artifacts/PROMOTE), gated on the manual promotion decision after the
# 1000-step screens are compared on VAL + synthetic. Same config as lane_head.sh, minus --stop_at. Then VAL eval and
# write artifacts/FULL_DONE, which releases the single final --allow_test read in lane_head2.sh.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
PSV='{"p_lod":0.6,"p_cens_r":0.4,"p_mcar":0.3,"p_well":0.4,"p_plate":0.3,"p_fail":0.25,"sub":1,"p_spec":0.6,"xor":0.3}'
VN=v7,v7_clean,v7_tail,v7_xorp,v7_cens,v7_mnar,v7_well,v7_meas
VAL=HAR,L1000_dose,CellPainting_dose,HVTN
base='"center": "attn", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true, "d_s": 64, "n_sab": 2, "isab": 16, "joint": 16, "flag_emb": true, "miss_emb": true'
COMMON="--task classification --steps 6000 --mode obs --freeze_backbone --lr 3e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set v7 --ps $PSV --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25 --dev_real $VAL --dev_every 500 --dev_kill_margin 0.5 --dev_kill_until 1000"

until [ -f artifacts/PROMOTE ]; do sleep 300; done
CK=$(cat artifacts/PROMOTE)
./supervise.sh $CK --rep_cfg "{$base}" $COMMON --shape_k 128
echo "[head_full] $(date -Is) $CK full run done" >> artifacts/logs/phase3.log
for args in "" "--subsample 256 --n_sub 4"; do
  $PY real_eval.py ${CK}_e8$( [ -n "$args" ] && echo _sub )=artifacts/pretrain/$CK/last.ckpt \
      --datasets $VAL --ks all --n_ens 8 $args --out artifacts/head_eval.jsonl >> artifacts/logs/eval_head.log 2>&1
done
$PY calib_summary.py artifacts/head_eval.jsonl >> artifacts/logs/eval_head.log 2>&1
touch artifacts/FULL_DONE
echo "[head_full] $(date -Is) $CK val-evaluated; FULL_DONE written" >> artifacts/logs/phase3.log
