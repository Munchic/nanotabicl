#!/usr/bin/env bash
# Shape learning, round 2 (after the final evaluation of V2b: robust location was the missing signal on cytometry).
#   V4a: V2b + backbone_loc huber (backbone sees Huber + sd columns; step 0 = the Huber + sd wrapper), 64 instances
#   V4b: same, trained with 256 instances per cell (--shape_k 256)
# Sequential (one shape run fits the 16 GB GPU). Dev kill switch as before; the learning check is logged only (it is
# mis-specified for variants that start from explicit summary columns; stated before results). 1000-step screens on
# the 6000-step schedule (resumable). Then dev evaluation on HAR + LINCS and on the five former final sets (now dev,
# flagged as already seen), 8 passes, with and without test-time subsample averaging (256 instances x 4).
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
VN=sh,sh_rep,sh_pop,sh_loc,sh_scale,sh_mix,sh_rare,sh_modesep,sh_tail,sh_state,sh_xor,sh_xorp
base='"center": "mean", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true, "d_s": 64, "n_sab": 2, "isab": 16, "sd_cols": true, "joint": 16, "backbone_loc": "huber"'
COMMON="--task classification --steps 6000 --stop_at 1000 --mode obs --freeze_backbone --lr 3e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set shape --ps {\"xor_pairs\":true} --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25 --dev_real HAR,L1000_dose,CellPainting_dose --dev_every 500 --dev_kill_margin 0.5 --dev_kill_until 1000 --learn_kill_at 1000 --learn_kill_min -1 --learn_kill_settings sh_scale,sh_mix,sh_rare,sh_modesep,sh_tail,sh_xorp"
for spec in "tx_shV4a_A|64" "tx_shV4b_A|256"; do
  name=${spec%%|*}; k=${spec#*|}
  ./supervise.sh $name --rep_cfg "{$base}" $COMMON --shape_k $k
  echo "[shape] $(date -Is) $name screen done" >> artifacts/logs/phase3.log
  for args in "" "--subsample 256 --n_sub 4"; do
    tag=${name}_e8$( [ -n "$args" ] && echo _sub )
    $PY real_eval.py $tag=artifacts/pretrain/$name/last.ckpt --datasets HAR,L1000_dose,CellPainting_dose,HVTN --ks all --n_ens 8 $args --out artifacts/shape_v4_eval.jsonl >> artifacts/logs/eval_shape_v4.log 2>&1
  done
  echo "[shape] $(date -Is) $name evaluated" >> artifacts/logs/phase3.log
done
