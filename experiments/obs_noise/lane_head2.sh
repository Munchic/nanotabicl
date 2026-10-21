#!/usr/bin/env bash
# Bitter-lesson queue, part 2 (after lane_head.sh): VAL references (CPU, no GPU) + the same v7 head in regression
# (Beijing dev). The single final --allow_test read is gated on FULL_DONE && PROMOTE (written by lane_head_full.sh),
# so test sets are never read during development.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
VPY=/tmp/claude-1002/-home-munchic-nanotabicl/b53eb5b4-8334-4c18-b331-49e90dd4cfcf/scratchpad/venv/bin/python
PSV='{"p_lod":0.6,"p_cens_r":0.4,"p_mcar":0.3,"p_well":0.4,"p_plate":0.3,"p_fail":0.25,"sub":1,"p_spec":0.6,"xor":0.3}'
VN=v7,v7_clean,v7_tail,v7_xorp,v7_cens,v7_mnar,v7_well,v7_meas
VAL=HAR,L1000_dose,CellPainting_dose,HVTN
base='"center": "attn", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true, "d_s": 64, "n_sab": 2, "isab": 16, "joint": 16, "flag_emb": true, "miss_emb": true'
COM="--mode obs --freeze_backbone --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set v7 --ps $PSV --val_every 250 --val_blocks 4 --log_every 25 --dev_every 500 --dev_kill_margin 0.5 --dev_kill_until 1000"

# 1. VAL references (CPU only, no GPU wait)
if [ -s artifacts/val_ref.jsonl ]; then
  echo "[head2] $(date -Is) val references already present: skipped" >> artifacts/logs/phase3.log
elif [ -x "$VPY" ]; then
  $VPY shape_features.py --datasets $VAL --n_est 8,1 --arms mean,mean_sd,huber_sd,sketch --device cpu \
      --out artifacts/val_ref.jsonl >> artifacts/logs/val_ref.log 2>&1
  $VPY calib_summary.py artifacts/val_ref.jsonl >> artifacts/logs/val_ref.log 2>&1
  echo "[head2] $(date -Is) val references done" >> artifacts/logs/phase3.log
else
  echo "[head2] $(date -Is) wrapper venv missing: val references SKIPPED" >> artifacts/logs/phase3.log
fi

# 2. Same v7 head in regression (Beijing dev; CRPS / PIT). Screen only.
until grep -q "tx_headM_A val-evaluated" artifacts/logs/phase3.log; do sleep 120; done
./supervise.sh tx_headM_reg_A --rep_cfg "{$base}" --task regression --bank artifacts/bank_reg --y_clip 30 \
    --steps 6000 --stop_at 1000 --lr 3e-4 $COM --val_noises $VN --dev_real Beijing --shape_k 128
echo "[head2] $(date -Is) tx_headM_reg_A screen done" >> artifacts/logs/phase3.log
for e in 1 8; do
  $PY real_eval_reg.py tx_headM_reg_A_e$e=artifacts/pretrain/tx_headM_reg_A/last.ckpt --datasets Beijing --ks all --n_ens $e \
      >> artifacts/logs/eval_head2.log 2>&1
done
echo "[head2] $(date -Is) tx_headM_reg_A val-evaluated" >> artifacts/logs/phase3.log

# 3. The one permitted test read, only after the promoted full run finishes.
if [ -f artifacts/FULL_DONE ] && [ -f artifacts/PROMOTE ]; then
  CK=$(cat artifacts/PROMOTE)
  $PY real_eval.py $CK=artifacts/pretrain/${CK}/last.ckpt --allow_test --datasets NK,PreE,Musk,PhysioNet,GasDrift,Tox21,UPS1_low,CPTAC \
      --ks all --n_ens 8 --out artifacts/final_test.jsonl >> artifacts/logs/final_test.log 2>&1
  $PY real_eval_reg.py ${CK}_reg=artifacts/pretrain/${CK}_reg/last.ckpt --allow_test --datasets GasConc,Tox21Pot,PhysioNetLOS \
      --ks all --n_ens 8 >> artifacts/logs/final_test.log 2>&1
  $PY calib_summary.py artifacts/final_test.jsonl >> artifacts/logs/final_test.log 2>&1
  echo "[head2] $(date -Is) FINAL test read done ($CK)" >> artifacts/logs/phase3.log
fi
