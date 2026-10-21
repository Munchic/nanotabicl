#!/usr/bin/env bash
# SHAPE.md item 6 (prepared, NOT yet launched): regression on the shape prior.
#   1. regression headroom on the shape prior (references for the learning kill switch)
#   2. training-free dev baselines on Beijing (wrapper: mean / mean + sd / Huber + sd / sketch)
#   3. V2b's configuration as frozen regression stage A (bank_reg, y_clip 30, xor_pairs prior), screened 1000 steps with
#      the Beijing dev kill switch (-100 x CRPS, margin 0.5 = 0.005 CRPS) and the regression learning check;
#      continued to 6000 steps only if both pass
#   4. dev evaluation with regression calibration (coverage / width / error per reliability tertile)
# Final regression evaluation (PhysioNetLOS, GasConc, Tox21Pot) is run separately, once.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/munchic/miniconda3/envs/ml/bin/python
S=/tmp/claude-1002/-home-munchic-nanotabicl/b53eb5b4-8334-4c18-b331-49e90dd4cfcf/scratchpad
VN=sh,sh_rep,sh_pop,sh_loc,sh_scale,sh_mix,sh_rare,sh_modesep,sh_tail,sh_state,sh_xorp
$PY shape_headroom.py --task regression --settings $VN >> artifacts/logs/shape_reg.log 2>&1
$S/venv/bin/python shape_features.py --datasets Beijing --n_est 8,1 --arms mean,mean_sd,huber_sd,sketch --out artifacts/shape_features_reg_dev.jsonl >> artifacts/logs/shape_reg.log 2>&1
base='"center": "mean", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true, "d_s": 64, "n_sab": 2, "isab": 16, "sd_cols": true, "joint": 16'
COMMON="--task regression --bank artifacts/bank_reg --y_clip 30 --steps 6000 --mode obs --freeze_backbone --lr 3e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set shape --ps {\"xor_pairs\":true} --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25 --dev_real Beijing --dev_every 500 --dev_kill_margin 0.5 --dev_kill_until 1000"
./supervise.sh tx_shR_A --rep_cfg "{$base}" $COMMON --stop_at 1000 --learn_kill_at 1000 --learn_kill_min 0.10 --learn_kill_ref shape_headroom_reg.json --learn_kill_settings sh_scale,sh_mix,sh_rare,sh_modesep,sh_tail,sh_xorp
if ! grep -q "learn_kill\|dev_kill" artifacts/pretrain/tx_shR_A/log.jsonl; then
  ./supervise.sh tx_shR_A --rep_cfg "{$base}" $COMMON
fi
echo "[shape] $(date -Is) tx_shR_A done" >> artifacts/logs/phase3.log
for e in 1 8; do
  $PY real_eval_reg.py tx_shR_A_e$e=artifacts/pretrain/tx_shR_A/last.ckpt --datasets Beijing --ks all --n_ens $e >> artifacts/logs/shape_reg.log 2>&1
done
echo "[shape] $(date -Is) tx_shR_A dev-evaluated" >> artifacts/logs/phase3.log
