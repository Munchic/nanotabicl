#!/usr/bin/env bash
# SHAPE.md screen, XOR fix: V2 (joint path) did not learn the joint signal because the original XOR tables code only
# one column pair (little joint-only headroom). Prior option xor_pairs: every column pair of an XOR table is XOR-coded
# (marginals uninformative; sh_xorp headroom: mean 0.589, sketch 0.618, oracle 0.801). Matched pair on that prior:
# V1b (sd columns) vs V2b (+ joint path), sequential after V3. Same schedule / kill switches; the learning check also
# covers sh_xorp.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VN=sh,sh_rep,sh_pop,sh_loc,sh_scale,sh_mix,sh_rare,sh_modesep,sh_tail,sh_state,sh_xor,sh_xorp
base='"center": "mean", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true, "dist_emb": true, "temp_head": true, "col_ctx": true, "d_s": 64, "n_sab": 2, "isab": 16, "sd_cols": true'
COMMON="--task classification --steps 6000 --stop_at 1000 --mode obs --freeze_backbone --lr 3e-4 --arch tabicl --batch_size 32 --micro_batch 2 --warmup 200 --noise_set shape --ps {\"xor_pairs\":true} --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25 --dev_real HAR,L1000_dose,CellPainting_dose --dev_every 500 --dev_kill_margin 0.5 --dev_kill_until 1000 --learn_kill_at 1000 --learn_kill_min 0.10 --learn_kill_settings sh_scale,sh_mix,sh_rare,sh_modesep,sh_tail,sh_xorp"
until grep -q "tx_shV3_A screen done" artifacts/logs/phase3.log; do sleep 60; done
for spec in "tx_shV1b_A|" "tx_shV2b_A|, \"joint\": 16"; do
  name=${spec%%|*}; extra=${spec#*|}
  ./supervise.sh $name --rep_cfg "{$base$extra}" $COMMON
  echo "[shape] $(date -Is) $name screen done" >> artifacts/logs/phase3.log
done
