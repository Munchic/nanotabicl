#!/usr/bin/env bash
# Phase 3 (Exp 5): like phase2.sh but trained with count regimes (row / column / context!=query) and up-to-50-replicate
# batches, for classification or regression. Usage: TASK=regression INIT=... P=r3_ phase3.sh <arm>...
cd "$(dirname "$0")"
TASK=${TASK:-classification}
if [ "$TASK" = regression ]; then BANK="--bank artifacts/bank_reg --y_clip 30"; else BANK=""; fi
INIT=${INIT:-artifacts/pretrain/stock/last.ckpt}
P=${P:-p3_}
REG=${REGIMES:-[0.4,0.2,0.2,0.2]}
COMMON="--task $TASK $BANK --init $INIT --steps ${STEPS:-4000} --batch_size 32 --lr 3e-4 --warmup 200 --val_every 1000 --regimes $REG --k50_prob ${K50:-0.2} --noise_set ${NOISESET:-v1}"
# Exp 7: LAYOUTS='[0.25,0.25,0.25,0.25]' -> heterogeneous per-column / per-row-group observation models
[ -n "$LAYOUTS" ] && COMMON="$COMMON --layouts $LAYOUTS"
[ -n "$EXTRA_N_BLOCKS" ] && COMMON="$COMMON --n_blocks $EXTRA_N_BLOCKS"  # e.g. 8000: the extended bank
BEST=${BEST:-'{"center": "attn", "med_feat": true, "std": "medmad", "zclip": 10.0}'}
with_e1() { python3 -c "import json,sys; c=json.loads(sys.argv[1]); c['single_emb']='scaled'; print(json.dumps(c))" "$1"; }
with_kv() { python3 -c "import json,sys; c=json.loads(sys.argv[1]); [c.__setitem__(k, True) for k in sys.argv[2:]]; print(json.dumps(c))" "$@"; }
for arm in "$@"; do
  case $arm in
    mean)      ./supervise.sh $P$arm --mode mean   $COMMON --micro_batch 16 ;;
    median)    ./supervise.sh $P$arm --mode median $COMMON --micro_batch 16 ;;
    stats)     ./supervise.sh $P$arm --mode stats  $COMMON --micro_batch 4 ;;
    obs)       ./supervise.sh $P$arm --mode obs    $COMMON --micro_batch 8 --rep_cfg "$BEST" ;;
    obs_e1)    ./supervise.sh $P$arm --mode obs    $COMMON --micro_batch 8 --rep_cfg "$(with_e1 "$BEST")" ;;
    obs_temp)      ./supervise.sh $P$arm --mode obs    $COMMON --micro_batch 8 --rep_cfg "$(with_kv "$BEST" temp_head)" ;;
    obs_dist_temp) ./supervise.sh $P$arm --mode obs    $COMMON --micro_batch 4 --rep_cfg "$(with_kv "$BEST" dist_emb temp_head)" ;;
    obs_dist)      ./supervise.sh $P$arm --mode obs    $COMMON --micro_batch 4 --rep_cfg "$(with_kv "$BEST" dist_emb)" ;;
    obs_dist_temp_v2s) ./supervise.sh $P$arm --mode obs $COMMON --micro_batch 4 --rep_cfg "$(with_kv "$BEST" dist_emb temp_head | python3 -c "import json,sys; c=json.load(sys.stdin); c['pool']='v2s'; print(json.dumps(c))")" ;;
    obs_dist_temp_row) ./supervise.sh $P$arm --mode obs $COMMON --micro_batch 4 --rep_cfg "$(with_kv "$BEST" dist_emb temp_head row_emb)" ;;
    obs_dist_temp_aux) ./supervise.sh $P$arm --mode obs $COMMON --micro_batch 4 --aux_w ${AUX_W:-0.3} --rep_cfg "$(with_kv "$BEST" dist_emb temp_head aux)" ;;
    obs_dist_temp_auxq) ./supervise.sh $P$arm --mode obs $COMMON --micro_batch 4 --aux_w ${AUX_W:-0.3} --rep_cfg "$(with_kv "$BEST" dist_emb temp_head aux | python3 -c "import json,sys; c=json.load(sys.stdin); c['aux_center']=False; print(json.dumps(c))")" ;;
    obs_dist_temp_col) ./supervise.sh $P$arm --mode obs $COMMON --micro_batch 4 --rep_cfg "$(with_kv "$BEST" dist_emb temp_head col_ctx)" ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  echo "[phase3] $(date -Is) $P$arm done ($?)" >> artifacts/logs/phase3.log
done
