#!/usr/bin/env bash
# Exp 8 stage B: unfreeze the TabICLv2 backbone, Muon (TabICLv2's optimizer) at a low lr, branch on AdamW at its stage-A
# lr; matched baseline = TabICLv2 + median imputation continued with exactly the same optimizer, steps and data.
# Usage: phase_txB.sh <classification|regression> <stage-A run to start from, e.g. tx_obs_colmed_A>
#        phase_txB.sh <classification|regression> median     # only the matched baseline (it needs no stage A)
# The baseline is skipped when it already finished, so it can be run early.
cd "$(dirname "$0")"
VN=gauss,cauchy,cauchy_hetero,censored_left_heavy,fail_floor,lod_logistic,plate_effect
if [ "$1" = regression ]; then T="--task regression --bank artifacts/bank_reg --y_clip 30"; s=_reg; else T="--task classification"; s=; fi
COMMON="--arch tabicl $T --optim muon --lr ${LR:-1.5e-5} --steps ${STEPS:-4000} --warmup 200 --batch_size 32 --micro_batch 4 --noise_set v2 --regimes [0.4,0.2,0.2,0.2] --k50_prob 0.2 --val_every 250 --val_blocks 4 --val_noises $VN --log_every 25"
median() {
  if grep -q "tx_median_B$s exited with 0" artifacts/logs/train_tx_median_B$s.log 2>/dev/null; then return; fi
  ./supervise.sh tx_median_B$s --mode median $COMMON
  echo "[phase_txB] $(date -Is) tx_median_B$s done ($?)" >> artifacts/logs/phase3.log
}
if [ "$2" = median ]; then
  median
  exit
fi
INIT=artifacts/pretrain/$2/last.ckpt
CFG=$(python3 -c "import json; print(json.load(open('artifacts/pretrain/$2/args.json'))['rep_cfg'])")  # the stage-A branch config
median &
./supervise.sh tx_obs_B$s --mode obs --init $INIT --rep_cfg "$CFG" --rep_lr_mult ${REP_MULT:-20} $COMMON
echo "[phase_txB] $(date -Is) tx_obs_B$s done ($?)" >> artifacts/logs/phase3.log
wait
