#!/usr/bin/env bash
# Exp 7 runs: auxiliary denoising loss (aux) and heterogeneous noise layouts (L), classification + regression.
# Same budget / init / mixture as p5 / r5 (8000 steps from stock, v2 modes, count regimes, 20% K <= 50 batches).
cd "$(dirname "$0")"
export STEPS=8000 NOISESET=v2 BEST='{"center": "attn", "med_feat": true, "std": "medmad", "zclip": 10.0, "robust_feats": true}'
L='[0.25,0.25,0.25,0.25]'
lane() {  # lane <a|b> <task>
  if [ "$2" = regression ]; then T=regression; I=artifacts/pretrain/stock_reg/last.ckpt; c=r; else T=classification; I=artifacts/pretrain/stock/last.ckpt; c=p; fi
  if [ "$1" = a ]; then
    TASK=$T INIT=$I P=${c}7_ ./phase3.sh obs_dist_temp_aux
    TASK=$T INIT=$I P=${c}7L_ LAYOUTS=$L ./phase3.sh obs_dist_temp_aux
  else
    TASK=$T INIT=$I P=${c}7L_ LAYOUTS=$L ./phase3.sh median obs_dist_temp
  fi
}
lane "$1" "$2"
