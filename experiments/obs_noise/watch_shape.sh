#!/usr/bin/env bash
# Wake-up condition for the shape / head runs (same pattern as watch_tx.sh): N new validation points, a crash or a
# non-finite loss, a kill switch, or a "done" marker. Then prints the progress table. GPU is queried through torch when
# nvidia-smi is unavailable (the NVML driver/library mismatch).
# Usage: watch_shape.sh [N] [--runs 'tx_shV4*|tx_headM*'] [--target 6000]
cd "$(dirname "$0")"
N=${1:-4}
RUNS=${2:-tx_shV4*}
TARGET=${3:-6000}
PAT=$(echo "$RUNS" | sed 's/,/|/g')
nv() { cat artifacts/pretrain/$PAT/log.jsonl 2>/dev/null | grep -c '"val"'; }
bad() { cat artifacts/logs/train_$PAT.log 2>/dev/null | grep -c "non-finite\|exited with [1-9]\|out of memory"; }
kill_() { cat artifacts/pretrain/$PAT/log.jsonl 2>/dev/null | grep -c '"dev_kill"\|"learn_kill"'; }
d0=$(grep -c "done" artifacts/logs/phase3.log); c0=$(nv); b0=$(bad); k0=$(kill_)
until [ $(nv) -ge $((c0 + N)) ] || [ $(bad) -gt $b0 ] || [ $(kill_) -gt $k0 ] || [ $(grep -c "done" artifacts/logs/phase3.log) -gt $d0 ]; do sleep 60; done
date -Is
tail -n 2 artifacts/logs/phase3.log
cat artifacts/logs/train_$PAT.log 2>/dev/null | grep "non-finite\|exited with [1-9]\|out of memory" | tail -2
/home/munchic/miniconda3/envs/ml/bin/python - <<'EOF'
import torch
if torch.cuda.is_available():
    print('gpu: %.1f/%.1f GiB used' % tuple(x / 2**30 for x in torch.cuda.mem_get_info()[::-1]))
EOF
/home/munchic/miniconda3/envs/ml/bin/python monitor_shape.py --target "$TARGET"
