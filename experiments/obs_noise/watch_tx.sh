#!/usr/bin/env bash
# wake-up condition for Exp 8 monitoring: N new validation points across the TabICLv2 runs, a finished run, or a crash
cd "$(dirname "$0")"
N=${1:-8}
nv() { cat artifacts/pretrain/tx_*/log.jsonl 2>/dev/null | grep -c '"val"'; }
live() { ls artifacts/logs/train_tx_*.log | grep -v -E "_bf16|_nan_oom"; }
bad() { cat $(live) 2>/dev/null | grep -c "exited with [1-9]\|non-finite"; }
c0=$(nv); b0=$(bad); d0=$(grep -c "done" artifacts/logs/phase3.log)
until [ $(nv) -ge $((c0 + N)) ] || [ $(bad) -gt $b0 ] || [ $(grep -c "done" artifacts/logs/phase3.log) -gt $d0 ]; do sleep 60; done
date; tail -n 3 artifacts/logs/phase3.log
cat $(live) | grep "exited with [1-9]\|non-finite" | tail -2
/home/munchic/miniconda3/envs/ml/bin/python monitor_tx.py 2>&1 | grep -E "^tx|step"
