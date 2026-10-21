"""Summary of artifacts/real_heldout.jsonl (heldout_eval.sh): per dataset, mean acc / AUC / log-loss over tasks (all
replicates), paired difference vs the reference (default: wrapper, mean input, 8 est.) as mean +- SE (n better / n), and
the replicate headroom (wrapper k = 1 -> all). Usage: python heldout_summary.py [--ref tabicl_mean_e8] [--metric auc]"""
import argparse, json, os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = ['tabicl_mean_e1', 'tabicl_mean_e8', 'tx_mean0p_e1', 'tx_mean0p_e8', 'tx_huber0p_e1', 'tx_huber0p_e8',
          'tx_v5gen_A_e1', 'tx_v5gen_A_e8', 'tx_obs_B_e1', 'tx_obs_B_e8']
DATASETS = ['L1000_dose', 'CellPainting_dose', 'GasDrift', 'Tox21', 'UPS1_low', 'CPTAC']

ap = argparse.ArgumentParser()
ap.add_argument('--ref', default='tabicl_mean_e8')
ap.add_argument('--file', default=os.path.join(ROOT, 'artifacts', 'real_heldout.jsonl'))
a = ap.parse_args()
R = [json.loads(l) for l in open(a.file)]
get = lambda m, ds, k='all': {r['task']: r for r in R if r['model'] == m and r['dataset'] == ds and r['k'] == k}
for ds in DATASETS:
    ref = get(a.ref, ds)
    if not ref:
        continue
    metric = 'auc' if ds in ('Tox21', 'UPS1_low', 'CPTAC') else 'acc'
    k1 = get('tabicl_mean_e8', ds, 1)
    head = np.mean([ref[t][metric] - k1[t][metric] for t in ref if t in k1]) if k1 else float('nan')
    print(f"\n## {ds} ({len(ref)} tasks; metric {metric}; wrapper 1 -> all replicates: {100 * head:+.1f})")
    print(f"{'model':18s} {metric:>6s} {'logloss':>8s}   Δ{metric} vs {a.ref}")
    for m in MODELS:
        cur = get(m, ds)
        if not cur:
            continue
        ts = [t for t in ref if t in cur]
        d = np.array([cur[t][metric] - ref[t][metric] for t in ts])
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        print(f"{m:18s} {100 * np.nanmean([cur[t][metric] for t in ts]):6.1f} {np.mean([cur[t]['log_loss'] for t in ts]):8.3f}"
              f"   {100 * d.mean():+5.1f} ± {100 * se:.1f} ({(d > 0).sum()}/{len(d)})")
