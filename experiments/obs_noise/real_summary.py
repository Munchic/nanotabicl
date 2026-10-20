"""Summary of real_eval.jsonl for one split: per model, mean over the 10 LINCS tasks (5 L1000 + 5 Cell Painting) of
accuracy / log-loss / ECE with all wells and with k wells, and paired deltas vs a reference arm.
Usage: python real_summary.py [--suffix _dose] [--ref tabicl_mean_e8]"""
import argparse, json, os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = ['tabicl_mean_e1', 'tabicl_mean_e8', 'tabicl_median_e1', 'tabicl_median_e8', 'tx_median0p', 'tx_median_B',
          'tx_obs_colmedg_A', 'tx_obs_B', 'p5_median', 'p7_col']

ap = argparse.ArgumentParser()
ap.add_argument('--suffix', default='_dose')
ap.add_argument('--ref', default='tabicl_mean_e8')
ap.add_argument('--models', default=','.join(MODELS))
a = ap.parse_args()
dsets = [f'L1000{a.suffix}', f'CellPainting{a.suffix}']
R = defaultdict(dict)
for r in map(json.loads, open(os.path.join(ROOT, 'artifacts', 'real_eval.jsonl'))):
    if r['dataset'] in dsets:
        R[(r['model'], str(r['k']))][(r['dataset'], r['task'])] = r
print(f'| model | acc, 1 well | acc, 2 | acc, 3 | acc, all | log-loss all | ECE all | Δacc all vs {a.ref} | Δlog-loss all vs {a.ref} |')
print('|---|---|---|---|---|---|---|---|---|')
for m in a.models.split(','):
    if not R[(m, 'all')]:
        continue
    ts = sorted(R[(m, 'all')])
    mean = lambda k, key: np.mean([R[(m, k)][t][key] for t in ts]) if R[(m, k)] else float('nan')
    d = np.array([R[(m, 'all')][t]['acc'] - R[(a.ref, 'all')][t]['acc'] for t in ts]) * 100
    dl = np.array([R[(m, 'all')][t]['log_loss'] - R[(a.ref, 'all')][t]['log_loss'] for t in ts])
    se = lambda v: v.std(ddof=1) / np.sqrt(len(v))
    print(f"| {m} | {mean('1', 'acc'):.3f} | {mean('2', 'acc'):.3f} | {mean('3', 'acc'):.3f} | {mean('all', 'acc'):.3f} | "
          f"{mean('all', 'log_loss'):.3f} | {mean('all', 'ece'):.3f} | {d.mean():+.1f} ± {se(d):.1f} ({(d > 0).sum()}/{len(d)}) | "
          f"{dl.mean():+.3f} ± {se(dl):.3f} ({(dl < 0).sum()}/{len(dl)}) |")
