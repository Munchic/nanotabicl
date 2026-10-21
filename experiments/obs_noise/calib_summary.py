"""Reliability-conditional calibration summary (records with 'rel', from real_eval.py or shape_features.py).
Per model / arm and dataset (all replicates, n_ens 8 where available): log-loss, conf_auroc (confidence separating
right from wrong; higher = uncertainty tracks error), and overconfidence in the least reliable tertiles (fewest
instances; most spread) vs the most reliable one. Usage: python calib_summary.py <jsonl> [<jsonl> ...]"""
import json, sys

import numpy as np

R = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
R = [r for r in R if 'rel' in r and r.get('k', 'all') == 'all']
key = lambda r: (r.get('model') or r.get('arm'), r.get('n_ens', r.get('n_est')), r['dataset'])
groups = {}
for r in R:
    groups.setdefault(key(r), []).append(r)
print(f"{'model/arm':24s} {'e':>2s} {'dataset':18s} {'logloss':>7s} {'confAUROC':>9s} | overconf: {'low-n':>6s} {'high-n':>6s} | {'hi-spread':>9s} {'lo-spread':>9s}")
for (m, e, ds), rs in sorted(groups.items(), key=lambda kv: (kv[0][2], str(kv[0][0]), kv[0][1] or 0)):
    oc = lambda which, b: np.mean([x['overconf'] for r in rs for x in r['rel'][which] if x['bin'] == b])
    print(f"{str(m):24s} {e:2d} {ds:18s} {np.mean([r['log_loss'] for r in rs]):7.3f} {np.nanmean([r['rel']['conf_auroc'] for r in rs]):9.3f} | "
          f"{' ':9s} {oc('by_n', 0):+6.3f} {oc('by_n', 2):+6.3f} | {oc('by_spread', 2):+9.3f} {oc('by_spread', 0):+9.3f}")
