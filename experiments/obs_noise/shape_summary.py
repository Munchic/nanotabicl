"""Summary of artifacts/shape_features.jsonl: per dataset and n_est, each arm's mean metric and its paired difference
vs the mean-only arm (mean +- SE, n better / n). Classification: acc (LINCS, GasDrift) or AUC (Tox21, UPS1_low, CPTAC),
higher is better; regression: CRPS, lower is better."""
import json, os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
import sys
R = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'artifacts', 'shape_features.jsonl'))]
for ds in dict.fromkeys(r['dataset'] for r in R):
    metric = 'crps' if 'crps' in next(r for r in R if r['dataset'] == ds) else ('auc' if ds in ('Tox21', 'UPS1_low', 'CPTAC') else 'acc')
    sign = -1 if metric == 'crps' else 1
    for e in sorted({r['n_est'] for r in R}):
        get = lambda arm: {r['task']: r for r in R if r['dataset'] == ds and r['arm'] == arm and r['n_est'] == e}
        base = get('mean')
        line = []
        for arm in ['mean', 'mean_sd', 'sketch', 'sketch_n', 'clusters', 'huber_sd', 'med_mad', 'huber_mad']:
            cur = get(arm)
            ts = [t for t in base if t in cur]
            if not ts:
                continue
            d = np.array([cur[t][metric] - base[t][metric] for t in ts])
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
            scale = 1 if metric == 'crps' else 100
            line.append(f"{arm} {scale * np.mean([cur[t][metric] for t in ts]):.3f}" + ('' if arm == 'mean' else
                        f" ({scale * d.mean():+.3f}±{scale * se:.3f}, {(sign * d > 0).sum()}/{len(d)})"))
        print(f"{ds:22s} {metric:4s} e{e}: " + ' | '.join(line))
