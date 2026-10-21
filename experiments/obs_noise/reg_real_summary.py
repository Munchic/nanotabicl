"""Summary of artifacts/real_eval_reg.jsonl: per dataset (all replicates), mean CRPS / R^2 / cov90 over tasks and the
paired CRPS difference vs the reference (default: the wrapper, mean input, 8 est.) as mean +- SE (n better / n; lower
CRPS is better), plus the replicate headroom (wrapper k = 1 -> all). Usage: python reg_real_summary.py [--ref ...]"""
import argparse, json, os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['L1000_dose_reg', 'CellPainting_dose_reg', 'GasConc', 'Tox21Pot']

ap = argparse.ArgumentParser()
ap.add_argument('--ref', default='tabicl_mean_e8')
ap.add_argument('--file', default=os.path.join(ROOT, 'artifacts', 'real_eval_reg.jsonl'))
a = ap.parse_args()
R = [json.loads(l) for l in open(a.file)]
models = list(dict.fromkeys(r['model'] for r in R))
get = lambda m, ds, k='all': {r['task']: r for r in R if r['model'] == m and r['dataset'] == ds and r['k'] == k}
for ds in DATASETS:
    ref = get(a.ref, ds)
    if not ref:
        continue
    k1 = get(a.ref, ds, 1)
    head = np.mean([ref[t]['crps'] - k1[t]['crps'] for t in ref if t in k1]) if k1 else float('nan')
    print(f"\n## {ds} ({len(ref)} tasks; wrapper CRPS 1 -> all replicates: {head:+.3f})")
    print(f"{'model':24s} {'crps':>6s} {'r2':>6s} {'cov90':>6s}   Δcrps vs {a.ref}")
    for m in sorted(models):
        cur = get(m, ds)
        if not cur:
            continue
        ts = [t for t in ref if t in cur]
        d = np.array([cur[t]['crps'] - ref[t]['crps'] for t in ts])
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        print(f"{m:24s} {np.mean([cur[t]['crps'] for t in ts]):6.3f} {np.mean([cur[t]['r2'] for t in ts]):6.3f} "
              f"{np.mean([cur[t]['cov90'] for t in ts]):6.3f}   {d.mean():+.4f} ± {se:.4f} ({(d < 0).sum()}/{len(d)})")
