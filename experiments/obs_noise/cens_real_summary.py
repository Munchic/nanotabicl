"""Summary of artifacts/cens_real.jsonl: per design, mean over the 12 Tox21 tasks of CRPS (observed queries; log10 AC50
units) and -log P(y > bound) (censored queries, natural designs), with paired deltas vs a reference arm."""
import argparse, collections, json, os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--ref', default='bound')
a = ap.parse_args()
R = collections.defaultdict(dict)
for l in open(os.path.join(ROOT, 'artifacts', 'cens_real.jsonl')):
    r = json.loads(l)
    R[(r['design'], r['model'])][r['task']] = r
for design in sorted({d for d, _ in R}):
    print(f'\n{design}  (n per task: {np.mean([r["n_train"] + r["n_test"] for r in R[(design, a.ref)].values()]):.0f}, '
          f'context censored {np.mean([r["cens_ctx"] for r in R[(design, a.ref)].values()]):.2f})')
    print('| arm | CRPS | Δ vs ' + a.ref + ' (wins / 12) | -log P(cens) |')
    print('|---|---|---|---|')
    ref = R[(design, a.ref)]
    for (d, m), t in sorted(R.items()):
        if d != design:
            continue
        ks = sorted(set(t) & set(ref))
        c = np.array([t[k]['crps'] for k in ks]); dc = c - np.array([ref[k]['crps'] for k in ks])
        nll = [t[k]['nll_cens'] for k in ks if 'nll_cens' in t[k]]
        se = dc.std(ddof=1) / np.sqrt(len(dc)) if len(dc) > 1 else 0
        print(f"| {m} | {c.mean():.4f} | {dc.mean():+.4f} ± {se:.4f} ({(dc < 0).sum()}) | {np.mean(nll):.3f} |" if nll else
              f"| {m} | {c.mean():.4f} | {dc.mean():+.4f} ± {se:.4f} ({(dc < 0).sum()}) | |")
