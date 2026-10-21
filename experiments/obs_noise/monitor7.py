"""Exp 7f early-stopping monitor: validation of running runs vs their references at the same step (col runs), and
stock2 / stock2_reg trajectories vs the final stock / stock_reg values. Usage: python monitor7.py"""
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
K = ['clean', 'gauss', 'cauchy', 'cauchy_hetero', 'censored_left_heavy', 'fail_floor', 'fail_shift', 'cv_lognormal',
     'lod_logistic', 'saturation', 'plate_effect']


def logv(name):
    p = os.path.join(ROOT, 'artifacts', 'pretrain', name, 'log.jsonl')
    return {r['step']: r['val'] for r in map(json.loads, open(p)) if 'val' in r} if os.path.exists(p) else {}


for run, ref, m in [('p7_obs_dist_temp_col', 'p5_obs_dist_temp', 'acc'), ('r7_obs_dist_temp_col', 'r5_obs_dist_temp', 'loss'),
                    ('r7L_obs_dist_temp', 'r5_obs_dist_temp', 'loss')]:
    A, B = logv(ref), logv(run)
    steps = [s for s in sorted(B) if s > 0 and s in A]
    for s in steps[-2:]:
        print(f'{run} @{s} Δ{m} vs {ref}: ' + ' '.join(f'{k[:9]} {B[s][k][m] - A[s][k][m]:+.3f}' for k in K))
for run, ref in [('stock2', 'stock'), ('stock2_reg', 'stock_reg')]:
    A, B = logv(ref), logv(run)
    if B:
        fin = A[max(A)]['clean']
        s = max(B)
        print(f'{run} @{s}: clean acc/R2 {B[s]["clean"]["acc"]:.4f} loss {B[s]["clean"]["loss"]:.4f} | {ref} final @{max(A)}: '
              f'{fin["acc"]:.4f} / {fin["loss"]:.4f} | {ref} @{s}: ' + (f'{A[s]["clean"]["acc"]:.4f}' if s in A else '-'))
