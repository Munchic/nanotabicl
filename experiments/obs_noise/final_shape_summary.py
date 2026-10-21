"""SHAPE.md section 7 summary: the final evaluation against the success criterion stated before it was run.
Reference: the wrapper on mean + sd, 8 est. Per dataset (all binary -> AUC is the primary metric; acc and log-loss also
shown): each arm's mean and its paired difference vs the reference per split, mean +- SE (n better / n).
Criterion: better than the reference by > 2 SE on at least half of the datasets, never significantly (> 2 SE) worse;
on paired datasets, better than the cluster-share wrapper. Calibration: confidence AUROC and overconfidence in the
least reliable tertiles (fewest instances, most spread)."""
import json, os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(ROOT, 'artifacts')
DATASETS = ['NK', 'HVTN', 'PreE', 'Musk', 'PhysioNet']
PAIRED = {'NK', 'HVTN', 'PreE', 'Musk'}
recs = []
for r in (json.loads(l) for l in open(os.path.join(A, 'final_shape_wrapper.jsonl'))):
    recs.append(dict(r, name=f"wrapper {r['arm']}", e=r['n_est']))
for r in (json.loads(l) for l in open(os.path.join(A, 'final_shape_ours.jsonl'))):
    recs.append(dict(r, name=r['model'].rsplit('_e', 1)[0], e=r['n_ens']))
get = lambda name, e, ds: {r['task']: r for r in recs if r['name'] == name and r['e'] == e and r['dataset'] == ds}
names = list(dict.fromkeys((r['name'], r['e']) for r in recs))
verdict = {}
for ds in DATASETS:
    ref = get('wrapper mean_sd', 8, ds)
    if not ref:
        continue
    print(f"\n## {ds} ({len(ref)} splits) — Δ AUC vs wrapper mean + sd (8 est.)")
    print(f"{'arm':28s} {'e':>2s} {'AUC':>6s} {'acc':>6s} {'logloss':>7s}   {'ΔAUC':>14s}  {'confAUROC':>9s} {'oc low-n':>8s} {'oc hi-spr':>9s}")
    for name, e in names:
        cur = get(name, e, ds)
        if not cur:
            continue
        ts = [t for t in ref if t in cur]
        d = np.array([cur[t]['auc'] - ref[t]['auc'] for t in ts])
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        oc = lambda key, b: np.nanmean([x['overconf'] for t in ts for x in cur[t]['rel'][key] if x['bin'] == b] or [np.nan])
        print(f"{name:28s} {e:2d} {100 * np.mean([cur[t]['auc'] for t in ts]):6.1f} {100 * np.mean([cur[t]['acc'] for t in ts]):6.1f} "
              f"{np.mean([cur[t]['log_loss'] for t in ts]):7.3f}   {100 * d.mean():+5.1f} ± {100 * se:3.1f} ({(d > 0).sum():2d}/{len(d)})  "
              f"{np.nanmean([cur[t]['rel']['conf_auroc'] for t in ts]):9.3f} {oc('by_n', 0):+8.3f} {oc('by_spread', 2):+9.3f}")
        if name == 'tx_shV2b_A' and e == 8:
            verdict[ds] = dict(d=d.mean(), se=se)
            hs = get('wrapper huber_sd', 8, ds)
            dh = np.array([cur[t]['auc'] - hs[t]['auc'] for t in ts if t in hs])
            verdict[ds]['vs_huber_sd'] = (dh.mean(), dh.std(ddof=1) / np.sqrt(len(dh)) if len(dh) > 1 else 0.0)
            if ds in PAIRED:
                cl = get('wrapper clusters', 8, ds)
                dc = np.array([cur[t]['auc'] - cl[t]['auc'] for t in ts if t in cl])
                verdict[ds]['vs_clusters'] = (dc.mean(), dc.std(ddof=1) / np.sqrt(len(dc)) if len(dc) > 1 else 0.0)
print('\n## Success criterion (stated in SHAPE.md before the run)')
wins = [ds for ds, v in verdict.items() if v['d'] > 2 * v['se']]
losses = [ds for ds, v in verdict.items() if v['d'] < -2 * v['se']]
print(f"better than mean + sd by > 2 SE: {wins} ({len(wins)}/{len(verdict)}); significantly worse: {losses or 'none'}")
print('secondary bar (stated before the run): vs the Huber + sd wrapper, 8 est.')
for ds, v in verdict.items():
    print(f"  {ds}: {100 * v['vs_huber_sd'][0]:+.1f} ± {100 * v['vs_huber_sd'][1]:.1f}")
for ds, v in verdict.items():
    if 'vs_clusters' in v:
        print(f"  {ds}: vs cluster shares {100 * v['vs_clusters'][0]:+.1f} ± {100 * v['vs_clusters'][1]:.1f}")
ok = len(wins) >= len(verdict) / 2 and not losses
print('CRITERION MET' if ok else 'criterion not met')
