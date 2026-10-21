"""Round-2 summary (V4a / V4b vs V2b, with and without test-time subsample averaging). The five former final sets
are dev now (already seen). Per dataset: AUC (binary sets) or accuracy (HAR, LINCS), log-loss, and paired
differences vs the wrapper on Huber + sd and on mean + sd (8 est.); calibration: confidence AUROC and overconfidence
in the most-spread tertile."""
import json, os

import numpy as np

A = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')
ours = [json.loads(l) for f in ['shape_v4_eval.jsonl', 'final_shape_ours.jsonl', 'shape_dev_eval.jsonl']
        if os.path.exists(os.path.join(A, f)) for l in open(os.path.join(A, f))]
wrap = [json.loads(l) for f in ['final_shape_wrapper.jsonl', 'shape_features_robust.jsonl', 'shape_features_dev_cal.jsonl']
        if os.path.exists(os.path.join(A, f)) for l in open(os.path.join(A, f))]
MODELS = ['tx_shV2b_A_e8', 'tx_shV2b_A_e8_sub', 'tx_shV4a_A_e8', 'tx_shV4a_A_e8_sub', 'tx_shV4b_A_e8', 'tx_shV4b_A_e8_sub', 'tx_shV4c_A_e8']
for ds in ['HAR', 'L1000_dose', 'CellPainting_dose', 'NK', 'HVTN', 'PreE', 'Musk', 'PhysioNet']:
    met = 'acc' if ds in ('HAR', 'L1000_dose', 'CellPainting_dose') else 'auc'
    ref = {a: {r['task']: r for r in wrap if r['dataset'] == ds and r.get('arm') == a and r.get('n_est') == 8 and r.get('k', 'all') == 'all'}
           for a in ('huber_sd', 'mean_sd')}
    print(f"\n## {ds} ({met})   wrapper Huber + sd {100 * np.mean([r[met] for r in ref['huber_sd'].values()]):.1f}, "
          f"mean + sd {100 * np.mean([r[met] for r in ref['mean_sd'].values()]):.1f}")
    for m in MODELS:
        cur = {}
        for r in ours:  # last record per task wins (re-runs)
            if r['model'] == m and r['dataset'] == ds and r['k'] == 'all':
                cur[r['task']] = r
        if not cur:
            continue
        line = f"{m:22s} {met} {100 * np.mean([r[met] for r in cur.values()]):5.1f} ll {np.mean([r['log_loss'] for r in cur.values()]):.3f}"
        for a in ('huber_sd', 'mean_sd'):
            ts = [t for t in cur if t in ref[a]]
            if ts:
                d = np.array([cur[t][met] - ref[a][t][met] for t in ts])
                line += f" | vs {a} {100 * d.mean():+5.1f} ± {100 * d.std(ddof=1) / np.sqrt(len(d)):.1f}"
        oc = np.mean([x['overconf'] for r in cur.values() for x in r['rel']['by_spread'] if x['bin'] == 2])
        line += f" | confAUROC {np.nanmean([r['rel']['conf_auroc'] for r in cur.values()]):.3f} oc hi-spread {oc:+.3f}"
        print(line)
