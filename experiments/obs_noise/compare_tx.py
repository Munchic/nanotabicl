"""Exp 8: TabICLv2-backbone runs vs the off-the-shelf TabICLv2 sklearn wrapper (1 / 8 estimators), TabICLv2 + median
through our pipeline, the best 1M model, and the Bayes-oracle input. 27 datasets; paired deltas per dataset averaged
over the group's settings, mean ± SE (n better). Usage: python compare_tx.py [cls_tag ...] -- [reg_tag ...]"""
import json, os, sys
from collections import defaultdict

import numpy as np

import calib

ROOT = os.path.dirname(os.path.abspath(__file__))
MAIN = ['gauss', 'cauchy', 'cauchy_hetero', 'censored_left_heavy', 'fail_floor', 'fail_shift', 'cv_lognormal',
        'lod_logistic', 'saturation', 'plate_effect']
HET = ['percol', 'pergroup', 'pergroup_vis', 'colgroup_vis']


def load(fs):
    R = defaultdict(dict)
    for f in fs:
        p = os.path.join(ROOT, 'artifacts', f + '.jsonl')
        if os.path.exists(p):
            for r in map(json.loads, open(p)):
                R[(r['model'], r['noise'])][r['dataset']] = r
    return R


def pdm(R, m, ss, key):
    if not all(R[(m, s)] for s in ss):
        return {}
    ds = set.intersection(*[set(R[(m, s)]) for s in ss])
    return {d: np.mean([R[(m, s)][d][key] for s in ss]) for d in ds}


def mean(R, m, ss, key):
    v = pdm(R, m, ss, key)
    return f'{np.mean(list(v.values())):.3f}' if v else '–'


def paired(R, a, b, ss, key, lower):
    A, B = pdm(R, a, ss, key), pdm(R, b, ss, key)
    ds = sorted(set(A) & set(B))
    if not ds or a == b:
        return ''
    d = np.array([A[x] - B[x] for x in ds])
    w = int(((d < 0) if lower else (d > 0)).sum())
    return f'{d.mean():+.3f}±{d.std(ddof=1) / np.sqrt(len(d)):.3f} ({w}/{len(d)})'


def main():
    args = sys.argv[1:]
    cls_extra, reg_extra = (args[:args.index('--')], args[args.index('--') + 1:]) if '--' in args else (args, [])
    R = load(['tabicl_eval', 'tx_eval', 'exp7_eval'])
    rows = [('TabICLv2 wrapper, 1 est.', 'tabicl_median_e1'), ('TabICLv2 wrapper, 8 est.', 'tabicl_median'),
            ('TabICLv2 + median, our pipeline (fp32)', 'tx_median0p_e1')] + [(t, t) for t in cls_extra] + \
           [('best 1M (p7 col), 8 passes', 'p7_obs_dist_temp_col_e8'), ('TabICLv2 + Bayes-oracle input, wrapper', 'tabicl_bayes')]
    print('## Classification (accuracy)\n')
    print('| model | clean | main (10) | grouped | het (4) | log-loss main | ECE main | Δ main vs wrapper 1-est. | Δ het vs wrapper |')
    print('|---|---|---|---|---|---|---|---|---|')
    for lab, m in rows:
        cs = [calib.calib_stats(p, y)['ece'] for s in MAIN for p, y in calib.load(m, s)]
        print(f"| {lab} | {mean(R, m, ['clean'], 'acc')} | {mean(R, m, MAIN, 'acc')} | {mean(R, m, ['grouped'], 'acc')} | "
              f"{mean(R, m, HET, 'acc')} | {mean(R, m, MAIN, 'log_loss')} | {np.mean(cs) if cs else float('nan'):.3f} | "
              f"{paired(R, m, 'tabicl_median_e1', MAIN, 'acc', False)} | {paired(R, m, 'tabicl_median_e1', HET, 'acc', False)} |")
    R = load(['tabicl_eval_reg', 'tx_eval_reg', 'exp7_eval_reg'])
    rows = [('TabICLv2 wrapper, 1 est.', 'tabicl_median_e1'), ('TabICLv2 wrapper, 8 est.', 'tabicl_median'),
            ('TabICLv2 + median, our pipeline (fp32)', 'tx_median0p_reg_e1')] + [(t, t) for t in reg_extra] + \
           [('best 1M (r7 col), 8 passes', 'r7_obs_dist_temp_col_e8'), ('TabICLv2 + Bayes-oracle input, wrapper', 'tabicl_bayes')]
    print('\n## Regression (CRPS, lower is better)\n')
    print('| model | clean | main (10) | grouped | het (4) | PIT err main | cov90 main | Δ main vs wrapper 1-est. | Δ het vs wrapper |')
    print('|---|---|---|---|---|---|---|---|---|')
    for lab, m in rows:
        print(f"| {lab} | {mean(R, m, ['clean'], 'crps')} | {mean(R, m, MAIN, 'crps')} | {mean(R, m, ['grouped'], 'crps')} | "
              f"{mean(R, m, HET, 'crps')} | {mean(R, m, MAIN, 'pit_ce')} | {mean(R, m, MAIN, 'cov90')} | "
              f"{paired(R, m, 'tabicl_median_e1', MAIN, 'crps', True)} | {paired(R, m, 'tabicl_median_e1', HET, 'crps', True)} |")


if __name__ == '__main__':
    main()
