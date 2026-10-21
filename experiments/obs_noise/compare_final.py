"""Head-to-head summary: our best model vs stock nanoTabICL vs full TabICLv2 (27 datasets per task, core suite)."""
import json, os
from collections import defaultdict

import numpy as np

import calib

ROOT = os.path.dirname(os.path.abspath(__file__))
MAIN = ['clean', 'gauss', 'cauchy', 'cauchy_hetero', 'censored_left_heavy', 'fail_floor', 'fail_shift', 'cv_lognormal',
        'lod_logistic', 'saturation', 'plate_effect', 'grouped']
NOISY = MAIN[1:]
MIX = ['gauss', 'cauchy', 'censored_left_heavy']
CLS = [('stock nanoTabICL (1M)', 'stock'), ('nanoTabICL + median imputation (1M, same training)', 'p5_median'),
       ('**ours** (1M)', 'p5_obs_dist_temp'), ('TabICLv2 (27M), mean input', 'tabicl_mean'),
       ('TabICLv2 (27M), median input', 'tabicl_median')]
REG = [('stock nanoTabICL (1M)', 'stock_reg'), ('nanoTabICL + median imputation (1M, same training)', 'r5_median'),
       ('**ours** (1M)', 'r5_obs_dist_temp'), ('TabICLv2 (27M), mean input', 'tabicl_mean'),
       ('TabICLv2 (27M), median input', 'tabicl_median')]


def load(paths):
    R = defaultdict(dict)
    for p in paths:
        for r in map(json.loads, open(os.path.join(ROOT, 'artifacts', p))):
            R[(r['model'], r['noise'])][r['dataset']] = r
    return R


def mean(R, m, s, k):
    v = [r[k] for r in R[(m, s)].values()]
    return np.mean(v) if v else np.nan


def cond_overconf(tag, nm, design, group):
    vals = []
    for e in os.listdir(os.path.join(ROOT, 'artifacts', 'probs', tag)):
        P = np.load(os.path.join(ROOT, 'artifacts', 'probs', tag, e))
        key = f'p__{nm}_{design}'
        if key not in P.files:
            continue
        name = e[:-4]
        te = np.load(os.path.join(ROOT, 'artifacts', 'data', 'eval', name + '.npz'))['test_idx']
        rc = np.load(os.path.join(ROOT, 'artifacts', 'data_ext2', 'eval', name + '.npz'))[f'{nm}_mix__rowcount'][te]
        p = P[key].astype(np.float64)
        oc = p.max(1) - (p.argmax(1) == P['y_test'])
        sel = {'single': rc == 1, 'all': np.ones(len(rc), bool)}[group]
        vals.append(oc[sel].mean())
    return np.mean(vals)


def main():
    R = load(['exp6_eval.jsonl', 'tabicl_eval.jsonl'])
    print('## Classification (27 datasets; accuracy / ROC-AUC / log-loss higher-lower as usual)\n')
    print('| model | clean acc | noisy acc (avg of 11) | noisy ROC-AUC | noisy log-loss | ECE (avg 12) | overconfidence (avg 12) | mix: single-sample rows | ctx n=10 → query n=1 |')
    print('|---|---|---|---|---|---|---|---|---|')
    for label, m in CLS:
        ece, oc = [], []
        for s in MAIN:
            per = [calib.calib_stats(p, y) for p, y in calib.load(m, s)]
            ece.append(np.mean([d['ece'] for d in per]))
            oc.append(np.mean([d['overconf'] for d in per]))
        single = np.mean([cond_overconf(m, nm, 'mix', 'single') for nm in MIX])
        ctxq = np.mean([cond_overconf(m, nm, 'ctx10_q1', 'all') for nm in MIX])
        print(f"| {label} | {mean(R, m, 'clean', 'acc'):.3f} | {np.mean([mean(R, m, s, 'acc') for s in NOISY]):.3f} | "
              f"{np.mean([mean(R, m, s, 'roc_auc') for s in NOISY]):.3f} | {np.mean([mean(R, m, s, 'log_loss') for s in NOISY]):.3f} | "
              f"{np.mean(ece):.3f} | {np.mean(oc):+.3f} | {single:+.3f} | {ctxq:+.3f} |")
    print('\nAccuracy by setting:\n')
    print('| model | ' + ' | '.join(MAIN) + ' |\n|' + '---|' * (len(MAIN) + 1))
    for label, m in CLS:
        print(f'| {label} | ' + ' | '.join(f"{mean(R, m, s, 'acc'):.3f}" for s in MAIN) + ' |')

    Rr = load(['exp5_reg_eval.jsonl', 'tabicl_eval_reg.jsonl'])
    print('\n## Regression (27 datasets; CRPS lower is better; PIT error lower is better; 90% interval coverage nominal 0.900)\n')
    print('| model | clean CRPS | noisy CRPS (avg of 11) | clean PIT err | noisy PIT err | clean cov90 | noisy cov90 |')
    print('|---|---|---|---|---|---|---|')
    for label, m in REG:
        print(f"| {label} | {mean(Rr, m, 'clean', 'crps'):.3f} | {np.mean([mean(Rr, m, s, 'crps') for s in NOISY]):.3f} | "
              f"{mean(Rr, m, 'clean', 'pit_ce'):.3f} | {np.mean([mean(Rr, m, s, 'pit_ce') for s in NOISY]):.3f} | "
              f"{mean(Rr, m, 'clean', 'cov90'):.3f} | {np.mean([mean(Rr, m, s, 'cov90') for s in NOISY]):.3f} |")
    print('\nCRPS by setting:\n')
    print('| model | ' + ' | '.join(MAIN) + ' |\n|' + '---|' * (len(MAIN) + 1))
    for label, m in REG:
        print(f'| {label} | ' + ' | '.join(f"{mean(Rr, m, s, 'crps'):.3f}" for s in MAIN) + ' |')


if __name__ == '__main__':
    main()
