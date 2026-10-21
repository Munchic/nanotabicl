"""Exp 7 summary: 1-pass vs 8-member ensembles, the auxiliary denoising loss and heterogeneous-noise training, against
matched median imputation, TabICLv2 (27M; 1 and 8 estimators) and the Bayes-oracle input. 27 datasets per task.

Setting groups: main = the 10 core noise modes (counts U{1..10}); grouped = Exp 3's visible-group setting; het = the
4 heterogeneous settings of gen_ext3.py (percol, pergroup, pergroup_vis, colgroup_vis). Paired deltas are per dataset
(averaged over the group's settings), mean ± SE over datasets, with the number of datasets where the row is better.
Usage: python compare_exp7.py [--ref p5_median_e8] [--ref_reg r5_median_e8]
"""
import argparse, json, os
from collections import defaultdict

import numpy as np

import calib

ROOT = os.path.dirname(os.path.abspath(__file__))
MAIN = ['gauss', 'cauchy', 'cauchy_hetero', 'censored_left_heavy', 'fail_floor', 'fail_shift', 'cv_lognormal',
        'lod_logistic', 'saturation', 'plate_effect']
HET = ['percol', 'pergroup', 'pergroup_vis', 'colgroup_vis']
GROUPS = {'clean': ['clean'], 'main': MAIN, 'grouped': ['grouped'], 'het': HET}
NOISY = MAIN + ['grouped'] + HET
CLS_FILES = ['exp6_eval', 'tabicl_eval', 'ens_eval', 'exp7_eval']
REG_FILES = ['exp5_reg_eval', 'tabicl_eval_reg', 'ens_eval_reg', 'exp7_eval_reg']
CLS_ROWS = [  # (label, tag); rows whose tag has no results are skipped
    ('stock nanoTabICL, 1 pass', 'stock_e1'), ('stock nanoTabICL, ens 8', 'stock_e8'),
    ('median imputation (p5), 1 pass', 'p5_median_e1'), ('median imputation (p5), ens 8', 'p5_median_e8'),
    ('ours p5, 1 pass', 'p5_obs_dist_temp_e1'), ('ours p5, ens 8', 'p5_obs_dist_temp_e8'),
    ('ours + aux (p7), 1 pass', 'p7_obs_dist_temp_aux_e1'), ('ours + aux (p7), ens 8', 'p7_obs_dist_temp_aux_e8'),
    ('median, het-trained (p7L), 1 pass', 'p7L_median_e1'), ('median, het-trained (p7L), ens 8', 'p7L_median_e8'),
    ('ours, het-trained (p7L), 1 pass', 'p7L_obs_dist_temp_e1'), ('ours, het-trained (p7L), ens 8', 'p7L_obs_dist_temp_e8'),
    ('ours + aux, het-trained (p7L), 1 pass', 'p7L_obs_dist_temp_aux_e1'),
    ('ours + aux, het-trained (p7L), ens 8', 'p7L_obs_dist_temp_aux_e8'),
    ('ours + column context (p7), 1 pass', 'p7_obs_dist_temp_col_e1'), ('ours + column context (p7), ens 8', 'p7_obs_dist_temp_col_e8'),
    ('stock2 (30k steps, 4x bank), 1 pass', 'stock2_e1'), ('stock2 (30k steps, 4x bank), ens 8', 'stock2_e8'),
    ('TabICLv2 27M median, 1 est.', 'tabicl_median_e1'), ('TabICLv2 27M median, 8 est.', 'tabicl_median'),
    ('TabICLv2 27M + Bayes-oracle input, 8 est.', 'tabicl_bayes'), ('nano + Bayes-oracle input, 1 pass', 'p2_mean+bayes')]
REG_ROWS = [(lab.replace('p5', 'r5').replace('p7', 'r7'), tag.replace('p5_', 'r5_').replace('p7', 'r7').replace('stock_', 'stock_reg_'))
            for lab, tag in CLS_ROWS[:-1]] + [('nano + Bayes-oracle input, 1 pass', 'stock_reg+bayes')]


def load(files):
    R = defaultdict(dict)  # (model, setting) -> dataset -> record (last one wins)
    for f in files:
        p = os.path.join(ROOT, 'artifacts', f + '.jsonl')
        if os.path.exists(p):
            for r in map(json.loads, open(p)):
                R[(r['model'], r['noise'])][r['dataset']] = r
    return R


def per_dataset(R, m, settings, key):  # dataset -> mean over settings (only datasets with all settings)
    ds = set.intersection(*[set(R[(m, s)]) for s in settings]) if all(R[(m, s)] for s in settings) else set()
    return {d: np.mean([R[(m, s)][d][key] for s in settings]) for d in ds}


def paired(R, m, ref, settings, key, lower_better):
    a, b = per_dataset(R, m, settings, key), per_dataset(R, ref, settings, key)
    ds = sorted(set(a) & set(b))
    if not ds or m == ref:
        return ''
    d = np.array([a[x] - b[x] for x in ds])
    wins = int(((d < 0) if lower_better else (d > 0)).sum())
    return f'{d.mean():+.3f}±{d.std(ddof=1) / np.sqrt(len(d)):.3f} ({wins}/{len(d)})'


def fmt(R, m, settings, key):
    v = per_dataset(R, m, settings, key)
    return f'{np.mean(list(v.values())):.3f}' if v else ''


def cls_calib(tag, settings):
    per = [calib.calib_stats(p, y) for s in settings for p, y in calib.load(tag, s)]
    if not per:
        return '', ''
    return f"{np.mean([d['ece'] for d in per]):.3f}", f"{np.mean([d['overconf'] for d in per]):+.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', default='p5_median_e8')
    ap.add_argument('--ref_reg', default='r5_median_e8')
    a = ap.parse_args()
    R = load(CLS_FILES)
    print(f'## Classification (accuracy; Δ = paired vs `{a.ref}`)\n')
    print('| model | clean | main (10) | grouped | het (4) | log-loss, all noisy | ECE, all | overconf., all | Δ acc main | Δ acc het |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for label, m in CLS_ROWS:
        if not R[(m, 'clean')] and not R[(m, 'percol')]:
            continue
        ece, oc = cls_calib(m, ['clean'] + NOISY)
        print(f"| {label} | {fmt(R, m, ['clean'], 'acc')} | {fmt(R, m, MAIN, 'acc')} | {fmt(R, m, ['grouped'], 'acc')} | "
              f"{fmt(R, m, HET, 'acc')} | {fmt(R, m, NOISY, 'log_loss')} | {ece} | {oc} | "
              f"{paired(R, m, a.ref, MAIN, 'acc', False)} | {paired(R, m, a.ref, HET, 'acc', False)} |")
    Rr = load(REG_FILES)
    print(f'\n## Regression (CRPS, lower is better; Δ = paired vs `{a.ref_reg}`)\n')
    print('| model | clean | main (10) | grouped | het (4) | PIT err, all noisy | cov90 clean / noisy | Δ CRPS main | Δ CRPS het |')
    print('|---|---|---|---|---|---|---|---|---|')
    for label, m in REG_ROWS:
        if not Rr[(m, 'clean')] and not Rr[(m, 'percol')]:
            continue
        print(f"| {label} | {fmt(Rr, m, ['clean'], 'crps')} | {fmt(Rr, m, MAIN, 'crps')} | {fmt(Rr, m, ['grouped'], 'crps')} | "
              f"{fmt(Rr, m, HET, 'crps')} | {fmt(Rr, m, NOISY, 'pit_ce')} | {fmt(Rr, m, ['clean'], 'cov90')} / "
              f"{fmt(Rr, m, NOISY, 'cov90')} | {paired(Rr, m, a.ref_reg, MAIN, 'crps', True)} | "
              f"{paired(Rr, m, a.ref_reg, HET, 'crps', True)} |")


if __name__ == '__main__':
    main()
