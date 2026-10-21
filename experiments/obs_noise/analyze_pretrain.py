"""Tables for obs_eval.py results: mean metric per (model, noise), and paired Δ vs a reference model (± SE, wins/27)."""
import argparse, json
from collections import defaultdict

import numpy as np

NOISES = ['clean', 'gauss', 'student_t3', 'censored', 'cauchy', 'cauchy_hetero', 'gauss_valhet', 'cauchy_valhet',
          'censored_left', 'censored_left_heavy']
NOISES += ['fail_floor', 'fail_shift', 'cv_lognormal', 'lod_logistic', 'saturation', 'plate_effect']
NOISES += [f'{n}_n{k}' for k in (20, 50) for n in NOISES[1:]] + ['grouped_clean', 'grouped']
NOISES += [f'{n}_{s}' for n in ('gauss', 'cauchy', 'censored_left_heavy') for s in ('k1', 'k2', 'k3', 'k5', 'k10', 'mix', 'ctx10_q1', 'ctx1_q10')]
SHORT = dict(clean='clean', gauss='gauss', student_t3='t3', censored='cens', cauchy='cauchy', cauchy_hetero='c_het',
             gauss_valhet='g_valhet', cauchy_valhet='c_valhet', censored_left='cens_L', censored_left_heavy='cens_L_hvy')
SHORT.update({f'{n}_n{k}': f'{SHORT[n]}_{k}' for k in (20, 50) for n in list(SHORT)[1:]})
SHORT.update(grouped_clean='grp_clean', grouped='grouped')
SHORT.update(fail_floor='f_floor', fail_shift='f_shift', cv_lognormal='cv_ln', lod_logistic='lod_lgt', saturation='satur', plate_effect='plate')


def load(paths):
    R = defaultdict(dict)  # (model, noise) -> dataset -> record (last one wins)
    for p in paths:
        for r in map(json.loads, open(p)):
            R[(r['model'], r['noise'])][r['dataset']] = r
    return R


LOWER_BETTER = {'log_loss', 'crps', 'rmse', 'pit_ce', 'pit_ks'}


def val(r, metric):  # derived metrics: covXX_err = |coverage - nominal|, tail_out_err = |tail_out - 0.10|
    if metric.startswith('cov') and metric.endswith('_err'):
        return abs(r[metric[:-4]] - int(metric[3:-4]) / 100)
    if metric == 'tail_out_err':
        return abs(r['tail_out'] - 0.10)
    return r[metric]


def table(R, models, metric, ref=None, only=None):
    noises = [n for n in (only or NOISES) if any((m, n) in R for m in models)]
    hdr = '| model | ' + ' | '.join(SHORT.get(n, n) for n in noises) + ' |'
    out = [hdr, '|' + '---|' * (len(noises) + 1)]
    for m in models:
        cells = []
        for n in noises:
            d = R.get((m, n), {})
            if not d:
                cells.append('')
                continue
            if ref is None or m == ref:
                cells.append(f"{np.mean([val(r, metric) for r in d.values()]):.3f}")
            else:
                ks = sorted(set(d) & set(R[(ref, n)]))
                diff = np.array([val(d[k], metric) - val(R[(ref, n)][k], metric) for k in ks])
                sign = -1 if (metric in LOWER_BETTER or metric.endswith('_err')) else 1
                cells.append(f"{diff.mean():+.3f}±{diff.std(ddof=1) / np.sqrt(len(diff)):.3f} ({(sign * diff > 0).sum()})")
        out.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='+')
    ap.add_argument('--models', required=True, help='comma-separated, in display order')
    ap.add_argument('--ref', default=None, help='model for paired Δ (cells: Δ ± SE (datasets where model is better))')
    ap.add_argument('--metrics', default='acc,roc_auc')
    ap.add_argument('--settings', default=None, help='comma-separated subset of settings (default: all present)')
    a = ap.parse_args()
    R = load(a.results)
    models = a.models.split(',')
    only = a.settings.split(',') if a.settings else None
    for metric in a.metrics.split(','):
        print(f'\n**{metric}** (mean over 27 datasets)\n')
        print(table(R, models, metric, only=only))
        if a.ref:
            print(f'\n**{metric}: paired Δ vs {a.ref}** (± SE; in parentheses: datasets where the model is better, of 27)\n')
            print(table(R, models, metric, ref=a.ref, only=only))
