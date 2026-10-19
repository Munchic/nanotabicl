"""Exp 4 analysis. (A) does predictive uncertainty shrink with more replicates? accuracy / confidence / overconfidence
vs n = 1..50 per noise model. (B) mixed single vs replicated rows. Reads artifacts/probs/<model>/<dataset>.npz."""
import argparse, json, os

import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
NOISES = ['gauss', 'student_t3', 'cauchy', 'cauchy_hetero', 'censored_left_heavy']
NS = [1, 2, 3, 5, 10, 20, 50]
MAN = json.load(open(os.path.join(ROOT, 'artifacts', 'data', 'manifest.json')))['eval']


def setting(nm, n):
    return f'{nm}_k{n}' if n <= 10 else f'{nm}_n{n}'


def row_stats(p, y):
    conf = p.max(1)
    return (p.argmax(1) == y).astype(float), conf, -(p * np.log(p)).sum(1) / np.log(p.shape[1])  # normalized entropy


def load(model, name):
    return np.load(os.path.join(ROOT, 'artifacts', 'probs', model, name + '.npz'))


def curve(models):
    res = {}  # (model, nm, n) -> list over datasets of (acc, conf, entropy)
    for e in MAN:
        for m in models:
            P = load(m, e['name'])
            for nm in NOISES:
                for n in NS + ['clean']:
                    key = 'p__clean' if n == 'clean' else f'p__{setting(nm, n)}'
                    if key not in P.files:
                        continue
                    a, c, h = row_stats(P[key].astype(np.float64), P['y_test'])
                    res.setdefault((m, nm, n), []).append((a.mean(), c.mean(), h.mean()))
    return res


def mixed(models):
    out = {}  # (model, nm, design, group) -> list over datasets of (acc, conf, n_rows)
    for e in MAN:
        z2 = np.load(os.path.join(ROOT, 'artifacts', 'data_ext2', 'eval', e['name'] + '.npz'))
        te = np.load(os.path.join(ROOT, 'artifacts', 'data', 'eval', e['name'] + '.npz'))['test_idx']
        for m in models:
            P = load(m, e['name'])
            for nm in NOISES:
                rc = z2[f'{nm}_mix__rowcount'][te]
                a, c, _ = row_stats(P[f'p__{nm}_mix'].astype(np.float64), P['y_test'])
                for grp, sel in [('mix: single rows', rc == 1), ('mix: replicated rows', rc > 1)]:
                    out.setdefault((m, nm, grp), []).append((a[sel].mean(), c[sel].mean()))
                for design, label in [(f'{nm}_k1', 'all single (k1)'), (f'{nm}_ctx10_q1', 'ctx n=10, query n=1'),
                                      (f'{nm}_k10', 'all n=10 (k10)'), (f'{nm}_ctx1_q10', 'ctx n=1, query n=10')]:
                    a, c, _ = row_stats(P[f'p__{design}'].astype(np.float64), P['y_test'])
                    out.setdefault((m, nm, label), []).append((a.mean(), c.mean()))
    return out


def fmt(v):
    v = np.asarray(v)
    return f'{v.mean():.3f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True)
    a = ap.parse_args()
    models = a.models.split(',')
    res = curve(models)
    for idx, label in [(0, 'accuracy'), (1, 'mean confidence (max prob)'), (2, 'mean normalized predictive entropy')]:
        print(f'\n**(A) {label} vs replicates per cell** (mean over 27 datasets; last column: clean data)\n')
        for nm in NOISES:
            print(f'\n*{nm}*\n\n| model | ' + ' | '.join(f'n={n}' for n in NS) + ' | clean |\n|' + '---|' * (len(NS) + 2))
            for m in models:
                print(f'| {m} | ' + ' | '.join(fmt([t[idx] for t in res.get((m, nm, n), [(np.nan,) * 3])]) for n in NS + ['clean']) + ' |')
    print('\n**(A) overconfidence (confidence − accuracy) vs n**, ± SE\n')
    for nm in NOISES:
        print(f'\n*{nm}*\n\n| model | ' + ' | '.join(f'n={n}' for n in NS) + ' |\n|' + '---|' * (len(NS) + 1))
        for m in models:
            cells = []
            for n in NS:
                d = np.array([t[1] - t[0] for t in res.get((m, nm, n), [])])
                cells.append(f'{d.mean():+.3f}±{d.std(ddof=1) / np.sqrt(len(d)):.3f}' if len(d) else '')
            print(f'| {m} | ' + ' | '.join(cells) + ' |')
    # figure: accuracy and confidence vs n
    fig, axs = plt.subplots(2, len(NOISES), figsize=(4 * len(NOISES), 7), sharex=True)
    for j, nm in enumerate(NOISES):
        for m in models:
            for i in (0, 1):
                ys = [np.mean([t[i] for t in res[(m, nm, n)]]) for n in NS]
                axs[i, j].plot(NS, ys, 'o-', ms=3, label=m)
        axs[0, j].set_title(nm)
        axs[1, j].set_xlabel('replicates per cell (n)')
        for i in (0, 1):
            axs[i, j].set_xscale('log')
            axs[i, j].set_xticks(NS, [str(n) for n in NS])
    axs[0, 0].set_ylabel('accuracy')
    axs[1, 0].set_ylabel('mean confidence')
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, 'figures', 'uncertainty_vs_n.png'), dpi=110)
    # (B)
    mx = mixed(models)
    groups = ['all single (k1)', 'mix: single rows', 'ctx n=10, query n=1', 'all n=10 (k10)', 'mix: replicated rows',
              'ctx n=1, query n=10']
    for idx, label in [(0, 'accuracy'), (1, 'mean confidence')]:
        print(f'\n**(B) {label}: single vs replicated rows** (mean over 27 datasets)\n')
        for nm in NOISES:
            print(f'\n*{nm}*\n\n| model | ' + ' | '.join(groups) + ' |\n|' + '---|' * (len(groups) + 1))
            for m in models:
                print(f'| {m} | ' + ' | '.join(fmt([t[idx] for t in mx[(m, nm, g)]]) for g in groups) + ' |')
    json.dump({'|'.join(map(str, k)): v for k, v in {**res, **mx}.items()}, open(os.path.join(ROOT, 'artifacts', 'unc.json'), 'w'))


if __name__ == '__main__':
    main()
