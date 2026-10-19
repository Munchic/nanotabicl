"""Step 8: aggregate results_*.jsonl -> tables (markdown) + figures. Paired comparisons across the same datasets."""
import argparse, json, os
from collections import defaultdict

import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
METRICS = ['acc', 'bal_acc', 'roc_auc', 'log_loss']


def load(path):
    R = defaultdict(dict)  # (arm, mode) -> dataset -> record
    for r in map(json.loads, open(path)):
        R[(r['arm'], r['mode'])][r['dataset']] = r
    return R


def paired(R, a, b, metric):
    """mean and SE of b - a over datasets present in both."""
    common = sorted(set(R[a]) & set(R[b]))
    d = np.array([R[b][k][metric] - R[a][k][metric] for k in common])
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), d, len(d)


def fmt(m, se):
    return f'{m:+.3f} ± {se:.3f}'


def pilot_table(pilot_dir):
    lines = ['| config | arm | n | Δacc (FT−ZS) | Δlog-loss (FT−ZS) | Δauc (FT−ZS) | weights changed | FT s/dataset |',
             '|---|---|---|---|---|---|---|---|']
    for fn in sorted(os.listdir(pilot_dir)):
        if not fn.startswith('results_'):
            continue
        R = load(os.path.join(pilot_dir, fn))
        for arm in sorted({a for a, _ in R}, key=lambda a: (a != 'clean', a)):
            zs, ft = (arm, 'zeroshot'), (arm, 'finetuned')
            cells = [fmt(*paired(R, zs, ft, m)[:2]) for m in ['acc', 'log_loss', 'roc_auc']]
            ch = np.mean([r['weights_changed'] for r in R[ft].values()])
            t = np.mean([r['ft_time_s'] for r in R[ft].values()])
            n = paired(R, zs, ft, 'acc')[3]
            lines.append(f"| {fn[8:-6]} | {arm} | {n} | {' | '.join(cells)} | {ch:.0%} | {t:.0f} |")
    return '\n'.join(lines)


def eval_tables(path, fig_dir):
    R = load(path)
    arms = sorted({a for a, _ in R}, key=lambda a: (a != 'clean', a))
    noises = [a for a in arms if a != 'clean']
    out = []

    out.append('### Absolute performance (mean over datasets)\n')
    out.append('| arm | mode | n | acc | bal_acc | roc_auc | log_loss |')
    out.append('|---|---|---|---|---|---|---|')
    for arm in arms:
        for mode in ['zeroshot', 'finetuned']:
            rs = list(R[(arm, mode)].values())
            out.append(f"| {arm} | {mode} | {len(rs)} | " + ' | '.join(f'{np.mean([r[m] for r in rs]):.3f}' for m in METRICS) + ' |')

    out.append('\n### Fine-tuning effect (FT − zero-shot, paired; ± = SE over datasets)\n')
    out.append('| arm | Δacc | Δroc_auc | Δlog_loss | FT wins (log-loss) | weights changed |')
    out.append('|---|---|---|---|---|---|')
    for arm in arms:
        zs, ft = (arm, 'zeroshot'), (arm, 'finetuned')
        _, _, dll, n = paired(R, zs, ft, 'log_loss')
        ch = np.mean([r['weights_changed'] for r in R[ft].values()])
        out.append(f"| {arm} | {fmt(*paired(R, zs, ft, 'acc')[:2])} | {fmt(*paired(R, zs, ft, 'roc_auc')[:2])} | "
                   f"{fmt(*paired(R, zs, ft, 'log_loss')[:2])} | {(dll < 0).sum()}/{n} | {ch:.0%} |")

    out.append('\n### Performance gap: noisy arm − clean (same mode, paired)\n')
    out.append('| arm | mode | Δacc | Δroc_auc | Δlog_loss | gap closed by FT (acc) |')
    out.append('|---|---|---|---|---|---|')
    gaps = {}
    for arm in noises:
        for mode in ['zeroshot', 'finetuned']:
            g = {m: paired(R, ('clean', mode), (arm, mode), m)[:2] for m in ['acc', 'roc_auc', 'log_loss']}
            gaps[(arm, mode)] = g
            closed = ''
            if mode == 'finetuned':
                g0, g1 = gaps[(arm, 'zeroshot')]['acc'][0], g['acc'][0]
                closed = f'{(g1 - g0) / -g0:+.0%}' if abs(g0) > 1e-3 else 'n/a'
            out.append(f"| {arm} | {mode} | {fmt(*g['acc'])} | {fmt(*g['roc_auc'])} | {fmt(*g['log_loss'])} | {closed} |")

    out.append('\n### Feature arm: full summary stats − mean only (same noise, same mode, paired)\n')
    out.append('| noise | mode | Δacc | Δroc_auc | Δlog_loss |')
    out.append('|---|---|---|---|---|')
    for nm in sorted({a.split('/')[0] for a in noises}, key=lambda s: [a.split('/')[0] for a in noises].index(s)):
        for mode in ['zeroshot', 'finetuned']:
            a, b = (f'{nm}/mean', mode), (f'{nm}/full', mode)
            out.append(f"| {nm} | {mode} | " + ' | '.join(fmt(*paired(R, a, b, m)[:2]) for m in ['acc', 'roc_auc', 'log_loss']) + ' |')

    # figure: gap per arm (accuracy + log loss)
    order = ['gauss', 'student_t3', 'censored', 'cauchy', 'cauchy_hetero']
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    for ax, metric in zip(axs, ['acc', 'log_loss']):
        x = np.arange(len(order))
        for i, (feats, mode, col, hatch) in enumerate([('mean', 'zeroshot', '#9ecae1', ''), ('mean', 'finetuned', '#3182bd', ''),
                                                       ('full', 'zeroshot', '#fdae6b', ''), ('full', 'finetuned', '#e6550d', '')]):
            ms, ses = zip(*[gaps[(f'{nm}/{feats}', mode)][metric] for nm in order])
            ax.bar(x + (i - 1.5) * 0.2, ms, 0.2, yerr=ses, color=col, label=f'{feats} / {mode}', capsize=2)
        ax.axhline(0, color='k', lw=0.7)
        ax.set_xticks(x, order)
        ax.set_title(f'{metric}: noisy − clean (paired, ±SE)')
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'gap.png'), dpi=110)

    # figure: per-dataset accuracy, zero-shot, clean vs each arm
    fig, axs = plt.subplots(1, len(order), figsize=(4 * len(order), 3.8), sharex=True, sharey=True)
    for ax, nm in zip(axs, order):
        for feats, col in [('mean', '#3182bd'), ('full', '#e6550d')]:
            ks = sorted(R[(f'{nm}/{feats}', 'zeroshot')])
            ax.scatter([R[('clean', 'zeroshot')][k]['acc'] for k in ks], [R[(f'{nm}/{feats}', 'zeroshot')][k]['acc'] for k in ks],
                       s=12, color=col, label=feats)
        ax.plot([0, 1], [0, 1], 'k--', lw=0.7)
        ax.set_title(nm)
        ax.set_xlabel('clean acc (zero-shot)')
    axs[0].set_ylabel('noisy acc (zero-shot)')
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'per_dataset_acc.png'), dpi=110)
    return '\n'.join(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=os.path.join(ROOT, 'artifacts', 'runs'))
    ap.add_argument('--stage', choices=['pilot', 'eval'], required=True)
    args = ap.parse_args()
    fig_dir = os.path.join(ROOT, 'figures')
    if args.stage == 'pilot':
        print(pilot_table(os.path.join(args.runs, 'pilot')))
    else:
        md = eval_tables(os.path.join(args.runs, 'eval', 'results_eval.jsonl'), fig_dir)
        open(os.path.join(fig_dir, 'results.md'), 'w').write(md + '\n')
        print(md)
