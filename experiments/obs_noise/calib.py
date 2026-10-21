"""Calibration of saved holdout probabilities (artifacts/probs/<tag>/<dataset>.npz): per (model, setting) mean over 27
datasets of top-label ECE (15 equal-width bins), overconfidence (mean confidence - accuracy), Brier score and NLL;
plus pooled reliability diagrams. Usage: python calib.py --models tag1,tag2 --settings clean,gauss,... [--fig name]"""
import argparse, glob, os

import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
BINS = np.linspace(0, 1, 16)


def calib_stats(p, y):
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == y).astype(float)
    idx = np.clip(np.digitize(conf, BINS) - 1, 0, len(BINS) - 2)
    ece = sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
              for b in range(len(BINS) - 1) if (idx == b).any())
    onehot = np.eye(p.shape[1])[y]
    return dict(ece=ece, overconf=conf.mean() - correct.mean(), brier=((p - onehot) ** 2).sum(1).mean(),
                nll=-np.log(p[np.arange(len(y)), y]).mean(), conf=conf, correct=correct)


def load(tag, setting):
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, 'artifacts', 'probs', tag, '*.npz'))):
        z = np.load(f)
        if f'p__{setting}' in z.files:
            out.append((z[f'p__{setting}'].astype(np.float64), z['y_test']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True)
    ap.add_argument('--settings', required=True)
    ap.add_argument('--fig', default=None, help='write a reliability-diagram grid to figures/<fig>.png')
    a = ap.parse_args()
    models, settings = a.models.split(','), a.settings.split(',')
    table = {}
    for m in models:
        for s in settings:
            per = [calib_stats(p, y) for p, y in load(m, s)]
            if per:
                table[(m, s)] = per
    for key, label in [('ece', 'ECE (top-label, 15 bins)'), ('overconf', 'overconfidence = mean confidence − accuracy'),
                       ('brier', 'Brier score')]:
        print(f'\n**{label}** (mean over datasets ± SE)\n')
        print('| model | ' + ' | '.join(settings) + ' |\n|' + '---|' * (len(settings) + 1))
        for m in models:
            cells = []
            for s in settings:
                v = np.array([d[key] for d in table.get((m, s), [])])
                cells.append(f'{v.mean():+.3f}±{v.std(ddof=1) / np.sqrt(len(v)):.3f}' if key == 'overconf' and len(v)
                             else (f'{v.mean():.3f}' if len(v) else ''))
            print(f'| {m} | ' + ' | '.join(cells) + ' |')
    if a.fig:
        fig, axs = plt.subplots(1, len(settings), figsize=(3.6 * len(settings), 3.6), sharey=True)
        for ax, s in zip(np.atleast_1d(axs), settings):
            for m in models:
                per = table.get((m, s), [])
                if not per:
                    continue
                conf = np.concatenate([d['conf'] for d in per])
                corr = np.concatenate([d['correct'] for d in per])
                idx = np.clip(np.digitize(conf, BINS) - 1, 0, len(BINS) - 2)
                xs = [conf[idx == b].mean() for b in range(len(BINS) - 1) if (idx == b).sum() >= 30]
                ys = [corr[idx == b].mean() for b in range(len(BINS) - 1) if (idx == b).sum() >= 30]
                ax.plot(xs, ys, 'o-', ms=3, label=m)
            ax.plot([0, 1], [0, 1], 'k--', lw=0.7)
            ax.set(title=s, xlabel='confidence', xlim=(0, 1), ylim=(0, 1))
        np.atleast_1d(axs)[0].set_ylabel('accuracy')
        np.atleast_1d(axs)[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(ROOT, 'figures', a.fig + '.png'), dpi=110)


if __name__ == '__main__':
    main()
