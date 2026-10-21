"""PIT histograms (regression, Exp 5): pooled PIT values u = F_hat(y) over the 27 datasets' test rows, per model and
setting, from artifacts/probs_reg/<model>/<dataset>.npz (keys u__<setting>). Flat = calibrated; U-shape = overconfident
(intervals too narrow); hump = underconfident; tilt = biased. Usage: python pit_fig.py --models a,b --settings s1,s2 --out name"""
import argparse, glob, os

import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))


def pooled(model, s):
    us = [np.load(f)[f'u__{s}'] for f in sorted(glob.glob(os.path.join(ROOT, 'artifacts', 'probs_reg', model, '*.npz')))
          if f'u__{s}' in np.load(f).files]
    return np.concatenate(us) if us else None


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True)
    ap.add_argument('--settings', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    models, settings = a.models.split(','), a.settings.split(',')
    fig, axs = plt.subplots(len(models), len(settings), figsize=(2.6 * len(settings), 2.0 * len(models)),
                            sharex=True, squeeze=False)
    bins = np.linspace(0, 1, 21)
    for i, m in enumerate(models):
        for j, s in enumerate(settings):
            ax = axs[i, j]
            u = pooled(m, s)
            if u is not None:
                ax.hist(u, bins=bins, density=True, color='#3182bd')
                ce = np.mean(np.abs([np.mean(u <= t) - t for t in np.arange(0.05, 0.951, 0.05)]))
                ax.set_title(f'{s}\nPIT err {ce:.3f}', fontsize=7)
            ax.axhline(1, color='k', lw=0.7, ls='--')
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(m, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, 'figures', a.out + '.png'), dpi=110)
