"""Summary statistics + plots of the stored eval datasets (clean and observed). Writes figures/ and figures/datasets.csv."""
import json, os

import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'artifacts', 'data')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(FIG, exist_ok=True)

m = json.load(open(os.path.join(DATA, 'manifest.json')))
NOISE = m['NOISE_MODELS']
ds = [(e, np.load(os.path.join(DATA, 'eval', e['name'] + '.npz'))) for e in m['eval']]


def corr_cols(A, B):
    return [np.corrcoef(A[:, j], B[:, j])[0, 1] for j in range(A.shape[1]) if A[:, j].std() > 0 and B[:, j].std() > 0]


# ----- per-dataset table -----
rows = []
for e, z in ds:
    X, y = z['X_clean'], z['y']
    frac = np.bincount(y) / len(y)
    sd = X.std(0)
    r = dict(name=e['name'], n=len(y), n_features=X.shape[1], n_classes=len(frac), majority_frac=frac.max(),
             minority_frac=frac.min(), median_abs_skew=np.median(np.abs(skew(X, axis=0))),
             median_excess_kurtosis=np.median(kurtosis(X, axis=0)),
             n_unique_median=int(np.median([len(np.unique(X[:, j])) for j in range(X.shape[1])])),
             near_constant_cols=int((sd < 1e-6).sum()))
    for nm in NOISE:
        f = z[f'{nm}__full'].reshape(len(y), X.shape[1], -1)
        r[f'r_mean_{nm}'] = np.median(corr_cols(f[:, :, 0], X))
        r[f'r_median_{nm}'] = np.median(corr_cols(f[:, :, 1], X))
    reps = z['censored__replicates']  # float32, so compare against float32 limits
    lo, hi = (np.quantile(X, q, 0).astype(np.float32)[None, :, None] for q in (0.1, 0.9))
    r['censored_frac'] = float(((reps == lo) | (reps == hi)).sum() / np.isfinite(reps).sum())
    rows.append(r)
keys = list(rows[0])
with open(os.path.join(FIG, 'datasets.csv'), 'w') as f:
    f.write(','.join(keys) + '\n')
    for r in rows:
        f.write(','.join(f'{r[k]:.4g}' if isinstance(r[k], float) else str(r[k]) for k in keys) + '\n')

# ----- fig 1: clean datasets, first two features coloured by class -----
fig, axs = plt.subplots(3, 9, figsize=(22, 7.5))
for ax, (e, z) in zip(axs.flat, ds):
    X, y = z['X_clean'], z['y']
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', vmin=0, vmax=9, s=4, linewidths=0)
    ax.set(xticks=[], yticks=[])
    ax.set_title(f"{e['name'][:7]} d={X.shape[1]} k={len(np.unique(y))}", fontsize=8)
fig.suptitle('Eval datasets (clean): x0 vs x1, coloured by class')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'datasets_clean.png'), dpi=110)

# ----- fig 2: observation models on one column -----
e, z = ds[12]
X = z['X_clean']
j = int(np.argmax(X.std(0)))
fig, axs = plt.subplots(2, len(NOISE), figsize=(4 * len(NOISE), 7.5), sharex=True)
lim = np.quantile(X[:, j], [0.001, 0.999]) + np.array([-1, 1]) * X[:, j].std()
for c, nm in enumerate(NOISE):
    f = z[f'{nm}__full'].reshape(len(X), X.shape[1], -1)
    for r, (si, sn) in enumerate([(0, 'mean'), (1, 'median')]):
        ax = axs[r, c]
        ax.scatter(X[:, j], f[:, j, si], c=z[f'{nm}__counts'][:, j], cmap='viridis', s=5, vmin=1, vmax=10)
        ax.plot(lim, lim, 'k--', lw=0.7)
        ax.set_ylim(lim)
        rr = np.corrcoef(X[:, j], f[:, j, si])[0, 1]
        ax.set_title(f'{nm}: x{j}_{sn}  (r={rr:.2f})', fontsize=9)
        if c == 0:
            ax.set_ylabel(f'observed {sn}')
        if r == 1:
            ax.set_xlabel(f'clean x{j}')
fig.suptitle(f"{e['name']}: observed summary vs clean value (colour = replicate count n, 1..10; y clipped)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'observation_models.png'), dpi=110)

# ----- fig 3: signal retention across all columns of all eval datasets -----
fig, ax = plt.subplots(figsize=(9, 4))
data, labels = [], []
for nm in NOISE:
    for si, sn in [(0, 'mean'), (1, 'median')]:
        data.append(np.concatenate([corr_cols(z[f'{nm}__full'].reshape(len(z['y']), z['X_clean'].shape[1], -1)[:, :, si],
                                              z['X_clean']) for _, z in ds]))
        labels.append(f'{nm}\n{sn}')
bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
for i, p in enumerate(bp['boxes']):
    p.set_facecolor('#9ecae1' if i % 2 == 0 else '#fdae6b')
ax.set_ylabel('corr(observed stat, clean value)')
ax.set_title('Signal retained per column (all 27 eval datasets); blue = mean, orange = median')
ax.tick_params(axis='x', labelsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'signal_retention.png'), dpi=110)

# ----- fig 4: per-cell error of the mean vs replicate count -----
fig, axs = plt.subplots(1, len(NOISE), figsize=(4 * len(NOISE), 3.4), sharey=True)
for ax, nm in zip(axs, NOISE):
    for si, sn, col in [(0, 'mean', '#3182bd'), (1, 'median', '#e6550d')]:
        err = {n: [] for n in range(1, 11)}
        for _, z in ds:
            X = z['X_clean']
            f = z[f'{nm}__full'].reshape(len(X), X.shape[1], -1)[:, :, si]
            sd = X.std(0)
            sd[sd < 1e-12] = 1
            ae = np.abs(f - X) / sd
            cnt = z[f'{nm}__counts']
            for n in range(1, 11):
                err[n].append(ae[cnt == n])
        med = [np.median(np.concatenate(err[n])) for n in range(1, 11)]
        ax.plot(range(1, 11), med, 'o-', color=col, label=sn, ms=3)
    ax.set_yscale('log')
    ax.set_title(nm, fontsize=9)
    ax.set_xlabel('replicate count n')
axs[0].set_ylabel('median |stat − clean| / sd(col)')
axs[0].legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'error_vs_count.png'), dpi=110)

print(open(os.path.join(FIG, 'datasets.csv')).read())
