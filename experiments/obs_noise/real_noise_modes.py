"""Which observation modes are present in real replicate data? (real-data to-do 4; LINCS L1000 + Cell Painting)

Cells = (compound, dose) x feature with >= 3 replicate wells (controls excluded). Leave-one-out residual of replicate k:
e_k = r_k - median(other replicates of the cell), standardized per feature by 1.4826 x the median |e| of that feature.
Signatures checked, against what each simulated mode would produce:
  heavy tails (cauchy / t)   : P(|e| > t) far above Gaussian; extremes symmetric in sign per feature
  fail_shift                 : extremes concentrated in ONE sign per feature, at a roughly fixed offset (a bump in
                               the |e| histogram, not a monotone tail)
  fail_floor / non-detects   : many exactly tied values at the column minimum
  well (replicate) failures  : one replicate is extreme in many features at once (paired across features, NOT
                               independent per cell as in the training prior)
Usage: python real_noise_modes.py [--max_features 300]
"""
import argparse, os

import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, 'artifacts', 'real_raw', 'lincs')
GAUSS_TAIL = {3: 2.7e-3, 5: 5.7e-7, 8: 1.2e-15}


def load(which, max_features, rng):
    if which == 'L1000':
        df = pd.read_csv(os.path.join(RAW, 'L1000_level4_cpd_replicates.csv.gz'))
        meta = ['replicate_id', 'sig_id', 'pert_id', 'pert_idose', 'det_plate', 'det_well', 'dose',
                'Metadata_broad_sample', 'pert_iname', 'moa']
        df = df[df['moa'].fillna('') != 'Control vehicle']
        key = df['pert_id'].astype(str) + '|' + df['dose'].astype(str)
        well = df['det_plate'].astype(str) + ':' + df['det_well'].astype(str)
    else:
        df = pd.read_csv(os.path.join(RAW, 'cp_level4_cpd_replicates_nonspherized.csv.gz'), low_memory=False)
        meta = [c for c in df.columns if c.startswith('Metadata_')]
        df = df[df['Metadata_broad_sample'].astype(str).str.upper() != 'DMSO']
        key = df['Metadata_broad_sample'].astype(str) + '|' + df['Metadata_dose_recode'].astype(str)
        well = df['Metadata_Plate'].astype(str) + ':' + df['Metadata_Well'].astype(str)
    feats = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    if len(feats) > max_features:
        feats = sorted(rng.choice(feats, max_features, replace=False))
    return df[feats].to_numpy(np.float64), key.to_numpy(), well.to_numpy(), feats


def residuals(X, key):
    """-> e (n_rep_rows, d) leave-one-out residuals (NaN for cells with < 3 replicates), group sizes per row."""
    e = np.full_like(X, np.nan)
    order = np.argsort(key, kind='stable')
    ks, starts = np.unique(key[order], return_index=True)
    bounds = list(starts) + [len(order)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        if len(idx) < 3:
            continue
        R = X[idx]
        for j, i in enumerate(idx):
            e[i] = R[j] - np.median(np.delete(R, j, axis=0), axis=0)
    return e


def report(which, X, key, well):
    e = residuals(X, key)
    ok = ~np.isnan(e).all(1)
    e, w = e[ok], well[ok]
    scale = 1.4826 * np.nanmedian(np.abs(e), axis=0)
    z = e / np.where(scale > 0, scale, np.nan)
    a = np.abs(z)
    n_cells = np.isfinite(z).sum()
    print(f'\n## {which}: {ok.sum()} replicate rows in cells with >= 3 replicates, {X.shape[1]} features, {n_cells} residuals')
    print('tail mass P(|e| > t), observed vs Gaussian: ' +
          ', '.join(f't={t}: {np.nanmean(a > t):.2e} vs {g:.0e}' for t, g in GAUSS_TAIL.items()))
    # sign concentration of extremes per feature (fail_shift -> near 0 or 1; heavy tail -> near 0.5)
    ext = a > 5
    frac_pos, n_ext = [], []
    for j in range(z.shape[1]):
        m = ext[:, j]
        if m.sum() >= 20:
            frac_pos.append(np.mean(z[m, j] > 0))
            n_ext.append(m.sum())
    frac_pos, n_ext = np.array(frac_pos), np.array(n_ext)
    if len(frac_pos):
        dev = np.abs(frac_pos - 0.5)
        null = np.median(0.5 * np.sqrt(1 / n_ext)) * 0.6745 * 2  # median |binomial deviation| under sign symmetry
        print(f'features with >= 20 extremes (|e| > 5): {len(frac_pos)}; median |frac positive - 0.5| = {np.median(dev):.3f} '
              f'(sign-symmetric null ~ {null:.3f}); features with > 90% one sign: {np.mean(dev > 0.4):.1%}')
    # |e| histogram beyond 3: monotone (tail) or bump (fixed shift)?
    edges = np.array([3, 4, 5, 6, 8, 10, 15, 25, np.inf])
    cnt = np.histogram(a[np.isfinite(a)], bins=edges)[0]
    dens = cnt / np.diff(np.minimum(edges, 50))
    print('|e| density per unit in bins ' + ', '.join(f'[{lo:g},{hi:g})' for lo, hi in zip(edges[:-1], edges[1:])) +
          ': ' + ', '.join(f'{d:.0f}' for d in dens) + ('  -> monotone (no fixed-offset bump)' if np.all(np.diff(dens[:-1]) <= 0)
                                                     else '  -> NON-monotone (possible bump)'))
    # ties at the column minimum (floor / non-detect substitution)
    tied = np.mean([np.mean(X[:, j] == np.nanmin(X[:, j])) for j in range(X.shape[1])])
    print(f'share of values tied at the column minimum: {tied:.2e}')
    # replicate (well)-level coupling: share of a replicate's features that are extreme
    p = np.nanmean(ext)
    per_rep = np.nanmean(ext, axis=1)
    print(f'replicate-level coupling: overall extreme rate {p:.2e}; replicates with > 10x that rate in their row: '
          f'{np.mean(per_rep > 10 * p):.1%}; they hold {ext[per_rep > 10 * p].sum() / max(ext.sum(), 1):.0%} of all extremes')
    return dict(dataset=which, tail5=float(np.nanmean(a > 5)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_features', type=int, default=300)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    for which in ['L1000', 'CellPainting']:
        X, key, well, feats = load(which, args.max_features, rng)
        report(which, X, key, well)
