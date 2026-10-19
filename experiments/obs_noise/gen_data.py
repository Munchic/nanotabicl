"""Step 0/1/4/5: generate prior datasets, apply observation-model noise, featurize replicates.

Everything is seeded per (dataset, noise model), so any single dataset can be regenerated independently.
Eval datasets are stored in full (clean X, raw replicates, featurized arms). The pilot pool (disjoint seeds, used only
to pick fine-tuning hyperparameters) stores clean X/y; its noise + features are regenerated deterministically via
`make_arm`.
"""
import argparse, json, os, sys, time
from multiprocessing import Pool

import numpy as np, torch
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, '..', '..'))
from prior import rand_dataset_filtered  # noqa: E402

N_ROWS, N_TRAIN = 1024, 512
FEATURE_GRID = [4, 8, 16]
CLASS_GRID = [2, 4, 8]
MIN_CLASS_COUNT = 16
MAX_REPS = 10
NOISE_SCALE = 0.5  # per-replicate noise scale, in units of the clean column's std

SEED_EVAL, SEED_PILOT, SEED_NOISE = 1_000, 2_000_000, 3_000_000
NOISE_MODELS = ['gauss', 'student_t3', 'censored', 'cauchy', 'cauchy_hetero',  # Exp 1 (easy -> hard)
                'gauss_valhet', 'cauchy_valhet', 'censored_left', 'censored_left_heavy']  # Exp 2 additions (appended:
# the noise seed includes the model's index, so appending keeps every earlier model's arrays bit-identical)
LOD_Q = {'censored_left': 0.3, 'censored_left_heavy': 0.6}  # lower detection limit = this quantile of the clean column
FEATURE_ARMS = {'full': ['mean', 'median', 'sd', 'iqr', 'sem', 'logn'], 'mean': ['mean'],  # 'mean' = what imputation collapses replicates to
                'median': ['median']}  # median-only (robust imputation); added in Exp 3


# ----- Step 1: prior datasets -----

def gen_dataset(seed: int, n_features: int, n_classes: int) -> dict:
    # resample with a derived seed until every class has enough rows for a stratified split
    for attempt in range(100):
        s = seed * 100 + attempt
        np.random.seed(s % 2**32)
        torch.manual_seed(s)
        cols = rand_dataset_filtered(x_cat_sizes=[0] * n_features, y_cat_sizes=[n_classes], n_samples=N_ROWS)
        X = torch.cat([cols[f'x_{i}'] for i in range(n_features)], dim=-1).double().numpy()
        y = np.unique(cols['y_0'].squeeze(-1).long().numpy(), return_inverse=True)[1]
        counts = np.bincount(y)
        if len(counts) >= 2 and counts.min() >= MIN_CLASS_COUNT:
            break
    else:
        raise RuntimeError(f'no valid dataset for seed {seed}')
    # stratified train/test split (seeded)
    train_idx, test_idx = train_test_split(np.arange(N_ROWS), train_size=N_TRAIN, stratify=y, random_state=seed)
    return dict(X=X, y=y, train_idx=train_idx, test_idx=test_idx, prior_seed=s, n_features=n_features,
                n_classes_requested=n_classes, n_classes=len(counts), attempts=attempt + 1)


def _gen_worker(args):
    torch.set_num_threads(1)
    return gen_dataset(*args)


# ----- Step 4: observation model -----

def observe(X: np.ndarray, noise: str, seed_key: list[int], n_fixed: int | None = None, scale_mult: float = 1.0,
            return_scale: bool = False):
    """Returns replicates (n, d, K) padded with NaN and counts (n, d). Default: K = MAX_REPS, counts ~ U{1..10} iid per
    cell. n_fixed: every cell gets exactly n_fixed replicates (K = n_fixed). scale_mult multiplies NOISE_SCALE.
    return_scale: also return the per-replicate noise scale (n, d, 1) and the row scale factor (for the Bayes oracle)."""
    rng = np.random.default_rng([SEED_NOISE, *seed_key, NOISE_MODELS.index(noise)])
    n, d = X.shape
    K = MAX_REPS if n_fixed is None else n_fixed
    counts = rng.integers(1, MAX_REPS + 1, size=(n, d)) if n_fixed is None else np.full((n, d), n_fixed)
    col_sd = X.std(axis=0)
    col_sd[col_sd < 1e-12] = 1.0
    scale = scale_mult * NOISE_SCALE * col_sd[None, :, None]  # (1, d, 1)

    if noise in ('gauss', 'censored', 'gauss_valhet', 'censored_left'):
        eps = rng.standard_normal((n, d, K))
    elif noise == 'student_t3':
        eps = rng.standard_t(3, size=(n, d, K))
    elif noise == 'censored_left_heavy':
        eps = rng.standard_t(2, size=(n, d, K))  # t2: most replicates near the value, some far outliers
    elif noise in ('cauchy', 'cauchy_hetero', 'cauchy_valhet'):
        eps = rng.standard_cauchy((n, d, K))
    else:
        raise ValueError(noise)
    if noise == 'cauchy_hetero':
        # heteroscedastic: per-row reliability, lognormal scale (median 1x, ~16% of rows >2.7x, ~16% <0.37x)
        scale = scale * np.exp(rng.standard_normal((n, 1, 1)))
    if noise.endswith('_valhet'):
        # value-dependent (one-directional) heteroscedasticity: scale = 2 * percentile rank of the clean value in its
        # column, so the smallest values are ~exact and the largest get 2x the base noise (average multiplier = 1)
        u = (np.argsort(np.argsort(X, axis=0, kind='stable'), axis=0, kind='stable') + 1) / n
        scale = scale * 2 * u[:, :, None]
    reps = X[:, :, None] + scale * eps
    if noise == 'censored':
        # assay limits at the 10th / 90th percentile of the clean column; values outside are reported AT the limit
        lo, hi = np.quantile(X, 0.1, axis=0), np.quantile(X, 0.9, axis=0)
        reps = np.clip(reps, lo[None, :, None], hi[None, :, None])
    if noise in LOD_Q:  # left-censoring at a lower detection limit (values below are reported AT the limit)
        reps = np.maximum(reps, np.quantile(X, LOD_Q[noise], axis=0)[None, :, None])
    reps[np.arange(K)[None, None, :] >= counts[:, :, None]] = np.nan
    if return_scale:
        return reps, counts, np.broadcast_to(scale, (n, d, 1)).copy()
    return reps, counts


# ----- Step 5: summary statistics -----

def summarize(reps: np.ndarray, counts: np.ndarray, stats: list[str]) -> tuple[np.ndarray, list[str]]:
    out = []
    for st in stats:
        if st == 'mean':
            v = np.nanmean(reps, axis=-1)
        elif st == 'median':
            v = np.nanmedian(reps, axis=-1)
        elif st == 'sd':
            v = np.where(counts > 1, np.nanstd(reps, axis=-1, ddof=1), 0.0)  # sd undefined for n=1 -> 0
        elif st == 'iqr':
            v = np.nanquantile(reps, 0.75, axis=-1) - np.nanquantile(reps, 0.25, axis=-1)
        elif st == 'sem':
            v = np.where(counts > 1, np.nanstd(reps, axis=-1, ddof=1) / np.sqrt(counts), 0.0)
        elif st == 'logn':
            v = np.log(counts)
        out.append(v)
    d = counts.shape[1]
    names = [f'x{j}_{st}' for j in range(d) for st in stats]
    return np.stack(out, axis=-1).reshape(counts.shape[0], -1), names  # column order: x0_*, x1_*, ...


def make_arm(X: np.ndarray, noise: str | None, feats: str | None, seed_key: list[int]):
    """Featurized design matrix for an arm. noise=None -> clean X."""
    if noise is None:
        return X, [f'x{j}' for j in range(X.shape[1])]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # nanstd on n=1 cells; masked by np.where
        reps, counts = observe(X, noise, seed_key)
        return summarize(reps, counts, FEATURE_ARMS[feats])


ARMS = [('clean', None, None)] + [(f'{nm}/{fa}', nm, fa) for nm in NOISE_MODELS for fa in FEATURE_ARMS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'artifacts', 'data'))
    ap.add_argument('--eval_seeds_per_config', type=int, default=3)
    ap.add_argument('--n_pilot', type=int, default=64)
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, 'eval'), exist_ok=True)

    configs = [(d, k) for d in FEATURE_GRID for k in CLASS_GRID]
    eval_jobs = [(SEED_EVAL + i * 1000 + r, d, k) for i, (d, k) in enumerate(configs)
                 for r in range(args.eval_seeds_per_config)]
    rng = np.random.default_rng(SEED_PILOT)
    pilot_jobs = [(SEED_PILOT + i, int(rng.choice(FEATURE_GRID)), int(rng.choice(CLASS_GRID))) for i in range(args.n_pilot)]

    t0 = time.time()
    with Pool(args.workers) as pool:
        eval_ds = pool.map(_gen_worker, eval_jobs, chunksize=1)
        print(f'eval: {len(eval_ds)} datasets in {time.time() - t0:.0f}s', flush=True)
        pilot_ds = pool.map(_gen_worker, pilot_jobs, chunksize=8)
        print(f'pilot pool: {len(pilot_ds)} datasets in {time.time() - t0:.0f}s', flush=True)

    manifest = []
    for i, ((seed, d, k), ds) in enumerate(zip(eval_jobs, eval_ds)):
        name = f'eval_{i:02d}_d{d}_k{k}_s{seed}'
        arrays = dict(X_clean=ds['X'], y=ds['y'], train_idx=ds['train_idx'], test_idx=ds['test_idx'])
        for nm in NOISE_MODELS:
            reps, counts = observe(ds['X'], nm, [seed])
            arrays[f'{nm}__replicates'] = reps.astype(np.float32)
            arrays[f'{nm}__counts'] = counts.astype(np.int8)
            for fa in FEATURE_ARMS:
                arrays[f'{nm}__{fa}'] = make_arm(ds['X'], nm, fa, [seed])[0]
        np.savez_compressed(os.path.join(args.out, 'eval', name + '.npz'), **arrays)
        manifest.append(dict(name=name, seed=seed, prior_seed=ds['prior_seed'], n_features=d,
                             n_classes_requested=k, n_classes=ds['n_classes'], attempts=ds['attempts'],
                             class_counts=np.bincount(ds['y']).tolist()))

    # prior output is float32, so storing float32 is lossless; X padded with NaN to max(FEATURE_GRID) columns
    X_pool = np.full((len(pilot_ds), N_ROWS, max(FEATURE_GRID)), np.nan, dtype=np.float32)
    for i, ds in enumerate(pilot_ds):
        X_pool[i, :, :ds['n_features']] = ds['X']
    np.savez_compressed(os.path.join(args.out, 'pilot_pool.npz'), X=X_pool,
                        n_features=np.array([ds['n_features'] for ds in pilot_ds]),
                        y=np.stack([ds['y'] for ds in pilot_ds]),
                        train_idx=np.stack([ds['train_idx'] for ds in pilot_ds]),
                        test_idx=np.stack([ds['test_idx'] for ds in pilot_ds]),
                        seed=np.array([s for s, _, _ in pilot_jobs]))
    meta = dict(N_ROWS=N_ROWS, N_TRAIN=N_TRAIN, MAX_REPS=MAX_REPS, NOISE_SCALE=NOISE_SCALE, MIN_CLASS_COUNT=MIN_CLASS_COUNT,
                SEED_EVAL=SEED_EVAL, SEED_PILOT=SEED_PILOT, SEED_NOISE=SEED_NOISE, NOISE_MODELS=NOISE_MODELS,
                FEATURE_ARMS=FEATURE_ARMS, eval=manifest,
                pilot_pool=dict(n=len(pilot_ds), configs=[[d, k] for _, d, k in pilot_jobs],
                             n_classes=[ds['n_classes'] for ds in pilot_ds]))
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
