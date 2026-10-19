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
TAG = os.environ.get('OBS_TAG', '')  # '' = classification artifacts, '_reg' = regression (Exp 5)
art = lambda name: os.path.join(ROOT, 'artifacts', name + TAG)
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
                'gauss_valhet', 'cauchy_valhet', 'censored_left', 'censored_left_heavy',  # Exp 2 additions (appended:
                'fail_floor', 'fail_shift',  # Exp 6: median-breaking failure modes (appended likewise)
                'cv_lognormal', 'lod_logistic', 'saturation', 'plate_effect']  # Exp 6: literature-based modes
# the noise seed includes the model's index, so appending keeps every earlier model's arrays bit-identical)
LOD_Q = {'censored_left': 0.3, 'censored_left_heavy': 0.6}  # lower detection limit = this quantile of the clean column
# failure modes: each replicate fails w.p. p_cell ~ U(0, FAIL_P_MAX) (per cell); good replicates = x + 0.25 sd * N(0,1);
# fail_floor: failed reading = column floor (clean min - 1 sd); fail_shift: failed reading = x + 3 sd + 0.25 sd * N(0,1)
FAIL_P_MAX, FAIL_GOOD_MULT, FAIL_SHIFT = 0.8, 0.5, 3.0
# literature-based modes (RESEARCH.md Exp 6), all with good-replicate noise 0.25 sd unless stated:
# cv_lognormal: constant-CV multiplicative error around a floor m = min - 0.25 sd: r = m + (x - m) exp(CV eps - CV^2/2)
# lod_logistic: intensity-dependent detection (logit-linear, Li & Smyth 2023): detected w.p. sigmoid((r - theta) / s),
#   theta = clean 35th pct, s = 0.3 sd; undetected replicates are reported at the substitution floor (clean 5th pct)
# saturation: right-censoring at an upper limit of quantification (clean 75th pct), Gaussian noise 0.5 sd
# plate_effect: replicate k comes from run/plate k, shared by every cell of the table: r = x + sd (b_k + c_kj) + 0.25 sd eps,
#   b_k ~ N(0, 0.5^2) (all columns), c_kj ~ N(0, 0.2^2) (per column)
CV_LN, LOD_THETA_Q, LOD_SLOPE, LOD_FLOOR_Q, ULOQ_Q, PLATE_SD, PLATE_COL_SD = 0.35, 0.35, 0.3, 0.05, 0.75, 0.5, 0.2
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
        if n_classes == 0:  # regression (Exp 5): continuous target, random split
            y = cols['y_0'].squeeze(-1).double().numpy()
            if np.isfinite(y).all() and y.std() > 1e-8:
                rng = np.random.default_rng([seed, 7])
                perm = rng.permutation(N_ROWS)
                return dict(X=X, y=y, train_idx=perm[:N_TRAIN], test_idx=perm[N_TRAIN:], prior_seed=s,
                            n_features=n_features, n_classes_requested=0, n_classes=0, attempts=attempt + 1)
            continue
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

    if noise in ('gauss', 'censored', 'gauss_valhet', 'censored_left', 'fail_floor', 'fail_shift', 'cv_lognormal',
                 'lod_logistic', 'saturation', 'plate_effect'):
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
    if noise.startswith('fail_') or noise in ('lod_logistic', 'plate_effect'):
        scale = scale * FAIL_GOOD_MULT  # good replicates are precise (0.25 sd)
    reps = X[:, :, None] + scale * eps
    if noise == 'cv_lognormal':
        m = (X.min(axis=0) - 0.25 * col_sd)[None, :, None]
        reps = m + (X[:, :, None] - m) * np.exp(CV_LN * eps - CV_LN ** 2 / 2)
    if noise == 'lod_logistic':
        theta, floor = (np.quantile(X, q, axis=0)[None, :, None] for q in (LOD_THETA_Q, LOD_FLOOR_Q))
        p_det = 1 / (1 + np.exp(-(reps - theta) / (LOD_SLOPE * col_sd[None, :, None])))
        reps = np.where(rng.random((n, d, K)) < p_det, reps, floor)
    if noise == 'saturation':
        reps = np.minimum(reps, np.quantile(X, ULOQ_Q, axis=0)[None, :, None])
    if noise == 'plate_effect':
        b = rng.normal(0, PLATE_SD, size=(1, 1, K)) + rng.normal(0, PLATE_COL_SD, size=(1, d, K))
        reps = reps + col_sd[None, :, None] * b
    if noise.startswith('fail_'):
        p_fail = rng.uniform(0, FAIL_P_MAX, size=(n, d, 1))
        failed = rng.random((n, d, K)) < p_fail
        if noise == 'fail_floor':
            floor = (X.min(axis=0) - col_sd)[None, :, None]
            reps = np.where(failed, floor, reps)
        else:
            reps = np.where(failed, reps + FAIL_SHIFT * col_sd[None, :, None], reps)
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
    ap.add_argument('--task', choices=['classification', 'regression'], default='classification')
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, 'eval'), exist_ok=True)

    regression = args.task == 'regression'
    if regression:  # 27 regression datasets: 3 feature counts x 9 seeds, seeds disjoint from classification
        configs, args.eval_seeds_per_config, args.n_pilot = [(d, 0) for d in FEATURE_GRID], 9, 0
    else:
        configs = [(d, k) for d in FEATURE_GRID for k in CLASS_GRID]
    base = SEED_EVAL + (500_000 if regression else 0)
    eval_jobs = [(base + i * 1000 + r, d, k) for i, (d, k) in enumerate(configs)
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
                             class_counts=None if regression else np.bincount(ds['y']).tolist(), task=args.task))

    # prior output is float32, so storing float32 is lossless; X padded with NaN to max(FEATURE_GRID) columns
    X_pool = np.full((len(pilot_ds), N_ROWS, max(FEATURE_GRID)), np.nan, dtype=np.float32)
    for i, ds in enumerate(pilot_ds):
        X_pool[i, :, :ds['n_features']] = ds['X']
    if pilot_ds:
        np.savez_compressed(os.path.join(args.out, 'pilot_pool.npz'), X=X_pool,
                            n_features=np.array([ds['n_features'] for ds in pilot_ds]),
                            y=np.stack([ds['y'] for ds in pilot_ds]),
                            train_idx=np.stack([ds['train_idx'] for ds in pilot_ds]),
                            test_idx=np.stack([ds['test_idx'] for ds in pilot_ds]),
                            seed=np.array([s for s, _, _ in pilot_jobs]))
    meta = dict(N_ROWS=N_ROWS, N_TRAIN=N_TRAIN, MAX_REPS=MAX_REPS, NOISE_SCALE=NOISE_SCALE, MIN_CLASS_COUNT=MIN_CLASS_COUNT,
                SEED_EVAL=SEED_EVAL, SEED_PILOT=SEED_PILOT, SEED_NOISE=SEED_NOISE, NOISE_MODELS=NOISE_MODELS,
                FEATURE_ARMS=FEATURE_ARMS, task=args.task, eval=manifest,
                pilot_pool=dict(n=len(pilot_ds), configs=[[d, k] for _, d, k in pilot_jobs],
                             n_classes=[ds['n_classes'] for ds in pilot_ds]))
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
