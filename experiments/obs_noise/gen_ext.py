"""Exp 3 extended eval settings on the same 27 eval datasets (same X, y, splits as artifacts/data):

  <noise>_n20, <noise>_n50 : every cell has exactly 20 / 50 replicates (training never saw more than 10)
  grouped                  : a visible group column g (4 groups, ~uniform); each group has its own observation model
                             (see GROUPS); counts ~ U{1..10}; the group column itself is exact (n = 1)
  grouped_clean            : clean X + the same group column (ceiling for `grouped`)

Each setting stores <s>__replicates (N, d, K), <s>__counts, <s>__mean, <s>__median, <s>__full (same featurization as
gen_data.py); grouped settings also store <s>__X_clean and grouped__group.
"""
import json, os, warnings

import numpy as np

from gen_data import NOISE_MODELS, FEATURE_ARMS, SEED_NOISE, observe, summarize

ROOT = os.path.dirname(os.path.abspath(__file__))
N_FIXED = [20, 50]
# (noise model, scale multiplier on NOISE_SCALE): from light & homoscedastic to heavy & strongly heteroscedastic
GROUPS = [('gauss', 0.5),           # g0: light homoscedastic Gaussian (0.25 sd)
          ('gauss_valhet', 1.0),    # g1: Gaussian, scale grows with the value (0 -> 1.0 sd, avg 0.5 sd)
          ('cauchy_valhet', 0.5),   # g2: Cauchy, scale grows with the value (avg 0.25 sd)
          ('cauchy_hetero', 1.0)]   # g3: Cauchy with lognormal per-row scale (median 0.5 sd)


def featurize(arrays, s, reps, counts):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        arrays[f'{s}__replicates'] = reps.astype(np.float32)
        arrays[f'{s}__counts'] = counts.astype(np.int8)
        for fa, stats in FEATURE_ARMS.items():
            arrays[f'{s}__{fa}'] = summarize(reps, counts, stats)[0]


def main():
    data = os.path.join(ROOT, 'artifacts', 'data')
    out = os.path.join(ROOT, 'artifacts', 'data_ext')
    os.makedirs(os.path.join(out, 'eval'), exist_ok=True)
    m = json.load(open(os.path.join(data, 'manifest.json')))
    for e in m['eval']:
        z = np.load(os.path.join(data, 'eval', e['name'] + '.npz'))
        X, seed = z['X_clean'], e['seed']
        arrays = {}
        for nm in NOISE_MODELS:
            for n in N_FIXED:
                reps, counts = observe(X, nm, [seed, 10_000 + n], n_fixed=n)
                featurize(arrays, f'{nm}_n{n}', reps, counts)
        # grouped heteroscedasticity: each row's group decides its observation model
        rng = np.random.default_rng([SEED_NOISE, seed, 777])
        g = rng.permutation(np.arange(len(X)) % len(GROUPS))
        reps = np.full(X.shape + (10,), np.nan)
        counts = np.zeros(X.shape, dtype=np.int64)
        for gi, (nm, mult) in enumerate(GROUPS):
            r_g, c_g = observe(X, nm, [seed, 20_000 + gi], scale_mult=mult)  # whole table, so ranks/sd are global
            reps[g == gi], counts[g == gi] = r_g[g == gi], c_g[g == gi]
        Xg = np.concatenate([X, g[:, None].astype(X.dtype)], axis=1)
        g_reps = np.full((len(X), 1, 10), np.nan)
        g_reps[:, 0, 0] = g
        featurize(arrays, 'grouped', np.concatenate([reps, g_reps], 1), np.concatenate([counts, np.ones((len(X), 1), int)], 1))
        clean_reps = np.full(Xg.shape + (10,), np.nan)
        clean_reps[..., 0] = Xg
        featurize(arrays, 'grouped_clean', clean_reps, np.ones(Xg.shape, int))
        arrays['grouped__X_clean'] = arrays['grouped_clean__X_clean'] = Xg
        arrays['grouped__group'] = g
        np.savez_compressed(os.path.join(out, 'eval', e['name'] + '.npz'), **arrays)
        print(e['name'], flush=True)
    settings = [f'{nm}_n{n}' for nm in NOISE_MODELS for n in N_FIXED] + ['grouped', 'grouped_clean']
    json.dump(dict(settings=settings, N_FIXED=N_FIXED, GROUPS=GROUPS, eval=m['eval']),
              open(os.path.join(out, 'manifest.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
