"""Exp 4 settings on the same 27 eval datasets (artifacts/data_ext2):

  <noise>_k<n>        : every cell has exactly n replicates, n in K_FIXED (with the n20/n50 of gen_ext.py: a 1..50 curve)
  <noise>_mix         : per ROW, all cells share n: single sample (n = 1) w.p. 0.5, else n ~ U{2..10}
  <noise>_ctx10_q1    : context (train) rows n = 10, query (test) rows n = 1
  <noise>_ctx1_q10    : context rows n = 1, query rows n = 10
Mixed designs draw 10 replicates per cell and keep the first n (iid, so this is exact). <s>__rowcount stores each row's n.
"""
import json, os, warnings

import numpy as np

from gen_data import observe, summarize, FEATURE_ARMS, art

ROOT = os.path.dirname(os.path.abspath(__file__))
NOISES = ['gauss', 'student_t3', 'cauchy', 'cauchy_hetero', 'censored_left_heavy']
K_FIXED = [1, 2, 3, 5, 10]


def featurize(arrays, s, reps, counts):
    reps = reps.copy()
    reps[np.arange(reps.shape[-1])[None, None] >= counts[..., None]] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        arrays[f'{s}__replicates'] = reps.astype(np.float32)
        arrays[f'{s}__counts'] = counts.astype(np.int8)
        for fa, stats in FEATURE_ARMS.items():
            arrays[f'{s}__{fa}'] = summarize(reps, counts, stats)[0]


def main():
    data, out = art('data'), art('data_ext2')
    os.makedirs(os.path.join(out, 'eval'), exist_ok=True)
    m = json.load(open(os.path.join(data, 'manifest.json')))
    for e in m['eval']:
        z = np.load(os.path.join(data, 'eval', e['name'] + '.npz'))
        X, seed, tr = z['X_clean'], e['seed'], z['train_idx']
        n, d = X.shape
        arrays = {}
        for nm in NOISES:
            for k in K_FIXED:
                reps, counts = observe(X, nm, [seed, 30_000 + k], n_fixed=k)
                featurize(arrays, f'{nm}_k{k}', reps, counts)
            reps10, _ = observe(X, nm, [seed, 40_000], n_fixed=10)  # one draw shared by the three mixed designs
            rng = np.random.default_rng([seed, 40_001, len(nm)])
            row_n = np.where(rng.random(n) < 0.5, 1, rng.integers(2, 11, size=n))
            is_ctx = np.isin(np.arange(n), tr)
            for s, rn in [(f'{nm}_mix', row_n), (f'{nm}_ctx10_q1', np.where(is_ctx, 10, 1)),
                          (f'{nm}_ctx1_q10', np.where(is_ctx, 1, 10))]:
                featurize(arrays, s, reps10, np.repeat(rn[:, None], d, axis=1))
                arrays[f'{s}__rowcount'] = rn.astype(np.int8)
        np.savez_compressed(os.path.join(out, 'eval', e['name'] + '.npz'), **arrays)
        print(e['name'], flush=True)
    settings = [f'{nm}_k{k}' for nm in NOISES for k in K_FIXED] + \
               [f'{nm}_{s}' for nm in NOISES for s in ('mix', 'ctx10_q1', 'ctx1_q10')]
    json.dump(dict(settings=settings, NOISES=NOISES, K_FIXED=K_FIXED), open(os.path.join(out, 'manifest.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
