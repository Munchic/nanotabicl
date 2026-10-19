"""Exp 7 settings: heterogeneous observation models within a table, on the same 27 eval datasets (artifacts/data_ext3).

  percol       : every column has its own noise model (one of the 10 core modes) and scale (0.5 sd x logU(0.5, 2));
                 20% of columns are exact (n = 1, noise-free, e.g. covariates); counts ~ U{1..10}
  pergroup     : rows fall into 3 hidden groups (labs / batches / sites); each group has its own model, scale and, w.p.
                 0.5, a replicate cap U{1..10} (one lab runs singles, another triplicates); the group is NOT observed
  pergroup_vis : the same draws, plus the exact group index as an extra column
  colgroup_vis : own model + scale per (group, column), group column visible, 20% exact columns

Cells with the same (model, scale) come from one gen_data.observe call on the whole table (so the noise matches the
main settings exactly, including the per-row scale of cauchy_hetero). Stores <s>__cellmodel (index into MODELS, -1 =
exact), <s>__cellmult, <s>__group and, for the visible settings, <s>__X_clean (X + group column), for the oracle.
"""
import json, os

import numpy as np

from gen_data import observe, art, SEED_NOISE
from gen_ext2 import featurize

MODELS = ['gauss', 'cauchy', 'cauchy_hetero', 'censored_left_heavy', 'fail_floor', 'fail_shift', 'cv_lognormal',
          'lod_logistic', 'saturation', 'plate_effect']  # = obs_eval.CORE_NOISE
SETTINGS = ['percol', 'pergroup', 'pergroup_vis', 'colgroup_vis']
N_GROUPS, P_EXACT = 3, 0.2


def draw_design(rng, n, d, kind):
    """-> per-cell model index (n, d) (-1 = exact), per-cell scale multiplier (n, d), group (n,), per-row cap (n,)."""
    mult = lambda *s: np.round(np.exp(rng.uniform(np.log(0.5), np.log(2.0), size=s)), 3)
    group = rng.permutation(np.arange(n) % N_GROUPS)
    cap = np.full(n, 10)
    if kind == 'percol':
        m, u = np.broadcast_to(rng.integers(len(MODELS), size=d), (n, d)), np.broadcast_to(mult(d), (n, d))
    elif kind == 'pergroup':
        mg, ug = rng.integers(len(MODELS), size=N_GROUPS), mult(N_GROUPS)
        cg = np.where(rng.random(N_GROUPS) < 0.5, rng.integers(1, 11, size=N_GROUPS), 10)
        m, u, cap = np.repeat(mg[group][:, None], d, 1), np.repeat(ug[group][:, None], d, 1), cg[group]
    else:  # colgroup
        mg, ug = rng.integers(len(MODELS), size=(N_GROUPS, d)), mult(N_GROUPS, d)
        m, u = mg[group], ug[group]
    m = np.array(m)
    if kind in ('percol', 'colgroup'):
        exact = rng.random(d) < P_EXACT
        if exact.all():
            exact[rng.integers(d)] = False
        m[:, exact] = -1
    return m, np.array(u, dtype=np.float64), group, cap


def realize(X, seed, m, u, cap, key):
    n, d = X.shape
    reps = np.full((n, d, 10), np.nan)
    counts = np.ones((n, d), dtype=np.int64)
    reps[..., 0] = X  # exact cells
    combos = sorted({(int(a), float(b)) for a, b in zip(m.ravel(), u.ravel()) if a >= 0})
    for h, (mi, mu) in enumerate(combos):
        r, c = observe(X, MODELS[mi], [seed, key, h], scale_mult=mu)
        sel = (m == mi) & (u == mu)
        reps[sel], counts[sel] = r[sel], c[sel]
    counts = np.where(m >= 0, np.minimum(counts, cap[:, None]), 1)
    return reps, counts


def main():
    data, out = art('data'), art('data_ext3')
    os.makedirs(os.path.join(out, 'eval'), exist_ok=True)
    manifest = json.load(open(os.path.join(data, 'manifest.json')))
    for e in manifest['eval']:
        X, seed = np.load(os.path.join(data, 'eval', e['name'] + '.npz'))['X_clean'], e['seed']
        n, d = X.shape
        arrays = {}
        for kind, key in [('percol', 50_000), ('pergroup', 51_000), ('colgroup', 52_000)]:
            rng = np.random.default_rng([SEED_NOISE, seed, key])
            m, u, group, cap = draw_design(rng, n, d, kind)
            reps, counts = realize(X, seed, m, u, cap, key)
            names = {'percol': ['percol'], 'pergroup': ['pergroup', 'pergroup_vis'], 'colgroup': ['colgroup_vis']}[kind]
            for s in names:
                if s.endswith('_vis'):  # exact group column appended
                    g_reps = np.full((n, 1, 10), np.nan)
                    g_reps[:, 0, 0] = group
                    featurize(arrays, s, np.concatenate([reps, g_reps], 1), np.concatenate([counts, np.ones((n, 1), int)], 1))
                    arrays[f'{s}__X_clean'] = np.concatenate([X, group[:, None].astype(X.dtype)], 1)
                else:
                    featurize(arrays, s, reps, counts)
                arrays[f'{s}__cellmodel'] = m.astype(np.int8)
                arrays[f'{s}__cellmult'] = u.astype(np.float32)
                arrays[f'{s}__group'] = group.astype(np.int8)
        np.savez_compressed(os.path.join(out, 'eval', e['name'] + '.npz'), **arrays)
        print(e['name'], flush=True)
    json.dump(dict(settings=SETTINGS, MODELS=MODELS, N_GROUPS=N_GROUPS, P_EXACT=P_EXACT),
              open(os.path.join(out, 'manifest.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
