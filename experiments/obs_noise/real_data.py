"""Real replicate data -> the eval format of the simulated settings (light loader).

Each task is written to artifacts/real/<dataset>/<task>.npz with
  replicates (n, d, K) float32, NaN-padded   counts (n, d) int8   y (n,) int64   train_idx, test_idx
  group (n,) str (compound|dose)              meta: dataset, task, classes, n_components
i.e. one row per experimental unit with its replicate measurements; the same keys gen_data.observe produces.

LINCS (Way et al. 2022, cpg0004): one row per (compound, dose); its replicate wells are the replicates.
  * features: z-scored and PCA-projected using the negative-control (DMSO) wells only, applied to every replicate well (linear, so each
    well keeps its own noise); n_components = 16 by default (the training prior has 2-16 features)
  * split: the paper's compound split (split_moas_cpds.csv): test compounds never appear in the context;
    --split dose: every compound is in the context, a seeded third of its (compound, dose) rows held out (signal
    from the compound's other doses, so replicate noise is a larger share of the error); replicates of a row stay together
  * task: MoA classification; MoAs with >= 2 train and >= 1 test compound, ranked by number of compounds, in
    consecutive blocks of 10 (TabICL's class limit) -> several independent 10-way tasks
  * excluded: compounds listed with several MoAs (ambiguous label), DMSO (dose 0), positive-control groups (> 10 wells: MG-132, bortezomib), (compound, dose) with < 1 well
Usage: python real_data.py [--datasets L1000,CellPainting] [--n_components 16] [--n_tasks 5]
"""
import argparse, json, os

import numpy as np, pandas as pd
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, 'artifacts', 'real_raw', 'lincs')
OUT = os.path.join(ROOT, 'artifacts', 'real')
K_MAX, MAX_WELLS, N_CLASSES = 10, 10, 10


def load_lincs(which):
    """-> features (wells, F), key columns DataFrame [pert_iname, dose]."""
    if which == 'L1000':
        df = pd.read_csv(os.path.join(RAW, 'L1000_level4_cpd_replicates.csv.gz'))
        meta = ['replicate_id', 'sig_id', 'pert_id', 'pert_idose', 'det_plate', 'det_well', 'dose',
                'Metadata_broad_sample', 'pert_iname', 'moa']
        key = df[['pert_iname', 'dose']].copy()
    else:
        df = pd.read_csv(os.path.join(RAW, 'cp_level4_cpd_replicates_nonspherized.csv.gz'), low_memory=False)
        meta = [c for c in df.columns if c.startswith('Metadata_')]
        ids = pd.read_csv(os.path.join(RAW, 'L1000_level4_cpd_replicates.csv.gz'), usecols=['pert_id', 'pert_iname'])
        names = ids.drop_duplicates('pert_id').set_index('pert_id')['pert_iname']
        key = pd.DataFrame({'pert_iname': df['Metadata_pert_id'].map(names), 'dose': df['Metadata_dose_recode']})
        key.loc[df['Metadata_broad_sample'].astype(str) == 'DMSO', 'pert_iname'] = 'DMSO'
    feats = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    return df[feats].to_numpy(np.float64), key


def build(which, n_components, n_tasks, split_mode='compound'):
    F, key = load_lincs(which)
    ok = np.isfinite(F).all(1)
    F, key = F[ok], key[ok].reset_index(drop=True)
    ctrl = (key['pert_iname'] == 'DMSO').to_numpy()
    mu, sd = F[ctrl].mean(0), F[ctrl].std(0)  # per-feature z-score on the negative controls (CP features differ in scale)
    use = sd > 1e-8
    F = (F[:, use] - mu[use]) / sd[use]
    pca = PCA(n_components=n_components, random_state=0).fit(F[ctrl])  # negative controls only: no label leakage
    Z = pca.transform(F)
    split = pd.read_csv(os.path.join(RAW, 'split_moas_cpds.csv'))
    split = split[~split['pert_iname'].duplicated(keep=False)].set_index('pert_iname')  # single-MoA compounds only
    keep = (~ctrl) & key['pert_iname'].isin(split.index).to_numpy() & (key['dose'] > 0).to_numpy()
    key = key[keep].reset_index(drop=True)
    Z = Z[keep]
    gid = key['pert_iname'].astype(str) + '|' + key['dose'].astype(str)
    sizes = gid.map(gid.value_counts())
    good = (sizes <= MAX_WELLS).to_numpy()  # drops positive-control groups plated on every plate
    key, Z, gid = key[good].reset_index(drop=True), Z[good], gid[good].reset_index(drop=True)
    cpd = key.drop_duplicates('pert_iname')['pert_iname']
    info = split.loc[cpd.values, ['moa', 'train', 'test']]
    counts = info.groupby('moa').agg(n=('train', 'size'), n_train=('train', 'sum'), n_test=('test', 'sum'))
    moas = counts[(counts.n_train >= 2) & (counts.n_test >= 1)].sort_values('n', ascending=False).index.tolist()
    out_dir = os.path.join(OUT, which if split_mode == 'compound' else f'{which}_dose')
    os.makedirs(out_dir, exist_ok=True)
    tasks = []
    for t in range(n_tasks):
        cls = moas[t * N_CLASSES:(t + 1) * N_CLASSES]
        if len(cls) < N_CLASSES:
            break
        rows = key['pert_iname'].map(split['moa']).isin(cls).to_numpy()
        groups = sorted(set(gid[rows]))
        n, d = len(groups), n_components
        reps = np.full((n, d, K_MAX), np.nan, dtype=np.float32)
        cnt = np.zeros((n, d), dtype=np.int8)
        y = np.zeros(n, dtype=np.int64)
        is_train = np.zeros(n, dtype=bool)
        index = {g: i for i, g in enumerate(groups)}
        for g, zrow in zip(gid[rows], Z[rows]):
            i = index[g]
            k = cnt[i, 0]
            if k < K_MAX:
                reps[i, :, k] = zrow
                cnt[i, :] = k + 1
        rng = np.random.default_rng([t, 7])
        held = {}  # dose split: per compound, a seeded third of its dose rows are held out (the compound stays in context)
        for g, i in index.items():
            name = g.split('|')[0]
            y[i] = cls.index(split.loc[name, 'moa'])
            if split_mode == 'compound':
                is_train[i] = bool(split.loc[name, 'train'])
            else:
                held.setdefault(name, []).append(i)
        for name, idx in held.items():
            test = rng.permutation(idx)[:max(1, len(idx) // 3)] if len(idx) > 1 else []
            is_train[idx] = True
            is_train[list(test)] = False
        name = f'moa{t + 1:02d}'
        np.savez_compressed(os.path.join(out_dir, name + '.npz'), replicates=reps, counts=cnt, y=y,
                            train_idx=np.where(is_train)[0], test_idx=np.where(~is_train)[0], group=np.array(groups),
                            meta=json.dumps(dict(dataset=which, task=name, classes=cls, n_components=n_components,
                                                 split=split_mode)))
        tasks.append(dict(task=name, n=n, n_train=int(is_train.sum()), n_test=int((~is_train).sum()),
                          wells_per_row=float(cnt[:, 0].mean()), classes=cls))
        print(which, name, f'rows {n} (train {is_train.sum()}, test {(~is_train).sum()}), wells/row {cnt[:, 0].mean():.2f}', flush=True)
    json.dump(dict(dataset=which, n_components=n_components, pca_explained=float(pca.explained_variance_ratio_.sum()),
                   tasks=tasks), open(os.path.join(out_dir, 'manifest.json'), 'w'), indent=1)


GAS = os.path.join(ROOT, 'artifacts', 'real_raw', 'gas_drift', 'Dataset')
GAS_TYPES = [[2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9] for t in range(4)]  # 0-based sensors per type (from the
# label-free correlation structure: two arrays of 8, each type = two adjacent sensors per array; within-type r 0.8-1.0)


def build_gas(max_rows=1000):
    """Gas sensor array drift (Vergara et al. 2012): rows = measurements; columns = the 8 features x 4 sensor types;
    replicates = the 4 physical units of a type (unit-to-unit bias included). Tasks: context = batch b, test = batch
    b + 1 (b = 1..9; the standard drift protocol), each side subsampled (seeded) to <= max_rows. Held-out test set."""
    from sklearn.datasets import load_svmlight_file
    batches = [load_svmlight_file(os.path.join(GAS, f'batch{b}.dat'), n_features=128) for b in range(1, 11)]
    out_dir = os.path.join(OUT, 'GasDrift')
    os.makedirs(out_dir, exist_ok=True)
    tasks = []
    for b in range(9):
        rng = np.random.default_rng([b, 11])
        parts = []
        for side in (b, b + 1):
            X, y = batches[side][0].toarray(), batches[side][1].astype(np.int64) - 1
            idx = rng.permutation(len(y))[:max_rows]
            parts.append((X[idx], y[idx]))
        X = np.concatenate([parts[0][0], parts[1][0]])
        y = np.concatenate([parts[0][1], parts[1][1]])
        S = X.reshape(len(X), 16, 8)  # (rows, sensor, feature)
        reps = np.stack([S[:, units, :] for units in GAS_TYPES], axis=1)  # (rows, type, unit, feature)
        reps = reps.transpose(0, 1, 3, 2).reshape(len(X), 32, 4)  # columns = (type, feature), replicates = units
        full = np.full((len(X), 32, K_MAX), np.nan, dtype=np.float32)
        full[..., :4] = reps
        n_tr = len(parts[0][1])
        name = f'b{b + 1:02d}to{b + 2:02d}'
        np.savez_compressed(os.path.join(out_dir, name + '.npz'), replicates=full, counts=np.full((len(X), 32), 4, np.int8),
                            y=y, train_idx=np.arange(n_tr), test_idx=np.arange(n_tr, len(X)), group=np.arange(len(X)).astype(str),
                            meta=json.dumps(dict(dataset='GasDrift', task=name, classes=sorted(set(y.tolist())))))
        tasks.append(dict(task=name, n_train=int(n_tr), n_test=int(len(X) - n_tr)))
        print('GasDrift', name, f'context {n_tr}, test {len(X) - n_tr}', flush=True)
    json.dump(dict(dataset='GasDrift', role='held-out test', tasks=tasks), open(os.path.join(out_dir, 'manifest.json'), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='L1000,CellPainting')
    ap.add_argument('--n_components', type=int, default=16)
    ap.add_argument('--n_tasks', type=int, default=5)
    ap.add_argument('--split', choices=['compound', 'dose'], default='compound',
                    help="compound: the paper's compound split; dose: seen compounds, a third of each compound's doses held out")
    a = ap.parse_args()
    for ds in a.datasets.split(','):
        if ds == 'GasDrift':
            build_gas()
        else:
            build(ds, a.n_components, a.n_tasks, a.split)
