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


TOX = os.path.join(ROOT, 'artifacts', 'real_raw', 'tox21')
TOX_ASSAYS = ['ar-mda-kb2-luc-agonist-p1', 'ar-bla-agonist-p1', 'ahr-p1', 'aromatase-p1', 'er-luc-bg1-4e2-agonist-p2',
              'er-bla-agonist-p2', 'pparg-bla-agonist-p1', 'are-bla-p1', 'elg1-luc-agonist-p1', 'hse-bla-p1', 'mitotox-p1',
              'p53-bla-p1']  # the 12 MoleculeNet Tox21 tasks (NR-AR ... SR-p53)


def build_tox21(max_rows=1000):
    """Tox21 qHTS: rows = substances (TOX21_ID); columns = per-run log10 AC50 [uM] and efficacy of 11 assays (ratio
    channel); replicates = the substance's screening runs (3-4; independent across assays). Inactive runs have no AC50:
    reported at the top tested concentration (right-censoring) with efficacy 0. Target: the 12th assay's call (active if
    the majority of its runs are active, inactive if none is, otherwise dropped); one task per held-out assay; context
    and test <= max_rows each, stratified by class (seeded). Held-out test set."""
    runs = {}
    for a in TOX_ASSAYS:
        d = pd.read_csv(os.path.join(TOX, f'tox21-{a}', f'tox21-{a}.txt'), sep='\t', index_col=False, low_memory=False)
        # the primary readout, one row per run: set1-3 (luciferase), ratio / ratio_1-3 (bla), antagonist1-3 (aromatase),
        # agonist (elg1); channel / viability rows are dropped
        d = d[d['SAMPLE_DATA_TYPE'].astype(str).str.match(r'^(set\d|ratio(_\d)?|antagonist\d|agonist)$')]
        conc = d[[c for c in d.columns if c.startswith('CONC')]].to_numpy(np.float64)
        top_um = np.nanmax(conc, 1) * 1e6
        ac50 = d['AC50'].to_numpy(np.float64)
        logac = np.log10(np.where(np.isfinite(ac50), ac50, top_um))  # inactive -> censored at the top concentration
        eff = np.where(np.isfinite(ac50), d['EFFICACY'].to_numpy(np.float64), 0.0)
        active = d['ASSAY_OUTCOME'].astype(str).str.startswith('active').to_numpy()
        runs[a] = pd.DataFrame({'id': d['TOX21_ID'].to_numpy(), 'logac': logac, 'eff': np.nan_to_num(eff), 'active': active})
        print('tox21', a, len(d), 'runs', flush=True)
    ids = sorted(set.intersection(*[set(r['id']) for r in runs.values()]))
    pos = {i: j for j, i in enumerate(ids)}
    n = len(ids)
    out_dir = os.path.join(OUT, 'Tox21')
    os.makedirs(out_dir, exist_ok=True)
    tasks = []
    for t, target in enumerate(TOX_ASSAYS):
        feats = [a for a in TOX_ASSAYS if a != target]
        reps = np.full((n, 2 * len(feats), K_MAX), np.nan, dtype=np.float32)
        cnt = np.zeros((n, 2 * len(feats)), dtype=np.int8)
        for j, a in enumerate(feats):
            r = runs[a][runs[a]['id'].isin(pos)]  # substances screened in all 12 assays
            for col, v in ((2 * j, 'logac'), (2 * j + 1, 'eff')):
                for i, val in zip(r['id'].map(pos), r[v]):
                    k = cnt[i, col]
                    if k < K_MAX:
                        reps[i, col, k] = val
                        cnt[i, col] = k + 1
        g = runs[target].groupby('id')['active'].agg(['mean', 'size'])
        lab = np.full(n, -1)
        for i_, (m, _) in g.iterrows():
            if i_ in pos:
                lab[pos[i_]] = 1 if m > 0.5 else (0 if m == 0 else -1)
        ok = np.where((lab >= 0) & (cnt.min(1) > 0))[0]
        rng = np.random.default_rng([t, 13])
        tr, te = [], []
        for c in (0, 1):  # stratified: each class split half / half, then capped
            idx = rng.permutation(ok[lab[ok] == c])
            half = len(idx) // 2
            tr.append(idx[:half]); te.append(idx[half:])
        tr, te = np.concatenate(tr), np.concatenate(te)
        tr = rng.permutation(tr)[:max_rows] if len(tr) > max_rows else tr
        te = rng.permutation(te)[:max_rows] if len(te) > max_rows else te
        rows = np.concatenate([tr, te])
        name = f'target_{t + 1:02d}'
        np.savez_compressed(os.path.join(out_dir, name + '.npz'), replicates=reps[rows], counts=cnt[rows], y=lab[rows],
                            train_idx=np.arange(len(tr)), test_idx=np.arange(len(tr), len(rows)), group=np.array(ids)[rows],
                            meta=json.dumps(dict(dataset='Tox21', task=name, target=target, features=feats)))
        tasks.append(dict(task=name, target=target, n_train=int(len(tr)), n_test=int(len(te)), pos_rate=float(lab[rows].mean())))
        print('Tox21', name, target, f'context {len(tr)}, test {len(te)}, active share {lab[rows].mean():.3f}', flush=True)
    json.dump(dict(dataset='Tox21', role='held-out test', tasks=tasks), open(os.path.join(out_dir, 'manifest.json'), 'w'), indent=1)


# ----- regression tasks (Exp 11; designs fixed before any model was evaluated on them) -----
# y float64 = the target; everything else as the classification tasks. real_eval_reg.py evaluates them.
GAS_CONC = os.path.join(ROOT, 'artifacts', 'real_raw', 'gas_drift_conc', 'raw')  # UCI 270: '<gas>;<ppmv>' labels


def build_gas_reg(max_rows=1000):
    """GasConc: Gas sensor array drift at different concentrations (Vergara et al. 2012, UCI 270). One task per gas
    (6): target log10 concentration [ppmv]; rows = measurements of that gas from all 10 batches; columns / replicates as
    build_gas (4 physical units per sensor type). Seeded half / half split, <= max_rows per side. Held-out test set."""
    from sklearn.datasets import load_svmlight_file
    X_all, gas, conc = [], [], []
    for b in range(1, 11):
        lines = open(os.path.join(GAS_CONC, f'batch{b}.dat')).read().splitlines()
        lab = [l.split(' ', 1)[0].split(';') for l in lines]
        import io
        X, _ = load_svmlight_file(io.BytesIO('\n'.join('0 ' + l.split(' ', 1)[1] for l in lines).encode()), n_features=128)
        X_all.append(X.toarray()); gas += [int(g) for g, _ in lab]; conc += [float(c) for _, c in lab]
    X_all, gas, conc = np.concatenate(X_all), np.array(gas), np.array(conc)
    out_dir = os.path.join(OUT, 'GasConc')
    os.makedirs(out_dir, exist_ok=True)
    tasks = []
    for g_ in sorted(set(gas.tolist())):
        idx = np.where((gas == g_) & (conc > 0))[0]
        rng = np.random.default_rng([g_, 17])
        idx = rng.permutation(idx)
        half = len(idx) // 2
        tr, te = idx[:half][:max_rows], idx[half:][:max_rows]
        rows = np.concatenate([tr, te])
        S = X_all[rows].reshape(len(rows), 16, 8)
        reps = np.stack([S[:, units, :] for units in GAS_TYPES], axis=1).transpose(0, 1, 3, 2).reshape(len(rows), 32, 4)
        full = np.full((len(rows), 32, K_MAX), np.nan, dtype=np.float32)
        full[..., :4] = reps
        y = np.log10(conc[rows])
        name = f'gas{g_}'
        np.savez_compressed(os.path.join(out_dir, name + '.npz'), replicates=full, counts=np.full((len(rows), 32), 4, np.int8),
                            y=y, train_idx=np.arange(len(tr)), test_idx=np.arange(len(tr), len(rows)),
                            group=np.arange(len(rows)).astype(str), meta=json.dumps(dict(dataset='GasConc', task=name, target='log10 ppmv')))
        tasks.append(dict(task=name, n_train=int(len(tr)), n_test=int(len(te)), y_sd=float(y.std())))
        print('GasConc', name, f'context {len(tr)}, test {len(te)}, log10 conc sd {y.std():.2f}', flush=True)
    json.dump(dict(dataset='GasConc', role='held-out test', tasks=tasks), open(os.path.join(out_dir, 'manifest.json'), 'w'), indent=1)


def build_tox21_reg(max_rows=1000, min_rows=100):
    """Tox21Pot: potency regression. Per target assay (12): rows = substances with a fitted AC50 in the majority of
    their runs of the target; y = median log10 AC50 [uM] over those fitted runs (uncensored target; censored targets are
    CENSORING.md's track). Features / replicates as build_tox21 (the other 11 assays' per-run log AC50, censored at the
    top concentration, and efficacy). Seeded half / half, <= max_rows per side; assays with < min_rows rows skipped."""
    runs = {}
    for a in TOX_ASSAYS:
        d = pd.read_csv(os.path.join(TOX, f'tox21-{a}', f'tox21-{a}.txt'), sep='\t', index_col=False, low_memory=False)
        d = d[d['SAMPLE_DATA_TYPE'].astype(str).str.match(r'^(set\d|ratio(_\d)?|antagonist\d|agonist)$')]
        conc = d[[c for c in d.columns if c.startswith('CONC')]].to_numpy(np.float64)
        ac50 = d['AC50'].to_numpy(np.float64)
        logac = np.log10(np.where(np.isfinite(ac50), ac50, np.nanmax(conc, 1) * 1e6))
        eff = np.where(np.isfinite(ac50), d['EFFICACY'].to_numpy(np.float64), 0.0)
        runs[a] = pd.DataFrame({'id': d['TOX21_ID'].to_numpy(), 'logac': logac, 'eff': np.nan_to_num(eff),
                                'fit': np.isfinite(ac50)})
    ids = sorted(set.intersection(*[set(r['id']) for r in runs.values()]))
    pos = {i: j for j, i in enumerate(ids)}
    n = len(ids)
    out_dir = os.path.join(OUT, 'Tox21Pot')
    os.makedirs(out_dir, exist_ok=True)
    tasks = []
    for t, target in enumerate(TOX_ASSAYS):
        feats = [a for a in TOX_ASSAYS if a != target]
        reps = np.full((n, 2 * len(feats), K_MAX), np.nan, dtype=np.float32)
        cnt = np.zeros((n, 2 * len(feats)), dtype=np.int8)
        for j, a in enumerate(feats):
            r = runs[a][runs[a]['id'].isin(pos)]
            for col, v in ((2 * j, 'logac'), (2 * j + 1, 'eff')):
                for i, val in zip(r['id'].map(pos), r[v]):
                    k = cnt[i, col]
                    if k < K_MAX:
                        reps[i, col, k] = val
                        cnt[i, col] = k + 1
        rt = runs[target][runs[target]['id'].isin(pos)]
        g = rt.groupby('id').agg(share=('fit', 'mean'))
        med = rt[rt['fit']].groupby('id')['logac'].median()
        y = np.full(n, np.nan)
        for i_, row in g.iterrows():
            if row['share'] > 0.5:
                y[pos[i_]] = med[i_]
        ok = np.where(np.isfinite(y) & (cnt.min(1) > 0))[0]
        name = f'target_{t + 1:02d}'
        if len(ok) < min_rows:
            print('Tox21Pot', name, target, f'{len(ok)} rows < {min_rows}: skipped', flush=True)
            continue
        rng = np.random.default_rng([t, 19])
        idx = rng.permutation(ok)
        half = len(idx) // 2
        tr, te = idx[:half][:max_rows], idx[half:][:max_rows]
        rows = np.concatenate([tr, te])
        np.savez_compressed(os.path.join(out_dir, name + '.npz'), replicates=reps[rows], counts=cnt[rows], y=y[rows],
                            train_idx=np.arange(len(tr)), test_idx=np.arange(len(tr), len(rows)), group=np.array(ids)[rows],
                            meta=json.dumps(dict(dataset='Tox21Pot', task=name, target=target, features=feats)))
        tasks.append(dict(task=name, target=target, n_train=int(len(tr)), n_test=int(len(te)), y_sd=float(y[rows].std())))
        print('Tox21Pot', name, target, f'context {len(tr)}, test {len(te)}, log AC50 sd {y[rows].std():.2f}', flush=True)
    json.dump(dict(dataset='Tox21Pot', role='held-out test', tasks=tasks), open(os.path.join(out_dir, 'manifest.json'), 'w'), indent=1)


def build_lincs_dose_reg(which):
    """<which>_dose_reg (dev): the <which>_dose tasks with target = the row's dose level (1-6, log-spaced doses);
    same rows, replicates and split (every compound in the context, a third of its doses held out)."""
    out_dir = os.path.join(OUT, f'{which}_dose_reg')
    os.makedirs(out_dir, exist_ok=True)
    for f in sorted(os.listdir(os.path.join(OUT, f'{which}_dose'))):
        if not f.endswith('.npz'):
            continue
        z = dict(np.load(os.path.join(OUT, f'{which}_dose', f)))
        z['y'] = np.array([float(g.split('|')[1]) for g in z['group']])
        z['meta'] = json.dumps(dict(dataset=f'{which}_dose_reg', task=f[:-4], target='dose level'))
        np.savez_compressed(os.path.join(out_dir, f), **z)
        print(f'{which}_dose_reg', f[:-4], f"rows {len(z['y'])}, dose levels {sorted(set(z['y'].tolist()))}", flush=True)


UPS = os.path.join(ROOT, 'artifacts', 'real_raw', 'proteomics_PXD002099', 'YEAST-Data-NonNormalized.csv')


def build_ups1(n_splits=10, keep_fmol=None, name='UPS1'):
    """Proteomics spike-in benchmark (PXD002099, Pursiheimo et al. 2015): UPS1 standard proteins spiked into a yeast
    background at 2 / 4 / 10 / 25 / 50 fmol, 3 replicate LC-MS runs each; label-free protein intensities.
    Rows = proteins; columns = the 5 concentrations; replicates = the 3 runs of each concentration (not paired across
    columns). Values: log2 intensity minus the protein's median over the included runs (its profile across concentrations).
    UPS1_low keeps only 2 and 4 fmol (a 2-fold change at about the replicate noise level, where replicates matter).
    Missing intensities are real non-detects (intensity-dependent): a cell with no detected run gets one value at the
    column's 1st percentile (detection-floor substitution, i.e. left-censoring). Label: UPS1 spike-in (changes with the
    concentration) vs yeast background (constant): differential-abundance detection with ground truth. 10 seeded
    stratified 50 / 50 context / test splits. Held-out test set."""
    d = pd.read_csv(UPS, sep=';', decimal=',')
    fmol = lambda c: int(c.split('fmol')[0].split('_')[-1])
    runs = [c for c in d.columns if 'fmol' in c and (keep_fmol is None or fmol(c) in keep_fmol)]
    conc = sorted({fmol(c) for c in runs})
    X = np.log2(d[runs].to_numpy(np.float64))
    with np.errstate(all='ignore'):
        X = X - np.nanmedian(X, 1, keepdims=True)  # centered on the INCLUDED runs only (no leak from other concentrations)
    y = d['Accession'].astype(str).str.contains('ups').to_numpy().astype(np.int64)
    n = len(d)
    reps = np.full((n, len(conc), K_MAX), np.nan, dtype=np.float32)
    cnt = np.zeros((n, len(conc)), dtype=np.int8)
    for j, c in enumerate(conc):
        cols = [i for i, r in enumerate(runs) if fmol(r) == c]
        V = X[:, cols]
        floor = np.nanquantile(V, 0.01)
        for i in range(n):
            v = V[i][np.isfinite(V[i])]
            if len(v) == 0:
                v = np.array([floor])  # non-detect in every run: detection-floor substitution
            reps[i, j, :len(v)] = v
            cnt[i, j] = len(v)
    out_dir = os.path.join(OUT, name)
    os.makedirs(out_dir, exist_ok=True)
    for t in range(n_splits):
        rng = np.random.default_rng([t, 17])
        tr, te = [], []
        for c in (0, 1):
            idx = rng.permutation(np.where(y == c)[0])
            tr.append(idx[:len(idx) // 2]); te.append(idx[len(idx) // 2:])
        tr, te = np.concatenate(tr), np.concatenate(te)
        rows = np.concatenate([tr, te])
        np.savez_compressed(os.path.join(out_dir, f'split{t + 1:02d}.npz'), replicates=reps[rows], counts=cnt[rows], y=y[rows],
                            train_idx=np.arange(len(tr)), test_idx=np.arange(len(tr), len(rows)),
                            group=d['Accession'].to_numpy().astype(str)[rows],
                            meta=json.dumps(dict(dataset=name, concentrations_fmol=conc, runs_per_cell=3)))
    print(name, f'proteins {n} (UPS1 {y.sum()}), cells with all runs missing {(np.isnan(X).reshape(n, len(conc), -1).all(-1)).mean():.3f}, '
          f'missing runs {np.isnan(X).mean():.3f}', flush=True)
    json.dump(dict(dataset=name, role='held-out test', n=int(n), n_ups=int(y.sum()), concentrations=conc, n_splits=n_splits),
              open(os.path.join(out_dir, 'manifest.json'), 'w'), indent=1)


CPTAC = os.path.join(ROOT, 'artifacts', 'real_raw', 'proteomics_cptac', 'peptides.txt')


def build_cptac(n_splits=10, max_rows=1000):
    """CPTAC study 6 spike-in (MaxQuant peptides.txt, lab 3, as distributed by statOmics / msqrob2): UPS1 proteins
    spiked into yeast at two levels (conditions 6A / 6B), 3 technical replicate runs each. Rows = peptides (reverse /
    contaminant hits removed); columns = the two conditions; replicates = the 3 runs. Values: log2 intensity minus the
    peptide's median over its detected runs. Missing intensities are real, intensity-dependent non-detects (82% for the
    low-abundance UPS1 peptides, 43% for yeast): a condition with no detected run gets one value at the detection floor
    (the run's 1st percentile, relative to the peptide's median); the `censored` mask marks those cells (for censoring
    flags). Label: UPS1 spike-in vs yeast. 10 seeded stratified splits of <= max_rows context / test. Held-out test set."""
    d = pd.read_csv(CPTAC, sep='\t', low_memory=False)
    d = d[(d.get('Reverse', '') != '+') & (d.get('Potential contaminant', '') != '+')].reset_index(drop=True)
    runs = [c for c in d.columns if c.startswith('Intensity ') and c[-3] in 'AB']
    V = np.log2(d[runs].replace(0, np.nan).to_numpy(np.float64))
    seen = np.isfinite(V).any(1)  # peptides detected in at least one of these runs
    d, V = d[seen].reset_index(drop=True), V[seen]
    floor = np.nanquantile(V, 0.01, axis=0)  # per-run detection floor (log2)
    med = np.nanmedian(V, 1, keepdims=True)
    y = d['Proteins'].astype(str).str.contains('ups', case=False).to_numpy().astype(np.int64)
    n = len(d)
    conds = sorted({r[-3] for r in runs})
    reps = np.full((n, len(conds), K_MAX), np.nan, dtype=np.float32)
    cnt = np.zeros((n, len(conds)), dtype=np.int8)
    cens = np.zeros((n, len(conds)), dtype=bool)
    for j, cnd in enumerate(conds):
        cols = [i for i, r in enumerate(runs) if r[-3] == cnd]
        for i in range(n):
            v = V[i, cols]
            v = v[np.isfinite(v)] - med[i, 0]
            if len(v) == 0:  # not detected in any run of this condition: detection floor
                v = np.array([np.mean(floor[cols]) - med[i, 0]])
                cens[i, j] = True
            reps[i, j, :len(v)] = v
            cnt[i, j] = len(v)
    out_dir = os.path.join(OUT, 'CPTAC')
    os.makedirs(out_dir, exist_ok=True)
    for t in range(n_splits):
        rng = np.random.default_rng([t, 19])
        tr, te = [], []
        for c in (0, 1):
            idx = rng.permutation(np.where(y == c)[0])
            half = len(idx) // 2
            k = int(round(max_rows * (y == c).mean()))  # keep the natural class rate in each side
            tr.append(idx[:half][:k]); te.append(idx[half:][:k])
        tr, te = np.concatenate(tr), np.concatenate(te)
        rows = np.concatenate([tr, te])
        np.savez_compressed(os.path.join(out_dir, f'split{t + 1:02d}.npz'), replicates=reps[rows], counts=cnt[rows], y=y[rows],
                            censored=cens[rows], train_idx=np.arange(len(tr)), test_idx=np.arange(len(tr), len(rows)),
                            group=d['Sequence'].to_numpy().astype(str)[rows], meta=json.dumps(dict(dataset='CPTAC', conditions=conds)))
    print('CPTAC', f'peptides {n} (UPS1 {y.sum()}), censored cells {cens.mean():.3f} (UPS1 {cens[y == 1].mean():.3f})', flush=True)
    json.dump(dict(dataset='CPTAC', role='held-out test', n=int(n), n_ups=int(y.sum()), n_splits=n_splits),
              open(os.path.join(out_dir, 'manifest.json'), 'w'), indent=1)


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
        elif ds == 'GasConc':
            build_gas_reg()
        elif ds == 'Tox21Pot':
            build_tox21_reg()
        elif ds.endswith('_dose_reg'):
            build_lincs_dose_reg(ds[:-len('_dose_reg')])
        elif ds == 'Tox21':
            build_tox21()
        elif ds == 'UPS1':
            build_ups1()
        elif ds == 'CPTAC':
            build_cptac()
        elif ds == 'UPS1_low':  # 2 vs 4 fmol: a 2-fold change at about the replicate noise level (low SNR)
            build_ups1(keep_fmol=(2, 4), name='UPS1_low')
        else:
            build(ds, a.n_components, a.n_tasks, a.split)
