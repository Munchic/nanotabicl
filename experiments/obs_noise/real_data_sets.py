"""Set-valued real data (SHAPE.md) -> the same task format as real_data.py (artifacts/real/<dataset>/<task>.npz):
  replicates (n, d, K) float32, NaN-padded   counts (n, d) int16   y   train_idx, test_idx   paired (bool)   meta
A row is one sample; its instances (time steps, conformations, cells, measurements) are the replicates. paired = True
when instance k is one physical unit measured on every column (a joint set), False when columns' sets are unrelated.
Sets larger than K_MAX are subsampled (seeded) to K_MAX; meta records the true sizes. Splits are seeded and fixed here,
before any model is evaluated on the data.
  HAR (dev)            UCI 240: windows of 128 steps x 9 inertial channels, 6 activities; subject-disjoint splits
  Musk (final eval)    UCI 75 (v2): molecules as bags of conformations (166 features), musk or not
  PhysioNet (final)    Challenge 2012 set-a: ICU stays, 36 time-varying variables over 48 h, in-hospital death
Usage: python real_data_sets.py --datasets HAR,Musk,PhysioNet
"""
import argparse, io, json, os, zipfile

import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, 'artifacts', 'real_raw')
OUT = os.path.join(ROOT, 'artifacts', 'real')
K_MAX = 256


def save(ds, name, reps, y, tr, te, paired, meta):
    out_dir = os.path.join(OUT, ds)
    os.makedirs(out_dir, exist_ok=True)
    cnt = np.isfinite(reps).sum(-1).astype(np.int16)
    rows = np.concatenate([tr, te])
    np.savez_compressed(os.path.join(out_dir, name + '.npz'), replicates=reps[rows].astype(np.float32), counts=cnt[rows],
                        y=y[rows], train_idx=np.arange(len(tr)), test_idx=np.arange(len(tr), len(rows)),
                        paired=np.array(paired), meta=json.dumps(dict(dataset=ds, task=name, **meta)))
    return dict(task=name, n_train=int(len(tr)), n_test=int(len(te)))


def to_sets(groups, X, k_max, rng):
    """groups: list of instance-index arrays per row; X (N_inst, d) -> (rows, d, k_max) NaN-padded, true sizes."""
    reps = np.full((len(groups), X.shape[1], k_max), np.nan, dtype=np.float32)
    sizes = np.array([len(gi) for gi in groups])
    for i, gi in enumerate(groups):
        gi = rng.permutation(gi)[:k_max] if len(gi) > k_max else gi
        reps[i, :, :len(gi)] = X[gi].T
    return reps, sizes


def build_har(n_tasks=5, max_rows=1000):
    """HAR (dev): row = one 2.56 s window; columns = the 9 inertial channels; instances = its 128 time steps (paired:
    a time step is measured on all channels). Order within the window is dropped (a set). Tasks: seeded split of the
    30 subjects into context / test halves, each side subsampled to <= max_rows windows."""
    base = os.path.join(RAW, 'uci_har', 'UCI HAR Dataset')
    chans = [f'{s}_{a}' for s in ('body_acc', 'body_gyro', 'total_acc') for a in 'xyz']
    sig, y, subj = [], [], []
    for part in ('train', 'test'):
        sig.append(np.stack([np.loadtxt(os.path.join(base, part, 'Inertial Signals', f'{c}_{part}.txt')) for c in chans], 1))
        y.append(np.loadtxt(os.path.join(base, part, f'y_{part}.txt'), dtype=np.int64) - 1)
        subj.append(np.loadtxt(os.path.join(base, part, f'subject_{part}.txt'), dtype=np.int64))
    reps, y, subj = np.concatenate(sig).astype(np.float32), np.concatenate(y), np.concatenate(subj)  # (N, 9, 128)
    tasks = []
    for t in range(n_tasks):
        rng = np.random.default_rng([t, 23])
        s = rng.permutation(np.unique(subj))
        ctx = np.isin(subj, s[:len(s) // 2])
        tr = rng.permutation(np.where(ctx)[0])[:max_rows]
        te = rng.permutation(np.where(~ctx)[0])[:max_rows]
        tasks.append(save('HAR', f'split{t + 1:02d}', reps, y, tr, te, True,
                          dict(columns=chans, instances='128 time steps', classes=6, role='dev')))
        print('HAR', tasks[-1], flush=True)
    json.dump(dict(dataset='HAR', role='dev', tasks=tasks), open(os.path.join(OUT, 'HAR', 'manifest.json'), 'w'), indent=1)


def build_musk(n_splits=10):
    """Musk v2 (final eval): row = molecule, instances = its conformations (166 features, paired), y = musk.
    10 seeded stratified half splits."""
    d = pd.read_csv(os.path.join(RAW, 'uci_musk', 'clean2.data'), header=None)
    mol, X, lab = d[0].to_numpy(), d.iloc[:, 2:168].to_numpy(np.float64), d[168].to_numpy().astype(np.int64)
    names = sorted(set(mol))
    groups = [np.where(mol == m)[0] for m in names]
    y = np.array([lab[gi[0]] for gi in groups])
    reps, sizes = to_sets(groups, X, K_MAX, np.random.default_rng(29))
    tasks = []
    for t in range(n_splits):
        rng = np.random.default_rng([t, 31])
        tr, te = [], []
        for c in (0, 1):
            idx = rng.permutation(np.where(y == c)[0])
            tr.append(idx[:len(idx) // 2]); te.append(idx[len(idx) // 2:])
        tasks.append(save('Musk', f'split{t + 1:02d}', reps, y, np.concatenate(tr), np.concatenate(te), True,
                          dict(features=166, set_sizes=[int(sizes.min()), int(np.median(sizes)), int(sizes.max())], role='final')))
    json.dump(dict(dataset='Musk', role='final eval', n=len(y), pos=int(y.sum()), tasks=tasks),
              open(os.path.join(OUT, 'Musk', 'manifest.json'), 'w'), indent=1)
    print('Musk', f'{len(y)} molecules ({y.sum()} musk), conformations per molecule min/median/max '
                  f'{sizes.min()}/{int(np.median(sizes))}/{sizes.max()}', flush=True)


PHYS_STATIC = {'RecordID', 'Age', 'Gender', 'Height', 'ICUType'}


def build_physionet(n_splits=10, target='death'):
    """PhysioNet 2012 set-a (final eval): row = ICU stay; columns = the time-varying variables (plus the static ones as
    exact n = 1 columns); instances = that variable's measurements over 48 h (marginal: times differ per variable;
    paired = False); y = in-hospital death. 10 seeded stratified half splits."""
    base = os.path.join(RAW, 'physionet2012')
    outc = pd.read_csv(os.path.join(base, 'Outcomes-a.txt')).set_index('RecordID')
    out = outc['In-hospital_death'] if target == 'death' else outc.loc[outc['Length_of_stay'] > 0, 'Length_of_stay']
    recs = {}
    with zipfile.ZipFile(os.path.join(base, 'set-a.zip')) as z:
        for f in z.namelist():
            if f.endswith('.txt'):
                d = pd.read_csv(io.BytesIO(z.read(f)))
                rid = int(d.loc[d['Parameter'] == 'RecordID', 'Value'].iloc[0])
                recs[rid] = d[d['Parameter'].notna()]
    params = sorted({p for d in recs.values() for p in d['Parameter'].unique()} - {'RecordID'})
    static = [p for p in params if p in PHYS_STATIC]
    dyn = [p for p in params if p not in PHYS_STATIC]
    cols = static + dyn
    ids = sorted(set(recs) & set(out.index))
    reps = np.full((len(ids), len(cols), K_MAX), np.nan, dtype=np.float32)
    for i, rid in enumerate(ids):
        d = recs[rid]
        for j, p in enumerate(cols):
            v = d.loc[d['Parameter'] == p, 'Value'].to_numpy(np.float64)
            v = v[v >= 0] if p in PHYS_STATIC else v  # -1 = missing for the static descriptors
            if p in PHYS_STATIC:
                v = v[:1]
            reps[i, j, :min(len(v), K_MAX)] = v[:K_MAX]
    if target == 'los':  # PhysioNetLOS (final eval, regression): log length of stay [days]; 10 seeded half splits
        y = np.log(out.loc[ids].to_numpy().astype(np.float64))
        tasks = []
        for t in range(n_splits):
            idx = np.random.default_rng([t, 47]).permutation(len(y))
            tasks.append(save('PhysioNetLOS', f'split{t + 1:02d}', reps, y, idx[:len(y) // 2][:1000], idx[len(y) // 2:][:1000],
                              False, dict(columns=cols, static=static, target='log length of stay', role='final')))
        json.dump(dict(dataset='PhysioNetLOS', role='final eval', n=len(y), tasks=tasks),
                  open(os.path.join(OUT, 'PhysioNetLOS', 'manifest.json'), 'w'), indent=1)
        print('PhysioNetLOS', f'{len(y)} stays, log LOS sd {y.std():.2f}', flush=True)
        return
    y = out.loc[ids].to_numpy().astype(np.int64)
    tasks = []
    for t in range(n_splits):
        rng = np.random.default_rng([t, 37])
        tr, te = [], []
        for c in (0, 1):
            idx = rng.permutation(np.where(y == c)[0])
            tr.append(idx[:len(idx) // 2][:500 if c == 0 else None]); te.append(idx[len(idx) // 2:][:500 if c == 0 else None])
        tasks.append(save('PhysioNet', f'split{t + 1:02d}', reps, y, np.concatenate(tr), np.concatenate(te), False,
                          dict(columns=cols, static=static, role='final')))
    cnt = np.isfinite(reps).sum(-1)
    json.dump(dict(dataset='PhysioNet', role='final eval', n=len(y), deaths=int(y.sum()), tasks=tasks),
              open(os.path.join(OUT, 'PhysioNet', 'manifest.json'), 'w'), indent=1)
    print('PhysioNet', f'{len(y)} stays ({y.sum()} deaths), {len(cols)} columns ({len(static)} static); measurements per '
                       f'cell median {np.median(cnt[:, len(static):]):.0f}, n=0 cells {(cnt == 0).mean():.2f}', flush=True)


BEIJING = os.path.join(RAW, 'uci_beijing_aq', 'PRSA_Data_20130301-20170228')
BJ_COLS = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']


def build_beijing(max_rows=1000):
    """Beijing (regression dev): UCI 501 multi-site air quality. Per station (12 tasks): row = day, columns = the 11
    numeric variables, instances = the day's 24 hourly readings (paired: an hour is measured on every variable; missing
    readings stay as gaps), y = log1p(next day's mean PM2.5) at the station. Temporal split: context = days up to
    2015-12-31 (seeded subsample of max_rows), test = days from 2016-01-01."""
    import glob as _g
    tasks = []
    for f in sorted(_g.glob(os.path.join(BEIJING, '*.csv'))):
        d = pd.read_csv(f)
        st = d['station'].iloc[0]
        d['date'] = pd.to_datetime(d[['year', 'month', 'day']])
        days = sorted(d['date'].unique())
        reps = np.full((len(days), len(BJ_COLS), 24), np.nan, dtype=np.float32)
        pos = {t: i for i, t in enumerate(days)}
        di, hi = d['date'].map(pos).to_numpy(), d['hour'].to_numpy()
        reps[di, :, hi] = d[BJ_COLS].to_numpy(np.float32)
        pm = np.nanmean(reps[:, 0], -1)
        y = np.log1p(np.r_[pm[1:], np.nan])  # next day's mean PM2.5
        ok = np.isfinite(y) & np.isfinite(reps).any(-1).all(-1)
        dt = np.array(days, dtype='datetime64[D]')
        ctx = np.where(ok & (dt < np.datetime64('2016-01-01')))[0]
        te = np.where(ok & (dt >= np.datetime64('2016-01-01')))[0]
        ctx = np.sort(np.random.default_rng(53).permutation(ctx)[:max_rows])
        tasks.append(save('Beijing', st, reps, y, ctx, te, True, dict(columns=BJ_COLS, instances='24 hourly readings',
                                                                       target='log1p next-day mean PM2.5', role='dev')))
        print('Beijing', st, tasks[-1], f'missing readings {np.isnan(reps[ok]).mean():.3f}', flush=True)
    json.dump(dict(dataset='Beijing', role='dev (regression)', tasks=tasks), open(os.path.join(OUT, 'Beijing', 'manifest.json'), 'w'), indent=1)


CYTO = os.path.join(RAW, 'cytometry_zenodo6546964')
TECH = {'Time', 'Cell_length', 'Dead', 'DNA1', 'DNA2', '(La139)Dd', 'Event_length', 'Center', 'Offset', 'Width', 'Residual',
        ' ', '', 'ViViD'}  # technical / scatter / viability channels


def build_cyto(name, fname, sample_key, label_key, n_splits=20, k=1024, group_of=None, group_key=None):
    """Cytometry (final eval; Zenodo 6546964 preprocessed AnnData): row = sample (donor), instances = its cells
    (paired: a cell is measured on every marker), seeded subsample of k cells per sample; technical channels dropped;
    y = the sample's label. n_splits seeded stratified half splits (small cohorts)."""
    import h5py
    h = h5py.File(os.path.join(CYTO, fname), 'r')
    var = h['var']
    dec = lambda a: [x.decode() if isinstance(x, bytes) else str(x) for x in a]
    if 'pns_label' in var and '__categories' in var and 'pns_label' in var['__categories']:  # categorical codes
        names = [dec(var['__categories']['pns_label'][:])[c] for c in var['pns_label'][:]]
    elif 'pns_label' in var:
        names = dec(var['pns_label'][:])
    else:
        names = [str(i) for i in range(h['X'].shape[1])]
    snames = dec(h['obs']['__categories'][sample_key][:]) if sample_key in h['obs']['__categories'] else None
    keep = [j for j, n_ in enumerate(names) if n_ not in TECH]
    sid, lab = h['obs'][sample_key][:], h['obs'][label_key][:]
    ids = np.unique(sid)
    rng = np.random.default_rng(41)
    X = h['X']
    reps = np.full((len(ids), len(keep), k), np.nan, dtype=np.float32)
    y, sizes = np.zeros(len(ids), np.int64), []
    for i, s_ in enumerate(ids):
        rows = np.where(sid == s_)[0]
        sizes.append(len(rows))
        pick = np.sort(rng.permutation(rows)[:k])
        reps[i, :, :len(pick)] = X[pick][:, keep].T
        ls = np.unique(lab[rows])
        assert len(ls) == 1, (s_, ls)
        y[i] = int(ls[0])
    if group_key is not None:  # subject id from an obs column (first cell of each sample)
        gcol = h['obs'][group_key][:]
        grp = np.array([str(int(gcol[np.argmax(sid == s_)])) for s_ in ids])
    else:
        grp = np.array([group_of(snames[int(s_)]) if group_of else str(int(s_)) for s_ in ids])
    tasks = []
    for t in range(n_splits):
        r2 = np.random.default_rng([t, 43])
        gs = r2.permutation(np.unique(grp))
        side = {}  # alternate groups between the two sides within each label pattern (stratified, grouped)
        pat = {g_: tuple(sorted(set(y[grp == g_].tolist()))) for g_ in gs}
        cnt = {}
        for g_ in gs:
            side[g_] = cnt.get(pat[g_], 0) % 2
            cnt[pat[g_]] = cnt.get(pat[g_], 0) + 1
        is_tr = np.array([side[g_] == 0 for g_ in grp])
        tasks.append(save(name, f'split{t + 1:02d}', reps, y, np.where(is_tr)[0], np.where(~is_tr)[0], True,
                          dict(markers=[names[j] for j in keep], cells_per_sample=k, role='final')))
    json.dump(dict(dataset=name, role='final eval', n=len(y), label_counts=np.bincount(y).tolist(), cells=sizes,
                   tasks=tasks), open(os.path.join(OUT, name, 'manifest.json'), 'w'), indent=1)
    print(name, f'groups {len(np.unique(grp))}, first split context/test {tasks[0]["n_train"]}/{tasks[0]["n_test"]}; ', end='')
    print(name, f'{len(y)} samples, labels {np.bincount(y).tolist()}, {len(keep)} markers, cells/sample '
                f'{min(sizes)}-{max(sizes)} (subsampled to {k})', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='HAR,Musk,PhysioNet')
    a = ap.parse_args()
    for ds in a.datasets.split(','):
        dict(HAR=build_har, Musk=build_musk, PhysioNet=build_physionet, PhysioNetLOS=lambda: build_physionet(target='los'),
             Beijing=build_beijing,
             NK=lambda: build_cyto('NK', 'nk_cell_preprocessed.h5ad', 'FCS_File', 'label'),
             HVTN=lambda: build_cyto('HVTN', 'hvtn_preprocessed.h5ad', 'FCS_File', 'label', group_key='Sample_Description'),
             PreE=lambda: build_cyto('PreE', 'preeclampsia_preprocessed.h5ad', 'FCS_File', 'label',
                                     group_of=lambda n: n.split('_')[1][:5]))[ds]()
