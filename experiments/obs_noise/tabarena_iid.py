"""IID guard (CENSORING.md): does a continued-trained TabICLv2 stay comparable to stock on clean, non-skewed IID data?

A few TabArena regression tasks (all TabArena splits are random / IID), picked from the leaderboard's per-dataset
results (artifacts/tabarena/frontier.csv) for |target skew| < 1, mostly numeric features, and the largest spread between
frontier models (TabICLv2, TabFM, TabPFN-2.6 / 3, RealTabPFN-2.5, LimiX-2, Mitra-v2) relative to the per-split SE, i.e.
the tasks most able to detect a regression. Data and the official OpenML 3-fold splits are fetched once to
artifacts/tabarena/<name>.npz (categoricals ordinal-coded, missing = NaN).

  python tabarena_iid.py fetch
  python tabarena_iid.py eval [name=ckpt ...] [--datasets ...] [--repeats 1]   # our pipeline (ml env)
  tabarena_wrapper.py                                                          # the sklearn wrapper (uv env)

Arms (our pipeline: raw module, fp32 + TF32, wrapper preprocessing, 1 pass, context = the full training fold):
  stock         released TabICLv2 regression weights
  <name>_nospec checkpoint, no spec (flags None)                  -> must equal stock up to continued training
  <name>_known  checkpoint, known all-observed spec (known models) -> "the user says nothing is censored"
RMSE in original units (TabArena's metric; point prediction = mean of the quantiles) and CRPS, per split, to
artifacts/tabarena_iid.jsonl. Compare with the TabArena 'TabICLv2 (default)' numbers (8-estimator wrapper) and the
wrapper run here, as anchors for the pipeline.
"""
import argparse, io, json, os, sys, urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'artifacts', 'tabarena')
TASKS = {  # name: OpenML task id (TabArena curated metadata)
    'houses': 363678, 'physiochemical_protein': 363693, 'superconductivity': 363705, 'wine_quality': 363708,
    'airfoil_self_noise': 363612, 'QSAR-TID-11': 363697,
}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'nanotabicl'})) as r:
        return r.read()


def fetch(name):
    import pandas as pd
    from scipy.io import arff
    path = os.path.join(OUT, f'{name}.npz')
    if os.path.exists(path):
        return path
    task = json.loads(get(f'https://www.openml.org/api/v1/json/task/{TASKS[name]}'))['task']
    inp = {x['name']: x for x in task['input']}
    did = inp['source_data']['data_set']['data_set_id']
    target = inp['source_data']['data_set']['target_feature']
    desc = json.loads(get(f'https://www.openml.org/api/v1/json/data/{did}'))['data_set_description']
    data, meta = arff.loadarff(io.StringIO(get(desc['url']).decode('utf-8')))
    df = pd.DataFrame(data)
    y = df.pop(target).to_numpy(np.float64)
    cols, cat = [], []
    for c in df.columns:
        v = df[c]
        if v.dtype == object:  # nominal: bytes -> ordinal codes (missing '?' -> NaN)
            s = v.map(lambda b: b.decode() if isinstance(b, bytes) else b)
            s = s.where(s != '?')
            codes, _ = pd.factorize(s, sort=True)
            cols.append(np.where(codes < 0, np.nan, codes).astype(np.float64)); cat.append(True)
        else:
            cols.append(v.to_numpy(np.float64)); cat.append(False)
    X = np.stack(cols, 1)
    sp, _ = arff.loadarff(io.StringIO(get(inp['estimation_procedure']['estimation_procedure']['data_splits_url']).decode()))
    sp = pd.DataFrame(sp)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(path, X=X, y=y, cat=np.array(cat), columns=np.array(list(df.columns)),
                        split_test=(sp['type'] == b'TEST').to_numpy(), split_row=sp['rowid'].to_numpy(np.int64),
                        split_repeat=sp['repeat'].to_numpy(np.int64), split_fold=sp['fold'].to_numpy(np.int64),
                        meta=json.dumps(dict(task_id=TASKS[name], dataset_id=did, target=target)))
    print(f'{name}: X {X.shape}, {sum(cat)} categorical, target {target}, splits {len(sp)}', flush=True)
    return path


def splits(z, repeats, max_ctx=0, name=''):
    """-> [(repeat, fold, train_idx, test_idx)] for repeats 0..repeats-1, all folds. max_ctx > 0: the context is a seeded
    max_ctx-row subsample of the training fold (the ICL stage is O(n_context^2) in memory; tabarena_wrapper.py draws the
    identical subsample), the test fold is complete."""
    out = []
    for r in range(repeats):
        for f in sorted(set(z['split_fold'][z['split_repeat'] == r].tolist())):
            m = (z['split_repeat'] == r) & (z['split_fold'] == f)
            tr, te = z['split_row'][m & ~z['split_test']], z['split_row'][m & z['split_test']]
            if max_ctx and len(tr) > max_ctx:
                tr = np.sort(np.random.default_rng([sum(map(ord, name)), r, f]).choice(tr, max_ctx, replace=False))
            out.append((r, f, tr, te))
    return out


def predict_ours(model, Xtr, ytr, Xte, flag=None, chunk=2048):
    """Our pipeline, one pass -> (T, NQ) quantiles in original units. Test rows are processed in chunks: the column and
    ICL stages only attend to context rows and the preprocessing is fitted on them, so chunking is exact."""
    import torch
    mu_x = np.nanmean(Xtr, 0)
    Xtr, Xte = (np.where(np.isnan(a), mu_x, a) for a in (Xtr, Xte))  # mean imputation, as the wrapper
    mu, sd = ytr.mean(), ytr.std() + 1e-8
    dev = 'cuda'
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
    Xtr_t, yc = t(Xtr), t((ytr - mu) / sd)
    qs = []
    for j in range(0, len(Xte), chunk):
        X = torch.cat([Xtr_t, t(Xte[j:j + chunk])])
        xf = yf = None
        if flag is not None:
            xf = torch.full(X.shape, flag, dtype=torch.long, device=dev)
            yf = torch.full(yc.shape, flag, dtype=torch.long, device=dev)
        with torch.no_grad():
            qs.append(model(X[None], yc[None], None if xf is None else xf[None], None if yf is None else yf[None])[0].float())
    return (torch.cat(qs) * sd + mu).cpu().numpy()


def scores(q, y):  # q (T, 999) quantiles at levels k / 1000; -> RMSE of the mean, CRPS at levels k / 100 (= the wrapper's)
    rmse = float(np.sqrt(((q.mean(1) - y) ** 2).mean()))
    q = q[:, 9::10]
    taus = np.arange(1, q.shape[1] + 1) / (q.shape[1] + 1)
    d = y[:, None] - q
    crps = 2 * np.maximum(taus * d, (taus - 1) * d).mean(1)
    return dict(rmse=rmse, crps=float(crps.mean()))


def evaluate(ckpts, datasets, repeats, mem_frac, max_ctx, chunk=2048):
    import torch
    sys.path.insert(0, os.path.join(ROOT, '..', '..'))
    from cens_model import CensNanoTabICL
    from tabicl_port import load_tabicl
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if mem_frac:  # fail in this process first, never push co-running training jobs into OOM
        torch.cuda.set_per_process_memory_fraction(mem_frac)
    cfg, sd0 = load_tabicl('regression')
    arms = []
    m0 = CensNanoTabICL(**cfg).cuda().eval()
    m0.load_state_dict(sd0, strict=False)
    arms.append(('stock', m0, None))
    for spec in ckpts:
        name, path = spec.split('=', 1)
        ck = torch.load(path, map_location='cuda', weights_only=False)
        known = ck.get('cens_cfg', {}).get('known', False)
        m = CensNanoTabICL(**ck['model_cfg'], **ck.get('cens_cfg', {})).cuda().eval()
        missing, unexpected = m.load_state_dict(ck['state_dict'], strict=False)  # plain backbones: cens_ maps stay 0
        assert not unexpected and all(k.startswith('cens_') for k in missing), (missing, unexpected)
        arms.append((f'{name}_nospec', m, None))
        if known:
            arms.append((f'{name}_known', m, 3))
    out = open(os.path.join(ROOT, 'artifacts', 'tabarena_iid.jsonl'), 'a')
    for ds in datasets:
        z = np.load(fetch(ds))
        X, y = z['X'], z['y']
        for r, f, tr, te in splits(z, repeats, max_ctx, ds):
            for tag, m, flag in arms:
                q = predict_ours(m, X[tr], y[tr], X[te], flag, chunk)
                rec = dict(model=tag, dataset=ds, repeat=r, fold=f, n_train=len(tr), n_test=len(te), max_ctx=max_ctx, **scores(q, y[te]))
                out.write(json.dumps(rec) + '\n'); out.flush()
                print(ds, r, f, tag, f"rmse {rec['rmse']:.4f} crps {rec['crps']:.4f}", flush=True)
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['fetch', 'eval'])
    ap.add_argument('ckpts', nargs='*')
    ap.add_argument('--datasets', default='airfoil_self_noise,wine_quality,houses,physiochemical_protein,superconductivity')
    ap.add_argument('--repeats', type=int, default=1, help='TabArena repeats to use (3 folds each)')
    ap.add_argument('--mem_frac', type=float, default=0.3, help='cap on this process\'s share of GPU memory')
    ap.add_argument('--max_ctx', type=int, default=4096, help='context subsample cap per training fold (0 = full)')
    ap.add_argument('--chunk', type=int, default=2048, help='test rows per forward pass (exact; smaller = less memory)')
    a = ap.parse_args()
    if a.cmd == 'fetch':
        for ds in TASKS:
            fetch(ds)
    else:
        evaluate(a.ckpts, a.datasets.split(','), a.repeats, a.mem_frac, a.max_ctx, a.chunk)


if __name__ == '__main__':
    main()
