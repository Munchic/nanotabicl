"""Regression evaluation on the real replicate regression tasks (real_data.py: GasConc, Tox21Pot, <L1000|CellPainting>_dose_reg).

Targets are standardized with the context mean / sd; metrics (reg_metrics: CRPS, RMSE, R^2, PIT, coverage) are in
context-sd units. Our checkpoints get the input their mode expects (real_eval.task_inputs); k = 1 uses each row's first
replicate only. With --wrapper, the off-the-shelf TabICLRegressor (mean-imputed input, n_estimators = --n_ens) instead
(run with the tabicl environment). Records -> artifacts/real_eval_reg.jsonl. Usage:
  python real_eval_reg.py tag=ckpt [tag=ckpt ...] --datasets GasConc,Tox21Pot [--ks 1,all] [--n_ens 1]
  python real_eval_reg.py --wrapper --datasets GasConc,Tox21Pot [--n_ens 8]
"""
import argparse, glob, json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from reg_metrics import reg_metrics, TAUS  # noqa: E402


def tasks(datasets):
    for ds in datasets:
        for f in sorted(glob.glob(os.path.join(ROOT, 'artifacts', 'real', ds, '*.npz'))):
            z = np.load(f)
            tr, te = z['train_idx'], z['test_idx']
            y = z['y'].astype(np.float64)
            y = (y - y[tr].mean()) / (y[tr].std() + 1e-8)
            yield ds, os.path.basename(f)[:-4], z, tr, te, y


def mean_input(z, k):
    counts = z['counts'].astype(np.int64) if k == 'all' else np.minimum(z['counts'], int(k)).astype(np.int64)
    mask = np.arange(z['replicates'].shape[-1])[None, None] < counts[..., None]
    with np.errstate(all='ignore'):
        return np.nanmean(np.where(mask, np.nan_to_num(z['replicates']), np.nan), -1)


def run_ckpts(specs, datasets, ks, n_ens):
    import torch
    import obs_eval
    from obs_train import make_model, set_quantiles
    from real_eval import task_inputs
    from tabicl_port import load_tabicl
    recs = []
    for spec in specs:
        tag, path = spec.split('=', 1)
        ck = torch.load(path, map_location='cuda', weights_only=False)
        assert ck.get('task') == 'regression', path
        set_quantiles(load_tabicl('regression')[0]['out_dim'])
        model = make_model(ck['mode'], ck['rep_cfg'], 'regression', cfg=ck.get('model_cfg')).cuda().eval()
        model.load_state_dict(ck['state_dict'])
        obs_eval.FP32 = ck.get('arch') == 'tabicl'
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = obs_eval.FP32
        for ds, task, z, tr, te, y in tasks(datasets):
            order = np.concatenate([tr, te])
            y_in = torch.tensor(y[tr], dtype=torch.float32, device='cuda')[None]
            for k in ks:
                with torch.no_grad():
                    q = obs_eval.ens_predict(model, task_inputs(z, ck['mode'], k, order, 'cuda'), y_in, ck['mode'], 'regression', None, n_ens)
                m, _ = reg_metrics(q, y[te])
                recs.append(dict(model=tag, dataset=ds, task=task, k=k, n_ens=n_ens, **m))
            torch.cuda.empty_cache()
        print_summary(recs, tag, datasets, ks)
    return recs


def run_wrapper(datasets, ks, n_ens):
    from tabicl import TabICLRegressor
    recs = []
    tag = f'tabicl_mean_e{n_ens}'
    for ds, task, z, tr, te, y in tasks(datasets):
        for k in ks:
            X = mean_input(z, k)
            reg = TabICLRegressor(random_state=0, device='cuda', n_estimators=n_ens).fit(X[tr], y[tr])
            q = np.asarray(reg.predict(X[te], output_type='quantiles', alphas=list(TAUS)))
            m, _ = reg_metrics(q, y[te])
            recs.append(dict(model=tag, dataset=ds, task=task, k=k, n_ens=n_ens, **m))
    print_summary(recs, tag, datasets, ks)
    return recs


def print_summary(recs, tag, datasets, ks):
    for ds in datasets:
        print(f'{tag:24s} {ds:22s} ' + ' '.join(
            f"k={k}: crps {np.mean([r['crps'] for r in recs if r['model'] == tag and r['dataset'] == ds and r['k'] == k]):.3f}"
            for k in ks), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='*')
    ap.add_argument('--wrapper', action='store_true')
    ap.add_argument('--datasets', default='GasConc,Tox21Pot')
    ap.add_argument('--ks', default='1,all')
    ap.add_argument('--n_ens', type=int, default=1)
    ap.add_argument('--out', default=os.path.join(ROOT, 'artifacts', 'real_eval_reg.jsonl'))
    a = ap.parse_args()
    ks = [k if k == 'all' else int(k) for k in a.ks.split(',')]
    ds = a.datasets.split(',')
    recs = run_wrapper(ds, ks, a.n_ens) if a.wrapper else run_ckpts(a.ckpts, ds, ks, a.n_ens)
    with open(a.out, 'a') as fh:
        for r in recs:
            fh.write(json.dumps(r) + '\n')
