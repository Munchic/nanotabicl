"""Evaluate checkpoints on the real replicate tasks of real_data.py (artifacts/real/<dataset>/<task>.npz).

Every checkpoint gets the input its mode expects: median / mean imputation -> per-cell median / mean of the replicate
wells; obs -> the wells themselves (+ mask). Replicate-subsampling curve: k = 1, 2, 3, ... uses only each row's first
k wells (k = 1: what a single-measurement table would give). Records go to artifacts/real_eval.jsonl:
{model, dataset, task, k, n_ens, acc, log_loss, ece, overconf}. Usage:
  python real_eval.py tag=ckpt [tag=ckpt ...] [--datasets L1000,CellPainting] [--ks 1,2,3,all] [--n_ens 1]
"""
import argparse, glob, json, os, sys

import numpy as np, torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import obs_eval  # noqa: E402
from obs_train import make_model  # noqa: E402
from calib import calib_stats  # noqa: E402


def auc(y, p):  # ROC-AUC: binary on p[:, 1]; multiclass one-vs-rest macro over the classes present in y
    present = np.unique(y)
    if len(present) < 2:
        return float('nan')
    if p.shape[1] == 2:
        return float(roc_auc_score(y, p[:, 1]))
    q = p[:, present] / p[:, present].sum(1, keepdims=True)
    return float(roc_auc_score(y, q, multi_class='ovr', labels=present)) if len(present) > 2 else float(roc_auc_score(y == present[1], q[:, 1]))


def task_inputs(z, mode, k, order, dev):
    reps, counts = z['replicates'], z['counts'].astype(np.int64)
    if k != 'all':
        counts = np.minimum(counts, int(k))
    mask = np.arange(reps.shape[-1])[None, None] < counts[..., None]
    r = np.where(mask, np.nan_to_num(reps), np.nan)
    t = lambda a: torch.tensor(a[order], dtype=torch.float32, device=dev)[None]
    if mode == 'huber':
        from huber import huber_impute
        return [t(huber_impute(r, mask, z['train_idx']))]
    if mode in ('median', 'clean', 'mean'):
        with np.errstate(all='ignore'):
            x = np.nanmedian(r, -1) if mode == 'median' else np.nanmean(r, -1)
        return [t(x)]
    return [t(np.nan_to_num(r)), torch.tensor(mask[order], device=dev)[None]]


@torch.no_grad()
def evaluate(tag, path, datasets, ks, n_ens, out):
    dev = 'cuda'
    ck = torch.load(path, map_location=dev, weights_only=False)
    mode = ck['mode']
    model = make_model(mode, ck['rep_cfg'], 'classification', cfg=ck.get('model_cfg')).to(dev).eval()
    model.load_state_dict(ck['state_dict'])
    obs_eval.FP32 = ck.get('arch') == 'tabicl'  # TabICLv2 backbone: fp32 + TF32; nano: bf16 autocast
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = obs_eval.FP32
    recs = []
    for ds in datasets:
        for f in sorted(glob.glob(os.path.join(ROOT, 'artifacts', 'real', ds, '*.npz'))):
            z = np.load(f)
            tr, te, y = z['train_idx'], z['test_idx'], z['y']
            order = np.concatenate([tr, te])
            n_cls = int(y.max()) + 1
            y_in = torch.tensor(y[tr], device=dev)[None]
            for k in ks:
                inp = task_inputs(z, mode, k, order, dev)
                p = np.clip(obs_eval.ens_predict(model, inp, y_in, mode, 'classification', n_cls, n_ens), 1e-7, 1)
                p /= p.sum(1, keepdims=True)
                c = calib_stats(p, y[te])
                recs.append(dict(model=tag, dataset=ds, task=os.path.basename(f)[:-4], k=k, n_ens=n_ens,
                                 acc=accuracy_score(y[te], p.argmax(1)), auc=auc(y[te], p), log_loss=log_loss(y[te], p, labels=np.arange(n_cls)),
                                 ece=float(c['ece']), overconf=float(c['overconf'])))
            torch.cuda.empty_cache()
    with open(out, 'a') as fh:
        for r in recs:
            fh.write(json.dumps(r) + '\n')
    return recs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='+')
    ap.add_argument('--datasets', default='L1000,CellPainting')
    ap.add_argument('--ks', default='1,2,3,all')
    ap.add_argument('--n_ens', type=int, default=1)
    ap.add_argument('--out', default=os.path.join(ROOT, 'artifacts', 'real_eval.jsonl'))
    a = ap.parse_args()
    ks = [k if k == 'all' else int(k) for k in a.ks.split(',')]
    for spec in a.ckpts:
        tag, path = spec.split('=', 1)
        recs = evaluate(tag, path, a.datasets.split(','), ks, a.n_ens, a.out)
        for ds in a.datasets.split(','):
            print(f'{tag:24s} {ds:12s} ' + ' '.join(
                f"k={k}: acc {np.mean([r['acc'] for r in recs if r['dataset'] == ds and r['k'] == k]):.3f}" for k in ks), flush=True)
