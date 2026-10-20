"""Off-the-shelf TabICLv2 sklearn wrapper on the real replicate tasks (real_data.py): replicate wells imputed by their
per-cell mean or median, 1 and 8 estimators, k = 1, 2, 3, all wells. Run with the tabicl environment.
Records -> artifacts/real_eval.jsonl (model = tabicl_<arm>_e<n_estimators>)."""
import glob, json, os, sys, warnings

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

BINS = np.linspace(0, 1, 16)


def auc(y, p):  # ROC-AUC: binary on p[:, 1]; multiclass one-vs-rest macro over the classes present in y
    present = np.unique(y)
    if len(present) < 2:
        return float('nan')
    if p.shape[1] == 2:
        return float(roc_auc_score(y, p[:, 1]))
    q = p[:, present] / p[:, present].sum(1, keepdims=True)
    return float(roc_auc_score(y, q, multi_class='ovr', labels=present)) if len(present) > 2 else float(roc_auc_score(y == present[1], q[:, 1]))


def calib_stats(p, y):  # = calib.calib_stats (top-label ECE, 15 bins; overconfidence); calib imports matplotlib
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == y).astype(float)
    idx = np.clip(np.digitize(conf, BINS) - 1, 0, len(BINS) - 2)
    ece = sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
              for b in range(len(BINS) - 1) if (idx == b).any())
    return dict(ece=ece, overconf=conf.mean() - correct.mean())


from run_tabicl import predict, base_ckpt_path  # noqa: E402

KS = [1, 2, 3, 'all']


def main():
    ckpt = base_ckpt_path()
    datasets = sys.argv[1].split(',') if len(sys.argv) > 1 else ['L1000', 'CellPainting']
    out = os.path.join(ROOT, 'artifacts', 'real_eval.jsonl')
    for ds in datasets:
        for f in sorted(glob.glob(os.path.join(ROOT, 'artifacts', 'real', ds, '*.npz'))):
            z = np.load(f)
            tr, te, y = z['train_idx'], z['test_idx'], z['y']
            n_cls = int(y.max()) + 1
            recs = []
            for k in KS:
                counts = z['counts'].astype(np.int64) if k == 'all' else np.minimum(z['counts'], k).astype(np.int64)
                mask = np.arange(z['replicates'].shape[-1])[None, None] < counts[..., None]
                r = np.where(mask, np.nan_to_num(z['replicates']), np.nan)
                for arm, X in [('mean', np.nanmean(r, -1)), ('median', np.nanmedian(r, -1))]:
                    for n_est in (1, 8):
                        seen = np.unique(y[tr])  # the wrapper's classes_ (LabelEncoder: sorted context labels)
                        p = np.zeros((len(te), n_cls))
                        p[:, seen] = predict(ckpt, X[tr], y[tr], X[te], n_est)  # classes absent from the context: 0
                        p = np.clip(p, 1e-7, 1)
                        p /= p.sum(1, keepdims=True)
                        c = calib_stats(p, y[te])
                        recs.append(dict(model=f'tabicl_{arm}_e{n_est}', dataset=ds, task=os.path.basename(f)[:-4], k=k,
                                         n_ens=n_est, acc=accuracy_score(y[te], p.argmax(1)), auc=auc(y[te], p),
                                         log_loss=log_loss(y[te], p, labels=np.arange(n_cls)), ece=float(c['ece']),
                                         overconf=float(c['overconf'])))
            with open(out, 'a') as fh:
                for r_ in recs:
                    fh.write(json.dumps(r_) + '\n')
            print(ds, os.path.basename(f), ' '.join(f"{r_['model']}/k{r_['k']} {r_['acc']:.3f}" for r_ in recs if r_['k'] == 'all'), flush=True)


if __name__ == '__main__':
    main()
