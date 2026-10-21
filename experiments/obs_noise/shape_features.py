"""Does the shape of a cell's replicate set carry signal beyond its mean? Training-free check with the off-the-shelf
TabICLv2 wrapper on the existing real tasks: each cell's replicates summarized as
  mean          : the mean (baseline, = what a user feeds the wrapper today)
  mean_sd       : + the replicate sd (0 for n = 1)
  sketch        : mean, median, sd, q25, q75, min, max
  sketch_n      : sketch + log n
All replicates, n_estimators 1 and 8. Classification: acc / AUC / log-loss; regression: CRPS (context-sd units).
Records -> artifacts/shape_features.jsonl. Run with the tabicl environment:
  python shape_features.py [--datasets ...] [--n_est 1,8]
"""
import argparse, glob, json, os, sys, warnings

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from reg_metrics import reg_metrics, TAUS  # noqa: E402
from real_tabicl import auc  # noqa: E402
from calib_rel import calib_by_reliability  # noqa: E402

REG = {'GasConc', 'Tox21Pot', 'CellPainting_dose_reg', 'L1000_dose_reg'}


def summaries(z, tr=None):
    counts = z['counts'].astype(np.int64)
    mask = np.isfinite(z['replicates'])  # (= arange < counts for packed sets; keeps gaps for paired sets)
    r = np.where(mask, np.nan_to_num(z['replicates']), np.nan)
    with np.errstate(all='ignore'):
        s = dict(mean=np.nanmean(r, -1), median=np.nanmedian(r, -1), sd=np.nan_to_num(np.nanstd(r, -1, ddof=1)),
                 q25=np.nanquantile(r, 0.25, -1), q75=np.nanquantile(r, 0.75, -1), min=np.nanmin(r, -1), max=np.nanmax(r, -1),
                 logn=np.log(counts.clip(min=1)).astype(np.float64))
        med = s['median']
        s['mad'] = 1.4826 * np.nanmedian(np.abs(r - med[..., None]), -1)
    if tr is not None:  # Huber location with the column-pooled scale (huber.py, = the tx_huber0p reference)
        from huber import huber_impute
        s['huber'] = huber_impute(r, mask, tr)
    return s


def cluster_freq(z, tr, k=16, seed=0):
    """Paired sets: k-means (k clusters) on the context rows' instances (columns standardized with context
    instances), each row -> its instances' cluster shares (the standard cytometry featurization)."""
    from sklearn.cluster import MiniBatchKMeans
    reps = z['replicates']  # (n, d, K)
    inst = np.transpose(reps, (0, 2, 1))  # (n, K, d)
    ok = np.isfinite(inst).all(-1)
    ctx = inst[tr][ok[tr]]
    mu, sd = ctx.mean(0), ctx.std(0) + 1e-8
    rng = np.random.default_rng(seed)
    fit = ctx[rng.permutation(len(ctx))[:200000]]
    km = MiniBatchKMeans(k, random_state=seed, n_init=3, batch_size=4096).fit((fit - mu) / sd)
    out = np.zeros((len(reps), k))
    for i in range(len(reps)):
        if ok[i].any():
            lab = km.predict((inst[i][ok[i]] - mu) / sd)
            out[i] = np.bincount(lab, minlength=k) / len(lab)
    return out


ARMS = {'mean': ['mean'], 'mean_sd': ['mean', 'sd'], 'sketch': ['mean', 'median', 'sd', 'q25', 'q75', 'min', 'max'],
        'sketch_n': ['mean', 'median', 'sd', 'q25', 'q75', 'min', 'max', 'logn'],
        'huber_sd': ['huber', 'sd'], 'med_mad': ['median', 'mad'], 'huber_mad': ['huber', 'mad']}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='L1000_dose,CellPainting_dose,GasDrift,Tox21,UPS1_low,CPTAC,'
                                          'CellPainting_dose_reg,GasConc,Tox21Pot')
    ap.add_argument('--n_est', default='1,8')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--arms', default='', help='comma-separated subset of ARMS (default: all)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'artifacts', 'shape_features.jsonl'))
    a = ap.parse_args()
    from tabicl import TabICLClassifier, TabICLRegressor
    from run_tabicl import base_ckpt_path
    ckpt = base_ckpt_path()
    for ds in a.datasets.split(','):
        for f in sorted(glob.glob(os.path.join(ROOT, 'artifacts', 'real', ds, '*.npz'))):
            z = np.load(f)
            tr, te = z['train_idx'], z['test_idx']
            s = summaries(z, tr)
            recs = []
            arms = {k: v for k, v in ARMS.items() if not a.arms or k in a.arms.split(',')}
            if 'paired' in z.files and bool(z['paired']):
                arms['clusters'] = ['mean', 'clusters']
                s['clusters'] = cluster_freq(z, tr)
            for arm, keys in arms.items():
                X = np.concatenate([s[k] for k in keys], axis=1)
                for n_est in map(int, a.n_est.split(',')):
                    r = dict(dataset=ds, task=os.path.basename(f)[:-4], arm=arm, n_est=n_est, n_feat=X.shape[1])
                    if ds in REG or z['y'].dtype.kind == 'f':  # regression target
                        y = z['y'].astype(np.float64)
                        y = (y - y[tr].mean()) / (y[tr].std() + 1e-8)
                        reg = TabICLRegressor(random_state=0, device=a.device, n_estimators=n_est).fit(X[tr], y[tr])
                        q = np.asarray(reg.predict(X[te], output_type='quantiles', alphas=list(TAUS)))
                        m, _ = reg_metrics(q, y[te])
                        r.update(crps=m['crps'], r2=m['r2'])
                    else:
                        y = z['y']
                        n_cls = int(y.max()) + 1
                        seen = np.unique(y[tr])
                        p = np.zeros((len(te), n_cls))
                        p[:, seen] = TabICLClassifier(model_path=ckpt, random_state=0, device=a.device,
                                                      n_estimators=n_est).fit(X[tr], y[tr]).predict_proba(X[te])
                        p = np.clip(p, 1e-7, 1); p /= p.sum(1, keepdims=True)
                        r.update(acc=accuracy_score(y[te], p.argmax(1)), auc=auc(y[te], p),
                                 log_loss=log_loss(y[te], p, labels=np.arange(n_cls)),
                                 rel=calib_by_reliability(p, y[te], z['replicates'], z['counts'], tr, te))
                    recs.append(r)
            with open(a.out, 'a') as fh:
                for r in recs:
                    fh.write(json.dumps(r) + '\n')
            key = 'crps' if (ds in REG or z['y'].dtype.kind == 'f') else 'acc'
            print(ds, os.path.basename(f), ' '.join(f"{r['arm']}/e{r['n_est']} {r[key]:.3f}" for r in recs), flush=True)
