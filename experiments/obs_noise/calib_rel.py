"""Reliability-conditional calibration (classification): does predicted uncertainty track the actual error where
samples are unreliable (few instances, or many but widely spread)?

Per test row: n_rel = mean over columns of log(instances); spread = mean over columns of the within-cell sd divided by
the column's sd of context cell means (0 for n = 1 cells). Rows are split into tertiles of each; per bin: accuracy,
mean confidence (max probability), overconfidence (= confidence - accuracy; 0 when calibrated), log-loss. Plus
conf_auroc: AUROC of the confidence for separating correct from wrong predictions (0.5 = uninformative).
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def row_reliability(reps, counts, tr, te):
    """reps (n, d, K) NaN-padded, counts (n, d) -> (log-n, spread) per test row."""
    with np.errstate(all='ignore'):
        mean = np.nanmean(reps, -1)
        sd = np.nan_to_num(np.nanstd(reps, -1, ddof=1))
        col_sd = np.nanstd(mean[tr], 0) + 1e-8
    logn = np.log(np.maximum(counts[te], 1)).mean(1)
    spread = (sd[te] / col_sd).mean(1)
    return logn, spread


def bins_report(p, y, key):
    conf, pred = p.max(1), p.argmax(1)
    ok = (pred == y).astype(float)
    ll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1))
    edges = np.quantile(key, [1 / 3, 2 / 3])
    b = np.digitize(key, edges)
    return [dict(bin=int(i), n=int((b == i).sum()), acc=float(ok[b == i].mean()), conf=float(conf[b == i].mean()),
                 overconf=float(conf[b == i].mean() - ok[b == i].mean()), log_loss=float(ll[b == i].mean()))
            for i in range(3) if (b == i).any()]


def calib_by_reliability(p, y, reps, counts, tr, te):
    logn, spread = row_reliability(reps, counts, tr, te)
    ok = p.argmax(1) == y
    auroc = float(roc_auc_score(ok, p.max(1))) if 0 < ok.mean() < 1 else float('nan')
    return dict(by_n=bins_report(p, y, logn), by_spread=bins_report(p, y, spread), conf_auroc=auroc)
