"""Regression metrics from predicted quantiles (Exp 5). All in units of the context target sd (y standardized with the
context mean / sd). Quantile levels TAUS = k / 100, k = 1..99.

PIT: u_i = F_hat(y_i), the predictive CDF at the observed target, by linear interpolation of the sorted predicted
quantiles (u = 0 below the 1% quantile, 1 above the 99% one). A calibrated model has u ~ Uniform(0, 1).
pit_ce = mean over alpha in {0.05, ..., 0.95} of |P(u <= alpha) - alpha|; pit_ks = max over a fine alpha grid.
"""
import numpy as np

NQ = 99
TAUS = np.arange(1, NQ + 1) / (NQ + 1)
LEVELS = [0.5, 0.8, 0.9, 0.95]
ALPHAS = np.arange(0.05, 0.951, 0.05)


def pit(q, y):  # q: (T, NQ) predicted quantiles, y: (T,) -> (T,) PIT values
    q = np.sort(q, axis=1)
    return np.array([np.interp(yi, qi, TAUS, left=0.0, right=1.0) for qi, yi in zip(q, y)])


def interval(q, level):  # central interval from the quantile curve, interpolated at (1 -/+ level) / 2
    q = np.sort(q, axis=1)
    lo_t, hi_t = (1 - level) / 2, (1 + level) / 2
    lo = np.array([np.interp(lo_t, TAUS, qi) for qi in q])
    hi = np.array([np.interp(hi_t, TAUS, qi) for qi in q])
    return lo, hi


def reg_metrics(q, y):
    q = np.sort(np.asarray(q, dtype=np.float64), axis=1)
    y = np.asarray(y, dtype=np.float64)
    med = q[:, NQ // 2]
    mse = np.mean((med - y) ** 2)
    diff = y[:, None] - q
    crps = 2 * np.mean(np.maximum(TAUS * diff, (TAUS - 1) * diff))  # quantile-score approximation of CRPS
    u = pit(q, y)
    emp = np.array([np.mean(u <= a) for a in ALPHAS])
    fine = np.linspace(0.01, 0.99, 99)
    out = dict(rmse=float(np.sqrt(mse)), r2=float(1 - mse / max(y.var(), 1e-12)), crps=float(crps),
               pit_ce=float(np.mean(np.abs(emp - ALPHAS))),
               pit_ks=float(np.max(np.abs(np.array([np.mean(u <= a) for a in fine]) - fine))),
               pit_mean=float(u.mean()))
    for lv in LEVELS:
        lo, hi = interval(q, lv)
        out[f'cov{int(lv * 100)}'] = float(np.mean((y >= lo) & (y <= hi)))
        out[f'width{int(lv * 100)}'] = float(np.mean(hi - lo))
    out['tail_out'] = float(np.mean((u < 0.05) | (u > 0.95)))  # nominal 0.10
    return out, u
