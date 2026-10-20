"""Huber M-estimate of each cell's location with a column-pooled robust scale (numpy): sigma_j = 1.4826 x median over
the column's replicated context cells of |replicate - cell median|, corrected for the small-sample bias of the median
residual by the Gaussian factor (estimated by simulation, see C_K). IRLS from the cell median, weights
w = min(1, c sigma_j / |r - mu|), c = 1.345 (95% Gaussian efficiency). n = 1 cells keep their value."""
import numpy as np

C_HUBER = 1.345


# 1 / median(|r - cell median|) for standard-normal cells of size n = 2..50 (simulated, 1e5 cells each); odd n are
# larger because the median replicate has residual 0
SCALE_FACTOR = np.array([np.nan, np.nan] + [2.0807, 3.1831, 2.0578, 2.0429, 1.9019, 1.8123, 1.7781, 1.7174, 1.7026, 1.6642, 1.6575, 1.6337, 1.6269, 1.6064, 1.6036, 1.593, 1.5868, 1.5759, 1.5772, 1.5673, 1.5686, 1.5619, 1.5572, 1.5536, 1.5514, 1.5462, 1.5455, 1.5417, 1.5426, 1.5402, 1.5382, 1.5332, 1.5337, 1.5306, 1.5298, 1.5262, 1.5299, 1.526, 1.5271, 1.5235, 1.5233, 1.5213, 1.5217, 1.5187, 1.5197, 1.5191, 1.5168, 1.5158, 1.5154])


def pooled_scale(R, M, ctx_rows):
    """R (n, d, K) NaN-padded, M (n, d, K) bool -> (d,) robust replicate sd per column: median over the column's
    replicated context cells of |replicate - cell median| x SCALE_FACTOR[cell size] (consistent for Gaussian noise)."""
    n = M.sum(-1)
    med = np.nanmedian(np.where(M, R, np.nan), -1, keepdims=True)
    res = np.abs(np.where(M, R - med, np.nan)) * SCALE_FACTOR[np.clip(n, 0, 50)][..., None]
    res = np.where((n >= 2)[..., None], res, np.nan)[ctx_rows]  # (rows, d, K)
    with np.errstate(all='ignore'):
        return np.nanmedian(res.transpose(1, 0, 2).reshape(R.shape[1], -1), 1)


def huber_impute(R, M, ctx_rows, iters=10):
    sig = pooled_scale(R, M, ctx_rows)  # (d,)
    X = np.where(M, R, np.nan)
    with np.errstate(all='ignore'):
        mu = np.nanmedian(X, -1)
        for _ in range(iters):
            r = X - mu[..., None]
            w = np.minimum(1.0, C_HUBER * sig[None, :, None] / np.maximum(np.abs(r), 1e-12))
            w = np.where(M, w, 0.0)
            mu = np.where(np.isfinite(sig)[None], np.nansum(w * np.nan_to_num(X), -1) / np.maximum(w.sum(-1), 1e-12), np.nanmean(X, -1))
    return mu
