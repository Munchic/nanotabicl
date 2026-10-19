"""Bayes-oracle per-cell denoiser: posterior mean E[x_ij | replicates] under the TRUE observation model.

Oracle knowledge: the noise family and scale, censoring limits, the value-dependent scale rule, the row-scale
distribution (cauchy_hetero), and the prior = the column's true empirical distribution (uniform over its N clean values,
which includes the cell's own value). It ignores dependence between columns and on y, so it is the Bayes-optimal
*per-cell* denoiser (MMSE estimate of each cell), not a Bayes-optimal classifier. Writes artifacts/bayes/<dataset>.npz.
"""
import json, math, os

import numpy as np, torch

from gen_data import (NOISE_MODELS, NOISE_SCALE, LOD_Q, art, FAIL_P_MAX, FAIL_GOOD_MULT, FAIL_SHIFT, CV_LN, LOD_THETA_Q,
                      LOD_SLOPE, LOD_FLOOR_Q, ULOQ_Q, PLATE_SD, PLATE_COL_SD)
from gen_ext import GROUPS

ROOT = os.path.dirname(os.path.abspath(__file__))
DEV = 'cuda'
NU = {'fail_floor': None, 'fail_shift': None, 'cv_lognormal': None, 'lod_logistic': None, 'saturation': None,
      'plate_effect': None, 'gauss': None, 'censored': None, 'gauss_valhet': None, 'censored_left': None, 'student_t3': 3.0,
      'censored_left_heavy': 2.0, 'cauchy': 1.0, 'cauchy_hetero': 1.0, 'cauchy_valhet': 1.0}
GH_T, GH_W = np.polynomial.hermite.hermgauss(24)  # row scale lambda = exp(sqrt(2) t), weights w / sqrt(pi)


def log_pdf(zs, nu):
    if nu is None:
        return -0.5 * zs ** 2 - 0.5 * math.log(2 * math.pi)
    return (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
            - (nu + 1) / 2 * torch.log1p(zs ** 2 / nu))


def log_cdf(zs, nu):  # closed forms for the censored families used here (Gaussian, t2)
    if nu is None:
        return torch.special.log_ndtr(zs)
    assert nu == 2.0
    return torch.log((0.5 + zs / (2 * torch.sqrt(2 + zs ** 2))).clamp(min=1e-38))


def special_loglik(r, m, s, sigma, noise, x_col):  # Exp 6 modes; sigma = base scale (0.5 sd), constant over s
    col_sd = float(sigma[0]) / NOISE_SCALE
    S, R_ = s[None, :, None], r[:, None, :]
    sig_f = float(sigma[0]) * (FAIL_GOOD_MULT if noise in ('fail_floor', 'fail_shift', 'lod_logistic', 'plate_effect') else 1.0)
    sig = torch.tensor(sig_f, device=r.device)
    lphi = lambda z, sc: -0.5 * z ** 2 - 0.5 * math.log(2 * math.pi) - torch.log(torch.as_tensor(sc, device=r.device))
    if noise in ('fail_floor', 'fail_shift'):
        G = 16
        out = []
        for g in range(G):
            p = FAIL_P_MAX * (g + 0.5) / G
            if noise == 'fail_floor':
                floor = float(np.float32(x_col.min() - col_sd))
                ll = torch.where(R_ == floor, math.log(p), math.log(1 - p) + lphi((R_ - S) / sig, sig))
            else:
                a = math.log(1 - p) + lphi((R_ - S) / sig, sig)
                b = math.log(p) + lphi((R_ - S - FAIL_SHIFT * col_sd) / sig, sig)
                ll = torch.logaddexp(a, b)
            out.append(torch.where(m[:, None, :], ll, 0.0).sum(-1))
        return torch.logsumexp(torch.stack(out), 0) - math.log(G)
    if noise == 'cv_lognormal':
        mm = float(x_col.min() - 0.25 * col_sd)
        lr = torch.log((R_ - mm).clamp(min=1e-12))
        z = (lr - torch.log((S - mm).clamp(min=1e-12)) + CV_LN ** 2 / 2) / CV_LN
        ll = -0.5 * z ** 2 - 0.5 * math.log(2 * math.pi) - math.log(CV_LN) - lr
    elif noise == 'lod_logistic':
        theta, floor = (float(np.float32(np.quantile(x_col, q))) for q in (LOD_THETA_Q, LOD_FLOOR_Q))
        sc = LOD_SLOPE * col_sd
        det = lphi((R_ - S) / sig, sig) + torch.nn.functional.logsigmoid((R_ - theta) / sc)
        a = (S - theta) / sc / math.sqrt(1 + math.pi * (sig_f / sc) ** 2 / 8)  # E[sigmoid] ~ logistic-normal approximation
        undet = torch.nn.functional.logsigmoid(-a).expand_as(det)
        ll = torch.where(R_ == floor, undet, det)
    elif noise == 'saturation':
        hi = float(np.float32(np.quantile(x_col, ULOQ_Q)))
        ll = torch.where(R_ == hi, torch.special.log_ndtr((S - hi) / sig), lphi((R_ - S) / sig, sig))
    else:  # plate_effect: APPROXIMATE oracle, per-cell Gaussian with the total variance (ignores run correlation)
        sig_t = col_sd * math.sqrt((NOISE_SCALE * FAIL_GOOD_MULT) ** 2 + PLATE_SD ** 2 + PLATE_COL_SD ** 2)
        ll = lphi((R_ - S) / sig_t, sig_t)
    return torch.where(m[:, None, :], ll, 0.0).sum(-1)


SPECIAL = ('fail_floor', 'fail_shift', 'cv_lognormal', 'lod_logistic', 'saturation', 'plate_effect')


def column_loglik(r, m, s, sigma, noise, x_col):
    """r, m: (R, K) replicates / mask for one column; s, sigma: (N,) support values / noise scale per support point.
    Returns (R, N) log-likelihood of each row's replicates for each support value."""
    if noise in SPECIAL:
        return special_loglik(r, m, s, sigma, noise, x_col)
    nu = NU[noise]
    z = (r[:, None, :] - s[None, :, None]) / sigma[None, :, None]  # (R, N, K)
    ll = log_pdf(z, nu) - torch.log(sigma)[None, :, None]
    if noise == 'censored':  # two-sided at the clean column's 10th / 90th percentile, reported at the limit
        lo, hi = (float(np.float32(np.quantile(x_col, q))) for q in (0.1, 0.9))
        ll = torch.where(r[:, None, :] == lo, log_cdf((lo - s[None, :, None]) / sigma[None, :, None], nu), ll)
        ll = torch.where(r[:, None, :] == hi, log_cdf((s[None, :, None] - hi) / sigma[None, :, None], nu), ll)
    if noise in LOD_Q:  # left-censored at the detection limit
        lod = float(np.float32(np.quantile(x_col, LOD_Q[noise])))
        ll = torch.where(r[:, None, :] == lod, log_cdf((lod - s[None, :, None]) / sigma[None, :, None], nu), ll)
    return torch.where(m[:, None, :], ll, 0.0).sum(-1)


@torch.no_grad()
def posterior_mean(X, reps, counts, noise, mult, rows, return_sd=False):
    """X: (N, d) clean; reps: (N, d, K); returns (len(rows), d) posterior means for the given row indices."""
    N, d = X.shape
    K = reps.shape[2]
    col_sd = X.std(axis=0)
    col_sd[col_sd < 1e-12] = 1.0
    r_all = torch.tensor(np.nan_to_num(reps[rows]), dtype=torch.float32, device=DEV)
    m_all = torch.tensor(np.arange(K)[None, None] < counts[rows][..., None], device=DEV)
    hetero = noise == 'cauchy_hetero'
    lams = np.exp(np.sqrt(2) * GH_T) if hetero else np.array([1.0])
    logw = np.log(GH_W / np.sqrt(np.pi)) if hetero else np.array([0.0])
    evid = torch.zeros(len(lams), len(rows), device=DEV)  # log p(row | lambda)
    cond_mean = torch.zeros(len(lams), len(rows), d, device=DEV)  # E[x_ij | row, lambda]
    cond_m2 = torch.zeros_like(cond_mean)  # E[x_ij^2 | row, lambda]
    for j in range(d):
        s = torch.tensor(X[:, j], dtype=torch.float32, device=DEV)
        base = mult * NOISE_SCALE * col_sd[j]
        if noise.endswith('_valhet'):
            u = (np.argsort(np.argsort(X[:, j], kind='stable'), kind='stable') + 1) / N
            sig0 = torch.tensor(base * 2 * u, dtype=torch.float32, device=DEV).clamp(min=1e-12)
        else:
            sig0 = torch.full((N,), base, dtype=torch.float32, device=DEV)
        for q, lam in enumerate(lams):
            L = column_loglik(r_all[:, j], m_all[:, j], s, sig0 * lam, noise, X[:, j])  # (R, N)
            evid[q] += torch.logsumexp(L, dim=1) - math.log(N)
            w = torch.softmax(L, dim=1)
            cond_mean[q, :, j] = w @ s
            cond_m2[q, :, j] = w @ (s * s)
    post_q = torch.softmax(evid + torch.tensor(logw, dtype=torch.float32, device=DEV)[:, None], dim=0)  # (Q, R)
    mean = (post_q[..., None] * cond_mean).sum(0)
    if not return_sd:
        return mean.cpu().numpy()
    var = ((post_q[..., None] * cond_m2).sum(0) - mean ** 2).clamp(min=0)
    return mean.cpu().numpy(), var.sqrt().cpu().numpy()


def main():
    data, ext = art('data'), art('data_ext')
    out = art('bayes')
    os.makedirs(out, exist_ok=True)
    ext_settings = json.load(open(os.path.join(ext, 'manifest.json')))['settings']
    for e in json.load(open(os.path.join(data, 'manifest.json')))['eval']:
        z, zx = (np.load(os.path.join(p, 'eval', e['name'] + '.npz')) for p in (data, ext))
        X = z['X_clean']
        allrows = np.arange(len(X))
        res = {}
        for nm in NOISE_MODELS:
            res[f'{nm}__bayes'] = posterior_mean(X, z[f'{nm}__replicates'], z[f'{nm}__counts'], nm, 1.0, allrows)
        for st in ext_settings:
            if st.startswith('grouped'):
                continue
            nm = st.rsplit('_n', 1)[0]
            res[f'{st}__bayes'] = posterior_mean(X, zx[f'{st}__replicates'], zx[f'{st}__counts'], nm, 1.0, allrows)
        g = zx['grouped__group']
        xb = np.zeros_like(X)
        for gi, (nm, mult) in enumerate(GROUPS):
            rows = np.where(g == gi)[0]
            xb[rows] = posterior_mean(X, zx['grouped__replicates'][:, :-1], zx['grouped__counts'][:, :-1], nm, mult, rows)
        res['grouped__bayes'] = np.concatenate([xb, g[:, None].astype(X.dtype)], 1)
        np.savez_compressed(os.path.join(out, e['name'] + '.npz'), **res)
        print(e['name'], flush=True)


def main_new(models):  # compute only the given (newly appended) noise models for the main data, merging into bayes/
    data, out = art('data'), art('bayes')
    for e in json.load(open(os.path.join(data, 'manifest.json')))['eval']:
        z = np.load(os.path.join(data, 'eval', e['name'] + '.npz'))
        pf = os.path.join(out, e['name'] + '.npz')
        res = dict(np.load(pf)) if os.path.exists(pf) else {}
        for nm in models:
            res[f'{nm}__bayes'] = posterior_mean(z['X_clean'], z[f'{nm}__replicates'], z[f'{nm}__counts'], nm, 1.0,
                                                 np.arange(len(z['X_clean'])))
        np.savez_compressed(pf, **res)
        print(e['name'], flush=True)


def main_ext2():  # Exp 4 settings (gen_ext2.py) -> artifacts/bayes_ext2/<dataset>.npz
    data, ext2 = art('data'), art('data_ext2')
    out = art('bayes_ext2')
    os.makedirs(out, exist_ok=True)
    m2 = json.load(open(os.path.join(ext2, 'manifest.json')))
    for e in json.load(open(os.path.join(data, 'manifest.json')))['eval']:
        X = np.load(os.path.join(data, 'eval', e['name'] + '.npz'))['X_clean']
        z = np.load(os.path.join(ext2, 'eval', e['name'] + '.npz'))
        res = {}
        for st in m2['settings']:
            nm = next(n for n in sorted(m2['NOISES'], key=len, reverse=True) if st.startswith(n + '_'))
            res[f'{st}__bayes'] = posterior_mean(X, z[f'{st}__replicates'], z[f'{st}__counts'], nm, 1.0, np.arange(len(X)))
        np.savez_compressed(os.path.join(out, e['name'] + '.npz'), **res)
        print(e['name'], flush=True)


if __name__ == '__main__':
    import sys
    if 'new' in sys.argv:
        main_new(sys.argv[sys.argv.index('new') + 1].split(','))
    else:
        main_ext2() if 'ext2' in sys.argv else main()
