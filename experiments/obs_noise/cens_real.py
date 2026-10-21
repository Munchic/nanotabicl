"""CENSORING.md, real data: Tox21 as censored regression (held-out test; nothing is tuned on it).

Rows = substances screened in all 12 assays (as real_data.build_tox21). Per (substance, assay), over its screening runs:
  observed  : the majority of runs have a fitted AC50 -> median log10 AC50 [uM] of those runs, flag 0
  censored  : no run has one -> median log10 top tested concentration (per sample, ~90-120 uM), flag 2 ('>')
  ambiguous : otherwise -> median log10 AC50 over all runs (values at the top included), flag 0
Features: the other 11 assays' log AC50 (flagged as above) and median efficacy (0 for censored runs).
Target: one assay's log AC50; ambiguous target rows are dropped. 12 tasks (one per target assay), two designs:
  natural c : context and query sampled with a censored share c of their rows (inactives are subsampled: raw Tox21 is
              84-98% censored, outside the prior); CRPS on the observed query rows, -log P(y > bound) on censored ones
  recens p  : observed (active) rows only. Context targets above the p-quantile t of the context's log AC50 are shown
              censored at t; every query is scored at its known value (ground truth for censoring on real data)
Arms: our checkpoints (spec on = target and feature flags; spec off = none) and the TabICLv2 baselines bound / drop /
impute (cens_baselines.py), all one forward pass, our pipeline (fp32 + TF32, wrapper preprocessing).
Usage: python cens_real.py [name=ckpt ...] [--designs natural30,natural50,recens70,recens50] [--out ...]"""
import argparse, json, os, sys

import numpy as np, pandas as pd, torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import cens_train as ct  # noqa: E402  (also puts the repo root on sys.path)
import obs_train as ot  # noqa: E402
from cens_baselines import trunc_mean  # noqa: E402
from cens_model import CensNanoTabICL  # noqa: E402
from real_data import TOX, TOX_ASSAYS  # noqa: E402
from tabicl_port import load_tabicl  # noqa: E402

CACHE = os.path.join(ROOT, 'artifacts', 'data_cens', 'tox21_reg.npz')


def substance_table():
    """-> ids, V (n, 12) log AC50, F (n, 12) flags {0, 2}, E (n, 12) efficacy, A (n, 12) bool ambiguous; cached."""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z['ids'], z['V'], z['F'], z['E'], z['A']
    per = []
    for a in TOX_ASSAYS:
        d = pd.read_csv(os.path.join(TOX, f'tox21-{a}', f'tox21-{a}.txt'), sep='\t', index_col=False, low_memory=False)
        d = d[d['SAMPLE_DATA_TYPE'].astype(str).str.match(r'^(set\d|ratio(_\d)?|antagonist\d|agonist)$')]  # as real_data
        conc = d[[c for c in d.columns if c.startswith('CONC')]].to_numpy(np.float64)
        ac50 = d['AC50'].to_numpy(np.float64)
        fit = np.isfinite(ac50)
        r = pd.DataFrame({'id': d['TOX21_ID'].to_numpy(), 'fit': fit,
                          'logac': np.log10(np.where(fit, ac50, np.nanmax(conc, 1) * 1e6)),
                          'eff': np.where(fit, np.nan_to_num(d['EFFICACY'].to_numpy(np.float64)), 0.0)})
        r['logac_fit'] = r['logac'].where(r['fit'])
        r['eff_fit'] = r['eff'].where(r['fit'])
        g = r.groupby('id').agg(n=('fit', 'size'), n_fit=('fit', 'sum'), med=('logac', 'median'),
                                med_fit=('logac_fit', 'median'), eff=('eff', 'median'), eff_fit=('eff_fit', 'median'))
        obs, cen = 2 * g['n_fit'] > g['n'], g['n_fit'] == 0
        per.append(pd.DataFrame({'v': np.where(obs, g['med_fit'], g['med']), 'f': np.where(cen, 2, 0),
                                 'e': np.where(obs, g['eff_fit'], np.where(cen, 0.0, g['eff'])), 'a': ~obs & ~cen},
                                index=g.index))
        print('tox21', a, len(g), 'substances', f"observed {obs.mean():.3f} censored {cen.mean():.3f}", flush=True)
    ids = sorted(set.intersection(*[set(p.index) for p in per]))
    V, F, E, A = (np.stack([p.loc[ids, c].to_numpy() for p in per], 1) for c in 'vfea')
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, ids=np.array(ids), V=V.astype(np.float32), F=F.astype(np.int64), E=E.astype(np.float32),
                        A=A.astype(bool))
    return np.array(ids), V.astype(np.float32), F.astype(np.int64), E.astype(np.float32), A.astype(bool)


def task(t, V, F, E, A, design, max_rows=1024, seed=0):
    """-> X (R, 22), x_flag (R, 22), y (R,) observed target (bound for censored), y_flag (R,), n_train; or None."""
    feats = [a for a in range(12) if a != t]
    X = np.concatenate([V[:, feats], E[:, feats]], 1)
    xf = np.concatenate([F[:, feats], np.zeros_like(F[:, feats])], 1)
    y, yf = V[:, t].copy(), F[:, t].copy()
    rng = np.random.default_rng([t, seed, 29])
    obs, cen = np.where(~A[:, t] & (yf == 0))[0], np.where(yf == 2)[0]
    if design.startswith('natural'):
        c = int(design[len('natural'):]) / 100
        n = min(max_rows, int(len(obs) / (1 - c)), int(len(cen) / c))
        n_c = int(round(c * n))
        rows = rng.permutation(np.concatenate([rng.choice(obs, n - n_c, replace=False), rng.choice(cen, n_c, replace=False)]))
        return X[rows], xf[rows], y[rows], yf[rows], n // 2
    p = int(design[len('recens'):]) / 100
    rows = rng.permutation(obs)[:max_rows]
    n_tr = len(rows) // 2
    y, yf = y[rows], yf[rows]
    thr = np.quantile(y[:n_tr], p)
    lat = y.copy()
    hide = np.arange(len(rows)) < n_tr
    hide &= y > thr
    y[hide], yf[hide] = thr, 2
    return X[rows], xf[rows], y, yf, n_tr, lat


def scores(q, y_lat, y_flag_te, cap=None):
    """q (T, NQ) quantiles in raw units, y_lat (T,) latent (or the bound for censored queries), y_flag_te (T,) ->
    CRPS on observed queries; mean -log P(y > bound) on censored ones (from the quantile levels above the bound)."""
    taus = ot.TAUS.to(q.device)
    crps = 2 * torch.maximum(taus * (y_lat[:, None] - q), (taus - 1) * (y_lat[:, None] - q)).mean(-1)
    o, c = y_flag_te == 0, y_flag_te == 2
    out = dict(crps=crps[o].mean().item())
    if cap is not None:  # proper score for censored data, over ALL query rows: CRPS of min(y, c) vs the prediction censored
        # at c (a censored query's bound is ~c; Tox21 bounds are 1.9-2.0 log uM); c = the task's median censoring bound
        qc, yc_ = q.clamp(max=cap), y_lat.clamp(max=cap)
        out['crps_cens_all'] = (2 * torch.maximum(taus * (yc_[:, None] - qc), (taus - 1) * (yc_[:, None] - qc)).mean(-1)).mean().item()
    if c.any():
        surv = (q[c] > y_lat[c, None]).float().mean(-1).clamp(1 / (len(taus) + 1), 1)
        out['nll_cens'] = (-surv.log()).mean().item()
    return out


@torch.no_grad()
def run_arm(model, arm, X, xf, y, yf, n_tr, known):
    """-> (T, NQ) raw-unit quantiles for the query rows. arm: spec_on / spec_off / bound / drop / impute."""
    mu, sd = y[:n_tr].mean(), y[:n_tr].std() + 1e-8  # observed context values, as cens_train.prepare
    yc = (y[:n_tr] - mu) / sd
    cz = yf[:n_tr] > 0
    if arm in ('spec_on', 'spec_off'):
        on = arm == 'spec_on'
        f = lambda fl: (torch.where(fl == 0, 3, fl) if known else fl) if on else torch.zeros_like(fl)
        q = model(X[None], yc[None], f(xf)[None], f(yf[:n_tr])[None])[0]
    elif arm == 'bound':
        q = model(X[None], yc[None])[0]
    elif arm == 'drop':
        keep = torch.cat([(~cz).nonzero()[:, 0], torch.arange(n_tr, len(y), device=y.device)])
        q = model(X[keep][None], yc[~cz][None])[0]
    elif arm == 'impute':
        unc, cen = (~cz).nonzero()[:, 0], cz.nonzero()[:, 0]
        q_c = model(X[torch.cat([unc, cen])][None], yc[unc][None])[0]
        y_imp = yc.clone()
        y_imp[cen] = trunc_mean(q_c, yc[cen], yf[cen])
        q = model(X[None], y_imp[None])[0]
    return q * sd + mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='*', help='name=path to cens_train checkpoints')
    ap.add_argument('--designs', default='natural30,natural50,recens70,recens50')
    ap.add_argument('--baselines', default='bound,drop,impute')
    ap.add_argument('--out', default=os.path.join(ROOT, 'artifacts', 'cens_real.jsonl'))
    a = ap.parse_args()
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ids, V, F, E, A = substance_table()
    cfg, sd0 = load_tabicl('regression')
    ot.set_quantiles(cfg['out_dim'])
    models = []
    if a.baselines:
        m = CensNanoTabICL(**cfg).to(dev).eval()
        m.load_state_dict(sd0, strict=False)
        models += [(arm, m, arm, False) for arm in a.baselines.split(',')]
    for spec in a.ckpts:
        name, path = spec.split('=', 1)
        ck = torch.load(path, map_location=dev, weights_only=False)
        known = ck.get('cens_cfg', {}).get('known', False)
        m = CensNanoTabICL(**ck['model_cfg'], **ck.get('cens_cfg', {})).to(dev).eval()
        m.load_state_dict(ck['state_dict'])
        models += [(f'{name}_on', m, 'spec_on', known), (f'{name}_off', m, 'spec_off', known)]
    out = open(a.out, 'a')
    for design in a.designs.split(','):
        for t in range(12):
            tk = task(t, V, F, E, A, design)
            X, xf, y, yf, n_tr = (torch.as_tensor(v, device=dev) if isinstance(v, np.ndarray) else v for v in tk[:5])
            y_true = torch.as_tensor(tk[5], device=dev) if len(tk) > 5 else y  # recens: known values; natural: observed
            cap = float(np.median(V[F[:, t] == 2, t])) if design.startswith('natural') else None
            for tag, m, arm, known in models:
                q = run_arm(m, arm, X.float(), xf, y.float(), yf, n_tr, known)
                rec = dict(model=tag, design=design, task=t, target=TOX_ASSAYS[t], n_train=n_tr, n_test=len(y) - n_tr,
                           cens_ctx=float((yf[:n_tr] > 0).float().mean()), **scores(q, y_true[n_tr:].float(), yf[n_tr:], cap))
                out.write(json.dumps(rec) + '\n'); out.flush()
            print(design, t, TOX_ASSAYS[t], f'n {len(y)}', flush=True)
    out.close()


if __name__ == '__main__':
    main()
