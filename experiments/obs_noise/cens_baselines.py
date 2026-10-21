"""CENSORING.md baselines (inference only) on the cens_train.py validation settings, with the identical censoring
draws (same seeds and RNG order as cens_train.validate), so every number pairs with a training run's validation log.
TabICLv2 regression (released weights) without spec channels:
  bound   (B1): censored context y at the bound (what the model sees with no special handling; = step 0 of R1 / R0)
  drop    (B2): censored context rows removed
  impute  (B3): one EM-style round: predict the censored context rows from the uncensored ones, replace each by the
                truncated mean of its predictive quantiles beyond the bound (the bound if none is), refit on all rows
  ind     (F2): features at the bound + one indicator column (-1 '<', 0, +1 '>') per column with a censored context cell
  latent  (B4): the latent context (the 'clean' validation setting is the same tables uncensored)
CRPS on the latent y in units of the table's latent-y sd, as in cens_train.validate.
Usage: python cens_baselines.py [--val_blocks 4] [--out artifacts/cens_baselines.json]"""
import argparse, json, os, sys

import numpy as np, torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import cens_train as ct  # noqa: E402  (also puts the repo root on sys.path)
import obs_train as ot  # noqa: E402
from cens_model import CensNanoTabICL  # noqa: E402
from tabicl_port import load_tabicl  # noqa: E402


def trunc_mean(q, b, side):  # q (n, NQ) predictive quantiles, b (n,) bound, side (n,) 1 '<' / 2 '>' -> (n,)
    beyond = torch.where(side[:, None] == 2, q > b[:, None], q < b[:, None])
    m = (q * beyond).sum(1) / beyond.sum(1).clamp(min=1)
    return torch.where(beyond.any(1), m, b)


@torch.no_grad()
def predict(model, X, yc, n_tr):  # one table: X (R, C), context y (n_tr,) -> quantiles for rows n_tr.. (R - n_tr, NQ)
    return model(X[None], yc[None])[0]


@torch.no_grad()
def run(model, bank, dev, n_blocks, arms):
    out = {}
    for name, (yd, xd) in ct.VAL_SETTINGS.items():
        if name == 'clean':
            continue
        g = torch.Generator(device=dev).manual_seed(12345)
        acc = {a: [] for a in arms}
        for f in bank.files()[:n_blocks]:
            X_all, y_all = bank.load(f, dev)
            for i in range(0, X_all.shape[0], 8):  # same chunking as validate (the RNG draws depend on it)
                X, y = X_all[i:i + 8, :1024], y_all[i:i + 8, :1024]
                B, R, C = X.shape
                n = 512
                y_des = None if yd is None else ct.fixed_spec(B, 1, dev, *yd)
                x_des = None if xd is None else ct.fixed_spec(B, C, dev, *xd[:3], col_share=xd[3], g=g)
                Xo, xf, yc, yf, yl, (mu, sd), _ = ct.prepare(X, y, n, g, y_des, x_des, True, True)
                unit = sd[:, 0] / (y.std(1) + 1e-8)
                for b in range(B):
                    crps = lambda q, rows=slice(n, None): (2 * ot.pinball(q[None], yl[b:b + 1, rows])[0]).mean().item() * unit[b].item()
                    if 'bound' in arms:
                        acc['bound'].append(crps(predict(model, Xo[b], yc[b], n)))
                    if yd is not None:
                        cz = yf[b] > 0
                        if 'drop' in arms:  # context = uncensored rows only (rows reordered: context first)
                            keep = torch.cat([(~cz).nonzero()[:, 0], torch.arange(n, R, device=dev)])
                            acc['drop'].append(crps(predict(model, Xo[b, keep], yc[b, ~cz], int((~cz).sum()))))
                        if 'impute' in arms:
                            unc, cen = (~cz).nonzero()[:, 0], cz.nonzero()[:, 0]
                            q_c = predict(model, Xo[b, torch.cat([unc, cen])], yc[b, unc], len(unc))
                            y_imp = yc[b].clone()
                            y_imp[cen] = trunc_mean(q_c, yc[b, cen], yf[b, cen])
                            acc['impute'].append(crps(predict(model, Xo[b], y_imp, n)))
                    if xd is not None and 'ind' in arms:
                        cols = (xf[b, :n] != 0).any(0)
                        ind = torch.sign(xf[b][:, cols].float() * 2 - 3) * (xf[b][:, cols] != 0)  # 1 -> -1, 2 -> +1
                        acc['ind'].append(crps(predict(model, torch.cat([Xo[b], ind], 1), yc[b], n)))
        out[name] = {a: float(np.mean(v)) for a, v in acc.items() if v}
        print(name, ' '.join(f'{a} {v:.4f}' for a, v in out[name].items()), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val_blocks', type=int, default=4)
    ap.add_argument('--arms', default='bound,drop,impute,ind')
    ap.add_argument('--out', default=os.path.join(ROOT, 'artifacts', 'cens_baselines.json'))
    ap.add_argument('--bank', default=os.path.join(ROOT, 'artifacts', 'bank_reg'))
    a = ap.parse_args()
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg, sd = load_tabicl('regression')
    ot.set_quantiles(cfg['out_dim'])
    model = CensNanoTabICL(**cfg).to(dev).eval()
    model.load_state_dict(sd, strict=False)
    res = run(model, ot.Bank(a.bank, 2000), dev, a.val_blocks, a.arms.split(','))
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
