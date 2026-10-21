"""Training-free headroom of the shape prior (SHAPE.md section 3): per validation setting (obs_train.PS_SETTINGS),
off-the-shelf TabICLv2 (fp32 port, tabicl preprocessing) on the cell mean / mean + sd / sketch (mean, median, sd, q25,
q75, min, max) / the clean x (oracle). 4 held-out bank blocks, 1024 rows, 512 context. -> accuracy and log-loss, and
the share of the mean -> oracle gap that mean + sd and the sketch close. Usage: python shape_headroom.py [--blocks 4]"""
import argparse, json, os, sys

import numpy as np, torch, torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.argv, argv = sys.argv[:1], sys.argv[1:]
import obs_train as T  # noqa: E402
from model import NanoTabICLv2  # noqa: E402
from tabicl_port import load_tabicl  # noqa: E402


def sketch(reps, mask, keys):
    n = mask.sum(-1).clamp(min=1).float()
    r = torch.where(mask, reps, 0.0)
    mean = r.sum(-1) / n
    sd = torch.sqrt((torch.where(mask, reps - mean[..., None], 0.0) ** 2).sum(-1) / (n - 1).clamp(min=1))
    q = lambda p: T.masked_quantile(reps, mask, p)
    f = dict(mean=mean, sd=sd, median=q(0.5), q25=q(0.25), q75=q(0.75), min=q(0.0), max=q(1.0))
    return torch.cat([f[k] for k in keys], dim=-1)


ARMS = {'mean': ['mean'], 'mean_sd': ['mean', 'sd'], 'sketch': ['mean', 'median', 'sd', 'q25', 'q75', 'min', 'max']}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--blocks', type=int, default=4)
    ap.add_argument('--settings', default=','.join(T.PS_SETTINGS))
    ap.add_argument('--out', default=None, help='default artifacts/shape_headroom[_reg].json')
    ap.add_argument('--task', choices=['classification', 'regression'], default='classification')
    a = ap.parse_args(argv)
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    reg = a.task == 'regression'
    out_path = a.out or os.path.join(ROOT, 'artifacts', 'shape_headroom_reg.json' if reg else 'shape_headroom.json')
    cfg, sd = load_tabicl(a.task)
    m = NanoTabICLv2(**cfg); m.load_state_dict(sd); m = m.to(dev).eval()
    bank = T.Bank(os.path.join(ROOT, 'artifacts', 'bank_reg' if reg else 'bank'), None)
    nq = cfg['out_dim']
    taus = (torch.arange(1, nq + 1, device=dev) / (nq + 1)).float()
    res = {}
    for st in a.settings.split(','):
        g = torch.Generator(device=dev).manual_seed(12345)
        acc = {k: [] for k in list(ARMS) + ['oracle']}
        ll = {k: [] for k in acc}
        for f in bank.files()[:a.blocks]:
            X_all, y_all = bank.load(f, dev)
            for i in range(0, X_all.shape[0], 4):
                X, y = X_all[i:i + 4, :1024], y_all[i:i + 4, :1024]
                n_tr = 512
                if reg:  # standardized with the context targets, soft-clipped as in training (y_clip 30)
                    y = T.standardize_y(y.float(), n_tr)
                    y = 30 * torch.tanh(y / 30)
                reps, mask, _, _ = T.observe_shape(X, g, n_tr, **T.PS_SETTINGS[st])
                y_te = y[:, n_tr:]
                valid = (y_te[..., None] == y[:, None, :n_tr]).any(-1) if not reg else torch.ones_like(y_te, dtype=torch.bool)
                for arm in acc:
                    inp = X if arm == 'oracle' else sketch(reps, mask, ARMS[arm])
                    with torch.no_grad():
                        lg = m(inp, y[:, :n_tr]).float()
                    if reg:  # 'loss' = CRPS (2 x mean pinball over the head's quantile levels), 'acc' = R^2 of the median
                        q = lg.sort(-1).values
                        diff = y_te[..., None] - q
                        ll[arm].append(2 * torch.maximum(taus * diff, (taus - 1) * diff).mean().item())
                        med = q[..., nq // 2]
                        acc[arm].append((1 - ((med - y_te) ** 2).mean(1) / y_te.var(1, unbiased=False).clamp(min=1e-8)).mean().item())
                        continue
                    acc[arm].append((lg.argmax(-1) == y_te)[valid].float().mean().item())
                    ll[arm].append(F.cross_entropy(lg[valid], y_te[valid]).item())
                torch.cuda.empty_cache()
        r = {k: dict(acc=float(np.mean(acc[k])), loss=float(np.mean(ll[k]))) for k in acc}
        gap = r['oracle']['loss'] - r['mean']['loss']
        for k in ('mean_sd', 'sketch'):
            r[k]['gap_closed'] = float((r[k]['loss'] - r['mean']['loss']) / gap) if abs(gap) > 1e-6 else float('nan')
        res[st] = r
        print(f"{st:11s} " + ' | '.join(f"{k} {v['acc']:.3f}/{v['loss']:.3f}" + (f" ({100 * v['gap_closed']:.0f}%)" if 'gap_closed' in v else '')
                                        for k, v in r.items()), flush=True)
    json.dump(res, open(out_path, 'w'), indent=1)
