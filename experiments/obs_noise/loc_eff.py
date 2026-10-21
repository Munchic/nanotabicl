"""How well does each per-cell location estimate recover the clean value? mean, median, the learned attention
location of ObsNanoTabICL (center='attn'), and the Bayes-oracle posterior mean. Errors in units of the column sd:
median |err| (robust) and RMSE with each cell's error clipped at 3 sd (so single Cauchy blow-ups don't dominate).
Usage: python loc_eff.py tag=ckpt [tag=ckpt ...]"""
import json, os, sys

import numpy as np, torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, '..', '..'))
sys.path.insert(0, ROOT)
from obs_train import make_model  # noqa: E402
from obs_eval import settings_for, DATA, EXT, BAYES  # noqa: E402


@torch.no_grad()
def learned_location(model, reps, counts, order, n_train):
    mask = np.arange(reps.shape[-1])[None, None] < counts[..., None]
    r = torch.tensor(np.nan_to_num(reps[order]), dtype=torch.float32, device='cuda')[None]
    m = torch.tensor(mask[order], device='cuda')[None]
    x_c, _ = model.cell_inputs(r, m, n_train)
    out = np.empty(reps.shape[:2])
    out[order] = x_c[0].float().cpu().numpy()
    return out


def main():
    models = {}
    for spec in sys.argv[1:]:
        tag, path = spec.split('=', 1)
        ck = torch.load(path, map_location='cuda', weights_only=False)
        mdl = make_model(ck['mode'], ck['rep_cfg']).cuda().eval()
        mdl.load_state_dict(ck['state_dict'])
        models[tag] = mdl
    est_names = ['mean', 'median'] + list(models) + ['bayes']
    res = {}  # (setting, estimator) -> list over datasets of (median abs err, clipped rmse)
    for e in json.load(open(os.path.join(DATA, 'manifest.json')))['eval']:
        zs = {'main': np.load(os.path.join(DATA, 'eval', e['name'] + '.npz')),
              'ext': np.load(os.path.join(EXT, 'eval', e['name'] + '.npz'))}
        zb = np.load(os.path.join(BAYES, e['name'] + '.npz'))
        order = np.concatenate([zs['main']['train_idx'], zs['main']['test_idx']])
        X0 = zs['main']['X_clean']
        sd = X0.std(0)
        sd[sd < 1e-12] = 1
        for src in ['main', 'ext']:
            for s in settings_for(src):
                if s in ('clean', 'grouped_clean'):
                    continue
                z = zs[src]
                d = X0.shape[1]  # grouped settings have the exact group column last: evaluate the d real columns
                ests = {'mean': z[f'{s}__mean'][:, :d], 'median': z[f'{s}__median'][:, :d], 'bayes': zb[f'{s}__bayes'][:, :d]}
                for tag, mdl in models.items():
                    ests[tag] = learned_location(mdl, z[f'{s}__replicates'], z[f'{s}__counts'], order, len(zs['main']['train_idx']))[:, :d]
                for k, v in ests.items():
                    err = np.abs(v - X0) / sd
                    res.setdefault((s, k), []).append((np.median(err), np.sqrt(np.mean(np.minimum(err, 3) ** 2))))
        print(e['name'], flush=True)
    settings = sorted({s for s, _ in res}, key=lambda s: list(dict.fromkeys(s for s, _ in res)).index(s))
    for i, label in enumerate(['median |error| / sd', 'RMSE / sd (per-cell error clipped at 3 sd)']):
        print(f'\n**{label}** (mean over 27 datasets)\n')
        print('| setting | ' + ' | '.join(est_names) + ' |\n|' + '---|' * (len(est_names) + 1))
        for s in settings:
            print(f'| {s} | ' + ' | '.join(f"{np.mean([t[i] for t in res[(s, k)]]):.3f}" for k in est_names) + ' |')
    json.dump({f'{s}|{k}': v for (s, k), v in res.items()}, open(os.path.join(ROOT, 'artifacts', 'loc_eff.json'), 'w'))


if __name__ == '__main__':
    main()
