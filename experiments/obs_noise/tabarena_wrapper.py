"""IID guard anchor: the off-the-shelf TabICLv2 sklearn wrapper (tabicl 2.2.0, defaults = TabArena's 'TabICLv2 (default)')
on the tabarena_iid.py tasks and splits. Runs in the isolated TabICL env (numpy + tabicl only):
  uv run --no-project --python 3.12 --with-requirements requirements-tabicl.txt \
         --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match python tabarena_wrapper.py
Records (model = tabicl_e<n_estimators>) -> artifacts/tabarena_iid.jsonl, same fields as tabarena_iid.py (RMSE of the
predicted mean; CRPS from the quantiles at levels k / 100)."""
import argparse, json, os

import numpy as np, torch

ROOT = os.path.dirname(os.path.abspath(__file__))
TAUS = np.arange(1, 100) / 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='airfoil_self_noise,wine_quality,houses,physiochemical_protein,superconductivity')
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--n_est', default='8,1')
    ap.add_argument('--mem_frac', type=float, default=0.3)
    ap.add_argument('--max_ctx', type=int, default=4096, help='as tabarena_iid.py (identical seeded subsample)')
    a = ap.parse_args()
    from tabicl import TabICLRegressor
    if a.mem_frac:
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    out = open(os.path.join(ROOT, 'artifacts', 'tabarena_iid.jsonl'), 'a')
    for ds in a.datasets.split(','):
        z = np.load(os.path.join(ROOT, 'artifacts', 'tabarena', f'{ds}.npz'))
        X, y = z['X'], z['y']
        for r in range(a.repeats):
            for f in sorted(set(z['split_fold'][z['split_repeat'] == r].tolist())):
                m = (z['split_repeat'] == r) & (z['split_fold'] == f)
                tr, te = z['split_row'][m & ~z['split_test']], z['split_row'][m & z['split_test']]
                if a.max_ctx and len(tr) > a.max_ctx:  # = tabarena_iid.splits
                    tr = np.sort(np.random.default_rng([sum(map(ord, ds)), r, f]).choice(tr, a.max_ctx, replace=False))
                for n in map(int, a.n_est.split(',')):
                    reg = TabICLRegressor(n_estimators=n, random_state=0, device='cuda').fit(X[tr], y[tr])
                    mean = np.asarray(reg.predict(X[te]))
                    q = np.asarray(reg.predict(X[te], output_type='quantiles', alphas=list(TAUS)))
                    d = y[te][:, None] - q
                    rec = dict(model=f'tabicl_e{n}', dataset=ds, repeat=r, fold=f, n_train=len(tr), n_test=len(te), max_ctx=a.max_ctx,
                               rmse=float(np.sqrt(((mean - y[te]) ** 2).mean())),
                               crps=float((2 * np.maximum(TAUS * d, (TAUS - 1) * d).mean(1)).mean()))
                    out.write(json.dumps(rec) + '\n'); out.flush()
                    print(ds, r, f, rec['model'], f"rmse {rec['rmse']:.4f} crps {rec['crps']:.4f}", flush=True)
                    del reg
                    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
