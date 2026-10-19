"""Zero-shot TabICLv2 on main + ext settings with arms mean / median / full / bayes (oracle-denoised); saves holdout
probabilities to artifacts/probs/tabicl_<arm>/<dataset>.npz and metrics to artifacts/tabicl_eval.jsonl (resumable)."""
import json, os, sys, warnings

import numpy as np

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from run_tabicl import predict, metrics, base_ckpt_path  # noqa: E402
from obs_eval import settings_for, DATA, SRC_DIR, BayesLookup, save_probs, PROBS, TAG  # noqa: E402
from reg_metrics import reg_metrics, TAUS  # noqa: E402

ARMS = ['mean', 'median', 'full', 'bayes']


def main():
    ckpt = base_ckpt_path()
    task = json.load(open(os.path.join(DATA, 'manifest.json'))).get('task', 'classification')
    if task == 'regression':
        from tabicl import TabICLRegressor  # regression checkpoint is downloaded on first use
    out = os.path.join(ROOT, 'artifacts', f'tabicl_eval{TAG}.jsonl')
    manifest = json.load(open(os.path.join(DATA, 'manifest.json')))
    arms = sys.argv[1].split(',') if len(sys.argv) > 1 else ARMS  # optional: subset of arms (parallel runs)
    sources = sys.argv[2].split(',') if len(sys.argv) > 2 else ['main', 'ext']
    for arm in arms:
        probs_dir = os.path.join(PROBS, f'tabicl_{arm}')
        os.makedirs(probs_dir, exist_ok=True)
        for e in manifest['eval']:
            pf = os.path.join(probs_dir, e['name'] + '.npz')
            marker = f'p__{settings_for(sources[-1])[-1]}'  # last setting of the last source = this (arm, dataset) done
            if os.path.exists(pf) and marker in np.load(pf).files:
                continue
            zs = {src: np.load(os.path.join(SRC_DIR[src], 'eval', e['name'] + '.npz')) for src in set(sources) | {'main'}}
            zb = BayesLookup(e['name'])
            tr, te, y = zs['main']['train_idx'], zs['main']['test_idx'], zs['main']['y']
            if task == 'regression':  # metrics in context-sd units, same as obs_eval
                y = (y - y[tr].mean()) / (y[tr].std() + 1e-8)
            saved, recs = dict(y_test=y[te]), []
            for src in sources:
                for s in settings_for(src):
                    z = zs[src]
                    X = z[f'{s}__X_clean'] if f'{s}__X_clean' in z.files else zs['main']['X_clean']
                    if s in ('clean', 'grouped_clean'):
                        Xa = X
                    elif arm == 'bayes':
                        Xa = zb[f'{s}__bayes']
                    else:
                        Xa = z[f'{s}__{arm}']
                    if task == 'regression':
                        reg = TabICLRegressor(random_state=0, device='cuda').fit(Xa[tr], y[tr])
                        q = np.asarray(reg.predict(Xa[te], output_type='quantiles', alphas=list(TAUS)))
                        m, u = reg_metrics(q, y[te])
                        saved[f'p__{s}'] = np.sort(q, 1).astype(np.float16)
                        saved[f'u__{s}'] = u.astype(np.float32)
                        recs.append(dict(model=f'tabicl_{arm}', task=task, source=src, dataset=e['name'], noise=s, **m))
                        continue
                    p = predict(ckpt, Xa[tr], y[tr], Xa[te])
                    saved[f'p__{s}'] = p.astype(np.float32)
                    recs.append(dict(model=f'tabicl_{arm}', source=src, dataset=e['name'], noise=s, **metrics(y[te], p)))
            with open(out, 'a') as f:  # metrics first; a rerun after a crash appends duplicates (analysis keeps last)
                for r in recs:
                    f.write(json.dumps(r) + '\n')
            save_probs(pf, saved)  # probs written last -> marks (arm, dataset) as done
            print(arm, e['name'], np.mean([r.get('acc', r.get('crps', np.nan)) for r in recs]).round(3), flush=True)


if __name__ == '__main__':
    main()
