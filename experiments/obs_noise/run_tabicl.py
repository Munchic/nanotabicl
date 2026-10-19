"""Steps 2/3/6/7: zero-shot and per-dataset fine-tuned TabICLv2 on every (dataset, arm).

All predictions go through one path: TabICLClassifier(model_path=<ckpt>, random_state=0).fit(context).predict_proba(query)
with context = X_arm[train_idx], query = X_arm[test_idx] (indices stored in each dataset's npz). Fine-tuned weights are
saved as slim checkpoints ({config, state_dict}, loadable via `model_path`) so any prediction can be re-inferred.
"""
import argparse, json, os, shutil, sys, tempfile, time, warnings

import numpy as np, torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score

warnings.filterwarnings('ignore')
from tabicl import TabICLClassifier, FinetunedTabICLClassifier  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from gen_data import ARMS, make_arm  # noqa: E402

BASE_CKPT = 'tabicl-classifier-v2-20260212.ckpt'
SEED = 0


def base_ckpt_path() -> str:
    return hf_hub_download('jingang/TabICL', BASE_CKPT)


def predict(ckpt: str, Xtr, ytr, Xte) -> np.ndarray:
    return TabICLClassifier(model_path=ckpt, random_state=SEED, device='cuda').fit(Xtr, ytr).predict_proba(Xte)


def metrics(y, p) -> dict:
    k = p.shape[1]
    return dict(acc=accuracy_score(y, p.argmax(1)), bal_acc=balanced_accuracy_score(y, p.argmax(1)),
                log_loss=log_loss(y, np.clip(p, 1e-7, 1), labels=np.arange(k)),
                roc_auc=roc_auc_score(y, p[:, 1]) if k == 2 else roc_auc_score(y, p, multi_class='ovr'))


def finetune(Xtr, ytr, hp: dict, out_ckpt: str | None, base_sd: dict) -> tuple[str, dict]:
    """Fine-tune on the context only (internal stratified val split for early stopping); returns ckpt path + info."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tmp = tempfile.mkdtemp()
    t = time.time()
    ft = FinetunedTabICLClassifier(epochs=hp['epochs'], learning_rate=hp['lr'], eval_metric='log_loss',
                                   validation_split_ratio=0.2, save_interval=10**9, random_state=SEED,
                                   device='cuda').fit(Xtr, ytr, output_dir=tmp)
    ft_time = time.time() - t
    ck = torch.load(os.path.join(tmp, 'best.ckpt'), map_location='cpu', weights_only=False)
    sd = ck['state_dict']
    delta = float(torch.sqrt(sum(((sd[k].float() - base_sd[k].float()) ** 2).sum() for k in sd)))
    info = dict(ft_time_s=ft_time, ft_best_val_neg_logloss=float(ck['best_metric']), weight_delta_l2=delta,
                weights_changed=delta > 0)
    path = os.path.join(tmp, 'slim.ckpt')
    torch.save({'config': ck['config'], 'state_dict': sd}, path)
    if out_ckpt is not None and delta > 0:  # unchanged weights = base checkpoint, no need to store a copy
        os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
        shutil.copy(path, out_ckpt)
    return path, info


def run(datasets: list[dict], arms: list[tuple], hp: dict, out_dir: str, store_ckpts: bool, tag: str,
        modes: tuple = ('zeroshot', 'finetuned')):
    """datasets: dicts with name, X (clean), y, train_idx, test_idx, seed_key, arrays (optional precomputed arms)."""
    base = base_ckpt_path()
    base_sd = torch.load(base, map_location='cpu', weights_only=False)['state_dict']
    os.makedirs(os.path.join(out_dir, 'preds'), exist_ok=True)
    res_path = os.path.join(out_dir, f'results_{tag}.jsonl')
    done = set()
    if os.path.exists(res_path):  # resume
        done = {(r['dataset'], r['arm'], r['mode']) for r in map(json.loads, open(res_path))}
    for ds in datasets:
        for arm, noise, feats in arms:
            key = arm.replace('/', '__')
            if 'arrays' in ds and noise is not None:
                X = ds['arrays'][f'{noise}__{feats}']
            else:
                X = make_arm(ds['X'], noise, feats, ds['seed_key'])[0]
            tr, te, y = ds['train_idx'], ds['test_idx'], ds['y']
            for mode in modes:
                if (ds['name'], arm, mode) in done:
                    continue
                t = time.time()
                info = {}
                if mode == 'zeroshot':
                    ckpt, ckpt_ref = base, BASE_CKPT
                else:
                    out_ckpt = os.path.join(out_dir, 'ckpts', ds['name'], key + '.ckpt') if store_ckpts else None
                    ckpt, info = finetune(X[tr], y[tr], hp, out_ckpt, base_sd)
                    ckpt_ref = os.path.relpath(out_ckpt, out_dir) if (out_ckpt and info['weights_changed']) else BASE_CKPT
                p = predict(ckpt, X[tr], y[tr], X[te])
                if mode == 'finetuned':
                    shutil.rmtree(os.path.dirname(ckpt))
                np.savez_compressed(os.path.join(out_dir, 'preds', f"{tag}__{ds['name']}__{key}__{mode}.npz"),
                                    test_idx=te, y_test=y[te], proba=p.astype(np.float32))
                rec = dict(dataset=ds['name'], arm=arm, noise=noise or 'clean', feats=feats or 'clean', mode=mode,
                           n_cols=X.shape[1], ckpt=ckpt_ref, hp=hp if mode == 'finetuned' else None,
                           seconds=time.time() - t, **info, **metrics(y[te], p))
                with open(res_path, 'a') as f:
                    f.write(json.dumps(rec) + '\n')
                print(f"[{tag}] {ds['name']:24s} {arm:22s} {mode:9s} acc={rec['acc']:.3f} ll={rec['log_loss']:.3f} "
                      f"auc={rec['roc_auc']:.3f} {'Δw=%.3g' % info['weight_delta_l2'] if info else ''} "
                      f"({rec['seconds']:.0f}s)", flush=True)


def load_eval(data_dir: str) -> list[dict]:
    m = json.load(open(os.path.join(data_dir, 'manifest.json')))
    out = []
    for e in m['eval']:
        z = np.load(os.path.join(data_dir, 'eval', e['name'] + '.npz'))
        out.append(dict(name=e['name'], X=z['X_clean'], y=z['y'], train_idx=z['train_idx'], test_idx=z['test_idx'],
                        seed_key=[e['seed']], arrays={k: z[k] for k in z.files if '__' in k}))
    return out


def load_pilot(data_dir: str, n: int) -> list[dict]:
    z = np.load(os.path.join(data_dir, 'pilot_pool.npz'))
    return [dict(name=f"pilot_{i:02d}_s{z['seed'][i]}", X=z['X'][i, :, :z['n_features'][i]].astype(np.float64),
                 y=z['y'][i], train_idx=z['train_idx'][i], test_idx=z['test_idx'][i], seed_key=[int(z['seed'][i])])
            for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['pilot', 'eval'], required=True)
    ap.add_argument('--data', default=os.path.join(ROOT, 'artifacts', 'data'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'artifacts', 'runs'))
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--n_pilot', type=int, default=12)
    ap.add_argument('--pilot_arms', default='clean,gauss/full,cauchy_hetero/full')
    ap.add_argument('--modes', default='zeroshot,finetuned')
    args = ap.parse_args()
    hp = dict(lr=args.lr, epochs=args.epochs)
    if args.stage == 'pilot':
        arms = [a for a in ARMS if a[0] in args.pilot_arms.split(',')]
        run(load_pilot(args.data, args.n_pilot), arms, hp, os.path.join(args.out, 'pilot'), store_ckpts=False,
            tag=f"pilot_lr{args.lr:g}_ep{args.epochs}", modes=tuple(args.modes.split(',')))
    else:
        run(load_eval(args.data), ARMS, hp, os.path.join(args.out, 'eval'), store_ckpts=True, tag='eval',
            modes=tuple(args.modes.split(',')))


if __name__ == '__main__':
    main()
