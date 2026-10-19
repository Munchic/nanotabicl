"""Evaluate obs_train.py checkpoints on the 27 eval datasets (same splits / stored replicates as the TabICL runs).

Settings: 'main' = clean + the 9 noise models (artifacts/data); 'ext' = n20/n50 + grouped (artifacts/data_ext).
Input per checkpoint mode: clean/mean -> cell means; median -> cell medians; stats -> the stored `<s>__full` columns;
obs -> replicates + counts. With --input bayes, clean/mean checkpoints get the Bayes-oracle posterior means instead.
One forward pass per dataset, no ensembling. Holdout probabilities are saved to artifacts/probs/<tag>/<dataset>.npz.
"""
import argparse, json, os, sys

import numpy as np, torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, '..', '..'))
sys.path.insert(0, ROOT)
from obs_train import make_model  # noqa: E402
from reg_metrics import reg_metrics  # noqa: E402
from gen_data import NOISE_MODELS as NOISES, art, TAG  # noqa: E402

DATA, EXT, EXT2, BAYES, BAYES2, PROBS = (art(p) for p in ('data', 'data_ext', 'data_ext2', 'bayes', 'bayes_ext2', 'probs'))
SRC_DIR = {'main': DATA, 'ext': EXT, 'ext2': EXT2}


# Exp 6 core suite: redundant settings dropped (t3, two-sided censoring, value-dependent gauss/cauchy and plain left
# censoring behaved like gauss / cauchy in Exps 1-5); SUITE=full evaluates everything
CORE_NOISE = ['gauss', 'cauchy', 'cauchy_hetero', 'censored_left_heavy', 'fail_floor', 'fail_shift', 'cv_lognormal',
              'lod_logistic', 'saturation', 'plate_effect']
CURVE_NOISE = ['gauss', 'cauchy', 'censored_left_heavy']
CORE = {'main': ['clean'] + CORE_NOISE,
        'ext': [f'{nm}_n{k}' for nm in CURVE_NOISE for k in (20, 50)] + ['grouped', 'grouped_clean'],
        'ext2': [f'{nm}_{s}' for nm in CURVE_NOISE for s in ('k1', 'k2', 'k3', 'k5', 'k10', 'mix', 'ctx10_q1', 'ctx1_q10')]}


def settings_for(source):
    full = ['clean'] + NOISES if source == 'main' else \
        json.load(open(os.path.join(SRC_DIR[source], 'manifest.json')))['settings']
    if os.environ.get('SUITE', 'core') == 'full':
        return full
    return [s for s in full if s in CORE[source]]


class BayesLookup:  # oracle posterior means from artifacts/bayes (main + ext) and artifacts/bayes_ext2
    def __init__(self, name):
        self.z = [np.load(os.path.join(d, name + '.npz')) for d in (BAYES, BAYES2) if os.path.exists(os.path.join(d, name + '.npz'))]

    def __getitem__(self, key):
        return next(z[key] for z in self.z if key in z.files)


def save_probs(path, new):  # merge into an existing per-dataset file (other sources' settings are kept)
    old = dict(np.load(path)) if os.path.exists(path) else {}
    old.update(new)
    np.savez_compressed(path, **old)


def inputs(z, s, mode, variant, zb, dev, X_main):
    X = z[f'{s}__X_clean'] if f'{s}__X_clean' in z.files else X_main
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)[None]
    if variant == 'bayes':
        assert mode in ('clean', 'mean'), 'bayes-denoised input is for single-value models'
        return (t(X if s in ('clean', 'grouped_clean') else zb[f'{s}__bayes']),)
    if mode in ('clean', 'mean', 'median'):
        return (t(X if s == 'clean' else z[f'{s}__{"median" if mode == "median" else "mean"}']),)
    if mode == 'stats':
        if s == 'clean':  # n = 1: mean = median = x, sd = iqr = sem = 0, log n = 0
            return (t(np.stack([X, X] + [0 * X] * 4, -1).reshape(len(X), -1)),)
        return (t(z[f'{s}__full']),)
    if s == 'clean':
        reps, counts = X[..., None], np.ones(X.shape, dtype=np.int64)
    else:
        reps, counts = np.nan_to_num(z[f'{s}__replicates']), z[f'{s}__counts'].astype(np.int64)
    mask = np.arange(reps.shape[-1])[None, None] < counts[..., None]
    return t(reps), torch.tensor(mask, device=dev)[None]


@torch.no_grad()
def evaluate(ckpt_path, out_path, tag, sources, variant=None):
    dev = 'cuda'
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    mode = ck['mode']
    task = ck.get('task', 'classification')
    model = make_model(mode, ck['rep_cfg'], task).to(dev).eval()
    model.load_state_dict(ck['state_dict'])
    manifest = json.load(open(os.path.join(DATA, 'manifest.json')))
    probs_dir = os.path.join(PROBS, tag)
    os.makedirs(probs_dir, exist_ok=True)
    recs = []
    for e in manifest['eval']:
        zs = {src: np.load(os.path.join(SRC_DIR[src], 'eval', e['name'] + '.npz')) for src in set(sources) | {'main'}}
        zb = BayesLookup(e['name']) if variant == 'bayes' else None
        tr, te, y = zs['main']['train_idx'], zs['main']['test_idx'], zs['main']['y']
        order = np.concatenate([tr, te])
        if task == 'regression':  # standardize with the context targets; metrics in context-sd units
            y = (y - y[tr].mean()) / (y[tr].std() + 1e-8)
            y_in = torch.tensor(y[tr], dtype=torch.float32, device=dev)[None]
        else:
            k = int(y.max()) + 1
            y_in = torch.tensor(y[tr], device=dev)[None]
        saved = dict(y_test=y[te])
        for src in sources:
            for s in settings_for(src):
                inp = [a[:, order] for a in inputs(zs[src], s, mode, variant, zb, dev, zs['main']['X_clean'])]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(*inp, y_in)
                base = dict(model=tag, ckpt=os.path.relpath(ckpt_path, ROOT), step=ck['step'], mode=mode, task=task,
                            input=variant or 'default', source=src, dataset=e['name'], noise=s)
                if task == 'regression':
                    q = logits[0].float().cpu().numpy()
                    m, u = reg_metrics(q, y[te])
                    saved[f'p__{s}'] = np.sort(q, 1).astype(np.float16)
                    saved[f'u__{s}'] = u.astype(np.float32)
                    recs.append(dict(base, **m))
                    continue
                p = torch.softmax(logits[0, :, :k].float(), -1).cpu().numpy().astype(np.float64)
                p = np.clip(p, 1e-7, 1)
                p /= p.sum(1, keepdims=True)
                saved[f'p__{s}'] = p.astype(np.float32)
                recs.append(dict(base, acc=accuracy_score(y[te], p.argmax(1)), log_loss=log_loss(y[te], p, labels=np.arange(k)),
                                 roc_auc=roc_auc_score(y[te], p[:, 1]) if k == 2 else roc_auc_score(y[te], p, multi_class='ovr')))
        save_probs(os.path.join(probs_dir, e['name'] + '.npz'), saved)
    with open(out_path, 'a') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    return recs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='+', help='tag=path pairs')
    ap.add_argument('--out', default=os.path.join(ROOT, 'artifacts', 'pretrain_eval_final.jsonl'))
    ap.add_argument('--sources', default='main', help='comma-separated: main, ext, ext2')
    ap.add_argument('--input', default=None, choices=[None, 'bayes'])
    args = ap.parse_args()
    for spec in args.ckpts:
        tag, path = spec.split('=', 1)
        recs = evaluate(path, args.out, tag, args.sources.split(','), args.input)
        by = {}
        for r in recs:
            by.setdefault(r['noise'], []).append(r['acc'] if 'acc' in r else r['crps'])
        print(f'{tag:28s} ' + ' '.join(f'{s} {np.mean(v):.3f}' for s, v in by.items()), flush=True)
