"""Conditional calibration in the mixed designs: overconfidence (confidence - accuracy) on single-sample vs replicated
query rows (mix), and on all queries for ctx10_q1 / ctx1_q10; mean ± SE over 27 datasets."""
import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MAN = json.load(open(os.path.join(ROOT, 'artifacts', 'data', 'manifest.json')))['eval']
NOISES = ['gauss', 'cauchy', 'censored_left_heavy']  # core-suite curve models
models = sys.argv[1].split(',')
cols = [('mix', 'single'), ('mix', 'repl'), ('ctx10_q1', 'all'), ('ctx1_q10', 'all')]
for nm in NOISES:
    print(f'\n*{nm}*: overconfidence (conf − acc) ± SE\n\n| model | mix: single rows | mix: replicated rows | ctx n=10 → query n=1 | ctx n=1 → query n=10 |\n|---|---|---|---|---|')
    for m in models:
        vals = {c: [] for c in cols}
        for e in MAN:
            P = np.load(os.path.join(ROOT, 'artifacts', 'probs', m, e['name'] + '.npz'))
            te = np.load(os.path.join(ROOT, 'artifacts', 'data', 'eval', e['name'] + '.npz'))['test_idx']
            rc = np.load(os.path.join(ROOT, 'artifacts', 'data_ext2', 'eval', e['name'] + '.npz'))[f'{nm}_mix__rowcount'][te]
            for design, grp in cols:
                p = P[f'p__{nm}_{design}'].astype(np.float64)
                oc = p.max(1) - (p.argmax(1) == P['y_test'])
                sel = {'single': rc == 1, 'repl': rc > 1, 'all': np.ones(len(rc), bool)}[grp]
                vals[(design, grp)].append(oc[sel].mean())
        print(f'| {m} | ' + ' | '.join(f'{np.mean(v):+.3f}±{np.std(v, ddof=1) / np.sqrt(len(v)):.3f}' for v in vals.values()) + ' |')
