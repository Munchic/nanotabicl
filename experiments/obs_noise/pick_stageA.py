"""Pick the stage-A run (mean start vs median start) with the larger final validation gain over TabICLv2 + median
(mean over the noisy validation settings: accuracy for classification, -CRPS for regression). Prints its name."""
import json, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
s = '_reg' if sys.argv[1] == 'regression' else ''
last = lambda n: [r for r in map(json.loads, open(os.path.join(ROOT, 'artifacts', 'pretrain', n, 'log.jsonl'))) if 'val' in r][-1]['val']
ref = last('tx_median0' + s)
score = {}
for run in ['tx_obs_col_A' + s, 'tx_obs_colmed_A' + s]:
    v = last(run)
    noisy = [k for k in ref if k != 'clean']
    key = 'loss' if s else 'acc'
    d = sum(v[k][key] - ref[k][key] for k in noisy) / len(noisy)
    score[run] = -d if s else d
best = max(score, key=score.get)
print(best, file=sys.stdout)
print({k: round(v, 5) for k, v in score.items()}, file=sys.stderr)
