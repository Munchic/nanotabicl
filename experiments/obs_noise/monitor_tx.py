"""Exp 8 monitor: each validation of the TabICLv2-branch runs vs the TabICLv2 + median reference (step 0, no training),
same 4 validation blocks and settings. Classification: accuracy / log-loss (Δ > 0 acc = better); regression: CRPS
(Δ < 0 = better). Usage: python monitor_tx.py [run ...]"""
import json, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = sys.argv[1:] or ['tx_obs_real_A', 'tx_obs_realB', 'tx_obs_cap_A', 'tx_obs_colmedg_A', 'tx_scr_gate10_A', 'tx_scr_cap_A', 'tx_scr_tok_A', 'tx_obs_colmedg_A_reg', 'tx_obs_B', 'tx_median_B',
                        'tx_obs_B_reg', 'tx_median_B_reg']


def logv(name):
    p = os.path.join(ROOT, 'artifacts', 'pretrain', name, 'log.jsonl')
    return [r for r in map(json.loads, open(p))] if os.path.exists(p) else []


for run in RUNS:
    reg = run.endswith('_reg')
    # reference: TabICLv2 + median at step 0 with the same precision / preprocessing as the run (fp32 + wrapper
    # preprocessing for runs started after the port: tx_median0p; the bf16 mean-start stage A: tx_median0)
    ref = logv(('tx_median0' if run.startswith('tx_obs_col_A') else 'tx_median0p_v3' if 'real' in run else 'tx_median0p')
               + ('_reg' if reg else ''))[-1]['val']
    if run.startswith('tx_obs_B'):  # stage B: vs the matched continued baseline at the same step (and clean vs step 0)
        B = {r['step']: r['val'] for r in logv(run.replace('tx_obs_B', 'tx_median_B')) if 'val' in r}
        for r in [r for r in logv(run) if 'val' in r and r['step'] in B][-3:]:
            v, b, k = r['val'], B[r['step']], ('loss' if reg else 'acc')
            noisy = [x for x in v if x != 'clean']
            d = sum(v[x][k] - b[x][k] for x in noisy) / len(noisy)
            print(f"{run} step {r['step']:5d}: Δ{'CRPS' if reg else 'acc'} vs tx_median_B (same step), mean noisy {d:+.4f} | "
                  f"clean {v['clean'][k]:.4f} (median_B {b['clean'][k]:.4f}, TabICLv2 step 0 {ref['clean'][k]:.4f})")
        continue
    L = logv(run)
    vals = [r for r in L if 'val' in r]
    tl = [r['train_loss'] for r in L if r.get('train_loss') is not None]
    if not vals:
        continue
    keys = list(ref)
    noisy = [k for k in keys if k != 'clean']
    print(f'{run} (train loss last 4: {" ".join(f"{x:.3f}" for x in tl[-4:])})')
    for r in vals[-3:]:
        v = r['val']
        if reg:
            d = {k: v[k]['loss'] - ref[k]['loss'] for k in keys}
            print(f"  step {r['step']:5d} ΔCRPS vs median: mean noisy {sum(d[k] for k in noisy) / len(noisy):+.4f} | " +
                  ' '.join(f'{k[:9]} {d[k]:+.3f}' for k in keys))
        else:
            d = {k: v[k]['acc'] - ref[k]['acc'] for k in keys}
            dl = {k: v[k]['loss'] - ref[k]['loss'] for k in keys}
            print(f"  step {r['step']:5d} Δacc vs median: mean noisy {100 * sum(d[k] for k in noisy) / len(noisy):+.2f} pts, "
                  f"Δlogloss {sum(dl[k] for k in noisy) / len(noisy):+.3f} | " + ' '.join(f'{k[:9]} {100 * d[k]:+.1f}' for k in keys))
