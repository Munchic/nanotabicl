"""Progress tracker for the shape / head runs (V4a, V4b, V4c, headM): reads artifacts/pretrain/<run>/log.jsonl only
(no GPU, no NVML: the driver/library mismatch breaks nvidia-smi). For each run: step, s/step, ETA to its target, train
loss trend, the dev kill-switch readout, and validation as a paired delta vs the run's own step 0 and (optionally) vs a
reference run. Usage: python monitor_shape.py [run ...] [--ref tx_shV4b_A] [--target 6000]
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
OPT = dict(a[2:].split('=', 1) if '=' in a else (a[2:], '1') for a in sys.argv[1:] if a.startswith('--'))
RUNS = ARGS or ['tx_shV4a_A', 'tx_shV4b_A', 'tx_shV4c_A', 'tx_headM_A']
REF = OPT.get('ref') or 'tx_shV4b_A'
TARGET = int(OPT.get('target', 6000))
LIVE = subprocess.run(['pgrep', '-af', 'obs_train.py'], capture_output=True, text=True).stdout


def logv(name):
    p = os.path.join(ROOT, 'artifacts', 'pretrain', name, 'log.jsonl')
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def val_delta(L, ref=None):
    """mean over noisy settings of (acc, loss) delta vs step 0 (or vs ref at the same step)."""
    by = {r['step']: r['val'] for r in L if 'val' in r}
    if not by:
        return None
    s0 = by.get(0)
    out = {}
    for st in sorted(by)[-2:]:
        v = by[st]
        keys = [k for k in v if k != 'clean']
        src = s0 if ref is None else by.get(st)
        if src is None:
            continue
        d = dict(acc=100 * sum(v[k]['acc'] - src[k]['acc'] for k in keys) / len(keys),
                 loss=sum(v[k]['loss'] - src[k]['loss'] for k in keys) / len(keys))
        out[st] = d
    return out


for run in RUNS:
    L = logv(run)
    if not L:
        print(f'{run}: no log yet'); continue
    steps = [r['step'] for r in L if r.get('step') is not None]
    tm = [r['time'] for r in L if r.get('step') is not None]
    st = max(steps)
    live = run in LIVE
    # s/step from the last two log points (resume restarts `time`, so only consecutive steps count)
    sp = (tm[-1] - tm[-2]) / (steps[-1] - steps[-2]) if len(steps) > 1 and steps[-1] > steps[-2] else None
    mil = 1000 if st < 1000 else TARGET  # the screen stops at 1000; the 6000-step continuation is a separate resume
    eta = (f'{(mil - st) * sp / 3600:.1f}h to {mil}' if sp and st < mil else ('stopped' if not live else ('screen done' if st < TARGET else 'done')))
    tl = [r['train_loss'] for r in L if r.get('train_loss') is not None]
    dev = [(r['step'], r['dev_real']) for r in L if 'dev_real' in r]
    d0 = next((v for s, v in dev if s == 0), None)
    kill = next((r.get('dev_kill') or r.get('learn_kill') for r in L if 'dev_kill' in r or 'learn_kill' in r), None)
    print(f'{run}  step {st}/{TARGET}  {"LIVE" if live else "stopped"}  '
          f'{"" if sp is None else f"{sp:.1f}s/step"}  ETA {eta}  train {tl[-1]:.3f} (last4: '
          f'{" ".join(f"{x:.3f}" for x in tl[-4:])})' if tl else f'{run} step {st}/{TARGET} {"LIVE" if live else "stopped"}')
    for s, v in dev[-3:]:
        m = '' if d0 is None or s == 0 else f'  ({v - d0:+.2f} vs step 0)'
        flag = '  KILL-WINDOW' if s <= 1000 and d0 is not None and v < d0 - 0.5 else ''
        print(f'   dev_real step {s:5d}: {v:7.2f}{m}{flag}')
    R = val_delta(logv(REF)) if run != REF else None
    vs0 = val_delta(L)
    if vs0:
        for s, d in vs0.items():
            print(f'   val step {s:5d}: vs step 0  Δacc {d["acc"]:+.2f} pts  Δloss {d["loss"]:+.3f}')
    if R:
        bym = {r['step']: r['val'] for r in logv(REF) if 'val' in r}
        for r in [r for r in L if 'val' in r][-1:]:
            if r['step'] in bym:
                keys = [k for k in r['val'] if k != 'clean']
                d = dict(acc=100 * sum(r['val'][k]['acc'] - bym[r['step']][k]['acc'] for k in keys) / len(keys),
                         loss=sum(r['val'][k]['loss'] - bym[r['step']][k]['loss'] for k in keys) / len(keys))
                print(f'   val step {r["step"]:5d} vs {REF} (same step): Δacc {d["acc"]:+.2f} pts  Δloss {d["loss"]:+.3f}')
    if kill:
        print(f'   KILL: {kill}')
