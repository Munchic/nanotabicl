"""Dev-set registry: every dataset this project has already looked at is DEV and is split into 'val' and 'test'.
Rule: model/prior/feature choices may be made on 'val' only; 'test' is read once, at the end, behind --allow_test.
New datasets that have never been evaluated go into 'heldout' (untouched; treated like 'test').

A dataset is the first token of a real_data / real_eval_reg name (e.g. 'L1000_dose' -> 'L1000').
Usage:  python devsplit.py                       # print the table
        python devsplit.py --check L1000_dose,HAR # exit 1 if any is 'test'/'heldout' and --allow_test is absent
        python devsplit.py --split val            # print the comma-separated val sets
"""
import argparse, sys

# val: fast, cheap, and (where possible) the set with real replicate headroom. test: everything else.
VAL = ['HAR', 'L1000', 'CellPainting', 'HVTN', 'Beijing']
TEST = ['NK', 'PreE', 'Musk', 'PhysioNet', 'GasDrift', 'GasConc', 'Tox21', 'Tox21Pot', 'UPS1', 'CPTAC']
HELDOUT = []  # datasets never evaluated by this project; add them here BEFORE the first run

ROLE = {**{d: 'val' for d in VAL}, **{d: 'test' for d in TEST}, **{d: 'test' for d in HELDOUT}}


def role(name):
    """Dataset role for a task name ('L1000_dose_reg' -> 'L1000'). Unknown name: treated as test (safe default)."""
    for d in sorted(ROLE, key=len, reverse=True):
        if name == d or name.startswith(d + '_'):
            return ROLE[d]
    return 'test'


def guard(names, allow_test=False):
    """Raise on a test/held-out read unless allow_test. Returns {name: role}."""
    rs = {n: role(n) for n in names}
    bad = [n for n, r in rs.items() if r == 'test' and not allow_test]
    if bad:
        raise SystemExit(f'devsplit: refusing to read test/held-out sets {bad} (pass --allow_test for the one final read)')
    return rs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', default='')
    ap.add_argument('--split', default='')
    ap.add_argument('--allow_test', action='store_true')
    a = ap.parse_args()
    if a.split:
        print(','.join(d for d, r in ROLE.items() if r == a.split))
    elif a.check:
        print(guard(a.check.split(','), a.allow_test))
    else:
        for r in ('val', 'test', 'heldout'):
            print(f'{r:8s}', ', '.join(d for d, rr in ROLE.items() if rr == r) or '-')
