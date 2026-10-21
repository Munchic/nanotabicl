"""PLAN 2 step 0a: cytometry scale check (data property only, no labels/metrics). Prints per-marker 1/50/99%
quantiles and the share of negative values of the raw AnnData X for NK, HVTN, PreE, so we can tell whether the
Zenodo 'preprocessed' files are on an arcsinh scale (CyTOF arcsinh(x/5) is roughly -1..8). Usage: python cyto_scale.py
"""
import os, h5py, numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CYTO = os.path.join(ROOT, 'artifacts', 'real_raw', 'cytometry_zenodo6546964')
FILES = {'NK': 'nk_cell_preprocessed.h5ad', 'HVTN': 'hvtn_preprocessed.h5ad', 'PreE': 'preeclampsia_preprocessed.h5ad'}
TECH = {'Time', 'Event_length', 'Center', 'Offset', 'Width', 'Residual', 'beadDist', 'bead1', 'bead2', 'bead3', 'bead4',
        'bead5', 'bead6', 'bead7', 'bead8', 'DNA1', 'DNA2', 'DNA3', 'DNA4', 'Viability', 'Cell_length', 'FSC', 'SSC'}

for ds, fn in FILES.items():
    p = os.path.join(CYTO, fn)
    if not os.path.exists(p):
        print(f'{ds}: missing {p}'); continue
    h = h5py.File(p, 'r')
    X = h['X'][:]
    names = None
    var = h['var']
    def dec(a): return [x.decode() if isinstance(x, bytes) else str(x) for x in a]
    if 'pns_label' in var:
        if '__categories' in var and 'pns_label' in var['__categories']:
            names = [dec(var['__categories']['pns_label'][:])[c] for c in var['pns_label'][:]]
        else:
            names = dec(var['pns_label'][:])
    else:
        names = [str(i) for i in range(X.shape[1])]
    keep = [j for j, n_ in enumerate(names) if n_ not in TECH]
    Xk = X[:, keep]
    q = np.nanpercentile(Xk, [1, 50, 99], axis=0)
    neg = (Xk < 0).mean(0)
    print(f'{ds}: {Xk.shape[0]} cells x {Xk.shape[1]} markers; per-marker min/median/max of the 1% quantile = '
          f'{q[0].min():.3f}/{np.median(q[0]):.3f}/{q[0].max():.3f}; median of 50% = {np.median(q[1]):.3f}; '
          f'median of 99% = {np.median(q[2]):.3f}; share-negative median = {np.median(neg):.2%} '
          f'(max {neg.max():.2%})')
    h.close()
