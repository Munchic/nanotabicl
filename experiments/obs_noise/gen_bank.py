"""Pretraining bank: clean nanoprior classification tables, stored in blocks that share n_features.

Block b has seed SEED_BANK + b; table t in the block uses prior seed (SEED_BANK + b) * 1000 + t (disjoint from the
eval/pilot seeds in gen_data.py). Each block file holds X (64, N_ROWS, d) float32 and y (64, N_ROWS) int8.
The observation model is applied on the GPU at training time (obs_train.py), so the bank stays clean.
"""
import argparse, os, sys, time
from multiprocessing import Pool

import numpy as np, torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, '..', '..'))
from prior import rand_dataset_filtered  # noqa: E402

SEED_BANK = 5_000_000
SEED_BANK_REG = 6_000_000  # regression bank (Exp 5), disjoint seeds
N_ROWS, BLOCK, MAX_FEATURES, MAX_CLASSES = 1024, 64, 16, 10


def gen_table(seed: int, n_features: int, n_classes: int):
    torch.set_num_threads(1)
    for attempt in range(100):
        s = seed * 100 + attempt
        np.random.seed(s % 2**32)
        torch.manual_seed(s)
        cols = rand_dataset_filtered(x_cat_sizes=[0] * n_features, y_cat_sizes=[n_classes], n_samples=N_ROWS)
        X = torch.cat([cols[f'x_{i}'] for i in range(n_features)], dim=-1).numpy()
        if n_classes == 0:  # regression: continuous target, require some spread
            y = cols['y_0'].squeeze(-1).numpy()
            if np.isfinite(y).all() and y.std() > 1e-8:
                return X.astype(np.float32), y.astype(np.float32)
            continue
        y = np.unique(cols['y_0'].squeeze(-1).long().numpy(), return_inverse=True)[1]
        if len(np.unique(y)) >= 2 and np.bincount(y).min() >= 4:
            X = torch.cat([cols[f'x_{i}'] for i in range(n_features)], dim=-1).numpy()
            return X.astype(np.float32), y.astype(np.int8)
    raise RuntimeError(seed)


def block_jobs(b: int, regression: bool = False):
    base = SEED_BANK_REG if regression else SEED_BANK
    rng = np.random.default_rng(base + b)
    d = int(rng.integers(2, MAX_FEATURES + 1))
    ks = [0] * BLOCK if regression else [2 if rng.random() < 0.5 else int(rng.integers(3, MAX_CLASSES + 1)) for _ in range(BLOCK)]
    return d, [((base + b) * 1000 + t, d, k) for t, k in enumerate(ks)]


def _worker(job):
    return gen_table(*job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'artifacts', 'bank'))
    ap.add_argument('--n_blocks', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--task', choices=['classification', 'regression'], default='classification')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        for b in range(args.n_blocks):
            path = os.path.join(args.out, f'block_{b:05d}.npz')
            if os.path.exists(path):
                continue
            d, jobs = block_jobs(b, args.task == 'regression')
            tables = pool.map(_worker, jobs, chunksize=1)
            np.savez(path + '.tmp.npz', X=np.stack([t[0] for t in tables]), y=np.stack([t[1] for t in tables]))
            os.replace(path + '.tmp.npz', path)  # atomic: training can read finished blocks while this runs
            if b % 50 == 0:
                print(f'block {b} d={d} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
