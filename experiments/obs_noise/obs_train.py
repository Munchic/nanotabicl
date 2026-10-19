"""Pretrain / continue-train nanoTabICL variants on the clean prior bank + a GPU-sampled observation model.

modes:
  clean : stock NanoTabICLv2 on clean X (phase 1, the "stock nanotabicl" checkpoint)
  mean  : stock NanoTabICLv2 on noisy cell means (= mean imputation)                [control]
  stats : stock NanoTabICLv2 on per-column summary stats (mean, median, sd, iqr, sem, log n)  [control]
  obs   : ObsNanoTabICL (replicate set encoder) on replicates                         [proposal]
Bank blocks 0..N_VAL-1 are held out for validation (fixed observation-model seed).
"""
import argparse, glob, json, math, os, sys, time

import numpy as np, torch, torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, '..', '..'))
from model import NanoTabICLv2  # noqa: E402
from obs_model import ObsNanoTabICL  # noqa: E402

K = 10
NOISES = ['gauss', 'student_t3', 'censored', 'cauchy', 'cauchy_hetero',
          'gauss_valhet', 'cauchy_valhet', 'censored_left', 'censored_left_heavy']  # same models as gen_data.py
NOISES_V2 = NOISES + ['fail_floor', 'fail_shift', 'cv_lognormal', 'lod_logistic', 'saturation', 'plate_effect']  # Exp 6
TRAIN_NOISES = NOISES  # set by --noise_set; v1 keeps the original training mixture and RNG stream
N_VAL = 16
VAL_CHUNK = 8
MODEL_CFG = dict(max_classes=10, out_dim=10, embed_dim=32, col_num_blocks=2, row_num_blocks=2, icl_num_blocks=6,
                 col_nhead=4, row_nhead=4, icl_nhead=4, n_cls_rows=32)  # ~1M params (NOTES.md compute budget)


# ----- observation model (batched, on GPU) -----

def observe(X, g, noise=None, scale=None, p_clean=0.0, censor_q=None, lod_q=None, k_max=None, regimes=None, n_train=None):
    """X: (B, R, C) clean. Returns reps (B, R, C, K), mask (B, R, C, K). Per table: clean w.p. p_clean, else a
    random noise model with scale ~ logU(0.1, 1) * sd(col) (or the given noise/scale); counts ~ U{1..K} per cell."""
    B, R, C = X.shape
    dev = X.device
    K = k_max or globals()['K']  # tensor width = max replicates per cell in this batch
    col_sd = X.std(dim=1, keepdim=True).clamp(min=1e-12)[..., None]  # (B, 1, C, 1)
    if scale is None:
        scale = torch.exp(torch.empty(B, 1, 1, 1, device=dev).uniform_(math.log(0.1), 0.0, generator=g))
    noise_ids = (torch.full((B,), NOISES_V2.index(noise), device=dev) if noise is not None
                 else torch.randint(len(TRAIN_NOISES), (B,), device=dev, generator=g))
    gauss = torch.randn(B, R, C, K, device=dev, generator=g)
    sq = torch.randn(B, R, C, K, 3, device=dev, generator=g) ** 2
    t3 = gauss / torch.sqrt(sq.sum(-1) / 3)
    t2 = gauss / torch.sqrt(sq[..., :2].sum(-1) / 2)
    cauchy = torch.tan(math.pi * (torch.rand(B, R, C, K, device=dev, generator=g) - 0.5))
    nid = noise_ids[:, None, None, None]
    eps = torch.where(nid == 1, t3, torch.where((nid == 3) | (nid == 4) | (nid == 6), cauchy,
                                                torch.where(nid == 8, t2, gauss)))
    row_scale = torch.where(nid == 4, torch.exp(torch.randn(B, R, 1, 1, device=dev, generator=g)), 1.0)
    # value-dependent heteroscedasticity (valhet): scale = 2 * percentile rank of the clean value in its column
    u = (X.argsort(dim=1, stable=True).argsort(dim=1, stable=True) + 1).to(X.dtype) / R
    val_scale = torch.where((nid == 5) | (nid == 6), 2 * u[..., None], 1.0)
    reps = X[..., None] + scale * col_sd * row_scale * val_scale * eps
    # censoring at per-column quantiles of the clean column (random limits in training, 10/90 in eval)
    if censor_q is None:
        q_lo = torch.empty(B, 1, 1, device=dev).uniform_(0.0, 0.2, generator=g)
        q_hi = torch.empty(B, 1, 1, device=dev).uniform_(0.8, 1.0, generator=g)
    else:
        q_lo, q_hi = (torch.full((B, 1, 1), q, device=dev) for q in censor_q)
    Xs = X.sort(dim=1).values
    lo = Xs.gather(1, (q_lo * (R - 1)).round().long().expand(B, 1, C))[..., None]
    hi = Xs.gather(1, (q_hi * (R - 1)).round().long().expand(B, 1, C))[..., None]
    reps = torch.where(nid == 2, torch.minimum(torch.maximum(reps, lo), hi), reps)
    # left-censoring at a lower detection limit (censored_left: q ~ U(0.1, 0.5), censored_left_heavy: q ~ U(0.4, 0.8))
    if lod_q is None:
        q_lod = torch.where(noise_ids == 8, 0.4, 0.1)[:, None, None] + 0.4 * torch.rand(B, 1, 1, device=dev, generator=g)
    else:
        q_lod = torch.full((B, 1, 1), lod_q, device=dev)
    lod = Xs.gather(1, (q_lod * (R - 1)).round().long().expand(B, 1, C))[..., None]
    reps = torch.where((nid == 7) | (nid == 8), torch.maximum(reps, lod), reps)
    if (noise_ids >= len(NOISES)).any():  # Exp 6 modes (only drawn with --noise_set v2 or when evaluating them)
        reps = observe_v2(X, reps, noise_ids, col_sd, scale, eps, g)
    counts = torch.randint(1, K + 1, (B, R, C, 1), device=dev, generator=g)
    if regimes is not None:  # per-table count regime (Exp 5); None keeps the original per-cell draw (and RNG stream)
        counts = count_regimes(counts, regimes, n_train, K, g)
    clean = (torch.rand(B, device=dev, generator=g) < p_clean)[:, None, None, None]
    counts = torch.where(clean, 1, counts)
    reps = torch.where(clean, X[..., None].expand_as(reps), reps)
    mask = torch.arange(K, device=dev) < counts
    return reps, mask


def masked_quantile(reps, mask, q):  # deterministic (sort-based) == numpy's default 'linear' nanquantile
    srt = torch.where(mask, reps, float('inf')).sort(dim=-1).values
    pos = q * (mask.sum(-1, keepdim=True) - 1).clamp(min=0).to(reps.dtype)
    lo, hi = pos.floor().long(), pos.ceil().long()
    v_lo, v_hi = srt.gather(-1, lo), srt.gather(-1, hi)
    return (v_lo + (pos - lo) * (v_hi - v_lo)).squeeze(-1)


def observe_v2(X, reps, noise_ids, col_sd, scale, eps, g):
    """GPU versions of gen_data.py's Exp 6 modes (same constants), overriding reps for tables with those noise ids.
    eps is the table's Gaussian draw (B, R, C, K); scale = base noise scale (B or 1, 1, 1, 1) in col-sd units."""
    from gen_data import (FAIL_P_MAX, FAIL_GOOD_MULT, FAIL_SHIFT, CV_LN, LOD_THETA_Q, LOD_SLOPE, LOD_FLOOR_Q, ULOQ_Q,
                          PLATE_SD, PLATE_COL_SD)
    B, R, C, K = eps.shape
    dev = X.device
    nid = noise_ids[:, None, None, None]
    sd = col_sd  # (B, 1, C, 1)
    good = X[..., None] + scale * FAIL_GOOD_MULT * sd * eps
    q = lambda p: torch.quantile(X, p, dim=1, keepdim=True)[..., None]  # (B, 1, C, 1)
    xmin = X.amin(dim=1, keepdim=True)[..., None]
    p_fail = FAIL_P_MAX * torch.rand(B, R, C, 1, device=dev, generator=g)
    failed = torch.rand(B, R, C, K, device=dev, generator=g) < p_fail
    fail_floor = torch.where(failed, (xmin - sd).expand_as(good), good)
    fail_shift = torch.where(failed, good + FAIL_SHIFT * sd, good)
    m = xmin - 0.25 * sd
    cv = m + (X[..., None] - m) * torch.exp(CV_LN * eps - CV_LN ** 2 / 2)
    p_det = torch.sigmoid((good - q(LOD_THETA_Q)) / (LOD_SLOPE * sd))
    lod = torch.where(torch.rand(B, R, C, K, device=dev, generator=g) < p_det, good, q(LOD_FLOOR_Q).expand_as(good))
    sat = torch.minimum(X[..., None] + scale * sd * eps, q(ULOQ_Q))
    b = PLATE_SD * torch.randn(B, 1, 1, K, device=dev, generator=g) + PLATE_COL_SD * torch.randn(B, 1, C, K, device=dev, generator=g)
    plate = good + sd * b
    out = reps
    for i, v in enumerate([fail_floor, fail_shift, cv, lod, sat, plate]):
        out = torch.where(nid == len(NOISES) + i, v, out)
    return out


REGIMES = ['cell', 'row', 'col', 'ctxq']


def count_regimes(counts, weights, n_train, K, g):
    """counts: (B, R, C, 1) per-cell U{1..K} draw. Per table pick a regime (weights over REGIMES):
    cell: keep; row: each row is single (n = 1) w.p. p ~ U(0.2, 0.8), else all its cells share n ~ U{2..K};
    col: each column is replicated w.p. 0.5 (cells n ~ U{2..K}), else single; ctxq: context rows share n_ctx and
    query rows share n_q, each independently 1 w.p. 0.5 else U{2..K}."""
    B, R, C, _ = counts.shape
    dev = counts.device
    rid = torch.multinomial(torch.tensor(weights, dtype=torch.float, device=dev), B, replacement=True, generator=g)
    rid = rid[:, None, None, None]
    many = lambda *shape: torch.randint(2, K + 1, shape, device=dev, generator=g)
    p_single = 0.2 + 0.6 * torch.rand(B, 1, 1, 1, device=dev, generator=g)
    row = torch.where(torch.rand(B, R, 1, 1, device=dev, generator=g) < p_single, 1, many(B, R, 1, 1)).expand(B, R, C, 1)
    col = torch.where(torch.rand(B, 1, C, 1, device=dev, generator=g) < 0.5, many(B, R, C, 1), 1)
    pick = lambda: torch.where(torch.rand(B, 1, 1, 1, device=dev, generator=g) < 0.5, 1, many(B, 1, 1, 1))
    n_ctx, n_q = pick(), pick()
    is_ctx = (torch.arange(R, device=dev) < n_train)[None, :, None, None]
    ctxq = torch.where(is_ctx, n_ctx, n_q).expand(B, R, C, 1)
    return torch.where(rid == 0, counts, torch.where(rid == 1, row, torch.where(rid == 2, col, ctxq)))


def summary_stats(reps, mask):  # -> (B, R, C*6): x0_mean, x0_median, x0_sd, x0_iqr, x0_sem, x0_logn, x1_mean, ...
    n = mask.sum(-1).float()
    mean = torch.where(mask, reps, 0.0).sum(-1) / n
    med = masked_quantile(reps, mask, 0.5)
    sd = torch.where(n > 1, torch.sqrt(((torch.where(mask, reps - mean[..., None], 0.0)) ** 2).sum(-1) / (n - 1).clamp(min=1)), 0.0)
    iqr = masked_quantile(reps, mask, 0.75) - masked_quantile(reps, mask, 0.25)
    return torch.stack([mean, med, sd, iqr, sd / n.sqrt(), n.log()], dim=-1).flatten(-2)


NQ = 99  # regression: predicted quantile levels tau_k = k / (NQ + 1)
TAUS = torch.arange(1, NQ + 1) / (NQ + 1)
TASK = 'classification'  # set by main() / make_model(task=...)


def model_cfg(task):
    return MODEL_CFG if task == 'classification' else dict(MODEL_CFG, max_classes=0, out_dim=NQ)


def make_model(mode, rep_cfg, task='classification'):
    cfg = model_cfg(task)
    return ObsNanoTabICL(**cfg, **rep_cfg) if mode == 'obs' else NanoTabICLv2(**cfg)


def standardize_y(y, n_train):  # regression: per-table standardization with the context targets
    mu = y[:, :n_train].mean(dim=1, keepdim=True)
    sd = y[:, :n_train].std(dim=1, keepdim=True) + 1e-8
    return (y - mu) / sd


def pinball(q_pred, y_test):  # (B, T, NQ), (B, T) -> (B, T) mean pinball loss over quantile levels
    taus = TAUS.to(q_pred.device)
    diff = y_test[..., None] - q_pred.float()
    return torch.maximum(taus * diff, (taus - 1) * diff).mean(-1)


def task_loss(out, y_test, y_ctx):  # -> per-test-row loss (B, T), valid mask (B, T)
    if TASK == 'regression':
        return pinball(out, y_test), torch.ones_like(y_test, dtype=torch.bool)
    valid = (y_test[..., None] == y_ctx[:, None, :]).any(-1)  # skip test rows of classes absent in context
    return ce_loss(out, y_test), valid


def model_forward(model, mode, X, y_train, g, **obs_kw):
    if mode == 'clean':
        return model(X, y_train)
    reps, mask = observe(X, g, **obs_kw)
    if mode == 'mean':
        return model(torch.where(mask, reps, 0).sum(-1) / mask.sum(-1), y_train)
    if mode == 'median':  # median imputation (numpy-equivalent median)
        return model(masked_quantile(reps, mask, 0.5), y_train)
    if mode == 'stats':
        return model(summary_stats(reps, mask), y_train)
    return model(reps, mask, y_train)


# ----- data -----

class Bank:
    def __init__(self, path, n_blocks=None):
        self.path = path
        self._files = sorted(glob.glob(os.path.join(self.path, 'block_*.npz')))
        if n_blocks is not None:  # reproducibility: train on a fixed, complete bank only
            assert len(self._files) == n_blocks, f'bank has {len(self._files)} blocks, expected {n_blocks}'

    def files(self):
        return self._files

    def load(self, f, dev):
        z = np.load(f)
        y = z['y'].astype(np.float32 if z['y'].dtype.kind == 'f' else np.int64)
        return torch.from_numpy(z['X']).to(dev), torch.from_numpy(y).to(dev)

    def batch(self, rng, dev, batch_size):
        files = self.files()[N_VAL:]
        X, y = self.load(files[rng.integers(len(files))], dev)
        tab = torch.from_numpy(rng.choice(X.shape[0], batch_size, replace=False)).to(dev)
        n_rows = int(rng.integers(256, X.shape[1] + 1))
        rows = torch.from_numpy(rng.choice(X.shape[1], n_rows, replace=False)).to(dev)
        X, y = X[tab][:, rows], y[tab][:, rows]
        n_train = int(n_rows * rng.uniform(0.3, 0.8))
        return X, y, n_train  # test rows whose class is absent from the context are masked out of the loss


def ce_loss(logits, y_test):
    return F.cross_entropy(logits.float().flatten(0, 1), y_test.flatten(), reduction='none').view(y_test.shape)


@torch.no_grad()
def validate(model, mode, bank, dev):
    model.eval()
    out = {}
    files = bank.files()[:N_VAL]
    settings = [('clean', None)] + [(nm, nm) for nm in TRAIN_NOISES]
    for name, noise in settings:
        g = torch.Generator(device=dev).manual_seed(12345)
        losses, accs = [], []
        for f in files:
            X_all, y_all = bank.load(f, dev)
            for i in range(0, X_all.shape[0], VAL_CHUNK):  # chunked: a full 64-table block would pin GPU memory
                X, y = X_all[i:i + VAL_CHUNK, :1024], y_all[i:i + VAL_CHUNK, :1024]
                n_train = 512
                if TASK == 'regression':
                    y = standardize_y(y, n_train)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    if noise is None and mode != 'clean':
                        reps = X[..., None].expand(*X.shape, K)
                        mask = (torch.arange(K, device=dev) < 1).expand(*X.shape, K)
                        if mode in ('mean', 'median'):
                            logits = model(X, y[:, :n_train])
                        elif mode == 'stats':
                            logits = model(summary_stats(reps, mask), y[:, :n_train])
                        else:
                            logits = model(reps, mask, y[:, :n_train])
                    elif noise is None:
                        logits = model(X, y[:, :n_train])
                    else:
                        m = 'mean' if mode == 'clean' else mode  # clean model sees noisy cell means
                        logits = model_forward(model, m, X, y[:, :n_train], g, noise=noise,
                                               scale=torch.tensor(0.5, device=dev), censor_q=(0.1, 0.9),
                                               lod_q={'censored_left': 0.3, 'censored_left_heavy': 0.6}.get(noise))
                y_test = y[:, n_train:]
                l, valid = task_loss(logits, y_test, y[:, :n_train])
                if TASK == 'regression':  # 'loss' = CRPS (~2 x mean pinball), 'acc' = R^2 of the predicted median
                    losses.append(2 * l.mean().item())
                    med = logits[..., NQ // 2].float()
                    accs.append((1 - ((med - y_test) ** 2).mean(1) / y_test.var(1, unbiased=False).clamp(min=1e-8)).mean().item())
                else:
                    losses.append(l[valid].mean().item())
                    accs.append((logits.argmax(-1) == y_test)[valid].float().mean().item())
        out[name] = dict(loss=float(np.mean(losses)), acc=float(np.mean(accs)))
    model.train()
    torch.cuda.empty_cache()  # give validation memory back to co-running jobs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['clean', 'mean', 'median', 'stats', 'obs'], required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--init', default=None, help='checkpoint to start from (strict=False for obs)')
    ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--p_clean', type=float, default=0.25)
    ap.add_argument('--val_every', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rep_cfg', default='{}', help='json kwargs for ObsNanoTabICL, e.g. {"use_sab": false}')
    ap.add_argument('--freeze_backbone', action='store_true', help='obs only: train only the replicate branch')
    ap.add_argument('--rep_lr_mult', type=float, default=1.0, help='obs only: lr multiplier for the replicate branch')
    ap.add_argument('--bank', default=os.path.join(ROOT, 'artifacts', 'bank'))
    ap.add_argument('--micro_batch', type=int, default=0, help='gradient accumulation micro-batch (0 = off)')
    ap.add_argument('--resume', action='store_true', help='continue from out_dir/last.ckpt if present')
    ap.add_argument('--n_blocks', type=int, default=2000, help='required bank size (0 = use whatever exists)')
    ap.add_argument('--nondeterministic', action='store_true', help='allow nondeterministic CUDA kernels')
    ap.add_argument('--stop_at', type=int, default=0, help='testing: exit right after the checkpoint at this step')
    ap.add_argument('--task', choices=['classification', 'regression'], default='classification')
    ap.add_argument('--regimes', default='', help='json weights over REGIMES (cell,row,col,ctxq); empty = per-cell only')
    ap.add_argument('--k50_prob', type=float, default=0.0, help='share of batches with up to 50 replicates per cell')
    ap.add_argument('--noise_set', choices=['v1', 'v2'], default='v1', help='v2 adds the Exp 6 failure / literature modes')
    ap.add_argument('--y_clip', type=float, default=0.0, help='regression: soft-clip standardized targets to +-y_clip in training')
    args = ap.parse_args()
    dev = 'cuda'
    global TASK, TRAIN_NOISES
    TASK = args.task
    TRAIN_NOISES = NOISES_V2 if args.noise_set == 'v2' else NOISES
    regimes = json.loads(args.regimes) if args.regimes else None
    if not args.nondeterministic:  # bitwise-reproducible training (also across exact resumes)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    out_dir = os.path.join(ROOT, 'artifacts', 'pretrain', args.name)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, 'args.json'), 'w'), indent=1)

    rep_cfg = json.loads(args.rep_cfg)
    model = make_model(args.mode, rep_cfg, args.task).to(dev)
    if args.init:
        missing, unexpected = model.load_state_dict(torch.load(args.init, map_location=dev)['state_dict'], strict=False)
        assert not unexpected and all(k.startswith('rep_') for k in missing), (missing, unexpected)
    params = [p for n, p in model.named_parameters() if not args.freeze_backbone or n.startswith('rep_')]
    rep = [p for n, p in model.named_parameters() if n.startswith('rep_')]
    rest = [p for n, p in model.named_parameters() if not n.startswith('rep_') and not args.freeze_backbone]
    groups = [g for g in [dict(params=rest, lr=args.lr), dict(params=rep, lr=args.lr * args.rep_lr_mult)] if g['params']]
    if args.rep_lr_mult == 1.0:  # single group == the original optimizer layout (keeps old checkpoints resumable)
        groups = params
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup) *
                                              (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps)))))
    bank = Bank(args.bank, args.n_blocks or None)
    start = 0
    last = os.path.join(out_dir, 'last.ckpt')
    if args.resume and os.path.exists(last):  # exact resume: ckpt at step s is taken before step s's update
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck['state_dict']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        rng.bit_generator.state = ck['np_rng']
        g.set_state(ck['torch_rng'].cpu())
        start = ck['step']
        print(f'resumed at step {start}', flush=True)
    log = open(os.path.join(out_dir, 'log.jsonl'), 'a')
    t0, run_loss = time.time(), []
    for step in range(start, args.steps + 1):
        if step % args.val_every == 0 or step == args.steps:
            v = validate(model, args.mode, bank, dev)
            rec = dict(step=step, time=time.time() - t0, train_loss=float(np.mean(run_loss)) if run_loss else None, val=v)
            log.write(json.dumps(rec) + '\n'); log.flush()
            print(f"step {step:6d} {rec['time']:6.0f}s train {rec['train_loss'] or float('nan'):.4f} | " +
                  ' '.join(f"{k} {x['acc']:.3f}/{x['loss']:.3f}" for k, x in v.items()), flush=True)
            run_loss = []
            torch.save(dict(state_dict=model.state_dict(), model_cfg=model_cfg(args.task), rep_cfg=rep_cfg, mode=args.mode, task=args.task,
                            step=step, args=vars(args), opt=opt.state_dict(), sched=sched.state_dict(),
                            np_rng=rng.bit_generator.state, torch_rng=g.get_state()), last + '.tmp')
            os.replace(last + '.tmp', last)
            if args.stop_at and step == args.stop_at:
                return
        if step == args.steps:
            break
        X, y, n_train = bank.batch(rng, dev, args.batch_size)
        if TASK == 'regression':
            y = standardize_y(y, n_train)
            if args.y_clip > 0:  # 0.2% of prior tables have |z| > 100 (max 1.5e5): they would dominate the pinball loss
                y = args.y_clip * torch.tanh(y / args.y_clip)
        y_test = y[:, n_train:]
        valid = (y_test[..., None] == y[:, None, :n_train]).any(-1) if TASK == 'classification' else \
            torch.ones_like(y_test, dtype=torch.bool)  # classification: skip test rows of classes absent in context
        obs_kw = dict(p_clean=args.p_clean)
        if args.k50_prob > 0:  # only draw when enabled: default runs keep their original RNG stream
            obs_kw['k_max'] = 50 if rng.random() < args.k50_prob else K
        if regimes is not None:
            obs_kw.update(regimes=regimes, n_train=n_train)
        opt.zero_grad(set_to_none=True)
        mb = args.micro_batch or args.batch_size
        if obs_kw.get('k_max', K) > K:  # up-to-50-replicate batches: 4x smaller micro-batches (same effective batch)
            mb = max(1, mb // 4)
        loss_sum, ok = 0.0, True
        for i in range(0, args.batch_size, mb):  # gradient accumulation; loss = mean over all valid test cells
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model_forward(model, args.mode, X[i:i + mb], y[i:i + mb, :n_train], g, **obs_kw)
            l = task_loss(logits, y_test[i:i + mb], y[i:i + mb, :n_train])[0][valid[i:i + mb]].sum() / valid.sum()
            if not torch.isfinite(l):
                ok = False
                break
            l.backward()
            loss_sum += l.item()
        if not ok:
            print(f'non-finite loss at step {step}, skipping', flush=True)
            continue
        loss = torch.tensor(loss_sum)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        run_loss.append(loss.item())


if __name__ == '__main__':
    main()
