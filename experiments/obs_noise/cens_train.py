"""CENSORING.md P1 + P2: continue TabICLv2 (regression) on the clean prior bank with censored targets and features.

Prior, per table (GPU, after the clean table is drawn):
  y: w.p. --p_ycens censored. Side left / right / both (0.4 / 0.4 / 0.2), fraction U(0.05, 0.5) per side (<= 0.35 each
     for both); bound = that quantile of the table's latent y, constant or (w.p. 0.5) jittered per row by 0.25 sd
     (a per-sample bound, like Tox21's varying top concentration). Censored values are reported AT the bound.
  X: w.p. --p_xcens a table censors a U(0.2, 0.6) share of its columns, same scheme with fraction U(0.05, 0.4).
  Spec dropout: per table and channel (y / X), flags hidden w.p. --spec_drop (the values stay at the bound).
  Loss: pinball on the LATENT y of every query row. Targets are standardized with the OBSERVED context values (bounds
  included), which are available identically with the spec on or off (and on real data).
Model: cens_model.CensNanoTabICL (zero-init flags; no flags == TabICLv2 exactly). Backbone on TabICLv2's Muon, flag
maps on AdamW at --lr x --cens_lr_mult. fp32 + TF32, deterministic, exact resume.
Validation: fixed censoring settings on held-out bank blocks, spec on and off, CRPS on the latent y in units of the
table's latent-y sd (comparable across settings; 'clean' = the latent context without a spec, i.e. the oracle
reference B4; with --known also 'clean_on' = the same tables with a known, all-observed spec).
"""
import argparse, json, math, os, sys, time

import numpy as np, torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, '..', '..'))
sys.path.insert(0, ROOT)
import obs_train as ot  # noqa: E402  (helpers only: Bank, pinball, quantile levels)
from cens_model import CensNanoTabICL  # noqa: E402
from tabicl_port import load_tabicl  # noqa: E402

LEFT, RIGHT, BOTH = 0, 1, 2


def censor(v, g, active, side, q_l, q_r, rowb, jit=0.25):
    """v (B, R, C) latent values; per (table, column), shape (B, 1, C): active bool, side {LEFT, RIGHT, BOTH}, fractions
    q_l / q_r, rowb bool (per-row jittered bounds). -> observed values (censored at the bound), flags {0, 1 '<', 2 '>'}."""
    B, R, C = v.shape
    s = v.sort(dim=1).values
    qtl = lambda q: s.gather(1, (q * (R - 1)).round().long().clamp(0, R - 1))
    jitter = torch.randn(B, R, C, device=v.device, generator=g) * (jit * v.std(dim=1, keepdim=True) * rowb)
    inf = torch.full_like(v, float('inf'))
    lo = torch.where(active & (side != RIGHT), qtl(q_l) + jitter, -inf)
    hi = torch.where(active & (side != LEFT), qtl(1 - q_r) + jitter, inf)
    lo = torch.minimum(lo, hi)
    flag = torch.where(v > hi, 2, torch.where(v < lo, 1, 0))
    return torch.minimum(torch.maximum(v, lo), hi), flag


def sample_spec(B, C, g, dev, p_table, col_share, frac):
    """Random censoring design, (B, 1, C) each. col_share: (lo, hi) share of censored columns in an active table."""
    u = lambda *s: torch.rand(*s, device=dev, generator=g)
    share = col_share[0] + (col_share[1] - col_share[0]) * u(B, 1, 1)
    active = (u(B, 1, 1) < p_table) & (u(B, 1, C) < share)
    r = u(B, 1, C)
    side = torch.where(r < 0.4, LEFT, torch.where(r < 0.8, RIGHT, BOTH))
    hi = torch.where(side == BOTH, torch.full_like(r, min(frac[1], 0.35)), torch.full_like(r, frac[1]))
    q_l, q_r = (frac[0] + (hi - frac[0]) * u(B, 1, C) for _ in range(2))
    return active, side, q_l, q_r, u(B, 1, C) < 0.5


def fixed_spec(B, C, dev, side, q, rowb, col_share=1.0, g=None):  # validation / eval designs
    full = lambda val: torch.full((B, 1, C), val, device=dev)
    active = torch.ones(B, 1, C, dtype=torch.bool, device=dev)
    if col_share < 1.0:
        active = torch.rand(B, 1, C, device=dev, generator=g) < col_share
    return active, full(side), full(q), full(q), full(rowb).bool()


def prepare(X, y, n_train, g, y_design=None, x_design=None, y_show=True, x_show=True, known=False, max_tail=0.0):
    """-> X_obs, x_flag, y_ctx (standardized observed context), y_flag, y_lat (standardized like y_ctx, all rows),
    (mu, sd) of the standardization. y_show / x_show: bool or (B,) bool tensor (spec visible per table). known: observed
    cells of a visible spec get flag 3 ("known observed") instead of 0 (= no spec), see cens_model. max_tail > 0: y
    censoring is undone in tables whose latent-y sd exceeds max_tail x the observed context sd (censoring a heavy tail
    of the prior leaves latent targets at up to 1e7 observed sd, which dominate the pinball loss)."""
    B, R, C = X.shape
    x_flag = torch.zeros_like(X, dtype=torch.long)
    if x_design is not None:
        X, x_flag = censor(X, g, *x_design)
    y_obs, y_flag = (y, torch.zeros_like(y, dtype=torch.long)) if y_design is None else \
        (t[..., 0] for t in censor(y[..., None], g, *y_design))
    if max_tail > 0 and y_design is not None:
        bad = y.std(1, keepdim=True) > max_tail * y_obs[:, :n_train].std(1, keepdim=True)
        y_obs, y_flag = torch.where(bad, y, y_obs), torch.where(bad, 0, y_flag)
    if known:
        x_flag, y_flag = (torch.where(f == 0, 3, f) for f in (x_flag, y_flag))
    mu = y_obs[:, :n_train].mean(1, keepdim=True)
    sd = y_obs[:, :n_train].std(1, keepdim=True) + 1e-8
    show = lambda s, f: f * (s if isinstance(s, torch.Tensor) else torch.tensor(bool(s), device=f.device)).view(-1, *[1] * (f.dim() - 1))
    return X, show(x_show, x_flag), (y_obs[:, :n_train] - mu) / sd, show(y_show, y_flag[:, :n_train]), (y - mu) / sd, \
        (mu, sd), y_flag


VAL_SETTINGS = {  # name: (y design (side, q, rowb) or None, x design (side, q, rowb, col_share) or None)
    'clean': (None, None),
    'yR30': ((RIGHT, 0.3, False), None),
    'yL30': ((LEFT, 0.3, False), None),
    'yR30row': ((RIGHT, 0.3, True), None),
    'yR50': ((RIGHT, 0.5, False), None),
    'xR30row': (None, (RIGHT, 0.3, True, 0.5)),
}


@torch.no_grad()
def validate(model, bank, dev, n_blocks, chunk=8, known=False):
    model.eval()
    out = {}
    for name, (yd, xd) in VAL_SETTINGS.items():
        for show in (([False, True] if known else [False]) if name == 'clean' else [True, False]):
            g = torch.Generator(device=dev).manual_seed(12345)
            crps, crps_c = [], []
            for f in bank.files()[:n_blocks]:
                X_all, y_all = bank.load(f, dev)
                for i in range(0, X_all.shape[0], chunk):
                    X, y = X_all[i:i + chunk, :1024], y_all[i:i + chunk, :1024]
                    B, R, C = X.shape
                    n_train = 512
                    y_des = None if yd is None else fixed_spec(B, 1, dev, *yd)
                    x_des = None if xd is None else fixed_spec(B, C, dev, *xd[:3], col_share=xd[3], g=g)
                    Xo, xf, yc, yf, yl, (mu, sd), yf_all = prepare(X, y, n_train, g, y_des, x_des, show, show, known)
                    q = model(Xo, yc, xf, yf)  # all-zero flags (spec off, or nothing censored without known) == None
                    l = 2 * ot.pinball(q, yl[:, n_train:])  # CRPS, standardized units
                    l = l * sd / (y.std(1, keepdim=True) + 1e-8)  # -> units of the table's latent-y sd
                    crps.append(l.mean().item())
                    c = (yf_all[:, n_train:] == 1) | (yf_all[:, n_train:] == 2)
                    if c.any():
                        crps_c.append((l[c].sum().item(), c.sum().item()))
            rec = dict(crps=float(np.mean(crps)))
            if crps_c:
                rec['crps_cens'] = sum(a for a, _ in crps_c) / sum(b for _, b in crps_c)
            out[name + (('_on' if show else '') if name == 'clean' else ('_on' if show else '_off'))] = rec
    model.train()
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--micro_batch', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1.5e-5, help='backbone (Muon, AdamW-matched scale)')
    ap.add_argument('--cens_lr_mult', type=float, default=20.0, help='flag maps: lr x this, AdamW')
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--freeze_backbone', action='store_true')
    ap.add_argument('--p_ycens', type=float, default=0.6)
    ap.add_argument('--p_xcens', type=float, default=0.5)
    ap.add_argument('--spec_drop', type=float, default=0.3, help='1.0 = flags never shown (matched control R0)')
    ap.add_argument('--known', action='store_true', help='4-state flags: "known observed" != "no spec" (cens_model)')
    ap.add_argument('--row_adapter', type=int, default=0, help='hidden size of the gated row adapter (cens_model; 0 = off)')
    ap.add_argument('--y_clip', type=float, default=30.0)
    ap.add_argument('--max_tail', type=float, default=0.0, help='undo y censoring where latent sd > this x observed sd')
    ap.add_argument('--lat_clip', type=float, default=0.0, help='soft-clip the latent query targets to +-lat_clip (0: y_clip)')
    ap.add_argument('--val_every', type=int, default=250)
    ap.add_argument('--val_blocks', type=int, default=4)
    ap.add_argument('--log_every', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bank', default=os.path.join(ROOT, 'artifacts', 'bank_reg'))
    ap.add_argument('--n_blocks', type=int, default=2000)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--stop_at', type=int, default=0, help='testing: exit after the checkpoint at this step')
    args = ap.parse_args()
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    out_dir = os.path.join(ROOT, 'artifacts', 'pretrain', args.name)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, 'args.json'), 'w'), indent=1)

    cfg, sd = load_tabicl('regression')
    ot.set_quantiles(cfg['out_dim'])
    cens_cfg = dict(known=args.known, row_adapter=args.row_adapter)
    model = CensNanoTabICL(**cfg, **cens_cfg).to(dev)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith('cens_') for k in missing), (missing, unexpected)
    cens = [p for n, p in model.named_parameters() if n.startswith('cens_')]
    rest = [p for n, p in model.named_parameters() if not n.startswith('cens_')]
    if args.freeze_backbone:
        for p in rest:
            p.requires_grad_(False)
        opt = torch.optim.AdamW(cens, lr=args.lr * args.cens_lr_mult, betas=(0.9, 0.95), weight_decay=0.01)
    else:
        from muon import Muon
        opt = Muon([dict(params=rest, lr=args.lr, use_muon=True, weight_decay=0.0),
                    dict(params=cens, lr=args.lr * args.cens_lr_mult, use_muon=False, weight_decay=0.01)],
                   lr=args.lr, weight_decay=0.0, matched_adamw_rms=0.2, momentum=0.9, nesterov=True, ns_steps=5,
                   adamw_betas=(0.9, 0.95), adamw_eps=1e-8)
    params = [p for p in model.parameters() if p.requires_grad]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup) *
                                              (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / max(args.steps, 1))))))
    bank = ot.Bank(args.bank, args.n_blocks or None)
    start, last = 0, os.path.join(out_dir, 'last.ckpt')
    if args.resume and os.path.exists(last):
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
            v = validate(model, bank, dev, args.val_blocks, known=args.known)
            rec = dict(step=step, time=time.time() - t0, train_loss=float(np.mean(run_loss)) if run_loss else None, val=v)
            log.write(json.dumps(rec) + '\n'); log.flush()
            print(f"step {step:5d} {rec['time']:6.0f}s train {rec['train_loss'] or float('nan'):.4f} | " +
                  ' '.join(f"{k} {x['crps']:.4f}" + (f"/{x['crps_cens']:.3f}" if 'crps_cens' in x else '') for k, x in v.items()), flush=True)
            run_loss = []
            torch.save(dict(state_dict=model.state_dict(), model_cfg=cfg, cens_cfg=cens_cfg, arch='tabicl', mode='cens', task='regression',
                            step=step, args=vars(args), opt=opt.state_dict(), sched=sched.state_dict(),
                            np_rng=rng.bit_generator.state, torch_rng=g.get_state()), last + '.tmp')
            os.replace(last + '.tmp', last)
            if args.stop_at and step == args.stop_at:
                return
        if step == args.steps:
            break
        X, y, n_train = bank.batch(rng, dev, args.batch_size)
        B, R, C = X.shape
        y_des = sample_spec(B, 1, g, dev, args.p_ycens, (1.0, 1.0), (0.05, 0.5))
        x_des = sample_spec(B, C, g, dev, args.p_xcens, (0.2, 0.6), (0.05, 0.4))
        y_show = torch.rand(B, device=dev, generator=g) >= args.spec_drop
        x_show = torch.rand(B, device=dev, generator=g) >= args.spec_drop
        Xo, xf, yc, yf, yl, _, _ = prepare(X, y, n_train, g, y_des, x_des, y_show, x_show, args.known, args.max_tail)
        if args.y_clip > 0:  # as obs_train: a few prior tables have |z| > 100
            yc, yl = (args.y_clip * torch.tanh(t / args.y_clip) for t in (yc, yl))
        if args.lat_clip > 0:
            yl = args.lat_clip * torch.tanh(yl / args.lat_clip)
        y_test = yl[:, n_train:]
        opt.zero_grad(set_to_none=True)
        loss_sum, ok = 0.0, True
        for i in range(0, B, args.micro_batch):
            sl = slice(i, i + args.micro_batch)
            q = model(Xo[sl], yc[sl], xf[sl], yf[sl])
            l = ot.pinball(q, y_test[sl]).sum() / y_test.numel()
            if not torch.isfinite(l):
                ok = False
                break
            l.backward()
            loss_sum += l.item()
        if not ok:
            print(f'non-finite loss at step {step}, skipping', flush=True)
            continue
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        run_loss.append(loss_sum)
        if args.log_every and (step + 1) % args.log_every == 0 and (step + 1) % args.val_every:
            log.write(json.dumps(dict(step=step + 1, time=time.time() - t0, train_loss=float(np.mean(run_loss[-args.log_every:])))) + '\n')
            log.flush()


if __name__ == '__main__':
    main()
