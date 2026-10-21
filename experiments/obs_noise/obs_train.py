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
from obs_model import ObsNanoTabICL, huber_location, fill_missing  # noqa: E402
from tabicl_port import load_tabicl  # noqa: E402

K = 10
NOISES = ['gauss', 'student_t3', 'censored', 'cauchy', 'cauchy_hetero',
          'gauss_valhet', 'cauchy_valhet', 'censored_left', 'censored_left_heavy']  # same models as gen_data.py
NOISES_V2 = NOISES + ['fail_floor', 'fail_shift', 'cv_lognormal', 'lod_logistic', 'saturation', 'plate_effect']  # Exp 6
# v3 (Exp 8, after the LINCS check): + 'contam', a contaminated Gaussian (per table p_out ~ U(0, 0.08) of the replicates
# replaced by Gaussian outliers logU(3, 30) x wider), the shape real LINCS replicate noise has (t2-like tails from
# 2-3% outliers, otherwise ~Gaussian); with --noise_weights the mixture can be tilted toward it
NOISES_V3 = NOISES_V2 + ['contam']
CONTAM_ID = NOISES_V3.index('contam')
# v5 (Exp 9, real-shaped): + 'gauss_rowhet', Gaussian with a lognormal(0, 0.5) per-row scale (rows / samples of unequal
# quality); used with large noise, few wells and a Gaussian-dominated mixture (see --noise_weights / --scale_min)
NOISES_V5 = NOISES_V3 + ['gauss_rowhet']
ROWHET_ID = NOISES_V5.index('gauss_rowhet')
SCALE_MIN = 0.1  # --scale_min
SCALE_MAX = 1.0  # --scale_max: noise scale ~ logU(0.1, SCALE_MAX) x column sd (1.0 = the original prior)
SMALL_K_PROB = 0.0  # --small_k_prob: share of tables whose cells get U{2..5} replicates (0 = the original stream)
NOISE_W = None  # --noise_weights: sampling weights over TRAIN_NOISES (None = uniform, the original stream)


def sample_ids(shape, g, dev):
    if NOISE_W is None:
        return torch.randint(len(TRAIN_NOISES), shape, device=dev, generator=g)
    w = NOISE_W.to(dev)
    return torch.multinomial(w, math.prod(shape), replacement=True, generator=g).view(shape)
TRAIN_NOISES = NOISES  # set by --noise_set; v1 keeps the original training mixture and RNG stream
NOISE_SET = 'v1'  # v6: training batches from observe_v6 (the continuous prior) instead of observe
N_VAL = 16
VAL_CHUNK = 8
VAL_BLOCKS, VAL_NOISES = N_VAL, None  # --val_blocks / --val_noises: cheaper validation (e.g. for the 27M model)
MODEL_CFG = dict(max_classes=10, out_dim=10, embed_dim=32, col_num_blocks=2, row_num_blocks=2, icl_num_blocks=6,
                 col_nhead=4, row_nhead=4, icl_nhead=4, n_cls_rows=32)  # ~1M params (NOTES.md compute budget)


# ----- observation model (batched, on GPU) -----

LAYOUTS = ['table', 'col', 'group', 'colgroup']
P_PRECISE = 0.2  # col / colgroup layouts: share of exact (n = 1, noise-free) columns, e.g. covariates or metadata


def noise_layout(layout, B, R, C, n_groups, g, dev):
    """Heterogeneous observation models within a table (Exp 7). -> noise ids and scales broadcastable to (B, R, C, 1),
    precise-column mask (B, 1, C, 1) or None, per-row group (B, R) or None, per-row replicate cap (B, R, 1, 1) or None.
    col: own noise model + scale per column; group: rows split into n_groups groups with their own model + scale;
    colgroup: own model + scale per (group, column). Group layouts: each group's replicate count is capped at
    U{1..K} w.p. 0.5 (e.g. one lab runs singles, another triplicates)."""
    n_id = len(TRAIN_NOISES)
    logu = lambda *shape: torch.exp(torch.empty(*shape, device=dev).uniform_(math.log(SCALE_MIN), math.log(SCALE_MAX), generator=g))
    precise = gid = cap = None
    if layout == 'col':
        ids, sc = sample_ids((B, 1, C, 1), g, dev), logu(B, 1, C, 1)
    else:
        gid = torch.randint(n_groups, (B, R), device=dev, generator=g)
        shape = (B, n_groups, 1) if layout == 'group' else (B, n_groups, C)
        ids_g, sc_g = sample_ids(shape, g, dev), logu(*shape)
        idx = gid[..., None].expand(B, R, shape[-1])
        ids, sc = ids_g.gather(1, idx)[..., None], sc_g.gather(1, idx)[..., None]
        k_cap = torch.where(torch.rand(B, n_groups, device=dev, generator=g) < 0.5,
                            torch.randint(1, K + 1, (B, n_groups), device=dev, generator=g), 10 ** 6)
        cap = k_cap.gather(1, gid)[..., None, None]
    if layout in ('col', 'colgroup'):
        precise = torch.rand(B, 1, C, 1, device=dev, generator=g) < P_PRECISE
    return ids, sc, precise, gid, cap


def observe(X, g, noise=None, scale=None, p_clean=0.0, censor_q=None, lod_q=None, k_max=None, regimes=None, n_train=None,
            layout=None, n_groups=3, group_visible=False):
    """X: (B, R, C) clean. Returns reps (B, R, C, K), mask (B, R, C, K). Per table: clean w.p. p_clean, else a
    random noise model with scale ~ logU(0.1, 1) * sd(col) (or the given noise/scale); counts ~ U{1..K} per cell.
    layout (Exp 7, see noise_layout): None / 'table' = one model + scale per table (the original stream);
    group layouts with group_visible append the exact group index as an extra column (C + 1)."""
    B, R, C = X.shape
    dev = X.device
    K = k_max or globals()['K']  # tensor width = max replicates per cell in this batch
    col_sd = X.std(dim=1, keepdim=True).clamp(min=1e-12)[..., None]  # (B, 1, C, 1)
    if scale is None:
        scale = torch.exp(torch.empty(B, 1, 1, 1, device=dev).uniform_(math.log(SCALE_MIN), math.log(SCALE_MAX), generator=g))
    noise_ids = (torch.full((B,), NOISES_V5.index(noise), device=dev) if noise is not None
                 else sample_ids((B,), g, dev))
    nid = noise_ids[:, None, None, None]
    precise = cap = None
    if layout not in (None, 'table'):
        nid, scale, precise, gid, cap = noise_layout(layout, B, R, C, n_groups, g, dev)
        if gid is not None and group_visible:
            X = torch.cat([X, gid[..., None].to(X.dtype)], dim=-1)
            C += 1
            col_sd = X.std(dim=1, keepdim=True).clamp(min=1e-12)[..., None]
            if nid.shape[2] > 1:  # colgroup: per-column draws for the C - 1 data columns; the group column is exact
                nid = torch.cat([nid, torch.zeros_like(nid[:, :, :1])], dim=2)
                scale = torch.cat([scale, torch.ones_like(scale[:, :, :1])], dim=2)
            vis = torch.zeros(B, 1, C, 1, dtype=torch.bool, device=dev)
            vis[:, :, -1] = True
            precise = vis if precise is None else torch.cat([precise, precise.new_zeros(B, 1, 1, 1)], dim=2) | vis
    gauss = torch.randn(B, R, C, K, device=dev, generator=g)
    sq = torch.randn(B, R, C, K, 3, device=dev, generator=g) ** 2
    t3 = gauss / torch.sqrt(sq.sum(-1) / 3)
    t2 = gauss / torch.sqrt(sq[..., :2].sum(-1) / 2)
    cauchy = torch.tan(math.pi * (torch.rand(B, R, C, K, device=dev, generator=g) - 0.5))
    eps = torch.where(nid == 1, t3, torch.where((nid == 3) | (nid == 4) | (nid == 6), cauchy,
                                                torch.where(nid == 8, t2, gauss)))
    if (nid == CONTAM_ID).any():  # contaminated Gaussian (extra draws only when present: earlier streams unchanged)
        p_out = 0.08 * torch.rand(B, 1, 1, 1, device=dev, generator=g)
        out = torch.rand(B, R, C, K, device=dev, generator=g) < p_out
        wide = torch.exp(torch.empty(B, R, C, K, device=dev).uniform_(math.log(3), math.log(30), generator=g))
        eps = torch.where((nid == CONTAM_ID) & out, gauss * wide, eps)
    row_draw = torch.randn(B, R, 1, 1, device=dev, generator=g)  # one draw for both (the earlier streams are unchanged)
    row_scale = torch.where(nid == 4, torch.exp(row_draw), torch.where(nid == ROWHET_ID, torch.exp(0.5 * row_draw), 1.0))
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
    if lod_q is None:  # one uniform per table; per cell, the limit follows the cell's model (light / heavy)
        u_lod = 0.4 * torch.rand(B, 1, 1, device=dev, generator=g)
        q_lod_lt, q_lod_hv = 0.1 + u_lod, 0.4 + u_lod
    else:
        q_lod_lt = q_lod_hv = torch.full((B, 1, 1), lod_q, device=dev)
    lod_at = lambda q_: Xs.gather(1, (q_ * (R - 1)).round().long().expand(B, 1, C))[..., None]
    lod = torch.where(nid == 8, lod_at(q_lod_hv), lod_at(q_lod_lt))
    reps = torch.where((nid == 7) | (nid == 8), torch.maximum(reps, lod), reps)
    if (nid >= len(NOISES)).any():  # Exp 6 modes (only drawn with --noise_set v2 or when evaluating them)
        reps = observe_v2(X, reps, nid, col_sd, scale, eps, g)
    counts = torch.randint(1, K + 1, (B, R, C, 1), device=dev, generator=g)
    if SMALL_K_PROB > 0 and regimes is not None:  # v4, training batches only (validation keeps its fixed design):
        # a share of tables with few replicates per cell, U{2..5} (as in real assays, 3-5 wells)
        small = torch.rand(B, 1, 1, 1, device=dev, generator=g) < SMALL_K_PROB
        counts = torch.where(small, torch.randint(2, 6, (B, R, C, 1), device=dev, generator=g), counts)
    if regimes is not None:  # per-table count regime (Exp 5); None keeps the original per-cell draw (and RNG stream)
        counts = count_regimes(counts, regimes, n_train, K, g)
    clean = (torch.rand(B, device=dev, generator=g) < p_clean)[:, None, None, None]
    if cap is not None:  # per-group replicate cap
        counts = torch.minimum(counts, cap)
    if precise is not None:
        clean = clean | precise
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


def observe_v2(X, reps, nid, col_sd, scale, eps, g):
    """GPU versions of gen_data.py's Exp 6 modes (same constants), overriding reps for tables with those noise ids.
    eps is the table's Gaussian draw (B, R, C, K); scale = base noise scale (B or 1, 1, 1, 1) in col-sd units."""
    from gen_data import (FAIL_P_MAX, FAIL_GOOD_MULT, FAIL_SHIFT, CV_LN, LOD_THETA_Q, LOD_SLOPE, LOD_FLOOR_Q, ULOQ_Q,
                          PLATE_SD, PLATE_COL_SD)
    B, R, C, K = eps.shape
    dev = X.device
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


# ----- v6: continuous replicate prior (Exp 10) -----
# Instead of one named noise model per table, every table draws its own observation process from continuous families
# and the components combine freely. P6 = probability that a table has each component (validation settings override
# them, see V6_SETTINGS); within a component every strength is drawn from a continuous range.
P6 = dict(p_tail=0.5, p_contam=0.35, p_rowhet=0.4, p_valhet=0.35, p_colhet=0.7, p_well=0.4, p_plate=0.3, p_fail=0.2,
          p_lod=0.35, p_uloq=0.12, p_mcar=0.3, p_spec=0.5, p_precise=0.3, p_test1=0.1, sub=None, scale=None)
V6_SETTINGS = {  # fixed-seed validation draws: the default mixture and single components at full probability
    'v6': {},
    'v6_gauss': dict(p_tail=0, p_contam=0, p_rowhet=0, p_valhet=0, p_well=0, p_plate=0, p_fail=0, p_lod=0, p_uloq=0, p_mcar=0),
    'v6_heavy': dict(p_tail=1, p_contam=1),
    'v6_paired': dict(p_well=1, p_plate=1),
    'v6_fail': dict(p_fail=1),
    'v6_mnar': dict(p_lod=1, sub=0, p_mcar=1),     # undetected replicates are absent (informative counts, n = 0 cells)
    'v6_lod_flag': dict(p_lod=1, sub=1, p_spec=1),  # undetected -> at the limit, flagged '<'
    'v6_lod_noflag': dict(p_lod=1, sub=1, p_spec=0),  # the same draws with the flags hidden
}
P6_TRAIN = dict(P6)  # --p6: the training mixture (validation keeps P6 + V6_SETTINGS, so runs stay comparable)
COUNT_DESIGN = [0.05, 0.2, 0.3, 0.15, 0.1, 0.1, 0.1]  # per table: n = 1, 2, 3, 4, 5, U{6..K} per table, U{1..K} per cell


def observe_v6(X, g, n_train, k_max=None, p_clean=0.0, **over):
    """X: (B, R, C) clean. -> reps (B, R, C, K), mask (B, R, C, K), flags (B, R, C, K) int64.
    reps: censored replicates reported AT their limit (or substituted, see 'sub'); mask: measured replicates, holes
    allowed, n = 0 cells allowed (all replicates missing); flags: 0 observed / not reported, 1 '<' (below a lower
    limit), 2 '>' (above an upper limit), shown only in tables with the spec (p_spec).
    Replicate index k is one physical well / run / sample of the row, the same for every column (paired tables)."""
    P = dict(P6, **over)
    B, R, C = X.shape
    dev = X.device
    K_ = k_max or K
    U = lambda *sh, lo=0.0, hi=1.0: torch.empty(*sh, device=dev).uniform_(lo, hi, generator=g)
    logU = lambda *sh, lo, hi: torch.exp(U(*sh, lo=math.log(lo), hi=math.log(hi)))
    N = lambda *sh: torch.randn(*sh, device=dev, generator=g)
    on = lambda key: U(B, 1, 1, 1) < P[key]  # (B, 1, 1, 1): table has this component
    col_sd = X.std(dim=1, keepdim=True).clamp(min=1e-12)[..., None]  # (B, 1, C, 1)
    Xs = X.sort(dim=1).values
    col_q = lambda q: Xs.gather(1, (q.clamp(0, 1) * (R - 1)).round().long().expand(B, 1, C))[..., None]  # q: (B, 1, C)
    # noise scale: per table logU(0.05, 2) x column sd, columns spread around it (p_colhet)
    sig = logU(B, 1, 1, 1, lo=0.05, hi=2.0) if P['scale'] is None else torch.full((B, 1, 1, 1), float(P['scale']), device=dev)
    sig = sig * torch.exp(on('p_colhet') * U(B, 1, 1, 1) * N(B, 1, C, 1)).clamp(max=4.0)
    # tails: eps = gauss / |gauss'|^a, a in [0, 1]: Gaussian (a = 0) ... Cauchy (a = 1), tail index ~ 1 / a
    a = on('p_tail') * U(B, 1, 1, 1) ** 1.5
    a = (a + 0.15 * N(B, 1, C, 1) * (a > 0)).clamp(0, 1)  # columns of a table share roughly the same tail
    eps = N(B, R, C, K_) / N(B, R, C, K_).abs().clamp(min=1e-4) ** a
    # contamination: a share p ~ U(0, 0.1) of replicates are outliers, logU(3, 30) x wider, symmetric or one-sided
    p_out = on('p_contam') * 0.1 * U(B, 1, 1, 1)
    out = U(B, R, C, K_) < p_out
    wide = logU(B, R, C, K_, lo=3.0, hi=30.0) * N(B, R, C, K_)
    side = torch.where(U(B, 1, 1, 1) < 0.3, torch.sign(N(B, 1, C, 1)), 0.0)  # one-sided outliers (sign per column)
    eps = torch.where(out, torch.where(side != 0, side * wide.abs(), wide), eps)
    # heteroscedasticity: per row (unequal sample quality) and with the value (intensity-dependent noise, either sign)
    row_scale = torch.exp(on('p_rowhet') * U(B, 1, 1, 1, hi=0.8) * N(B, R, 1, 1))
    beta = on('p_valhet') * U(B, 1, 1, 1, lo=-0.8, hi=0.8)
    zc = ((X - Xs[:, R // 2:R // 2 + 1]) / col_sd[..., 0]).clamp(-3, 3)[..., None]
    reps = X[..., None] + sig * col_sd * row_scale * torch.exp(beta * zc) * eps
    # paired effects: a well / sample effect shared by all columns of the row's replicate k (loadings per column), and a
    # run / plate effect shared by all rows' replicate k (per column + common part)
    omega = on('p_well') * U(B, 1, 1, 1)
    reps = reps + omega * sig * col_sd * U(B, 1, C, 1, lo=0.3, hi=1.5) * N(B, R, 1, K_)
    pi = on('p_plate') * U(B, 1, 1, 1, hi=0.8)
    reps = reps + pi * sig * col_sd * (0.7 * N(B, 1, C, K_) + 0.7 * N(B, 1, 1, K_))
    # replicate design (per table): mostly a fixed small n, sometimes varying by row; aligned across columns
    design = torch.multinomial(torch.tensor(COUNT_DESIGN, device=dev), B, replacement=True, generator=g)[:, None, None, None]
    n_tab = torch.where(design < 5, design + 1, torch.randint(min(6, K_), K_ + 1, (B, 1, 1, 1), device=dev, generator=g))
    n_row = (n_tab - torch.where(U(B, 1, 1, 1) < 0.3, torch.randint(0, 3, (B, R, 1, 1), device=dev, generator=g), 0)).clamp(min=1)
    n_cell = torch.where(design == 6, torch.randint(1, K_ + 1, (B, R, C, 1), device=dev, generator=g), n_row.expand(B, R, C, 1))
    test1 = (U(B, 1, 1, 1) < P['p_test1']) & (torch.arange(R, device=dev) >= n_train)[None, :, None, None]
    n_cell = torch.where(test1, 1, n_cell).clamp(max=K_)  # some tables: new samples measured once
    mask = torch.arange(K_, device=dev) < n_cell
    # failed wells: floor value, a large shift, or the well is lost; a failed well fails for all columns w.p. 0.5
    p_f = on('p_fail') * 0.15 * U(B, 1, 1, 1)
    failed = torch.where(U(B, 1, 1, 1) < 0.5, U(B, R, 1, K_) < p_f, U(B, R, C, K_) < p_f)
    f_kind = torch.randint(0, 3, (B, 1, 1, 1), device=dev, generator=g)
    floor = Xs[:, :1, :, None] - U(B, 1, C, 1, lo=0.5, hi=2.0) * col_sd
    shift = reps + torch.sign(N(B, 1, C, 1)) * U(B, 1, 1, 1, lo=2.0, hi=6.0) * col_sd
    reps = torch.where(failed & (f_kind == 0), floor, torch.where(failed & (f_kind == 1), shift, reps))
    mask = mask & ~(failed & (f_kind == 2))
    # missing at random: replicates lost independently
    mask = mask & ~(U(B, R, C, K_) < on('p_mcar') * 0.3 * U(B, 1, 1, 1))
    # lower detection limit (quantile per column), soft or sharp detection probability on the measured value;
    # undetected replicates: absent (sub 0: informative counts), at the limit (1, flagged '<' with the spec), or
    # substituted by a floor below the limit (2, never flagged: how many pipelines report non-detects)
    flags = torch.zeros(B, R, C, K_, dtype=torch.long, device=dev)
    lod_on = on('p_lod')
    lod = col_q(U(B, 1, 1, lo=0.02, hi=0.6) + 0.1 * N(B, 1, C))
    slope = logU(B, 1, 1, 1, lo=0.02, hi=1.0)
    undet = lod_on & (U(B, R, C, K_) > torch.sigmoid((reps - lod) / (slope * col_sd)))
    sub = (torch.multinomial(torch.tensor([0.35, 0.4, 0.25], device=dev), B, replacement=True, generator=g)
           if P['sub'] is None else torch.full((B,), int(P['sub']), device=dev))[:, None, None, None]
    lod_floor = lod - U(B, 1, C, 1, lo=0.2, hi=1.5) * col_sd
    mask = mask & ~(undet & (sub == 0))
    reps = torch.where(undet & (sub == 1), lod, torch.where(undet & (sub == 2), lod_floor, reps))
    flags = torch.where(undet & (sub == 1), 1, flags)
    # upper limit (saturation / ULOQ): reported at the limit, flagged '>' with the spec
    hi = col_q(U(B, 1, 1, lo=0.8, hi=0.99) + 0.05 * N(B, 1, C))
    sat = on('p_uloq') & (reps > hi)
    reps = torch.where(sat, hi, reps)
    flags = torch.where(sat, 2, flags)
    flags = torch.where(U(B, 1, 1, 1) < P['p_spec'], flags, 0)
    # exact columns (covariates / metadata, n = 1, no noise) and exact tables
    precise = on('p_precise') & (U(B, 1, C, 1) < 0.5 * U(B, 1, 1, 1))
    exact = precise | (U(B, 1, 1, 1) < p_clean)
    first = torch.arange(K_, device=dev) == 0
    reps = torch.where(exact, X[..., None].expand_as(reps), reps)
    mask = torch.where(exact, first.expand_as(mask), mask)
    flags = torch.where(exact, 0, flags) * mask
    return reps, mask, flags


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


NQ = 99  # regression: predicted quantile levels tau_k = k / (NQ + 1); the TabICLv2 head has 999 (set in main)
TAUS = torch.arange(1, NQ + 1) / (NQ + 1)
TASK = 'classification'  # set by main() / make_model(task=...)
ARCH = 'nano'  # 'nano' (~1M, MODEL_CFG) or 'tabicl' (27.55M, the released TabICLv2 weights via tabicl_port.py)


def model_cfg(task, arch=None):
    if (arch or ARCH) == 'tabicl':
        return load_tabicl(task)[0]
    return MODEL_CFG if task == 'classification' else dict(MODEL_CFG, max_classes=0, out_dim=NQ)


def make_model(mode, rep_cfg, task='classification', cfg=None):
    cfg = cfg or model_cfg(task)
    return ObsNanoTabICL(**cfg, **rep_cfg) if mode == 'obs' else NanoTabICLv2(**cfg)


LOC_PARAMS = ('rep_gate', 'rep_enc.loc_head', 'rep_alpha')
PRECISION = 'bf16'  # 'bf16' autocast (the ~1M nano runs) or 'fp32' with TF32 matmuls (TabICLv2: its training default)


def amp():
    return torch.autocast('cuda', dtype=torch.bfloat16, enabled=PRECISION == 'bf16')


def set_quantiles(nq):  # regression head size: 99 (nano) or 999 (TabICLv2); levels k / (nq + 1)
    global NQ, TAUS
    NQ, TAUS = nq, torch.arange(1, nq + 1) / (nq + 1)


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


def model_forward(model, mode, X, y_train, g, aux=False, **obs_kw):
    """-> logits, or (logits, auxiliary denoising loss) when aux (obs mode, model built with aux=True)."""
    if mode == 'clean':
        return model(X, y_train)
    flags = None
    if obs_kw.get('noise') in V6_SETTINGS or (NOISE_SET == 'v6' and obs_kw.get('noise') is None):  # v6 settings / training
        over = V6_SETTINGS[obs_kw['noise']] if obs_kw.get('noise') in V6_SETTINGS else P6_TRAIN
        reps, mask, flags = observe_v6(X, g, y_train.shape[1], k_max=obs_kw.get('k_max'), p_clean=obs_kw.get('p_clean', 0.0), **over)
        if mode != 'obs':
            reps, mask, _ = fill_missing(reps, mask, y_train.shape[1])
    else:
        reps, mask = observe(X, g, **obs_kw)
    if mode == 'mean':
        return model(torch.where(mask, reps, 0).sum(-1) / mask.sum(-1), y_train)
    if mode == 'median':  # median imputation (numpy-equivalent median)
        return model(masked_quantile(reps, mask, 0.5), y_train)
    if mode == 'huber':  # Huber imputation, column-pooled scale from the context rows
        return model(huber_location(reps, mask, y_train.shape[1]), y_train)
    if mode == 'stats':
        return model(summary_stats(reps, mask), y_train)
    out = model(reps, mask, y_train) if flags is None else model(reps, mask, y_train, flags=flags)
    if not aux:
        return out
    if reps.shape[2] > X.shape[2]:  # visible group column (exact) appended by observe
        X = torch.cat([X, reps[:, :, X.shape[2]:, 0]], dim=-1)
    return out, model.aux_loss(X)


# ----- data -----

class Bank:
    def __init__(self, path, n_blocks=None):
        self.path = path
        self._files = sorted(glob.glob(os.path.join(self.path, 'block_*.npz')))
        if n_blocks is not None:  # reproducibility: train on a fixed, complete bank prefix only (blocks are seeded by
            # index, so extending the bank leaves blocks 0..n_blocks-1 and every earlier run unchanged)
            assert len(self._files) >= n_blocks, f'bank has {len(self._files)} blocks, expected {n_blocks}'
            assert os.path.basename(self._files[n_blocks - 1]) == f'block_{n_blocks - 1:05d}.npz', 'bank prefix incomplete'
            self._files = self._files[:n_blocks]

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
    files = bank.files()[:VAL_BLOCKS]
    settings = [('clean', None)] + [(nm, nm) for nm in (VAL_NOISES or TRAIN_NOISES)]
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
                with amp():
                    if noise is None and mode != 'clean':
                        reps = X[..., None].expand(*X.shape, K)
                        mask = (torch.arange(K, device=dev) < 1).expand(*X.shape, K)
                        if mode in ('mean', 'median', 'huber'):
                            logits = model(X, y[:, :n_train])
                        elif mode == 'stats':
                            logits = model(summary_stats(reps, mask), y[:, :n_train])
                        else:
                            logits = model(reps, mask, y[:, :n_train])
                    elif noise is None:
                        logits = model(X, y[:, :n_train])
                    else:
                        m = 'mean' if mode == 'clean' else mode  # clean model sees noisy cell means
                        fixed = {} if noise in V6_SETTINGS else dict(  # named modes: fixed scale / limits
                            scale=torch.tensor(0.5, device=dev), censor_q=(0.1, 0.9),
                            lod_q={'censored_left': 0.3, 'censored_left_heavy': 0.6}.get(noise))
                        logits = model_forward(model, m, X, y[:, :n_train], g, noise=noise, **fixed)
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


_DEV_CACHE = {}


@torch.no_grad()
def dev_real_eval(model, mode, datasets, dev):
    """Mean accuracy (points) over the real replicate tasks of `datasets` (real_data.py npz files), all replicates,
    1 pass; regression: -100 x mean CRPS (context-sd units), so higher is better and a kill margin of 0.5 = 0.005 CRPS.
    Used only as a kill switch (--dev_real), never to select a checkpoint."""
    from real_eval import task_inputs  # (imports this module under its own name; only task_inputs is used)
    model.eval()
    accs = []
    for ds in datasets:
        for f in sorted(glob.glob(os.path.join(ROOT, 'artifacts', 'real', ds, '*.npz'))):
            z = _DEV_CACHE.setdefault(f, dict(np.load(f)))
            tr, te, y = z['train_idx'], z['test_idx'], z['y']
            order = np.concatenate([tr, te])
            inp = task_inputs(z, mode, 'all', order, dev)
            if TASK == 'regression':
                from reg_metrics import reg_metrics
                ys = (y - y[tr].mean()) / (y[tr].std() + 1e-8)
                with amp():
                    q = model(*inp, torch.tensor(ys[tr], dtype=torch.float32, device=dev)[None])[0].float().cpu().numpy()
                q = np.sort(q, 1)
                accs.append(-reg_metrics(q[:, 9::10] if q.shape[1] == 999 else q, ys[te])[0]['crps'])
                continue
            n_cls = int(y.max()) + 1
            with amp():
                logits = model(*inp, torch.tensor(y[tr], device=dev)[None])
            accs.append(float((logits[0, :, :n_cls].float().argmax(-1).cpu().numpy() == y[te]).mean()))
    model.train()
    torch.cuda.empty_cache()
    return 100 * float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['clean', 'mean', 'median', 'huber', 'stats', 'obs'], required=True)
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
    ap.add_argument('--noise_set', choices=['v1', 'v2', 'v3', 'v5', 'v6'], default='v1',
                    help='v2 adds the Exp 6 failure / literature modes; v3 adds the contaminated Gaussian (contam)')
    ap.add_argument('--noise_weights', default='', help='json {noise: weight} over the noise set (others weight 1)')
    ap.add_argument('--scale_max', type=float, default=1.0, help='noise scale ~ logU(scale_min, scale_max) x column sd')
    ap.add_argument('--scale_min', type=float, default=0.1)
    ap.add_argument('--p6', default='', help='v6: json overrides of P6 (component probabilities), e.g. {"p_tail": 0.2}')
    ap.add_argument('--small_k_prob', type=float, default=0.0, help='share of tables with U{2..5} replicates per cell')
    ap.add_argument('--y_clip', type=float, default=0.0, help='regression: soft-clip standardized targets to +-y_clip in training')
    ap.add_argument('--layouts', default='', help='json weights over LAYOUTS (table,col,group,colgroup), drawn per batch; '
                                                  'empty = one model per table (original stream)')
    ap.add_argument('--group_vis_prob', type=float, default=0.5, help='group layouts: share of batches with the group column')
    ap.add_argument('--aux_w', type=float, default=0.0, help='obs: weight of the auxiliary denoising loss (needs rep_cfg aux)')
    ap.add_argument('--arch', choices=['nano', 'tabicl'], default='nano', help='tabicl: TabICLv2 (27.55M) backbone; '
                    'without --init it starts from the released checkpoint (tabicl_port.py)')
    ap.add_argument('--val_blocks', type=int, default=N_VAL, help='validation blocks (<= N_VAL)')
    ap.add_argument('--val_noises', default='', help='comma-separated validation noise models (default: the training set)')
    ap.add_argument('--log_every', type=int, default=0, help='also log the running train loss every N steps')
    ap.add_argument('--optim', choices=['adamw', 'muon'], default='adamw', help='muon: backbone with TabICLv2\'s Muon')
    ap.add_argument('--muon_wd', type=float, default=0.0, help='Muon weight decay (tabicl.train default 0)')
    ap.add_argument('--dev_real', default='', help='comma-separated real dev datasets (e.g. L1000_dose,CellPainting_dose): '
                    'evaluated at step 0 and every --dev_every steps; KILL SWITCH ONLY (no checkpoint selection)')
    ap.add_argument('--dev_every', type=int, default=500)
    ap.add_argument('--dev_kill_margin', type=float, default=0.5, help='stop if dev accuracy < step 0 - margin (points)')
    ap.add_argument('--dev_kill_until', type=int, default=1000, help='the kill rule applies at dev checks up to this step')
    ap.add_argument('--gate_lr_mult', type=float, default=1.0, help='lr multiplier for the location parameters (per-cell '
                    'gate rep_gate, location scores rep_enc.loc_head): both start at zero and gate each other\'s gradient')
    ap.add_argument('--precision', choices=['bf16', 'fp32'], default=None,
                    help='default: fp32 (+TF32 matmuls) for --arch tabicl, bf16 autocast for nano. bf16 costs the released '
                         'TabICLv2 2-4 accuracy points (e.g. clean 0.742 -> 0.720)')
    args = ap.parse_args()
    dev = 'cuda'
    global TASK, TRAIN_NOISES, ARCH, VAL_BLOCKS, VAL_NOISES, PRECISION, NOISE_SET
    TASK, ARCH = args.task, args.arch
    PRECISION = args.precision or ('fp32' if ARCH == 'tabicl' else 'bf16')
    if PRECISION == 'fp32':  # TF32 tensor-core matmuls, as tabicl.train sets (10-bit mantissa vs bf16's 7)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    VAL_BLOCKS = min(args.val_blocks, N_VAL)
    VAL_NOISES = args.val_noises.split(',') if args.val_noises else None
    if ARCH == 'tabicl' and TASK == 'regression':
        set_quantiles(load_tabicl('regression')[0]['out_dim'])
    NOISE_SET = args.noise_set
    TRAIN_NOISES = {'v1': NOISES, 'v2': NOISES_V2, 'v3': NOISES_V3, 'v5': NOISES_V5, 'v6': list(V6_SETTINGS)}[args.noise_set]
    regimes = json.loads(args.regimes) if args.regimes else None
    layouts = (lambda w: np.asarray(w) / np.sum(w))(json.loads(args.layouts)) if args.layouts else None
    use_aux = args.aux_w > 0
    assert not use_aux or (args.mode == 'obs' and json.loads(args.rep_cfg).get('aux')), '--aux_w needs obs + rep_cfg aux'
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
    for k, v in rep_cfg.pop('_args', {}).items():  # training-argument overrides carried in the branch config (screens)
        assert hasattr(args, k), k
        setattr(args, k, v)
    global NOISE_W, SCALE_MAX, SCALE_MIN, SMALL_K_PROB
    SCALE_MAX, SCALE_MIN, SMALL_K_PROB = args.scale_max, args.scale_min, args.small_k_prob
    NOISE_SET = args.noise_set
    TRAIN_NOISES = {'v1': NOISES, 'v2': NOISES_V2, 'v3': NOISES_V3, 'v5': NOISES_V5, 'v6': list(V6_SETTINGS)}[args.noise_set]  # (after the overrides)
    if args.p6:  # training draws only: validation settings are defined relative to the default P6 (V6_SETTINGS)
        P6_TRAIN.update(json.loads(args.p6) if isinstance(args.p6, str) else args.p6)
        assert set(P6_TRAIN) == set(P6), args.p6
    if args.noise_weights:
        w = json.loads(args.noise_weights) if isinstance(args.noise_weights, str) else args.noise_weights
        assert set(w) <= set(TRAIN_NOISES), w
        NOISE_W = torch.tensor([float(w.get(nm, 1.0)) for nm in TRAIN_NOISES])
    model = make_model(args.mode, rep_cfg, args.task).to(dev)
    if args.init or ARCH == 'tabicl':
        sd = torch.load(args.init, map_location=dev)['state_dict'] if args.init else load_tabicl(TASK)[1]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith('rep_') for k in missing), (missing, unexpected)
    if args.freeze_backbone:  # no weight gradients for the backbone at all (saves memory and backward time)
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith('rep_'))
    params = [p for n, p in model.named_parameters() if not args.freeze_backbone or n.startswith('rep_')]
    rep = [p for n, p in model.named_parameters() if n.startswith('rep_')]
    rest = [p for n, p in model.named_parameters() if not n.startswith('rep_') and not args.freeze_backbone]
    groups = [g for g in [dict(params=rest, lr=args.lr), dict(params=rep, lr=args.lr * args.rep_lr_mult)] if g['params']]
    if args.rep_lr_mult == 1.0:  # single group == the original optimizer layout (keeps old checkpoints resumable)
        groups = params
    loc = [p for n, p in model.named_parameters() if n.startswith(LOC_PARAMS)]
    if args.gate_lr_mult != 1.0 and loc:  # location learning (per-cell gate + attention scores) at its own lr
        ids = {id(p) for p in loc}
        rep = [p for p in rep if id(p) not in ids]
        groups = [g for g in [dict(params=rest, lr=args.lr), dict(params=rep, lr=args.lr * args.rep_lr_mult),
                              dict(params=loc, lr=args.lr * args.rep_lr_mult * args.gate_lr_mult)] if g['params']]
    if args.optim == 'muon':  # TabICLv2's own optimizer for the backbone (tabicl.train: one Muon group for all its
        # parameters, matched_adamw_rms 0.2, Nesterov, momentum beta1, wd 0); the replicate branch stays on AdamW
        # (Muon's internal AdamW, our betas / wd), as in stage A
        from muon import Muon
        groups = [g for g in [dict(params=rest, lr=args.lr, use_muon=True, weight_decay=args.muon_wd),
                              dict(params=rep, lr=args.lr * args.rep_lr_mult, use_muon=False, weight_decay=0.01)] +
                  ([dict(params=loc, lr=args.lr * args.rep_lr_mult * args.gate_lr_mult, use_muon=False, weight_decay=0.01)]
                   if args.gate_lr_mult != 1.0 and loc else [])
                  if g['params']]
        opt = Muon(groups, lr=args.lr, weight_decay=args.muon_wd, matched_adamw_rms=0.2, momentum=0.9, nesterov=True,
                   ns_steps=5, adamw_betas=(0.9, 0.95), adamw_eps=1e-8)
    else:
        opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup) *
                                              (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / max(args.steps, 1))))))
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
            kill = False
            if args.dev_real and step % args.dev_every == 0:
                rec['dev_real'] = dev_real_eval(model, args.mode, args.dev_real.split(','), dev)
                prev = [json.loads(l) for l in open(os.path.join(out_dir, 'log.jsonl'))] if os.path.exists(os.path.join(out_dir, 'log.jsonl')) else []
                base0 = next((r['dev_real'] for r in prev if r.get('step') == 0 and 'dev_real' in r), rec['dev_real'] if step == 0 else None)
                if base0 is not None and 0 < step <= args.dev_kill_until and rec['dev_real'] < base0 - args.dev_kill_margin:
                    kill = True
                    rec['dev_kill'] = f"dev {rec['dev_real']:.2f} < step-0 {base0:.2f} - {args.dev_kill_margin}"
                print(f"dev_real step {step}: {rec['dev_real']:.2f} (step 0: {base0})" + ('  -> KILL' if kill else ''), flush=True)
            log.write(json.dumps(rec) + '\n'); log.flush()
            print(f"step {step:6d} {rec['time']:6.0f}s train {rec['train_loss'] or float('nan'):.4f} | " +
                  ' '.join(f"{k} {x['acc']:.3f}/{x['loss']:.3f}" for k, x in v.items()), flush=True)
            run_loss = []
            torch.save(dict(state_dict=model.state_dict(), model_cfg=model_cfg(args.task), arch=ARCH, rep_cfg=rep_cfg, mode=args.mode, task=args.task,
                            step=step, args=vars(args), opt=opt.state_dict(), sched=sched.state_dict(),
                            np_rng=rng.bit_generator.state, torch_rng=g.get_state()), last + '.tmp')
            os.replace(last + '.tmp', last)
            if args.stop_at and step == args.stop_at:
                return
            if kill:
                print('stopped by the dev kill switch', flush=True)
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
        if layouts is not None:  # heterogeneous observation models (Exp 7): one layout per batch
            lay = LAYOUTS[rng.choice(len(LAYOUTS), p=layouts)]
            obs_kw.update(layout=lay, n_groups=int(rng.integers(2, 5)), group_visible=bool(rng.random() < args.group_vis_prob))
        opt.zero_grad(set_to_none=True)
        mb = args.micro_batch or args.batch_size
        if obs_kw.get('k_max', K) > K:  # up-to-50-replicate batches: 4x smaller micro-batches (same effective batch)
            mb = max(1, mb // 4)
        loss_sum, ok = 0.0, True
        for i in range(0, args.batch_size, mb):  # gradient accumulation; loss = mean over all valid test cells
            with amp():
                logits = model_forward(model, args.mode, X[i:i + mb], y[i:i + mb, :n_train], g, aux=use_aux, **obs_kw)
            if use_aux:
                logits, l_aux = logits
            l = task_loss(logits, y_test[i:i + mb], y[i:i + mb, :n_train])[0][valid[i:i + mb]].sum() / valid.sum()
            if use_aux:  # per-micro-batch mean, weighted by the micro-batch's share of the batch
                l = l + args.aux_w * l_aux * (X[i:i + mb].shape[0] / args.batch_size)
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
        if args.log_every and (step + 1) % args.log_every == 0 and (step + 1) % args.val_every:
            log.write(json.dumps(dict(step=step + 1, time=time.time() - t0, train_loss=float(np.mean(run_loss[-args.log_every:])))) + '\n')
            log.flush()


if __name__ == '__main__':
    main()
