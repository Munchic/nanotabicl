import math, torch, torch.nn as nn, torch.utils.checkpoint
from model import NanoTabICLv2, get_mlp


class ReplicateEncoder(nn.Module):  # set encoder over the replicates of one cell -> (d_s + 1)-dim vector
    def __init__(self, d_s: int = 32, n_heads: int = 4, use_sab: bool = True, use_null: bool = True,
                 med_feat: bool = False, loc_head: bool = False, robust_feats: bool = False, pool: str = 'attn',
                 col_ctx: bool = False, n_sab: int = 1, col_tok: int = 0, flag_emb: bool = False, isab: int = 0):
        super().__init__()
        # isab > 0: the n_sab self-attention blocks over replicates are replaced by induced set attention blocks with
        # `isab` inducing points (linear in K: populations of hundreds of instances, SHAPE.md)
        self.isab_blocks = nn.ModuleList([ISAB(d_s, n_heads, isab) for _ in range(n_sab)]) if isab else None
        # flag_emb: user-supplied per-replicate censoring flags {'<' below a limit, '>' above a limit} (the value is
        # reported AT the limit), bias-free zero-init map of the one-hot added to every replicate embedding; unflagged
        # replicates (or flags=None) contribute exactly 0
        self.flag_emb = nn.Linear(2, d_s, bias=False) if flag_emb else None
        if flag_emb:
            nn.init.zeros_(self.flag_emb.weight)
        # pool: 'attn' = learned seed query + null key/value (default); 'v2s' = Vec2Summ-style moment summary of the
        # replicate embeddings (Li et al. 2025): LN(W [mean_k h_k, log(var_k h_k + eps)])
        self.pool = pool
        if pool == 'v2s':
            self.v2s_proj = nn.Linear(2 * d_s, d_s)
        self.n_heads, self.use_sab, self.use_null, self.med_feat = n_heads, use_sab, use_null, med_feat
        self.robust_feats = robust_feats
        # per-replicate embedding of (z, z - z_bar[, z - z_med][, robust residual, within-cell rank, limit-tie flag])
        self.phi = get_mlp(2 + med_feat + 3 * robust_feats, 2 * d_s, d_s)
        # col_ctx: separate per-replicate DeepSets embedding, averaged over a cell's replicates; the caller pools it
        # over the column's replicated context cells into a column token (see ObsNanoTabICL.cell_inputs)
        self.phi_col = get_mlp(2 + med_feat + 3 * robust_feats, 2 * d_s, d_s) if col_ctx else None
        self.loc_head = nn.Linear(d_s, 1) if loc_head else None  # per-replicate location score (zero-init: mean)
        if loc_head:
            nn.init.zeros_(self.loc_head.weight)
            nn.init.zeros_(self.loc_head.bias)
        if use_sab:  # one pre-LN masked self-attention block: replicates attend to their siblings
            self.sab_ln, self.sab_qkv, self.sab_out = nn.LayerNorm(d_s), nn.Linear(d_s, 3 * d_s), nn.Linear(d_s, d_s)
            self.sab_mlp_ln, self.sab_mlp = nn.LayerNorm(d_s), get_mlp(d_s, 2 * d_s, d_s)
        # n_sab > 1: further pre-LN self-attention blocks over the replicates (capacity for the 27M backbone)
        self.sab_extra = nn.ModuleList([SAB(d_s, n_heads) for _ in range(max(0, n_sab - 1))]) if use_sab else nn.ModuleList()
        # col_tok > 0: replicates cross-attend to the column's col_tok pooled tokens (zero-init output: step 0 unchanged)
        if col_tok:
            self.xa_ln, self.xa_q, self.xa_kv = nn.LayerNorm(d_s), nn.Linear(d_s, d_s), nn.Linear(d_s, 2 * d_s)
            self.xa_out = nn.Linear(d_s, d_s)
            nn.init.zeros_(self.xa_out.weight)
            nn.init.zeros_(self.xa_out.bias)
        self.pool_ln, self.pool_kv = nn.LayerNorm(d_s), nn.Linear(d_s, 2 * d_s)
        self.seed = nn.Parameter(0.02 * torch.randn(1, 1, d_s))  # single learned pooling query
        self.null_kv = nn.Parameter(0.02 * torch.randn(1, 1, 2 * d_s))  # learned null key/value (shrink to prior)
        self.out_ln = nn.LayerNorm(d_s)

    def heads(self, t):  # (M, L, d) -> (M, H, L, d/H)
        return t.unflatten(-1, (self.n_heads, -1)).transpose(1, 2)

    def feats(self, z: torch.Tensor, mask: torch.Tensor, tie: torch.Tensor | None = None):
        """-> per-replicate input features f (M, K, n_feat), n (M, 1)."""
        z = torch.where(mask, z, 0.0)
        n = mask.sum(-1, keepdim=True)
        z_bar = z.sum(-1, keepdim=True) / n
        feats = [z, z - z_bar]
        z_med = masked_median(z, mask)[..., None] if (self.med_feat or self.robust_feats) else None
        if self.med_feat:
            feats.append(z - z_med)
        if self.robust_feats:
            mad = masked_median((z - z_med).abs(), mask)[..., None]
            resid = (z - z_med) / (mad + 0.05)  # "how many robust SDs from the cell median"
            ranks = torch.where(mask, z, float('inf')).argsort(dim=-1, stable=True).argsort(dim=-1, stable=True)
            feats += [10 * torch.tanh(resid / 10), ranks.to(z.dtype) / (n - 1).clamp(min=1).to(z.dtype),
                      tie.to(z.dtype)]
        return torch.stack(feats, dim=-1), n

    def col_summary(self, z: torch.Tensor, mask: torch.Tensor, tie: torch.Tensor | None = None):
        """(M, d_s) per-cell DeepSets summary for the column context (col_ctx); pooled over cells by the caller."""
        f, n = self.feats(z, mask, tie)
        return (self.phi_col(f) * mask[..., None].to(f.dtype)).sum(1) / n.to(f.dtype)

    def forward(self, z: torch.Tensor, mask: torch.Tensor, tie: torch.Tensor | None = None, ctx_fn=None, ctx_tokens=None,
                flags: torch.Tensor | None = None, ctx_vec: torch.Tensor | None = None):
        # z: (M, K) standardized replicates (padding arbitrary), mask: (M, K) bool, every row has >= 2 valid entries
        # tie: (M, K) bool, replicate equals the column's extreme context value (robust_feats only)
        # flags: (M, K, 2) float one-hot of '<' / '>' (flag_emb only)
        # ctx_vec: (M, d_s) precomputed column-context term (= ctx_fn(col_summary(...)); lets the caller chunk cells)
        # returns (cell vector (M, d_s + 1), location scores (M, K) or None)
        f, n = self.feats(z, mask, tie)
        z = torch.where(mask, z, 0.0)
        h = self.phi(f)  # (M, K, d_s)
        if flags is not None and self.flag_emb is not None:
            h = h + self.flag_emb(flags.to(h.dtype))
        if ctx_vec is not None:
            h = h + ctx_vec[:, None].to(h.dtype)
        elif ctx_fn is not None:  # column context: (M, d_s) per-cell summary -> (M, d_s) additive term for every replicate
            u = (self.phi_col(f) * mask[..., None].to(h.dtype)).sum(1) / n.to(h.dtype)
            h = h + ctx_fn(u)[:, None].to(h.dtype)
        if ctx_tokens is not None:  # (M, T, d_s) column tokens: cross-attention, every replicate queries its column's tokens
            q = self.heads(self.xa_q(self.xa_ln(h)))
            k, v = self.heads(self.xa_kv(ctx_tokens.to(h.dtype))).chunk(2, dim=-1)
            h = h + self.xa_out(nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(-2))
        key_mask = mask[:, None, None, :]  # (M, 1, 1, K): attend only to real replicates
        if self.isab_blocks is not None:
            for blk in self.isab_blocks:
                h = blk(h, key_mask)
        elif self.use_sab:
            q, k, v = self.heads(self.sab_qkv(self.sab_ln(h))).chunk(3, dim=-1)
            a = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=key_mask)
            h = h + self.sab_out(a.transpose(1, 2).flatten(-2))
            h = h + self.sab_mlp(self.sab_mlp_ln(h))
            for blk in self.sab_extra:
                h = blk(h, key_mask)
        if self.pool == 'v2s':
            w = mask[..., None].to(h.dtype)
            mu = (h * w).sum(1) / n.to(h.dtype)
            var = (((h - mu[:, None]) ** 2) * w).sum(1) / n.to(h.dtype)
            pooled_v = self.out_ln(self.v2s_proj(torch.cat([mu, torch.log(var + 1e-4)], dim=-1)))
            vec = torch.cat([pooled_v, torch.log(n.to(pooled_v.dtype))], dim=-1)
            return vec, (self.loc_head(h).squeeze(-1) if self.loc_head is not None else None)
        kv = self.pool_kv(self.pool_ln(h))  # (M, K, 2 d_s)
        if self.use_null:
            kv = torch.cat([self.null_kv.expand(kv.shape[0], -1, -1), kv], dim=1)
            key_mask = torch.cat([key_mask.new_ones(key_mask.shape[:-1] + (1,)), key_mask], dim=-1)
        k, v = self.heads(kv).chunk(2, dim=-1)
        q = self.heads(self.seed.expand(kv.shape[0], -1, -1))
        pooled = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=key_mask).transpose(1, 2).flatten(-2)
        vec = torch.cat([self.out_ln(pooled[:, 0]), torch.log(n.to(pooled.dtype))], dim=-1)  # + log n (counting)
        return vec, (self.loc_head(h).squeeze(-1) if self.loc_head is not None else None)


class SAB(nn.Module):  # pre-LN masked self-attention block over a cell's replicates (same form as the encoder's first)
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.ln, self.qkv, self.out = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp_ln, self.mlp = nn.LayerNorm(d), get_mlp(d, 2 * d, d)

    def forward(self, h, key_mask):
        q, k, v = self.qkv(self.ln(h)).unflatten(-1, (self.n_heads, -1)).transpose(1, 2).chunk(3, dim=-1)
        a = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=key_mask)
        h = h + self.out(a.transpose(1, 2).flatten(-2))
        return h + self.mlp(self.mlp_ln(h))


class ISAB(nn.Module):  # pre-LN induced set attention block (Lee et al. 2019): m inducing points, linear in K
    def __init__(self, d: int, n_heads: int, m: int):
        super().__init__()
        self.n_heads = n_heads
        self.ind = nn.Parameter(0.02 * torch.randn(1, m, d))
        self.ln1, self.q1, self.kv1, self.o1 = nn.LayerNorm(d), nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.ln2, self.q2, self.kv2, self.o2 = nn.LayerNorm(d), nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.mlp_ln, self.mlp = nn.LayerNorm(d), get_mlp(d, 2 * d, d)

    def heads(self, t):
        return t.unflatten(-1, (self.n_heads, -1)).transpose(1, 2)

    def forward(self, h, key_mask):  # h (M, K, d), key_mask (M, 1, 1, K)
        hn = self.ln1(h)
        k, v = self.heads(self.kv1(hn)).chunk(2, dim=-1)
        ind = self.ind.expand(h.shape[0], -1, -1).to(h.dtype)
        a = nn.functional.scaled_dot_product_attention(self.heads(self.q1(ind)), k, v, attn_mask=key_mask)
        H = ind + self.o1(a.transpose(1, 2).flatten(-2))  # (M, m, d): summaries of the set
        k, v = self.heads(self.kv2(self.ln2(H))).chunk(2, dim=-1)
        a = nn.functional.scaled_dot_product_attention(self.heads(self.q2(hn)), k, v)
        h = h + self.o2(a.transpose(1, 2).flatten(-2))
        return h + self.mlp(self.mlp_ln(h))


class JointEncoder(nn.Module):
    """SHAPE.md joint path, for paired sets (instance k = one physical unit measured on every column): per instance,
    one pre-LN attention layer ACROSS its columns (co-expression), then attention pooling over the instances of each
    (row, column) -> (B, R, C, d_out), through a zero-init output map. Columns are anonymous (any C)."""
    def __init__(self, d: int, d_out: int, n_heads: int = 2):
        super().__init__()
        self.n_heads = n_heads
        self.phi = get_mlp(2, 2 * d, d)  # (z, observed)
        self.ln, self.qkv, self.o = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp_ln, self.mlp = nn.LayerNorm(d), get_mlp(d, 2 * d, d)
        self.pool_ln, self.pool_w = nn.LayerNorm(d), nn.Linear(d, 1)
        self.out = nn.Linear(d, d_out, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, z, mask):  # z, mask: (b, r, C, K) -> (b, r, C, d_out)
        b, r, C, K = z.shape
        zt, mt = z.permute(0, 1, 3, 2), mask.permute(0, 1, 3, 2)  # (b, r, K, C)
        h = self.phi(torch.stack([torch.where(mt, zt, 0.0), mt.to(z.dtype)], -1)).flatten(0, 2)  # (b r K, C, d)
        km = mt.flatten(0, 2)
        q, k, v = self.qkv(self.ln(h)).unflatten(-1, (self.n_heads, -1)).transpose(1, 2).chunk(3, dim=-1)
        a = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=(km | ~km.any(-1, keepdim=True))[:, None, None, :])
        h = h + self.o(a.transpose(1, 2).flatten(-2))
        h = h + self.mlp(self.mlp_ln(h))
        h = h.view(b, r, K, C, -1).transpose(2, 3)  # (b, r, C, K, d)
        s = self.pool_w(self.pool_ln(h)).squeeze(-1).float().masked_fill(~mask, float('-inf'))
        w = torch.nan_to_num(torch.softmax(s, -1)).to(h.dtype)
        return self.out((w[..., None] * h).sum(-2))


def col_quantile(v: torch.Tensor, q: float) -> torch.Tensor:  # (B, R, C) -> (B, 1, C), linear interpolation, sort-based
    srt = v.sort(dim=1).values
    pos = q * (v.shape[1] - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return srt[:, lo:lo + 1] + (pos - lo) * (srt[:, hi:hi + 1] - srt[:, lo:lo + 1])


def linear_median(v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """numpy-style (linear interpolation) median over valid entries of the last dim; same arithmetic as
    obs_train.masked_quantile(v, mask, 0.5), so median-start models equal median imputation at step 0."""
    srt = torch.where(mask, v, float('inf')).sort(dim=-1).values
    pos = 0.5 * (mask.sum(-1, keepdim=True) - 1).clamp(min=0).to(v.dtype)
    lo, hi = pos.floor().long(), pos.ceil().long()
    v_lo, v_hi = srt.gather(-1, lo), srt.gather(-1, hi)
    return (v_lo + (pos - lo) * (v_hi - v_lo)).squeeze(-1)


# 1 / median(|r - cell median|) for standard-normal cells of size n = 2..50 (experiments/obs_noise/huber.py)
HUBER_SCALE_FACTOR = [float('nan'), float('nan')] + [2.0807, 3.1831, 2.0578, 2.0429, 1.9019, 1.8123, 1.7781, 1.7174, 1.7026, 1.6642, 1.6575, 1.6337, 1.6269, 1.6064, 1.6036, 1.593, 1.5868, 1.5759, 1.5772, 1.5673, 1.5686, 1.5619, 1.5572, 1.5536, 1.5514, 1.5462, 1.5455, 1.5417, 1.5426, 1.5402, 1.5382, 1.5332, 1.5337, 1.5306, 1.5298, 1.5262, 1.5299, 1.526, 1.5271, 1.5235, 1.5233, 1.5213, 1.5217, 1.5187, 1.5197, 1.5191, 1.5168, 1.5158, 1.5154]
HUBER_C = 1.345  # 95% Gaussian efficiency


@torch.no_grad()
def huber_location(reps: torch.Tensor, mask: torch.Tensor, n_train: int, iters: int = 10) -> torch.Tensor:
    """(B, R, C, K) replicates + mask -> (B, R, C) Huber M-estimate per cell, scale pooled per column over the replicated
    CONTEXT cells (median of |replicate - cell median| x the Gaussian small-n factor of the cell size). About the mean
    under Gaussian noise (95% efficiency), about the median under outliers; n = 1 cells keep their value; columns without
    replicated context cells get the mean. Same estimator as experiments/obs_noise/huber.py (numpy)."""
    B, R, C, K = reps.shape
    x = torch.where(mask, reps, 0.0).double()
    n = mask.sum(-1)
    med = linear_median(x, mask)
    fac = torch.tensor(HUBER_SCALE_FACTOR + [HUBER_SCALE_FACTOR[-1]] * max(0, K + 1 - len(HUBER_SCALE_FACTOR)),
                       dtype=x.dtype, device=x.device)[n.clamp(max=len(HUBER_SCALE_FACTOR) - 1)]
    res = (x - med[..., None]).abs() * fac[..., None]
    ok = mask & (n >= 2)[..., None] & (torch.arange(R, device=x.device) < n_train)[None, :, None, None]
    flat = torch.where(ok, res, float('inf')).transpose(1, 2).reshape(B, C, -1)  # (B, C, R*K)
    cnt = ok.transpose(1, 2).reshape(B, C, -1).sum(-1)
    srt = flat.sort(-1).values
    lo, hi = ((cnt - 1).clamp(min=0) // 2), (cnt // 2).clamp(max=flat.shape[-1] - 1)
    sig = 0.5 * (srt.gather(-1, lo[..., None]) + srt.gather(-1, hi[..., None])).squeeze(-1)  # (B, C) pooled median
    sig = torch.where(cnt > 0, sig, float('inf'))[:, None, :, None]
    mu = med
    for _ in range(iters):
        w = torch.where(mask, (HUBER_C * sig / (x - mu[..., None]).abs().clamp(min=1e-12)).clamp(max=1.0), 0.0)
        mu = (w * x).sum(-1) / w.sum(-1).clamp(min=1e-12)
    mean = x.sum(-1) / n.clamp(min=1)
    mu = torch.where((sig[..., 0] <= 1e-12).expand_as(mu), med, mu)  # degenerate scale (ties): the median limit
    mu = torch.where(torch.isfinite(sig[..., 0]).expand_as(mu), mu, mean)
    return torch.where(n > 1, mu, mean).to(reps.dtype)


def pooled_scale(reps: torch.Tensor, mask: torch.Tensor, n_train: int) -> torch.Tensor:
    """(B, 1, C) pooled robust column scale from the replicated CONTEXT cells (the huber_location scale, in raw
    units): median over the column of |replicate - cell median| x the Gaussian small-n factor. inf where a column has
    no replicated context cells."""
    B, R, C, K = reps.shape
    x = torch.where(mask, reps, 0.0).double()
    n = mask.sum(-1)
    med = linear_median(x, mask)
    fac = torch.tensor(HUBER_SCALE_FACTOR + [HUBER_SCALE_FACTOR[-1]] * max(0, K + 1 - len(HUBER_SCALE_FACTOR)),
                       dtype=x.dtype, device=x.device)[n.clamp(max=len(HUBER_SCALE_FACTOR) - 1)]
    res = (x - med[..., None]).abs() * fac[..., None]
    ok = mask & (n >= 2)[..., None] & (torch.arange(R, device=x.device) < n_train)[None, :, None, None]
    flat = torch.where(ok, res, float('inf')).transpose(1, 2).reshape(B, C, -1)
    cnt = ok.transpose(1, 2).reshape(B, C, -1).sum(-1)
    srt = flat.sort(-1).values
    lo, hi = ((cnt - 1).clamp(min=0) // 2), (cnt // 2).clamp(max=flat.shape[-1] - 1)
    sig = 0.5 * (srt.gather(-1, lo[..., None]) + srt.gather(-1, hi[..., None])).squeeze(-1)  # (B, C)
    return torch.where(cnt > 0, sig, float('inf'))[:, None]  # (B, 1, C)


def fill_missing(reps: torch.Tensor, mask: torch.Tensor, n_train: int):
    """n = 0 cells (no measurement) -> one replicate at the column mean of the CONTEXT cell means (= the TabICLv2
    wrapper's NaN imputation). -> reps, mask, missing (B, R, C) bool or None (no n = 0 cell: inputs returned as is)."""
    n = mask.sum(-1)
    missing = n == 0
    if not missing.any():
        return reps, mask, None
    cm = torch.where(mask, reps, 0.0).sum(-1) / n.clamp(min=1)
    ok = (~missing[:, :n_train]).to(reps.dtype)
    fill = (cm[:, :n_train] * ok).sum(1, keepdim=True) / ok.sum(1, keepdim=True).clamp(min=1)  # (B, 1, C)
    first = (torch.arange(reps.shape[-1], device=reps.device) == 0) & missing[..., None]
    return torch.where(first, fill[..., None], reps), mask | first, missing


def masked_quantile_cell(v: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    """per-cell linear-interpolated quantile over the valid entries of the last dim (numpy 'linear'), sort-based."""
    srt = torch.where(mask, v, float('inf')).sort(dim=-1).values
    pos = q * (mask.sum(-1, keepdim=True) - 1).clamp(min=0).to(v.dtype)
    lo, hi = pos.floor().long(), pos.ceil().long()
    a, b = srt.gather(-1, lo), srt.gather(-1, hi)
    return (a + (pos - lo) * (b - a)).squeeze(-1)


def masked_median(v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # lower median over valid entries (last dim)
    srt = torch.where(mask, v, float('inf')).sort(dim=-1).values
    return srt.gather(-1, ((mask.sum(-1) - 1).clamp(min=0) // 2)[..., None]).squeeze(-1)


class ObsNanoTabICL(NanoTabICLv2):
    """NanoTabICLv2 + replicate-level cell encoder. Input: replicates (B, R, C, K) + validity mask (B, R, C, K).

    The backbone sees the cell means exactly as the stock model sees a single value. Cells with n >= 2 additionally
    get a set-encoder vector; vectors are grouped at the same column offsets as the backbone (0, 1, 3) and added to the
    token through a zero-initialized, bias-free linear map. So: step 0 == stock model, and n = 1 cells contribute 0.
    """
    def __init__(self, *args, d_s: int = 32, rep_heads: int = 4, use_sab: bool = True, use_null: bool = True,
                 center: str = 'mean', med_feat: bool = False, std: str = 'meansd', zclip: float = 0.0,
                 robust_feats: bool = False, single_emb: str = 'none', temp_head: bool = False, dist_emb: bool = False,
                 pool: str = 'attn', row_emb: bool = False, aux: bool = False, aux_center: bool = True,
                 col_ctx: bool = False, n_sab: int = 1, col_tok: int = 0, col_tok_target: bool = False,
                 gate_colstats: bool = False, flag_emb: bool = False, miss_emb: bool = False, row_attn: bool = False,
                 isab: int = 0, sd_cols: bool = False, joint: int = 0, sd_learn: int = 0, backbone_loc: str = '',
                 rep_emb: bool = True, summ: str = '',
                 **kwargs):
        super().__init__(*args, **kwargs)
        # Side information (Exp 10), both optional, zero-init, bias-free -> tables without it equal the model without it
        # at any weights. flag_emb: per-replicate censoring flags (forward(..., flags=)) into the set encoder, plus the
        # cell's share of '<' / '>' replicates mapped into the cell vector (reaches n = 1 cells too). miss_emb: learned
        # vector for n = 0 cells (no measurement). n = 0 cells are always filled (see fill_missing): the backbone gets
        # the context column mean, as the TabICLv2 wrapper does for a NaN cell.
        self.flag_emb, self.miss_emb = flag_emb, miss_emb
        # sd_cols (SHAPE.md V1): the backbone also sees each replicated column's per-cell replicate sd (0 for n = 1) as
        # an extra feature column, appended after the C value columns (zero extra embedding). Only columns with
        # replicated context cells get one, so an all-single table is the stock input exactly; step 0 == the
        # "mean + sd" wrapper input.
        self.sd_cols = sd_cols
        # joint > 0 (SHAPE.md V2): JointEncoder of width `joint` for paired tables (forward(..., paired=)); its output
        # (zero-init) is added to the cell vector of n > 1 cells only -> step 0 and all-single tables unchanged
        # sd_learn = S >= 1 (with sd_cols): S ADDITIONAL per-cell summary columns, zero-init linear readouts of the cell
        # vector (0 outside replicated columns / n = 1 cells) -> a learned, hook-free summary block for any backbone.
        # S = 1 reproduces SHAPE.md V3 plus the explicit sd column. Step 0 == sd_cols exactly.
        self.sd_learn = int(sd_learn)
        # rep_emb=False: Tier-0 head only (no additive embedding into the backbone -> a head that also ports to backbones
        # without an injection hook; x_c carries the explicit columns instead). Nothing else changes.
        self.rep_emb = rep_emb
        # summ='sketch': the explicit block also carries [median, q25, q75, min, max, log n] per replicated column, i.e.
        # step 0 == the 7-column sketch wrapper (the strongest hand arm on HAR / HVTN in the val references).
        self.summ = summ
        # backbone_loc='huber' (center 'mean' only): the backbone's value columns are the cells' Huber locations
        # (huber_location, column-pooled scale) instead of their means; n = 1 cells keep their value -> all-single tables
        # unchanged; with sd_cols, step 0 == the "Huber + sd" wrapper input (the best hand-built pair on cytometry)
        self.backbone_loc = backbone_loc
        if backbone_loc == 'learned_m':  # PLAN 2 step 1: column-aware per-replicate weights (IRLS), starts at the mean
            assert col_ctx, 'learned_m needs col_ctx (its column token)'
            self.loc_c0 = nn.Parameter(torch.tensor(math.log(math.e - 1.0)))  # softplus -> 1
            self.loc_w = nn.Sequential(nn.Linear(d_s + 2, 16), nn.Tanh(), nn.Linear(16, 1, bias=False))
            nn.init.zeros_(self.loc_w[-1].weight)
            self._loc_w = None
        if self.sd_learn:
            self.rep_sdcol = nn.Linear(d_s + 1, self.sd_learn, bias=False)
            nn.init.zeros_(self.rep_sdcol.weight)
        self._cell = None
        self.joint = joint
        if joint:
            self.rep_joint = JointEncoder(joint, d_s + 1)
        # aux: training-only denoising targets for replicated cells (the clean value is known in pretraining).
        # rep_aux: linear head on the cell vector -> quantiles AUX_TAUS of the clean z; the learned center is also
        # pulled toward the clean value. Not used in forward -> both guarantees are unaffected.
        self.aux, self.aux_center = aux, aux_center  # aux_center=False: quantile head only, no pull on the center
        self._aux_state = None
        # col_ctx (backlog A1): column token = mean over the column's replicated CONTEXT cells of the encoder's
        # phi_col summaries -> LayerNorm -> zero-init linear, added to every replicate embedding of the column's
        # replicated cells (a learned, column-level empirical Bayes: failure floors / shifts, limits, noise shape).
        # n = 1 cells are untouched and step 0 is unchanged.
        self.col_ctx = col_ctx
        # temp_head: per-query-row temperature T = exp(W2 tanh(W1 f_row)) (bias-free, W2 zero-init) from row reliability
        # features that are exactly 0 for all-single tables -> stock behaviour there is preserved exactly.
        # classification: logits / T; regression: quantiles spread around the median x T.
        self.temp_head = temp_head
        self.is_regression = type(self.y_embed_in) is nn.Linear  # ClassEmbedding (a Linear subclass) for classification
        # single_emb: learned vector for n = 1 cells, ONLY in columns whose context has n > 1 cells (all-single columns
        # and tables keep exactly zero -> stock). 'const': one vector; 'scaled': MLP(log column replicate spread).
        self.single_emb = single_emb
        # std: column standardization of replicates for the encoder: 'meansd' (mean / sd of the context cell centers)
        # or 'medmad' (median / max(1.4826 MAD, IQR / 1.349, 1e-3 sd)). zclip > 0: soft clip z -> zclip * tanh(z / zclip)
        self.std, self.zclip = std, zclip
        # per-cell value fed to the backbone (n = 1 -> x in all cases): 'mean' (= imputation, the spec), 'median'
        # (robust), 'attn' (learned attention-weighted location over replicates; zero-init scores -> starts at the mean)
        self.center = center
        # 'attn_med': median start. center = median + ALPHA_SCALE * rep_alpha * (attention center - median), rep_alpha
        # zero-init -> step 0 feeds the backbone exactly the (numpy, linear) cell median, i.e. step 0 == median
        # imputation; the gate gets a gradient at once and the attention scores learn as soon as it moves
        if center == 'attn_med':
            self.rep_alpha = nn.Parameter(torch.zeros(1))
        # 'attn_medg': the same median start with a PER-CELL gate g = rep_gate(cell vector), zero-init. A single global
        # gate stays ~0 because the noise families disagree (Cauchy wants the median, censoring a mean-like center);
        # per cell, each cell can choose. Step 0 == median imputation exactly.
        # 'attn_meang': the mirror image, mean start: center = mean + gate * (attention center - mean); step 0 == mean
        # imputation. Leaving the mean is triggered by outliers a cell can see in its own residuals, whereas leaving
        # the median toward the mean needs knowing the noise is Gaussian (not identifiable from 3 replicates).
        # (two gates for the mean start: with zero-init scores the attention center starts AT the mean, so a single gate
        # on (attention center - mean) would get no gradient (deadlock); the median term breaks it from step 0:
        # center = mean + g1 (median - mean) + g2 (attention center - mean))
        # 'attn_huberg': Huber start (huber_location, column-pooled scale): center = huber + g1 (median - huber) +
        # g2 (attention center - huber); step 0 == Huber imputation (about the mean under Gaussian noise, the median under
        # outliers: the strongest simple baseline on the LINCS dev tasks, +1.0 over the mean)
        # gate_colstats: the gate also sees N_COLSTATS column-level noise statistics of the context replicates (pooled
        # robust scale, outlier share, tail quantile, limit-tie share; column_noise_stats), so that it can keep the Huber /
        # mean behaviour on Gaussian-like columns and robustify on heavy-tailed ones (a cell's own 2-5 replicates cannot
        # tell the two apart)
        self.gate_colstats = gate_colstats
        if center in ('attn_medg', 'attn_meang', 'attn_huberg'):
            self.rep_gate = nn.Linear(d_s + 1 + self.N_COLSTATS * gate_colstats, 1 if center == 'attn_medg' else 2)
            nn.init.zeros_(self.rep_gate.weight)
            nn.init.zeros_(self.rep_gate.bias)
        self.rep_enc = ReplicateEncoder(d_s=d_s, n_heads=rep_heads, use_sab=use_sab, use_null=use_null,
                                        med_feat=med_feat, loc_head=center in ('attn', 'attn_med', 'attn_medg', 'attn_meang', 'attn_huberg'), robust_feats=robust_feats, pool=pool,
                                        col_ctx=col_ctx, n_sab=n_sab, col_tok=col_tok, flag_emb=flag_emb, isab=isab)
        if flag_emb:
            self.rep_flag_cell = nn.Linear(2, d_s + 1, bias=False)
            nn.init.zeros_(self.rep_flag_cell.weight)
        if miss_emb:
            self.rep_missing = nn.Parameter(torch.zeros(d_s + 1))
        # col_tok: richer column context. Every CONTEXT cell of a column (replicated or single) is summarized as
        # [median z, log n, log(1 + MAD z), share of limit ties, n == 1] (+ its row's label with col_tok_target, context
        # rows only -> leak-free), embedded, and pooled by col_tok learned seed queries into col_tok tokens (several modes,
        # e.g. a floor cluster and the bulk); replicates cross-attend to them (ReplicateEncoder.xa_*, zero-init output)
        self.col_tok, self.col_tok_target = col_tok, col_tok_target
        if col_tok:
            self.rep_tok_phi = get_mlp(5, 2 * d_s, d_s)
            self.rep_tok_seed = nn.Parameter(0.02 * torch.randn(1, col_tok, d_s))
            self.rep_tok_ln, self.rep_tok_q, self.rep_tok_kv = nn.LayerNorm(d_s), nn.Linear(d_s, d_s), nn.Linear(d_s, 2 * d_s)
            self.rep_tok_out = nn.LayerNorm(d_s)
            if col_tok_target:
                n_cls = self.y_embed_in.in_features  # ClassEmbedding(max_classes, .) or Linear(1, .) for regression
                self.rep_tok_y = nn.Linear(n_cls, d_s, bias=False)
        if col_ctx:
            self.rep_col = nn.Sequential(nn.LayerNorm(d_s), nn.Linear(d_s, d_s))
            nn.init.zeros_(self.rep_col[-1].weight)
            nn.init.zeros_(self.rep_col[-1].bias)
        self.rep_proj = nn.Linear(self.feature_group_size * (d_s + 1), self.x_embed.out_features, bias=False)
        nn.init.zeros_(self.rep_proj.weight)
        # dist_emb: fixed probabilistic cell descriptors (no learned encoder needed to express uncertainty), for every
        # cell -- including n = 1 -- in columns whose context has replicated cells, via a zero-init bias-free map:
        #  empirical-Bayes Gaussian posterior: pooled column variance s2_j (median of (1.4826 MAD_cell)^2 over replicated
        #  context cells), shrunk cell variance v = (NU0 s2_j + (n-1) s2_cell) / (NU0 + n - 1), posterior sd sqrt(v / n)
        #  -> [log s_post, (median - mean) / s_post]; kernel mean embedding: mean_k cos(w_m z_k + b_m), m = 1..N_RFF
        self.dist_emb = dist_emb
        if dist_emb:
            gen = torch.Generator().manual_seed(1234)
            self.register_buffer('rff_w', torch.randn(self.N_RFF, generator=gen) * 1.5, persistent=False)
            self.register_buffer('rff_b', torch.rand(self.N_RFF, generator=gen) * 2 * math.pi, persistent=False)
            self.rep_dist = nn.Linear(2 + self.N_RFF, d_s + 1, bias=False)
            nn.init.zeros_(self.rep_dist.weight)
        self._s_post = None
        if temp_head:
            self.rep_temp = nn.Sequential(nn.Linear(7, 16, bias=False), nn.Tanh(), nn.Linear(16, 1, bias=False))
            nn.init.zeros_(self.rep_temp[-1].weight)
        # row_emb (NOTES.md Stage 2, implicit form): the same row reliability features, mapped (bias-free, zero-init)
        # into each row's ICL token, so in-context attention can learn to discount unreliable context rows
        self.row_emb = row_emb
        # row_attn (A2, Stage 2 proper): the same features -> a learned logit bias per CONTEXT row in every ICL attention
        # (bias-free, zero-init; soft-clipped to +-5), so queries can down-weight unreliable context rows. Features are 0
        # for all-single tables -> the bias is skipped there (stock exactly); step 0 == the model without it.
        self.row_attn = row_attn
        if row_attn:
            self.rep_row_attn = nn.Sequential(nn.Linear(7, 32, bias=False), nn.Tanh(), nn.Linear(32, 1, bias=False))
            nn.init.zeros_(self.rep_row_attn[-1].weight)
        if row_emb:
            self.rep_row = nn.Sequential(nn.Linear(7, 32, bias=False), nn.Tanh(),
                                         nn.Linear(32, self.out_ln.normalized_shape[0], bias=False))
            nn.init.zeros_(self.rep_row[-1].weight)
        if aux:
            self.rep_aux = nn.Linear(d_s + 1, len(self.AUX_TAUS))
        if single_emb != 'none':
            self.rep_single = get_mlp(1, 16, d_s + 1)  # input: log robust spread (scaled) or a constant 0 (const)
            nn.init.zeros_(self.rep_single[-1].weight)
            nn.init.zeros_(self.rep_single[-1].bias)

    def forward(self, reps: torch.Tensor, mask: torch.Tensor, y: torch.Tensor, flags: torch.Tensor | None = None,
                paired: torch.Tensor | None = None) -> torch.Tensor:
        """flags: optional (B, R, C, K) int {0 observed / not given, 1 '<', 2 '>'} (needs flag_emb to have an effect)."""
        n_train = y.shape[1]
        reps, mask, missing = fill_missing(reps, mask, n_train)
        if flags is not None and missing is not None:
            flags = torch.where(missing[..., None], 0, flags)
        x_c, extra_emb = self.cell_inputs(reps, mask, n_train=n_train, y=y, flags=flags, missing=missing, paired=paired)
        if self.sd_cols:
            n = mask.sum(-1)
            rep_col = (n[:, :n_train] > 1).any(1).any(0)  # (C,): columns with replicated context cells (any table)
            if rep_col.any():
                r = torch.where(mask, reps, 0.0)
                if self.backbone_loc == 'learned_m' and self._loc_w is not None:
                    w = self._loc_w.to(r.dtype)
                    mu = x_c  # the learned weighted location, the same value the backbone sees
                    var = ((w * (reps - mu[..., None]) ** 2).sum(-1)) / (w.sum(-1) - 1).clamp(min=1)
                    sd = torch.sqrt(var.clamp_min(1e-8))  # exact ties (censoring limits) -> 0 variance, finite grad
                    sd = torch.where(n > 1, sd, 0.0)
                else:
                    mu = r.sum(-1, keepdim=True) / n.clamp(min=1)[..., None]
                    var = (torch.where(mask, reps - mu, 0.0) ** 2).sum(-1) / (n - 1).clamp(min=1)
                    sd = torch.sqrt(var.clamp_min(1e-8))
                base_c = x_c.shape[-1]
                cols = [sd[..., rep_col].to(x_c.dtype)]
                if self.sd_learn:  # S learned summary columns, replicated cells only (n = 1 -> 0)
                    d = self.rep_sdcol(self._cell.to(self.rep_sdcol.weight.dtype)).to(x_c.dtype)
                    d = d * (n > 1).to(x_c.dtype)[..., None]
                    cols.append(d[:, :, rep_col].flatten(-2))
                if self.summ == 'sketch':  # hand sketch, raw scale, 0 for n = 1 cells (as the wrapper's sketch arm)
                    rf = torch.where(mask, reps, 0.0)
                    q25, med, q75 = (masked_quantile_cell(rf, mask, q) for q in (0.25, 0.5, 0.75))
                    lo = torch.where(mask, reps, float('inf')).amin(-1)
                    hi = torch.where(mask, reps, float('-inf')).amax(-1)
                    lg = torch.log(n.clamp(min=1).to(reps.dtype))
                    for t in (med, q25, q75, lo, hi, lg):
                        cols.append(t[..., rep_col].to(x_c.dtype))
                x_c = torch.cat([x_c] + cols, dim=-1)
                added = x_c.shape[-1] - base_c
                extra_emb = torch.cat([extra_emb, extra_emb.new_zeros(*extra_emb.shape[:2], added, extra_emb.shape[-1])], dim=2)
        feats = self.row_features(reps, mask, n_train) if (self.temp_head or self.row_emb or self.row_attn) else None
        row_extra = self.rep_row(feats.to(extra_emb.dtype)) if self.row_emb else None
        icl_bias = None
        if self.row_attn and bool(feats.abs().amax() > 0):
            icl_bias = 5 * torch.tanh(self.rep_row_attn(feats.to(extra_emb.dtype)).squeeze(-1) / 5)  # (B, R)
        out = super().forward(x_c, y, extra_emb=extra_emb, extra_row_emb=row_extra, icl_key_bias=icl_bias)
        if self.temp_head:
            log_t = self.rep_temp(feats[:, n_train:].to(out.dtype)).clamp(-3, 3)
            if self.is_regression:
                mid = out.shape[-1] // 2
                med = out[..., mid:mid + 1]
                out = out + (out - med) * torch.expm1(log_t)  # exactly out when log_t == 0
            else:
                out = out * torch.exp(-log_t)
        return out

    N_RFF, NU0 = 8, 3.0
    ENC_CHUNK = 1 << 20  # replicate values per encoder chunk (see cell_inputs)
    N_COLSTATS = 4

    @torch.no_grad()
    def column_noise_stats(self, z, mask, n, tie, n_train):
        """(B, 1, C, 4) column noise statistics from the replicated context cells, in the encoder's standardized units:
        log pooled robust scale (as huber_location), share of replicates > 3 pooled sd from their cell median, log 90th
        percentile of the standardized residuals, share of replicates tied at a column extreme (0 without robust_feats)."""
        B, R, C, K = z.shape
        zf = torch.where(mask, z, 0.0).float()
        med = linear_median(zf, mask)
        fac = torch.tensor(HUBER_SCALE_FACTOR + [HUBER_SCALE_FACTOR[-1]] * max(0, K + 1 - len(HUBER_SCALE_FACTOR)),
                           device=z.device)[n.clamp(max=len(HUBER_SCALE_FACTOR) - 1)]
        res = (zf - med[..., None]).abs() * fac[..., None]
        ok = mask & (n >= 2)[..., None] & (torch.arange(R, device=z.device) < n_train)[None, :, None, None]
        okf = ok.transpose(1, 2).reshape(B, C, -1)
        cnt = okf.sum(-1).clamp(min=1)
        srt = torch.where(okf, res.transpose(1, 2).reshape(B, C, -1), float('inf')).sort(-1).values
        q = lambda p: srt.gather(-1, ((p * (cnt - 1)).long())[..., None]).squeeze(-1)
        sig = q(0.5).clamp(min=1e-4)
        stdz = res / sig[:, None, :, None]
        out = (torch.where(ok, stdz > 3, False)).transpose(1, 2).reshape(B, C, -1).sum(-1) / cnt
        tie_s = ((tie & ok).transpose(1, 2).reshape(B, C, -1).sum(-1) / cnt) if tie is not None else torch.zeros_like(out)
        f = torch.stack([torch.log(sig), 10 * out, torch.log((q(0.9) / sig).clamp(min=1e-3)), tie_s], dim=-1)
        f = torch.where((okf.sum(-1) > 0)[..., None], f, 0.0)  # columns without replicated context cells: 0
        return f[:, None]  # (B, 1, C, 4)
    ALPHA_SCALE = 10.0  # gate parameterization: Adam moves rep_alpha by ~lr per step -> the gate by ~10 lr
    AUX_TAUS = (0.1, 0.25, 0.5, 0.75, 0.9)

    def aux_loss(self, x_clean: torch.Tensor) -> torch.Tensor:
        """x_clean: (B, R, C) clean values of the last forward's cells (training only). Mean over replicated cells of
        the pinball loss of rep_aux's quantiles + the absolute error of the learned center, both in the encoder's
        standardized (and soft-clipped) units."""
        multi, vec, x_c, mu, sd = self._aux_state
        self._aux_state = None
        if vec is None:
            return x_clean.new_zeros(())
        clip = (lambda t: self.zclip * torch.tanh(t / self.zclip)) if self.zclip > 0 else (lambda t: t)
        zt = clip((x_clean.float() - mu.float()) / sd.float())[multi]  # (M,)
        q = self.rep_aux(vec.to(self.rep_aux.weight.dtype)).float()
        taus = torch.tensor(self.AUX_TAUS, device=q.device)
        diff = zt[:, None] - q
        loss = torch.maximum(taus * diff, (taus - 1) * diff).mean()
        if self.center in ('attn', 'attn_med', 'attn_medg', 'attn_meang', 'attn_huberg') and self.aux_center:
            zc = clip((x_c.float() - mu.float()) / sd.float())[multi]
            loss = loss + (zc - zt).abs().mean()
        return loss

    def distributional(self, z, mask, n, n_train, dtype):
        """Fixed probabilistic descriptors (see __init__); returns the additive cell vector (zero outside replicated
        columns, and zero overall unless dist_emb). Also stores the posterior sd for the temperature head."""
        B, R, C = n.shape
        flag = (n[:, :n_train] > 1).any(dim=1, keepdim=True)  # (B, 1, C)
        zf = torch.where(mask, z, 0.0).float()
        med = masked_median(zf, mask)
        mean = zf.sum(-1) / n.clamp(min=1)
        mad = masked_median((zf - med[..., None]).abs(), mask)
        s2_cell = (1.4826 * mad) ** 2
        ok = (n > 1) & (torch.arange(R, device=z.device) < n_train)[None, :, None]
        srt = torch.where(ok, s2_cell, float('inf')).sort(dim=1).values
        s2_col = srt.gather(1, ((ok.sum(1, keepdim=True) - 1).clamp(min=0) // 2)).clamp(min=1e-4, max=1e4)  # (B, 1, C)
        nf = n.float()
        v = (self.NU0 * s2_col + (nf - 1).clamp(min=0) * s2_cell) / (self.NU0 + (nf - 1).clamp(min=0))
        s_post = torch.sqrt(v / nf.clamp(min=1))
        self._s_post = torch.where(flag, torch.log(s_post), 0.0)  # (B, R, C), 0 outside replicated columns
        if not self.dist_emb:
            return 0.0
        disagree = ((med - mean) / s_post).clamp(-5, 5)
        rff = (torch.cos(zf[..., None] * self.rff_w + self.rff_b) * mask[..., None]).sum(-2) / n.clamp(min=1)[..., None]
        desc = torch.cat([torch.log(s_post)[..., None], disagree[..., None], rff], dim=-1)
        out = self.rep_dist(desc.to(self.rep_dist.weight.dtype))
        return torch.where(flag[..., None], out, 0.0).to(dtype)

    def row_features(self, reps, mask, n_train):
        """(B, R, 5) per-row reliability features, all exactly 0 when every cell has n = 1: mean log n, share of
        single-sample cells in replicated columns, mean log(1 + within-cell MAD) (z units), and the context means of
        the first two (so a query can be compared with the context)."""
        n = mask.sum(-1)
        f1 = torch.log(n.clamp(min=1).float()).mean(-1)
        flag = (n[:, :n_train] > 1).any(dim=1, keepdim=True)
        f2 = ((n == 1) & flag).float().mean(-1)
        r = torch.where(mask, reps, 0.0)
        x_s = masked_median(r, mask)
        ctx = x_s[:, :n_train]
        sd = (col_quantile(ctx, 0.75) - col_quantile(ctx, 0.25)).clamp(min=1e-6)[..., None] / 1.349
        z = r / sd
        med = masked_median(z, mask)[..., None]
        f3 = torch.log1p(masked_median((z - med).abs(), mask)).float().mean(-1)
        g1 = f1[:, :n_train].mean(1, keepdim=True).expand_as(f1)
        g2 = f2[:, :n_train].mean(1, keepdim=True).expand_as(f2)
        f4 = self._s_post.mean(-1) if self._s_post is not None else torch.zeros_like(f1)  # row mean log posterior sd
        g4 = f4[:, :n_train].mean(1, keepdim=True).expand_as(f4)
        return torch.stack([f1, f2, f3, g1, g2, f4, g4], dim=-1)

    def column_tokens(self, z, mask, n, tie, n_train, y):
        """-> (B, C, col_tok, d_s) pooled tokens from all context cells of each column (see __init__)."""
        B, R, C, K = z.shape
        zf = torch.where(mask, z, 0.0).float()
        med = masked_median(zf, mask)
        mad = masked_median((zf - med[..., None]).abs(), mask)
        nf = n.clamp(min=1).float()
        tie_share = (tie & mask).sum(-1).float() / nf if tie is not None else torch.zeros_like(nf)
        f = torch.stack([med, torch.log(nf), torch.log1p(mad), tie_share, (n == 1).float()], dim=-1)[:, :n_train]
        u = self.rep_tok_phi(f.to(self.rep_tok_seed.dtype))  # (B, Rc, C, d_s)
        if self.col_tok_target:
            if self.is_regression:
                yc = y[:, :n_train, None].to(u.dtype)
            else:
                yc = nn.functional.one_hot(y[:, :n_train].long(), self.rep_tok_y.in_features).to(u.dtype)
            u = u + self.rep_tok_y(yc)[:, :, None]
        u = self.rep_tok_ln(u).transpose(1, 2).flatten(0, 1)  # (B*C, Rc, d_s)
        H = self.rep_enc.n_heads
        hd = lambda t: t.unflatten(-1, (H, -1)).transpose(1, 2)
        k, v = hd(self.rep_tok_kv(u)).chunk(2, dim=-1)
        q = hd(self.rep_tok_q(self.rep_tok_seed)).expand(k.shape[0], -1, -1, -1)
        tok = nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(-2)  # (B*C, col_tok, d_s)
        return self.rep_tok_out(tok).view(B, C, self.col_tok, -1)

    def cell_inputs(self, reps: torch.Tensor, mask: torch.Tensor, n_train: int, y: torch.Tensor | None = None,
                    flags: torch.Tensor | None = None, missing: torch.Tensor | None = None,
                    paired: torch.Tensor | None = None):
        """-> (per-cell value fed to the backbone (B, R, C), additive encoder embedding (B, R, C, embed_dim))."""
        n_batch, n_rows, n_cols, _ = reps.shape
        reps = torch.where(mask, reps, 0.0)
        n = mask.sum(-1)
        flag_oh = None
        if flags is not None and self.flag_emb:
            flag_oh = (nn.functional.one_hot(flags.long().clamp(0, 2), 3)[..., 1:] * mask[..., None]).float()  # (B, R, C, K, 2)
        x = reps.sum(-1) / n.clamp(min=1)  # cell means -> backbone, exactly like a single observed value
        # standardize replicates with the training rows' column statistics: of the cell means (the spec), or of the
        # cell medians for center in (median, attn) -- robust to cell means blown up by heavy-tailed replicates
        x_s = x if self.center == 'mean' else masked_median(reps, mask)
        ctx = x_s[:, :n_train]
        if self.std == 'medmad':
            mu = col_quantile(ctx, 0.5)
            mad = col_quantile((ctx - mu).abs(), 0.5)
            iqr = col_quantile(ctx, 0.75) - col_quantile(ctx, 0.25)
            sd = torch.maximum(torch.maximum(1.4826 * mad, iqr / 1.349), 1e-3 * ctx.std(dim=1, unbiased=False, keepdim=True))
            mu, sd = mu[..., None], sd[..., None] + 1e-8
        else:
            mu = ctx.mean(dim=1, keepdim=True)[..., None]
            sd = ctx.std(dim=1, unbiased=False, keepdim=True)[..., None] + 1e-8
        z = (reps - mu) / sd
        if self.zclip > 0:
            z = self.zclip * torch.tanh(z / self.zclip)
        tie = None
        if self.rep_enc.robust_feats:  # replicate sits at the column's extreme context value (e.g. a censoring limit)
            ctx_r = reps[:, :n_train]
            lo = torch.where(mask[:, :n_train], ctx_r, float('inf')).amin(dim=(1, 3), keepdim=True)  # (B, 1, C, 1)
            hi = torch.where(mask[:, :n_train], ctx_r, float('-inf')).amax(dim=(1, 3), keepdim=True)
            tie = ((reps == lo) | (reps == hi)) & mask

        multi = n > 1  # gate: only cells with >= 2 replicates go through the set encoder
        cell = z.new_zeros(n_batch, n_rows, n_cols, self.rep_proj.in_features // self.feature_group_size)
        x_c = x if self.center == 'mean' else x_s
        if self.backbone_loc == 'huber':
            x_c = huber_location(reps, mask, n_train).to(x.dtype)
        vec = None
        ctx_fn = None
        if self.col_ctx:
            use = multi & (torch.arange(n_rows, device=reps.device) < n_train)[None, :, None]  # replicated context cells

            def ctx_fn(u):
                full = u.new_zeros(n_batch, n_rows, n_cols, u.shape[-1])
                full[multi] = u
                w = use[..., None].to(u.dtype)
                tok = (full * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1)  # (B, 1, C, d_s)
                return self.rep_col(tok).expand(n_batch, n_rows, n_cols, -1)[multi]
        if multi.any():
            ctx_tokens = None
            if self.col_tok:
                tok = self.column_tokens(z, mask, n, tie, n_train, y)  # (B, C, T, d_s)
                ctx_tokens = tok[:, None].expand(n_batch, n_rows, n_cols, -1, -1)[multi]
            zm, mm, tm = z[multi], mask[multi], None if tie is None else tie[multi]
            fm = None if flag_oh is None else flag_oh[multi]
            ckpt = self.training and torch.is_grad_enabled()  # activation checkpointing: per-replicate activations
            # (cells x K x d_s, attention maps) are recomputed in backward instead of stored; same math
            run = (lambda fn, *a, **k: torch.utils.checkpoint.checkpoint(fn, *a, **k, use_reentrant=False)) if ckpt else \
                (lambda fn, *a, **k: fn(*a, **k))
            # cells in chunks whose replicate tensors hold <= ENC_CHUNK values (populations: K up to hundreds), each
            # chunk checkpointed separately -> peak memory of one chunk; the column context (needs all cells) is
            # computed in a cheap first pass. Identical math to one call.
            step = max(1, self.ENC_CHUNK // zm.shape[-1])
            sl = [slice(i, i + step) for i in range(0, zm.shape[0], step)]
            ctx_vec = None
            if ctx_fn is not None:
                u = torch.cat([run(self.rep_enc.col_summary, zm[c], mm[c], None if tm is None else tm[c]) for c in sl])
                ctx_vec = ctx_fn(u)
                if self.backbone_loc == 'learned_m':  # column token over ALL replicated cells (context + query, no labels)
                    full = u.new_zeros(n_batch, n_rows, n_cols, u.shape[-1])
                    full[multi] = u
                    w_all = multi[..., None].to(u.dtype)
                    loc_tok = (full * w_all).sum(1, keepdim=True) / w_all.sum(1, keepdim=True).clamp(min=1)  # (B,1,C,d_s)
            outs = [run(self.rep_enc, zm[c], mm[c], None if tm is None else tm[c],
                        ctx_tokens=None if ctx_tokens is None else ctx_tokens[c], flags=None if fm is None else fm[c],
                        ctx_vec=None if ctx_vec is None else ctx_vec[c]) for c in sl]
            vec = torch.cat([o[0] for o in outs])
            scores = None if outs[0][1] is None else torch.cat([o[1] for o in outs])
            cell[multi] = vec.to(cell.dtype)
            if self.center in ('attn_medg', 'attn_meang', 'attn_huberg'):
                gate_in = vec.to(self.rep_gate.weight.dtype)
                if self.gate_colstats:
                    cs = self.column_noise_stats(z, mask, n, tie, n_train).expand(n_batch, n_rows, n_cols, -1)[multi]
                    gate_in = torch.cat([gate_in, cs.to(gate_in.dtype)], dim=-1)
            if self.center in ('attn', 'attn_med', 'attn_medg', 'attn_meang', 'attn_huberg'):  # learned location
                w = torch.softmax(scores.float().masked_fill(~mask[multi], float('-inf')), dim=-1)
                att = (w * reps[multi]).sum(-1)
                x_c = x.clone()
                if self.center == 'attn_med':
                    med = linear_median(reps[multi], mask[multi])
                    att = med + self.ALPHA_SCALE * self.rep_alpha.float() * (att - med)
                elif self.center == 'attn_medg':
                    med = linear_median(reps[multi], mask[multi])
                    att = med + self.rep_gate(gate_in).float().squeeze(-1) * (att - med)
                elif self.center in ('attn_meang', 'attn_huberg'):
                    mu = (x[multi] if self.center == 'attn_meang' else huber_location(reps, mask, n_train)[multi]).float()
                    gt = self.rep_gate(gate_in).float()
                    med = linear_median(reps[multi], mask[multi])
                    att = mu + gt[:, 0] * (med - mu) + gt[:, 1] * (att - mu)
                x_c[multi] = att.to(x.dtype)
            if self.backbone_loc == 'learned_m':  # PLAN 2 step 1: weighted location from per-replicate residual weights
                x_c = x.clone()
                # pooled robust scale: a CONTEXT statistic, so it is not differentiated (the wrapper's preprocessing is
                # fitted the same way); columns without replicated context cells fall back to the context column sd
                s_c = pooled_scale(reps.detach(), mask, n_train).to(reps.dtype)
                cm = torch.where(mask, reps.detach(), 0.0).sum(-1) / n.clamp(min=1)
                ctx_sd = cm[:, :n_train].std(1, unbiased=False, keepdim=True).clamp(min=1e-3)  # (B, 1, C)
                s_c = torch.where(torch.isfinite(s_c), s_c, ctx_sd).clamp_min(0.05 * ctx_sd)
                mu = linear_median(torch.where(mask, reps, 0.0), mask)
                ln = torch.log(n.clamp(min=1).to(reps.dtype))[..., None, None]
                for _ in range(3):
                    q = ((reps - mu[..., None]).abs() / (s_c[..., None] + 1e-6)).clamp(max=50.0)[..., None]
                    feat = torch.cat([q, loc_tok[..., None, :].expand(n_batch, n_rows, n_cols, reps.shape[-1], -1),
                                      ln.expand(n_batch, n_rows, n_cols, reps.shape[-1], 1)], dim=-1)
                    w = torch.nn.functional.softplus(self.loc_c0 + self.loc_w(feat.to(self.loc_w[0].weight.dtype)).squeeze(-1))
                    w = torch.where(mask, w.to(reps.dtype), 0.0)
                    mu = (w * reps).sum(-1) / w.sum(-1).clamp(min=1e-3)
                x_c = torch.where(multi, mu, x_c)
                self._loc_w = w
        if self.aux and self.training:  # kept for aux_loss(); mu / sd: (B, 1, C) column standardization
            self._aux_state = (multi, vec, x_c, mu[..., 0], sd[..., 0])
        if self.dist_emb or self.temp_head:
            cell = cell + self.distributional(z, mask, n, n_train, cell.dtype)
        if self.single_emb != 'none':
            flag = (n[:, :n_train] > 1).any(dim=1, keepdim=True)  # (B, 1, C): column has replicated context cells
            use = (n == 1) & flag
            if use.any():
                if self.single_emb == 'scaled':  # column noise level: median within-cell MAD (z units), context cells
                    z_med = masked_median(z, mask)[..., None]
                    cell_mad = masked_median((z - z_med).abs(), mask)  # (B, R, C)
                    ok = (n > 1) & (torch.arange(n_rows, device=reps.device) < n_train)[None, :, None]
                    srt = torch.where(ok, cell_mad, float('inf')).sort(dim=1).values
                    idx = ((ok.sum(1, keepdim=True) - 1).clamp(min=0) // 2)
                    inp = torch.log(srt.gather(1, idx).clamp(min=1e-3, max=1e3))[..., None]  # (B, 1, C, 1)
                else:
                    inp = cell.new_zeros(n_batch, 1, n_cols, 1)
                e1 = self.rep_single(inp.to(cell.dtype)).to(cell.dtype)  # (B, 1, C, d_s + 1)
                cell = torch.where(use[..., None], e1.expand_as(cell), cell)
        if self.joint and paired is not None and bool(torch.as_tensor(paired).any()) and multi.any():
            pt = torch.as_tensor(paired, device=reps.device).reshape(-1)
            pt = pt.expand(n_batch) if pt.numel() == 1 else pt
            ckpt = self.training and torch.is_grad_enabled()
            rows = max(1, self.ENC_CHUNK // max(1, n_cols * z.shape[-1] * 4))
            jv = []
            for b_ in range(n_batch):
                if not bool(pt[b_]):
                    jv.append(cell.new_zeros(1, n_rows, n_cols, cell.shape[-1]))
                    continue
                parts = []
                for i in range(0, n_rows, rows):
                    a_ = (z[b_:b_ + 1, i:i + rows], mask[b_:b_ + 1, i:i + rows])
                    parts.append(torch.utils.checkpoint.checkpoint(self.rep_joint, *a_, use_reentrant=False) if ckpt
                                 else self.rep_joint(*a_))
                jv.append(torch.cat(parts, 1).to(cell.dtype))
            cell = cell + torch.cat(jv, 0) * multi[..., None].to(cell.dtype)
        if flag_oh is not None:  # share of '<' / '>' replicates of the cell (all cells, n = 1 included)
            cell = cell + self.rep_flag_cell(flag_oh.sum(-2) / n.clamp(min=1)[..., None].to(flag_oh.dtype)).to(cell.dtype)
        if missing is not None and self.miss_emb:
            cell = cell + missing[..., None].to(cell.dtype) * self.rep_missing.to(cell.dtype)
        self._cell = cell  # (B, R, C, d_s + 1): per-cell vectors (sd_learn reads them)
        idxs = torch.arange(n_cols, device=reps.device)
        grouped = torch.cat([cell[:, :, (idxs + (2 ** i - self.group_shift)) % n_cols]
                             for i in range(self.feature_group_size)], dim=-1)
        return x_c, (self.rep_proj(grouped) if self.rep_emb else grouped.new_zeros(*grouped.shape[:3], self.rep_proj.out_features))


if __name__ == '__main__':  # step 0 == stock model, and n = 1 everywhere == stock model at any weights
    torch.manual_seed(0)
    cfg = dict(max_classes=10, out_dim=10, embed_dim=32, col_num_blocks=2, row_num_blocks=2, icl_num_blocks=6,
               col_nhead=4, row_nhead=4, icl_nhead=4, n_cls_rows=32)
    stock, obs = NanoTabICLv2(**cfg), ObsNanoTabICL(**cfg)
    obs.load_state_dict(stock.state_dict(), strict=False)
    if '--col_tok' in __import__('sys').argv:
        obs = ObsNanoTabICL(**cfg, center='attn_medg', med_feat=True, std='medmad', zclip=10.0, robust_feats=True,
                            dist_emb=True, temp_head=True, col_ctx=True, n_sab=2, d_s=64, col_tok=4, col_tok_target=True)
        obs.load_state_dict(stock.state_dict(), strict=False)
    elif '--col_ctx' in __import__('sys').argv:
        obs = ObsNanoTabICL(**cfg, center='attn', med_feat=True, std='medmad', zclip=10.0, robust_feats=True,
                            dist_emb=True, temp_head=True, col_ctx=True)
        obs.load_state_dict(stock.state_dict(), strict=False)
    B, R, C, K, n_train = 2, 40, 5, 10, 24
    reps = torch.randn(B, R, C, K)
    mask = torch.arange(K) < torch.randint(1, K + 1, (B, R, C, 1))
    y = torch.randint(10, (B, n_train))
    xm = torch.where(mask, reps, 0).sum(-1) / mask.sum(-1)
    if obs.center in ('attn_med', 'attn_medg'):  # median start: step 0 == median imputation
        xm = linear_median(torch.where(mask, reps, 0), mask)
    print('step0 max|diff| =', (obs(reps, mask, y) - stock(xm, y)).abs().max().item())
    nn.init.normal_(obs.rep_proj.weight)
    for m_ in [getattr(obs, 'rep_col', None), getattr(obs, 'rep_dist', None), getattr(obs, 'rep_temp', None),
               getattr(obs.rep_enc, 'xa_out', None), getattr(obs, 'rep_gate', None)]:
        if m_ is not None:  # randomize every zero-init map: n = 1 tables must still equal stock
            [nn.init.normal_(p_) for p_ in m_.parameters()]
    m1 = torch.zeros_like(mask); m1[..., 0] = True
    print('n=1, trained U: max|diff| =', (obs(reps, m1, y) - stock(reps[..., 0], y)).abs().max().item())
    print('n>1, trained U: max|diff| =', (obs(reps, mask, y) - stock(xm, y)).abs().max().item(), '(should be > 0)')
    print('params: stock %.3fM, replicate branch %.3fM' % (sum(p.numel() for p in stock.parameters()) / 1e6,
          sum(p.numel() for n_, p in obs.named_parameters() if n_.startswith('rep_')) / 1e6))
    if '--side' in __import__('sys').argv:  # side information: flags + n = 0 cells
        side = ObsNanoTabICL(**cfg, center='attn', med_feat=True, std='medmad', zclip=10.0, robust_feats=True,
                             dist_emb=True, temp_head=True, col_ctx=True, n_sab=2, d_s=64, flag_emb=True, miss_emb=True)
        side.load_state_dict(stock.state_dict(), strict=False)
        holes = mask & (torch.rand(B, R, C, K) > 0.3)  # holes anywhere, some cells n = 0
        flags = torch.randint(0, 3, (B, R, C, K)) * (torch.rand(B, R, C, K) < 0.2)
        hm = torch.where(holes, reps, 0).sum(-1) / holes.sum(-1).clamp(min=1)
        miss = holes.sum(-1) == 0
        fill = (hm[:, :n_train] * ~miss[:, :n_train]).sum(1, keepdim=True) / (~miss[:, :n_train]).sum(1, keepdim=True)
        hm = torch.where(miss, fill, hm)
        print(f'side: {miss.float().mean():.1%} n=0 cells; step0 with flags + n=0 vs stock(mean, column-mean fill):',
              (side(reps, holes, y, flags=flags) - stock(hm, y)).abs().max().item())
        for m_ in [side.rep_proj, side.rep_col, side.rep_dist, side.rep_temp, side.rep_flag_cell, side.rep_enc.flag_emb]:
            [nn.init.normal_(p_) for p_ in m_.parameters()]
        nn.init.normal_(side.rep_missing)
        a = side(reps, mask, y)
        print('side, random weights: zero flags == no flags:', (side(reps, mask, y, flags=torch.zeros_like(flags)) - a).abs().max().item(),
              '| flags change output:', (side(reps, mask, y, flags=flags) - a).abs().max().item(), '(> 0)',
              '| n=1 + flags change output:', (side(reps, m1, y, flags=flags) - side(reps, m1, y)).abs().max().item(), '(> 0)')
    if '--row_attn' in __import__('sys').argv:  # learned context-row attention bias
        ra = ObsNanoTabICL(**cfg, center='attn', med_feat=True, std='medmad', zclip=10.0, robust_feats=True,
                           dist_emb=True, temp_head=True, col_ctx=True, row_attn=True)
        ra.load_state_dict(stock.state_dict(), strict=False)
        print('row_attn: step0 vs stock(mean):', (ra(reps, mask, y) - stock(torch.where(mask, reps, 0).sum(-1) / mask.sum(-1), y)).abs().max().item())
        for m_ in [ra.rep_proj, ra.rep_col, ra.rep_dist, ra.rep_temp]:
            [nn.init.normal_(p_) for p_ in m_.parameters()]
        a = ra(reps, mask, y)
        [nn.init.normal_(p_) for p_ in ra.rep_row_attn.parameters()]
        print('row_attn, random weights: n=1 vs stock:', (ra(reps, m1, y) - stock(reps[..., 0], y)).abs().max().item(),
              '| bias changes n>1 output:', (ra(reps, mask, y) - a).abs().max().item(), '(> 0)')
    if '--shape' in __import__('sys').argv:  # ISAB encoder + sd columns
        sh = ObsNanoTabICL(**cfg, center='mean', med_feat=True, std='medmad', zclip=10.0, robust_feats=True,
                           dist_emb=True, temp_head=True, col_ctx=True, isab=16, n_sab=2, sd_cols=True)
        sh.load_state_dict(stock.state_dict(), strict=False)
        mean = torch.where(mask, reps, 0).sum(-1) / mask.sum(-1)
        n_ = mask.sum(-1)
        sdv = torch.sqrt((torch.where(mask, reps - mean[..., None], 0) ** 2).sum(-1) / (n_ - 1).clamp(min=1))
        print('shape: step0 vs stock(mean, sd):', (sh(reps, mask, y) - stock(torch.cat([mean, sdv], -1), y)).abs().max().item())
        for m_ in [sh.rep_proj, sh.rep_col, sh.rep_dist, sh.rep_temp]:
            [nn.init.normal_(p_) for p_ in m_.parameters()]
        print('shape, random weights: n=1 vs stock:', (sh(reps, m1, y) - stock(reps[..., 0], y)).abs().max().item(),
              '| n>1 differs:', (sh(reps, mask, y) - stock(torch.cat([mean, sdv], -1), y)).abs().max().item(), '(> 0)')
    if '--joint' in __import__('sys').argv:  # joint path (paired sets)
        jt = ObsNanoTabICL(**cfg, center='mean', med_feat=True, std='medmad', zclip=10.0, robust_feats=True,
                           dist_emb=True, temp_head=True, col_ctx=True, isab=16, n_sab=2, sd_cols=True, joint=16)
        jt.load_state_dict(stock.state_dict(), strict=False)
        mean = torch.where(mask, reps, 0).sum(-1) / mask.sum(-1)
        n_ = mask.sum(-1)
        sdv = torch.sqrt((torch.where(mask, reps - mean[..., None], 0) ** 2).sum(-1) / (n_ - 1).clamp(min=1))
        pr = torch.tensor([True, True])
        print('joint: step0 (paired) vs stock(mean, sd):', (jt(reps, mask, y, paired=pr) - stock(torch.cat([mean, sdv], -1), y)).abs().max().item())
        for m_ in [jt.rep_proj, jt.rep_col, jt.rep_dist, jt.rep_temp, jt.rep_joint.out]:
            [nn.init.normal_(p_) for p_ in m_.parameters()]
        a = jt(reps, mask, y, paired=torch.tensor([False, False]))
        print('joint, random weights: n=1 paired vs stock:', (jt(reps, m1, y, paired=pr) - stock(reps[..., 0], y)).abs().max().item(),
              '| paired changes n>1:', (jt(reps, mask, y, paired=pr) - a).abs().max().item(), '(> 0)',
              '| unpaired == no joint:', (a - jt(reps, mask, y)).abs().max().item())
    if '--sd_learn' in __import__('sys').argv:
        sl = ObsNanoTabICL(**cfg, center='mean', med_feat=True, std='medmad', zclip=10.0, robust_feats=True,
                           dist_emb=True, temp_head=True, col_ctx=True, isab=16, n_sab=2, sd_cols=True, sd_learn=1)
        sl.load_state_dict(stock.state_dict(), strict=False)
        mean = torch.where(mask, reps, 0).sum(-1) / mask.sum(-1)
        n_ = mask.sum(-1)
        sdv = torch.sqrt((torch.where(mask, reps - mean[..., None], 0) ** 2).sum(-1) / (n_ - 1).clamp(min=1))
        zc = torch.zeros_like(sdv)  # S learned columns are 0 at step 0
        print('sd_learn: step0 vs stock(mean, sd, 0):', (sl(reps, mask, y) - stock(torch.cat([mean, sdv, zc], -1), y)).abs().max().item())
        nn.init.normal_(sl.rep_sdcol.weight)
        print('sd_learn, random readout: n=1 vs stock:', (sl(reps, m1, y) - stock(reps[..., 0], y)).abs().max().item(),
              '| n>1 differs:', (sl(reps, mask, y) - stock(torch.cat([mean, sdv, zc], -1), y)).abs().max().item(), '(> 0)')
    if '--learned_m' in __import__('sys').argv:  # PLAN 2 step 1: column-aware per-replicate weights
        lm = ObsNanoTabICL(**cfg, center='mean', med_feat=True, std='medmad', zclip=10.0, robust_feats=True,
                           dist_emb=True, temp_head=True, col_ctx=True, isab=16, n_sab=2, sd_cols=True, backbone_loc='learned_m')
        lm.load_state_dict(stock.state_dict(), strict=False)
        mean = torch.where(mask, reps, 0).sum(-1) / mask.sum(-1)
        n_ = mask.sum(-1)
        sdv = torch.sqrt((torch.where(mask, reps - mean[..., None], 0) ** 2).sum(-1) / (n_ - 1).clamp(min=1))
        d0 = (lm(reps, mask, y) - stock(torch.cat([mean, sdv], -1), y)).abs().max().item()
        print('learned_m: step0 vs stock(mean, sd):', d0, '(exact except all-tied cells, sd clamped at 1e-4)',
              '| w at init all 1:', bool((lm._loc_w[mask].abs() - 1).max().item() < 1e-5), '| d0 < 1e-4:', d0 < 1e-4)
        nn.init.normal_(lm.loc_w[-1].weight)  # randomize the weight network
        a = lm(reps, mask, y)
        print('learned_m, random g: n=1 vs stock:', (lm(reps, m1, y) - stock(reps[..., 0], y)).abs().max().item(),
              '| n>1 differs:', (lm(reps, mask, y) - stock(torch.cat([mean, sdv], -1), y)).abs().max().item(), '(> 0)',
              '| weights vary:', bool(lm._loc_w[mask].std().item() > 0))
    if '--huber_loc' in __import__('sys').argv:
        hl = ObsNanoTabICL(**cfg, center='mean', med_feat=True, std='medmad', zclip=10.0, robust_feats=True,
                           dist_emb=True, temp_head=True, col_ctx=True, isab=16, n_sab=2, sd_cols=True, joint=16, backbone_loc='huber')
        hl.load_state_dict(stock.state_dict(), strict=False)
        hub = huber_location(reps, mask, n_train).float()
        mean = torch.where(mask, reps, 0).sum(-1) / mask.sum(-1)
        n_ = mask.sum(-1)
        sdv = torch.sqrt((torch.where(mask, reps - mean[..., None], 0) ** 2).sum(-1) / (n_ - 1).clamp(min=1))
        print('huber_loc: step0 vs stock(huber, sd):', (hl(reps, mask, y) - stock(torch.cat([hub, sdv], -1), y)).abs().max().item())
        for m_ in [hl.rep_proj, hl.rep_col, hl.rep_dist, hl.rep_temp]:
            [nn.init.normal_(p_) for p_ in m_.parameters()]
        print('huber_loc, random weights: n=1 vs stock:', (hl(reps, m1, y) - stock(reps[..., 0], y)).abs().max().item())
