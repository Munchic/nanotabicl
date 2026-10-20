import math, torch, torch.nn as nn
from model import NanoTabICLv2, get_mlp


class ReplicateEncoder(nn.Module):  # set encoder over the replicates of one cell -> (d_s + 1)-dim vector
    def __init__(self, d_s: int = 32, n_heads: int = 4, use_sab: bool = True, use_null: bool = True,
                 med_feat: bool = False, loc_head: bool = False, robust_feats: bool = False, pool: str = 'attn',
                 col_ctx: bool = False):
        super().__init__()
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
        self.pool_ln, self.pool_kv = nn.LayerNorm(d_s), nn.Linear(d_s, 2 * d_s)
        self.seed = nn.Parameter(0.02 * torch.randn(1, 1, d_s))  # single learned pooling query
        self.null_kv = nn.Parameter(0.02 * torch.randn(1, 1, 2 * d_s))  # learned null key/value (shrink to prior)
        self.out_ln = nn.LayerNorm(d_s)

    def heads(self, t):  # (M, L, d) -> (M, H, L, d/H)
        return t.unflatten(-1, (self.n_heads, -1)).transpose(1, 2)

    def forward(self, z: torch.Tensor, mask: torch.Tensor, tie: torch.Tensor | None = None, ctx_fn=None):
        # z: (M, K) standardized replicates (padding arbitrary), mask: (M, K) bool, every row has >= 2 valid entries
        # tie: (M, K) bool, replicate equals the column's extreme context value (robust_feats only)
        # returns (cell vector (M, d_s + 1), location scores (M, K) or None)
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
        f = torch.stack(feats, dim=-1)
        h = self.phi(f)  # (M, K, d_s)
        if ctx_fn is not None:  # column context: (M, d_s) per-cell summary -> (M, d_s) additive term for every replicate
            u = (self.phi_col(f) * mask[..., None].to(h.dtype)).sum(1) / n.to(h.dtype)
            h = h + ctx_fn(u)[:, None].to(h.dtype)
        key_mask = mask[:, None, None, :]  # (M, 1, 1, K): attend only to real replicates
        if self.use_sab:
            q, k, v = self.heads(self.sab_qkv(self.sab_ln(h))).chunk(3, dim=-1)
            a = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=key_mask)
            h = h + self.sab_out(a.transpose(1, 2).flatten(-2))
            h = h + self.sab_mlp(self.sab_mlp_ln(h))
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
                 col_ctx: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
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
        self.rep_enc = ReplicateEncoder(d_s=d_s, n_heads=rep_heads, use_sab=use_sab, use_null=use_null,
                                        med_feat=med_feat, loc_head=center in ('attn', 'attn_med'), robust_feats=robust_feats, pool=pool,
                                        col_ctx=col_ctx)
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

    def forward(self, reps: torch.Tensor, mask: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n_train = y.shape[1]
        x_c, extra_emb = self.cell_inputs(reps, mask, n_train=n_train)
        feats = self.row_features(reps, mask, n_train) if (self.temp_head or self.row_emb) else None
        row_extra = self.rep_row(feats.to(extra_emb.dtype)) if self.row_emb else None
        out = super().forward(x_c, y, extra_emb=extra_emb, extra_row_emb=row_extra)
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
        if self.center in ('attn', 'attn_med') and self.aux_center:
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

    def cell_inputs(self, reps: torch.Tensor, mask: torch.Tensor, n_train: int):
        """-> (per-cell value fed to the backbone (B, R, C), additive encoder embedding (B, R, C, embed_dim))."""
        n_batch, n_rows, n_cols, _ = reps.shape
        reps = torch.where(mask, reps, 0.0)
        n = mask.sum(-1)
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
            vec, scores = self.rep_enc(z[multi], mask[multi], None if tie is None else tie[multi], ctx_fn=ctx_fn)
            cell[multi] = vec.to(cell.dtype)
            if self.center in ('attn', 'attn_med'):  # learned location: softmax-weighted replicates (raw scale), n = 1 keep x
                w = torch.softmax(scores.float().masked_fill(~mask[multi], float('-inf')), dim=-1)
                att = (w * reps[multi]).sum(-1)
                x_c = x.clone()
                if self.center == 'attn_med':
                    med = linear_median(reps[multi], mask[multi])
                    att = med + self.ALPHA_SCALE * self.rep_alpha.float() * (att - med)
                x_c[multi] = att.to(x.dtype)
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
        idxs = torch.arange(n_cols, device=reps.device)
        grouped = torch.cat([cell[:, :, (idxs + (2 ** i - self.group_shift)) % n_cols]
                             for i in range(self.feature_group_size)], dim=-1)
        return x_c, self.rep_proj(grouped)


if __name__ == '__main__':  # step 0 == stock model, and n = 1 everywhere == stock model at any weights
    torch.manual_seed(0)
    cfg = dict(max_classes=10, out_dim=10, embed_dim=32, col_num_blocks=2, row_num_blocks=2, icl_num_blocks=6,
               col_nhead=4, row_nhead=4, icl_nhead=4, n_cls_rows=32)
    stock, obs = NanoTabICLv2(**cfg), ObsNanoTabICL(**cfg)
    obs.load_state_dict(stock.state_dict(), strict=False)
    if '--col_ctx' in __import__('sys').argv:
        obs = ObsNanoTabICL(**cfg, center='attn', med_feat=True, std='medmad', zclip=10.0, robust_feats=True,
                            dist_emb=True, temp_head=True, col_ctx=True)
        obs.load_state_dict(stock.state_dict(), strict=False)
    B, R, C, K, n_train = 2, 40, 5, 10, 24
    reps = torch.randn(B, R, C, K)
    mask = torch.arange(K) < torch.randint(1, K + 1, (B, R, C, 1))
    y = torch.randint(10, (B, n_train))
    xm = torch.where(mask, reps, 0).sum(-1) / mask.sum(-1)
    print('step0 max|diff| =', (obs(reps, mask, y) - stock(xm, y)).abs().max().item())
    nn.init.normal_(obs.rep_proj.weight)
    for m_ in [getattr(obs, 'rep_col', None), getattr(obs, 'rep_dist', None), getattr(obs, 'rep_temp', None)]:
        if m_ is not None:  # randomize every zero-init map: n = 1 tables must still equal stock
            [nn.init.normal_(p_) for p_ in m_.parameters()]
    m1 = torch.zeros_like(mask); m1[..., 0] = True
    print('n=1, trained U: max|diff| =', (obs(reps, m1, y) - stock(reps[..., 0], y)).abs().max().item())
    print('n>1, trained U: max|diff| =', (obs(reps, mask, y) - stock(xm, y)).abs().max().item(), '(should be > 0)')
    print('params: stock %.3fM, replicate branch %.3fM' % (sum(p.numel() for p in stock.parameters()) / 1e6,
          sum(p.numel() for n_, p in obs.named_parameters() if n_.startswith('rep_')) / 1e6))
