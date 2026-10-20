import typing, math, torch, torch.nn as nn


class NanoTabICLv2(nn.Module):
    def __init__(self, max_classes: int, out_dim: int, embed_dim: int = 128,
                 col_num_blocks: int = 3, row_num_blocks: int = 3, icl_num_blocks: int = 12,
                 col_nhead: int = 8, row_nhead: int = 8, icl_nhead: int = 8,
                 feature_group_size: int = 3, n_cls_cols: int = 4, n_cls_rows: int = 128, group_shift: int = 1,
                 preproc: str = 'standard'):
        # classification: max_classes = out_dim (= 10 typically); regression: max_classes = 0, out_dim = n_quantiles
        super().__init__()
        self.feature_group_size = feature_group_size
        self.group_shift = group_shift  # feature i is grouped with features i + 2^k - group_shift (TabICLv2: 0)
        # input preprocessing, fitted on the training rows: 'standard' (z-score) or 'tabicl' (the TabICLv2 sklearn
        # wrapper's 'none' pipeline: z-score clipped to +-100, then two-stage 4-sd outlier soft clipping)
        self.preproc = preproc
        icl_dim = embed_dim * n_cls_cols

        self.x_embed = nn.Linear(feature_group_size, embed_dim)
        self.y_embed_in = ClassEmbedding(max_classes, embed_dim) if max_classes > 0 else nn.Linear(1, embed_dim)
        self.y_embed_icl = ClassEmbedding(max_classes, icl_dim) if max_classes > 0 else nn.Linear(1, icl_dim)

        self.col_blocks = nn.ModuleList([
            InducedTransformerBlock(embed_dim=embed_dim, num_heads=col_nhead, n_inducing=n_cls_rows, ssmax=True)
            for _ in range(col_num_blocks)])
        self.row_blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=row_nhead, use_rope=True) for _ in range(row_num_blocks)])
        self.icl_blocks = nn.ModuleList([
            TransformerBlock(embed_dim=icl_dim, num_heads=icl_nhead, ssmax=True) for _ in range(icl_num_blocks)])

        self.row_cls_tokens = nn.Parameter(0.02 * torch.randn(1, 1, n_cls_cols, embed_dim))
        self.row_ln = nn.LayerNorm(embed_dim)
        self.out_ln = nn.LayerNorm(icl_dim)
        self.out_mlp = get_mlp(icl_dim, icl_dim * 2, out_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor, extra_emb: typing.Optional[torch.Tensor] = None,
                extra_row_emb: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        n_batch, n_rows, n_cols = x.shape
        n_batch, n_train = y.shape

        # ----- Embedding: standardize -> repeated feature grouping -> x embedding -> add y embedding to train
        if self.preproc == 'tabicl':
            x = tabicl_preprocess(x, n_train)
        else:
            x = (x - x[:, :n_train].mean(dim=1, keepdim=True)) / (x[:, :n_train].std(dim=1, unbiased=False, keepdim=True) + 1e-8)
        idxs = torch.arange(n_cols, dtype=torch.long, device=x.device)
        x = torch.stack([x[:, :, (idxs + (2 ** i - self.group_shift)) % n_cols] for i in range(self.feature_group_size)], dim=-1)
        emb = self.x_embed(x)  # emb.shape = (n_batch, n_rows, n_cols, embed_dim)
        if extra_emb is not None:  # optional extra per-cell embedding, e.g. from a replicate encoder (obs_model.py)
            emb = emb + extra_emb
        emb[:, :n_train] += self.y_embed_in(y[:, :, None, None])

        # ----- TF_col: induced self-attention within each column
        for block in self.col_blocks:
            emb = block.col_attn(emb, kv_max_idx=n_train)  # all rows only attend to training rows

        # ----- TF_row: concat CLS tokens as extra columns -> row attention -> norm + merge cls tokens
        emb = torch.cat([self.row_cls_tokens.expand(n_batch, n_rows, -1, -1), emb], dim=2)
        for block in self.row_blocks[:-1]:
            emb = block.row_attn(emb)
        emb = self.row_blocks[-1].row_attn(emb, q_max_idx=self.row_cls_tokens.size(-2))  # need only the cls token values
        emb = self.row_ln(emb).flatten(-2, -1)  # norm + merge cls tokens into one bigger token
        if extra_row_emb is not None:  # optional per-row embedding, e.g. row reliability (obs_model.py)
            emb = emb + extra_row_emb

        # ----- TF_icl: add y embedding -> self-attention
        # now emb.shape = (n_batch, n_rows, icl_dim)
        emb[:, :n_train] += self.y_embed_icl(y[:, :, None])  # add y embeddings again
        for block in self.icl_blocks[:-1]:
            emb = block(emb, kv_max_idx=n_train)  # all rows only attend to training rows
        emb = self.icl_blocks[-1](emb[:, n_train:], emb[:, :n_train])  # need only test predictions

        return self.out_mlp(self.out_ln(emb))  # output MLP


def tabicl_preprocess(x: torch.Tensor, n_train: int, threshold: float = 4.0) -> torch.Tensor:
    """Torch port of tabicl's PreprocessingPipeline(normalization_method='none') (tabicl 2.2.0, _sklearn/preprocessing.py),
    per table and column, fitted on the first n_train rows: CustomStandardScaler (population sd + 1e-6, clip +-100),
    then OutlierRemover (bounds mean +- threshold * sd with ddof=1; refit mean / sd without the outliers; soft clip
    x = max(L - log1p|x|, x), x = min(U + log1p|x|, x)). Not ported: UniqueFeatureFilter (columns constant in the
    training rows are dropped by tabicl; here they become all-zero columns)."""
    # the fitted statistics are constants (as in tabicl, which fits on the training rows): gradients flow through x only.
    # Differentiating through them is also numerically unsafe: a constant training column gives sd = sqrt(0) -> NaN.
    tr = x[:, :n_train].detach()
    x = ((x - tr.mean(dim=1, keepdim=True)) / (tr.std(dim=1, unbiased=False, keepdim=True) + 1e-6)).clamp(-100, 100)
    tr = x[:, :n_train].detach()
    ddof = 1 if n_train > 1 else 0
    sd = tr.std(dim=1, correction=ddof, keepdim=True).clamp(min=1e-6)
    mu = tr.mean(dim=1, keepdim=True)
    ok = ((tr >= mu - threshold * sd) & (tr <= mu + threshold * sd)).to(x.dtype)
    cnt = ok.sum(dim=1, keepdim=True)
    mu = (tr * ok).sum(dim=1, keepdim=True) / cnt.clamp(min=1)
    sd = torch.sqrt((((tr - mu) ** 2) * ok).sum(dim=1, keepdim=True) / (cnt - ddof).clamp(min=1)).clamp(min=1e-6)
    lo, hi = mu - threshold * sd, mu + threshold * sd
    x = torch.maximum(-torch.log1p(x.abs()) + lo, x)
    return torch.minimum(torch.log1p(x.abs()) + hi, x)


class ClassEmbedding(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(torch.nn.functional.one_hot(input.squeeze(-1), self.in_features).float())


def get_mlp(n_in: int, n_hidden: int, n_out: int):
    return nn.Sequential(nn.Linear(n_in, n_hidden), nn.GELU(), nn.Linear(n_hidden, n_out))


class TableAttnBase(nn.Module):  # base class with functions to apply attention on 2D tables instead of 1D sequences
    def row_attn(self, q, kv=None, **kwargs):  # apply attention within each row separately
        n_batch, n_rows, n_cols, embed_dim = q.shape
        q, kv = (None if t is None else t.flatten(0, 1) for t in [q, kv])  # merge rows dim into batch dim
        # apply attention -> unmerge rows dim from batch dim; dimension -2 might differ because of q_max_idx in kwargs
        return self(q, kv, **kwargs).reshape(n_batch, n_rows, -1, embed_dim)

    def col_attn(self, q, kv=None, **kwargs):  # apply attention within each column separately
        return self.row_attn(q.transpose(1, 2), None if kv is None else kv.transpose(1, 2), **kwargs).transpose(1, 2)


class InducedTransformerBlock(TableAttnBase):
    def __init__(self, embed_dim: int, num_heads: int, n_inducing: int, ssmax: bool = False):
        super().__init__()
        self.tfm1 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ssmax=ssmax)
        self.tfm2 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads)  # fixed nb of ind. vectors -> no ssmax
        self.inducing_vectors = nn.Parameter(0.02 * torch.randn(1, n_inducing, embed_dim))

    def forward(self, q, kv=None, q_max_idx: typing.Optional[int] = None, kv_max_idx: typing.Optional[int] = None):
        kv = self.tfm1(self.inducing_vectors.expand(q.shape[0], -1, -1), q if kv is None else kv, kv_max_idx=kv_max_idx)
        return self.tfm2(q, kv, q_max_idx=q_max_idx)


class TransformerBlock(nn.MultiheadAttention, TableAttnBase):
    def __init__(self, embed_dim: int, num_heads: int, use_rope: bool = False, ssmax: bool = False):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads)
        self.rope = Rope(head_dim=embed_dim // num_heads, theta=100_000.0) if use_rope else None
        self.ssmax_layer = QASSMax(num_heads=num_heads, head_dim=embed_dim // num_heads) if ssmax else None
        self.mlp = get_mlp(embed_dim, embed_dim * 2, embed_dim)
        self.ln_attn = nn.LayerNorm(embed_dim)
        self.ln_mlp = nn.LayerNorm(embed_dim)

    def forward(self, q, kv=None, q_max_idx: typing.Optional[int] = None, kv_max_idx: typing.Optional[int] = None):
        # q.shape: (batch_size, q_len, embed_dim), kv.shape: (batch_size, kv_len, embed_dim)
        x, q = q, self.ln_attn(q)
        kv = q if kv is None else self.ln_attn(kv)
        if kv_max_idx is not None: kv = kv[..., :kv_max_idx, :]
        if q_max_idx is not None: x, q = x[..., :q_max_idx, :], q[..., :q_max_idx, :]

        x = x + self.attn(q, kv)
        del q, kv  # save memory during inference
        return x + self.mlp(self.ln_mlp(x))  # we use pre-norm here and for the attention as well

    def attn(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # Joint projection of (q, k, v), then transpose heads to (batch_size, num_heads, len, head_dim)
        q, k, v = nn.functional._in_projection_packed(q, k, k, self.in_proj_weight, self.in_proj_bias)
        q, k, v = (t.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2) for t in [q, k, v])

        q = q if self.ssmax_layer is None else self.ssmax_layer(q=q, n=k.size(-2))  # SSMax (optional)
        q, k = (t if self.rope is None else self.rope(t) for t in [q, k])  # RoPE (optional)

        # attention with heads in batch dim (maybe needed for FlashAttention) -> put the head dim back -> out projection
        attn_output = nn.functional.scaled_dot_product_attention(*[t.flatten(0, 1) for t in (q, k, v)]).view(q.shape)
        del q, k, v  # save memory during inference
        return self.out_proj(attn_output.transpose(-3, -2).flatten(-2, -1))  # (batch_size, q_len, embed_dim)


class Rope(nn.Module):  # rotary positional encoding
    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        self.half = head_dim // 2
        self.register_buffer("inv_freq", theta ** torch.linspace(0.0, -1.0, self.half + 1)[:-1], persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)
        self.register_buffer("cos", torch.empty(0), persistent=False)

    @torch.autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape

        if self.cos.numel() == 0 or self.cos.device != x.device or self.cos.size(0) < seq_len:  # need to extend cache
            pos = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)  # (seq_len,)
            angles = pos[:, None] * self.inv_freq[None, :]  # (seq_len, half_head_dim)
            self.sin, self.cos = angles.sin(), angles.cos()  # (seq_len, half_head_dim)

        sin, cos = self.sin[:seq_len], self.cos[:seq_len]
        x1, x2 = x[..., :self.half], x[..., self.half:]  # (batch_size, num_heads, seq_len, half_head_dim)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


class QASSMax(nn.Module):  # query-aware scalable softmax for better context length scaling
    def __init__(self, num_heads: int, head_dim: int, n_hidden: int = 64):
        super().__init__()
        self.base_mlp = get_mlp(1, n_hidden, num_heads * head_dim)
        self.query_mlp = get_mlp(head_dim, n_hidden, head_dim)
        nn.init.zeros_(self.query_mlp[-1].weight)  # ensures initial modulation is zero
        nn.init.zeros_(self.query_mlp[-1].bias)

    def forward(self, q: torch.Tensor, n: int) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = q.shape
        logn = q.new_tensor(math.log(max(1, n))).view(1, 1)
        return self.base_mlp(logn).view(1, num_heads, 1, head_dim) * (1 + torch.tanh(self.query_mlp(q))) * q


if __name__ == '__main__':  # check that forward pass works
    n_batch, n_train, n_test, n_cols = 2, 16, 8, 3
    model = NanoTabICLv2(max_classes=10, out_dim=10)
    x = torch.randn(n_batch, n_train + n_test, n_cols)
    y = torch.randint(10, size=(n_batch, n_train))
    print(f'{model(x,y).shape=}')
