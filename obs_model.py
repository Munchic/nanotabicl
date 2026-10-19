import math, torch, torch.nn as nn
from model import NanoTabICLv2, get_mlp


class ReplicateEncoder(nn.Module):  # set encoder over the replicates of one cell -> (d_s + 1)-dim vector
    def __init__(self, d_s: int = 32, n_heads: int = 4, use_sab: bool = True, use_null: bool = True,
                 med_feat: bool = False, loc_head: bool = False):
        super().__init__()
        self.n_heads, self.use_sab, self.use_null, self.med_feat = n_heads, use_sab, use_null, med_feat
        self.phi = get_mlp(3 if med_feat else 2, 2 * d_s, d_s)  # per-replicate embedding of (z, z - z_bar[, z - z_med])
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

    def forward(self, z: torch.Tensor, mask: torch.Tensor):
        # z: (M, K) standardized replicates (padding arbitrary), mask: (M, K) bool, every row has >= 2 valid entries
        # returns (cell vector (M, d_s + 1), location scores (M, K) or None)
        z = torch.where(mask, z, 0.0)
        n = mask.sum(-1, keepdim=True)
        z_bar = z.sum(-1, keepdim=True) / n
        feats = [z, z - z_bar]
        if self.med_feat:
            feats.append(z - masked_median(z, mask)[..., None])
        h = self.phi(torch.stack(feats, dim=-1))  # (M, K, d_s)
        key_mask = mask[:, None, None, :]  # (M, 1, 1, K): attend only to real replicates
        if self.use_sab:
            q, k, v = self.heads(self.sab_qkv(self.sab_ln(h))).chunk(3, dim=-1)
            a = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=key_mask)
            h = h + self.sab_out(a.transpose(1, 2).flatten(-2))
            h = h + self.sab_mlp(self.sab_mlp_ln(h))
        kv = self.pool_kv(self.pool_ln(h))  # (M, K, 2 d_s)
        if self.use_null:
            kv = torch.cat([self.null_kv.expand(kv.shape[0], -1, -1), kv], dim=1)
            key_mask = torch.cat([key_mask.new_ones(key_mask.shape[:-1] + (1,)), key_mask], dim=-1)
        k, v = self.heads(kv).chunk(2, dim=-1)
        q = self.heads(self.seed.expand(kv.shape[0], -1, -1))
        pooled = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=key_mask).transpose(1, 2).flatten(-2)
        vec = torch.cat([self.out_ln(pooled[:, 0]), torch.log(n.to(pooled.dtype))], dim=-1)  # + log n (counting)
        return vec, (self.loc_head(h).squeeze(-1) if self.loc_head is not None else None)


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
                 center: str = 'mean', med_feat: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # per-cell value fed to the backbone (n = 1 -> x in all cases): 'mean' (= imputation, the spec), 'median'
        # (robust), 'attn' (learned attention-weighted location over replicates; zero-init scores -> starts at the mean)
        self.center = center
        self.rep_enc = ReplicateEncoder(d_s=d_s, n_heads=rep_heads, use_sab=use_sab, use_null=use_null,
                                        med_feat=med_feat, loc_head=center == 'attn')
        self.rep_proj = nn.Linear(self.feature_group_size * (d_s + 1), self.x_embed.out_features, bias=False)
        nn.init.zeros_(self.rep_proj.weight)

    def forward(self, reps: torch.Tensor, mask: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_c, extra_emb = self.cell_inputs(reps, mask, n_train=y.shape[1])
        return super().forward(x_c, y, extra_emb=extra_emb)

    def cell_inputs(self, reps: torch.Tensor, mask: torch.Tensor, n_train: int):
        """-> (per-cell value fed to the backbone (B, R, C), additive encoder embedding (B, R, C, embed_dim))."""
        n_batch, n_rows, n_cols, _ = reps.shape
        reps = torch.where(mask, reps, 0.0)
        n = mask.sum(-1)
        x = reps.sum(-1) / n.clamp(min=1)  # cell means -> backbone, exactly like a single observed value
        # standardize replicates with the training rows' column statistics: of the cell means (the spec), or of the
        # cell medians for center in (median, attn) -- robust to cell means blown up by heavy-tailed replicates
        x_s = x if self.center == 'mean' else masked_median(reps, mask)
        mu = x_s[:, :n_train].mean(dim=1, keepdim=True)[..., None]
        sd = x_s[:, :n_train].std(dim=1, unbiased=False, keepdim=True)[..., None] + 1e-8
        z = (reps - mu) / sd

        multi = n > 1  # gate: only cells with >= 2 replicates go through the set encoder
        cell = z.new_zeros(n_batch, n_rows, n_cols, self.rep_proj.in_features // self.feature_group_size)
        x_c = x if self.center == 'mean' else x_s
        if multi.any():
            vec, scores = self.rep_enc(z[multi], mask[multi])
            cell[multi] = vec.to(cell.dtype)
            if self.center == 'attn':  # learned location: softmax-weighted replicates (raw scale), n = 1 cells keep x
                w = torch.softmax(scores.float().masked_fill(~mask[multi], float('-inf')), dim=-1)
                x_c = x.clone()
                x_c[multi] = (w * reps[multi]).sum(-1).to(x.dtype)
        idxs = torch.arange(n_cols, device=reps.device)
        grouped = torch.cat([cell[:, :, (idxs + (2 ** i - 1)) % n_cols] for i in range(self.feature_group_size)], dim=-1)
        return x_c, self.rep_proj(grouped)


if __name__ == '__main__':  # step 0 == stock model, and n = 1 everywhere == stock model at any weights
    torch.manual_seed(0)
    cfg = dict(max_classes=10, out_dim=10, embed_dim=32, col_num_blocks=2, row_num_blocks=2, icl_num_blocks=6,
               col_nhead=4, row_nhead=4, icl_nhead=4, n_cls_rows=32)
    stock, obs = NanoTabICLv2(**cfg), ObsNanoTabICL(**cfg)
    obs.load_state_dict(stock.state_dict(), strict=False)
    B, R, C, K, n_train = 2, 40, 5, 10, 24
    reps = torch.randn(B, R, C, K)
    mask = torch.arange(K) < torch.randint(1, K + 1, (B, R, C, 1))
    y = torch.randint(10, (B, n_train))
    xm = torch.where(mask, reps, 0).sum(-1) / mask.sum(-1)
    print('step0 max|diff| =', (obs(reps, mask, y) - stock(xm, y)).abs().max().item())
    nn.init.normal_(obs.rep_proj.weight)
    m1 = torch.zeros_like(mask); m1[..., 0] = True
    print('n=1, trained U: max|diff| =', (obs(reps, m1, y) - stock(reps[..., 0], y)).abs().max().item())
    print('n>1, trained U: max|diff| =', (obs(reps, mask, y) - stock(xm, y)).abs().max().item(), '(should be > 0)')
    print('params: stock %.3fM, replicate branch %.3fM' % (sum(p.numel() for p in stock.parameters()) / 1e6,
          sum(p.numel() for n_, p in obs.named_parameters() if n_.startswith('rep_')) / 1e6))
