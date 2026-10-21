"""Censoring spec channels for NanoTabICLv2 / TabICLv2 (CENSORING.md P1 + P2).

Per-cell feature flags and per-context-row target flags in {0: no spec / observed, 1: left-censored '<',
2: right-censored '>', 3: observed AND the spec is known (known=True models only)}. A censored value is reported AT its
bound, so the bound itself needs no extra channel; the flag says "the true value is beyond this". All maps are bias-free
over a one-hot whose state 0 is the zero vector, and zero-initialized, so
  (i)  step 0 == the backbone exactly, and
  (ii) a table without flags (flags=None or all 0) == the backbone exactly at ANY weights.
row_adapter=h (> 0): a gated, zero-init MLP on [the row's token after the row stage (row_ln output), flag one-hot,
observed y] for context rows with a target flag in {1, 2} only, added to that row's ICL token (forward hook on row_ln).
It lets a frozen backbone represent a censored row as "its latent value given its features", which three bias-free
linear maps cannot (they only shift every censored row by the same vector). Unflagged rows get exactly 0.
known=True separates "spec given, nothing censored here" (3) from "no spec" (0). Without it both are 0, and training
with spec dropout teaches the model that unflagged data may be censored, so it hedges on all clean data (R1, step 500:
clean CRPS +0.006). With it, "no spec" still equals the backbone, and a known-clean table gets a learned embedding.
The backbone's forward is not modified: the feature flags enter through its `extra_emb` hook, the target flags through
forward hooks on `y_embed_in` (column stage) and `y_embed_icl` (ICL stage), the two places where y enters.
"""
import torch, torch.nn as nn, torch.nn.functional as F

from model import NanoTabICLv2


def flag_onehot(flag, n=2):  # (...,) int {0..3} -> (..., n) float: states 1..n; 0 (and 3 when n = 2) -> 0
    return F.one_hot(flag.long(), 4)[..., 1:1 + n].float()


class CensNanoTabICL(NanoTabICLv2):
    def __init__(self, *args, known=False, row_adapter=0, **kw):
        super().__init__(*args, **kw)
        self.n_flag = n = 3 if known else 2
        d, icl_dim = self.x_embed.out_features, self.y_embed_icl.out_features
        G = self.feature_group_size
        self.cens_x = nn.Linear(n * G, d, bias=False)  # cell flags, grouped like the features (same column offsets)
        self.cens_y_in = nn.Linear(n, d, bias=False)
        self.cens_y_icl = nn.Linear(n, icl_dim, bias=False)
        for m in (self.cens_x, self.cens_y_in, self.cens_y_icl):
            nn.init.zeros_(m.weight)
        self._y_oh = self._y_ctx = None
        if row_adapter:
            tok = self.row_ln.normalized_shape[0] * self.row_cls_tokens.shape[-2]
            self.cens_row = nn.Sequential(nn.Linear(tok + 3, row_adapter), nn.GELU(), nn.Linear(row_adapter, tok))
            nn.init.zeros_(self.cens_row[-1].weight)
            nn.init.zeros_(self.cens_row[-1].bias)
            self.row_ln.register_forward_hook(self._row_hook)
        self.y_embed_in.register_forward_hook(lambda m, i, out: out if self._y_oh is None else
                                              out + self.cens_y_in(self._y_oh)[:, :, None].to(out.dtype))
        self.y_embed_icl.register_forward_hook(lambda m, i, out: out if self._y_oh is None else
                                               out + self.cens_y_icl(self._y_oh).to(out.dtype))

    def _row_hook(self, m, i, out):  # out (B, R, n_cls, d): add the adapter to censored context rows (others: 0)
        if self._y_ctx is None:
            return out
        y, flag = self._y_ctx
        n = y.shape[1]
        cen = ((flag == 1) | (flag == 2)).to(out.dtype)[..., None]  # (B, n, 1)
        tok = out[:, :n].flatten(-2)
        inp = torch.cat([tok, F.one_hot(flag.long().clamp(max=2), 3)[..., 1:].to(out.dtype), y[..., None].to(out.dtype)], -1)
        delta = (self.cens_row(inp) * cen).view_as(out[:, :n])
        return torch.cat([out[:, :n] + delta, out[:, n:]], 1)

    def x_flag_emb(self, x_flag):  # (B, R, C) -> (B, R, C, d), grouped as NanoTabICLv2.forward groups x
        C = x_flag.shape[-1]
        idxs = torch.arange(C, device=x_flag.device)
        oh = flag_onehot(x_flag, self.n_flag)
        oh = torch.cat([oh[:, :, (idxs + (2 ** i - self.group_shift)) % C] for i in range(self.feature_group_size)], -1)
        return self.cens_x(oh)

    def forward(self, x, y, x_flag=None, y_flag=None, **kw):
        """x (B, R, C) observed values (censored cells at their bound); y (B, n_train) observed context targets;
        x_flag (B, R, C), y_flag (B, n_train): int {0..3} or None (= spec not given)."""
        if x_flag is not None:
            e = self.x_flag_emb(x_flag)
            kw['extra_emb'] = e if kw.get('extra_emb') is None else kw['extra_emb'] + e
        self._y_oh = None if y_flag is None else flag_onehot(y_flag, self.n_flag)
        self._y_ctx = None if (y_flag is None or not hasattr(self, 'cens_row')) else (y, y_flag)
        try:
            return super().forward(x, y, **kw)
        finally:
            self._y_oh = self._y_ctx = None


if __name__ == '__main__':  # guarantees (i) and (ii), regression and classification heads, TabICLv2 grouping and nano's
    torch.manual_seed(0)
    for kw in [dict(max_classes=0, out_dim=99, group_shift=0, preproc='tabicl'), dict(max_classes=10, out_dim=10),
               dict(max_classes=0, out_dim=99, group_shift=0, preproc='tabicl', known=True),
               dict(max_classes=0, out_dim=99, group_shift=0, preproc='tabicl', row_adapter=32)]:
        known, ra = kw.pop('known', False), kw.pop('row_adapter', 0)
        cfg = dict(embed_dim=32, col_num_blocks=2, row_num_blocks=2, icl_num_blocks=2, col_nhead=4, row_nhead=4,
                   icl_nhead=4, n_cls_rows=16, **kw)
        base, m = NanoTabICLv2(**cfg).eval(), CensNanoTabICL(**cfg, known=known, row_adapter=ra).eval()
        missing, unexpected = m.load_state_dict(base.state_dict(), strict=False)
        assert not unexpected and all(k.startswith('cens_') for k in missing), (missing, unexpected)
        B, n_tr, n_te, C = 3, 40, 20, 7
        x = torch.randn(B, n_tr + n_te, C)
        y = torch.randn(B, n_tr) if kw['max_classes'] == 0 else torch.randint(10, (B, n_tr))
        xf, yf = torch.randint(4 if known else 3, (B, n_tr + n_te, C)), torch.randint(4 if known else 3, (B, n_tr))
        with torch.no_grad():
            ref = base(x, y)
            d0 = (m(x, y, xf, yf) - ref).abs().max().item()  # (i) zero-init, flags present
            for p in (m.cens_x, m.cens_y_in, m.cens_y_icl) + ((m.cens_row[-1],) if ra else ()):
                nn.init.normal_(p.weight)
            d1 = (m(x, y) - ref).abs().max().item()  # (ii) random weights, no spec
            d2 = (m(x, y, torch.zeros_like(xf), torch.zeros_like(yf)) - ref).abs().max().item()  # (ii) all observed
            d3 = (m(x, y, xf, yf) - ref).abs().max().item()  # flags matter once the maps are non-zero
            d4 = (m(x, y, xf, None) - ref).abs().max().item(), (m(x, y, None, yf) - ref).abs().max().item()
            d5 = (m(x, y, torch.full_like(xf, 3), torch.full_like(yf, 3)) - ref).abs().max().item()  # known-observed
            if ra:  # the adapter alone (linear maps back at 0) must act on censored rows only
                for p in (m.cens_x, m.cens_y_in, m.cens_y_icl):
                    nn.init.zeros_(p.weight)
                d6 = (m(x, y, None, torch.where(yf == 3, 0, yf)) - ref).abs().max().item()
                d7 = (m(x, y, None, torch.zeros_like(yf)) - ref).abs().max().item()
                assert d6 > 0 and d7 == 0.0, (d6, d7)
        print(f"{'regression' if kw['max_classes'] == 0 else 'classification'}{' known' if known else ''}{' adapter' if ra else ''}: step0 {d0:.1e}  no-spec {d1:.1e}  "
              f"all-observed {d2:.1e}  | random maps + flags {d3:.2f}, x only {d4[0]:.2f}, y only {d4[1]:.2f}, known-observed {d5:.2f}")
        assert d0 == 0.0 and d1 == 0.0 and d2 == 0.0 and d3 > 0 and min(d4) > 0 and (d5 > 0) == known
    print('ok')
