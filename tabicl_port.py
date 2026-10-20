"""Load the released TabICLv2 checkpoints (HF jingang/TabICL) into NanoTabICLv2 / ObsNanoTabICL.

The architectures are the same (27.55M parameters); only parameter names differ, plus two reshapes. The TabICL rope
`freqs` buffer equals nano's `inv_freq` and is dropped. The regression checkpoint uses bias-free LayerNorms
(`bias_free_ln`); they load as zero biases, which is exactly equivalent. Quantile head: 999 levels k / 1000.

    cfg, sd = load_tabicl('classification')   # cfg: NanoTabICLv2 kwargs; sd: nano state_dict
    python tabicl_port.py                     # equivalence check vs tabicl's own module (needs the tabicl package)
"""
import glob, os, re

import torch

CKPT = {'classification': 'tabicl-classifier-v2-20260212.ckpt', 'regression': 'tabicl-regressor-v2-20260212.ckpt'}


def ckpt_path(task):
    hits = glob.glob(os.path.expanduser(f'~/.cache/huggingface/hub/models--jingang--TabICL/snapshots/*/{CKPT[task]}'))
    if not hits:
        from huggingface_hub import hf_hub_download
        return hf_hub_download('jingang/TabICL', CKPT[task])
    return hits[0]


def nano_config(tc):  # TabICL config -> NanoTabICLv2 kwargs
    assert tc['col_feature_group'] == 'same' and tc['norm_first'] and tc['ff_factor'] == 2 and tc['activation'] == 'gelu'
    assert tc['row_rope_base'] == 100000 and not tc['row_rope_interleaved'] and tc['col_target_aware']
    reg = tc['max_classes'] == 0
    return dict(max_classes=tc['max_classes'], out_dim=tc['num_quantiles'] if reg else tc['max_classes'],
                embed_dim=tc['embed_dim'], col_num_blocks=tc['col_num_blocks'], row_num_blocks=tc['row_num_blocks'],
                icl_num_blocks=tc['icl_num_blocks'], col_nhead=tc['col_nhead'], row_nhead=tc['row_nhead'],
                icl_nhead=tc['icl_nhead'], feature_group_size=tc['col_feature_group_size'], n_cls_cols=tc['row_num_cls'],
                n_cls_rows=tc['col_num_inds'], group_shift=0)  # TabICL groups feature i with i + 1, i + 2, i + 4


def convert(sd):
    out = {}
    for k, v in sd.items():
        if k == 'row_interactor.tf_row.rope.freqs':
            continue  # == nano's (non-persistent) Rope.inv_freq
        n = k.replace('col_embedder.in_linear', 'x_embed').replace('col_embedder.y_encoder', 'y_embed_in')
        n = re.sub(r'col_embedder\.tf_col\.blocks\.(\d+)\.ind_vectors', r'col_blocks.\1.inducing_vectors', n)
        n = re.sub(r'col_embedder\.tf_col\.blocks\.(\d+)\.multihead_attn([12])\.', r'col_blocks.\1.tfm\2.', n)
        n = n.replace('row_interactor.cls_tokens', 'row_cls_tokens').replace('row_interactor.out_ln', 'row_ln')
        n = re.sub(r'row_interactor\.tf_row\.blocks\.(\d+)\.', r'row_blocks.\1.', n)
        n = re.sub(r'icl_predictor\.tf_icl\.blocks\.(\d+)\.', r'icl_blocks.\1.', n)
        n = n.replace('icl_predictor.ln.', 'out_ln.').replace('icl_predictor.y_encoder', 'y_embed_icl')
        n = n.replace('icl_predictor.decoder.', 'out_mlp.')
        n = n.replace('.attn.ssmax_layer.', '.ssmax_layer.').replace('.attn.in_proj', '.in_proj').replace('.attn.out_proj', '.out_proj')
        n = n.replace('.norm1.', '.ln_attn.').replace('.norm2.', '.ln_mlp.').replace('.linear1.', '.mlp.0.').replace('.linear2.', '.mlp.2.')
        if n.endswith('inducing_vectors'):
            v = v[None]
        if n == 'row_cls_tokens':
            v = v[None, None]
        out[n] = v
    for n in [n for n in out if n.endswith('.weight') and re.search(r'(ln|ln_attn|ln_mlp|row_ln|out_ln)\.weight$', n)]:
        out.setdefault(n[:-len('weight')] + 'bias', torch.zeros_like(out[n]))  # bias-free LayerNorm == zero bias
    return out


def load_tabicl(task):
    ck = torch.load(ckpt_path(task), map_location='cpu', weights_only=False)
    return nano_config(ck['config']), convert(ck['state_dict'])


if __name__ == '__main__':  # equivalence vs tabicl._model.tabicl.TabICL's training forward (fp32, pre-standardized X)
    # Result (2026-10-19): max |diff| <= 1e-4 on logits / quantiles for 3-16 features, both tasks. TabICL's own
    # eval-mode inference path differs from its training path by 0.01-0.07 (column embedder inference code path).
    import inspect, warnings
    warnings.filterwarnings('ignore')
    from model import NanoTabICLv2
    from tabicl._model.tabicl import TabICL
    for task in ['classification', 'regression']:
        ck = torch.load(ckpt_path(task), map_location='cpu', weights_only=False)
        cfg, sd = load_tabicl(task)
        nano = NanoTabICLv2(**cfg)
        nano.load_state_dict(sd, strict=True)
        tab = TabICL(**{k: v for k, v in ck['config'].items() if k in inspect.signature(TabICL.__init__).parameters})
        tab.load_state_dict(ck['state_dict'])
        nano, tab = nano.cuda().eval(), tab.cuda().train()  # dropout = 0: train mode is deterministic
        for C in [3, 7, 16]:
            torch.manual_seed(C)
            B, R, nt = 2, 300, 180
            X = torch.randn(B, R, C, device='cuda')
            X = (X - X[:, :nt].mean(1, keepdim=True)) / X[:, :nt].std(1, unbiased=False, keepdim=True)
            y = torch.randint(0, 4, (B, nt), device='cuda') if task == 'classification' else torch.randn(B, nt, device='cuda')
            with torch.no_grad():
                a, b = nano(X, y).float(), tab._train_forward(X, y.float()).float()
            if task == 'regression':
                a, b = a.sort(-1).values, b.sort(-1).values
            print(f'{task}, {C} features: max |nano - TabICL| = {(a - b).abs().max().item():.1e}')
