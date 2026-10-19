"""Training-step and inference cost of each phase-2 architecture on a fixed batch (run with the GPU otherwise idle)."""
import json, sys, time

import torch

sys.path.insert(0, '.')
from obs_train import make_model, model_forward, ce_loss  # noqa: E402

dev = 'cuda'
torch.manual_seed(0)
ARCHS = {'stock / mean (C tokens)': ('mean', {}), 'stats (6C tokens)': ('stats', {}),
         'obs spec (mean, SAB)': ('obs', {}), 'obs attn+med_feat (SAB)': ('obs', {'center': 'attn', 'med_feat': True}),
         'obs attn+med_feat (no SAB)': ('obs', {'center': 'attn', 'med_feat': True, 'use_sab': False})}


def bench(mode, cfg, B, R, C, train):
    model = make_model(mode, cfg).to(dev).train(train)
    X = torch.randn(B, R, C, device=dev)
    y = torch.randint(4, (B, R), device=dev)
    g = torch.Generator(device=dev).manual_seed(0)
    n_train = R // 2

    def step():
        with torch.autocast('cuda', dtype=torch.bfloat16), torch.set_grad_enabled(train):
            out = model_forward(model, mode, X, y[:, :n_train], g, p_clean=0.0)
            if train:
                ce_loss(out, y[:, n_train:]).mean().backward()
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(10):
        step()
    torch.cuda.synchronize()
    return (time.time() - t) / 10 * 1000, torch.cuda.max_memory_allocated() / 2**30


rows = []
for name, (mode, cfg) in ARCHS.items():
    tr = bench(mode, cfg, 8, 1024, 16, True)
    inf = bench(mode, cfg, 1, 1024, 16, False)
    rows.append(dict(arch=name, train_ms=tr[0], train_gib=tr[1], infer_ms=inf[0], infer_gib=inf[1]))
    print(f'| {name} | {tr[0]:.0f} | {tr[1]:.2f} | {inf[0]:.1f} | {inf[1]:.2f} |', flush=True)
json.dump(rows, open('artifacts/bench_cost.json', 'w'), indent=1)
