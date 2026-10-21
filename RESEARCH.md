# RESEARCH — observation-model noise vs. TabICLv2

Log of quick experiments supporting [NOTES.md](NOTES.md) (ObsPFN). Code: [experiments/obs_noise/](experiments/obs_noise/).

---

## Exp 1 — How much does replicate noise cost a pretrained TabICL, and do summary statistics or fine-tuning recover it? (2026-10-18)

**Question.** If replicated, noisy measurements are collapsed into one value per cell (as imputation does, `x_mean`)
or into a set of summary columns (`x_mean, x_median, x_sd, x_iqr, x_sem, x_logn`), how much accuracy does
TabICLv2 lose relative to the clean table, and does per-dataset fine-tuning close the gap?
This is the "summary stats" baseline that NOTES.md Stage 1 must beat.

### Setup

| | |
|---|---|
| Model | TabICLv2 classifier, `tabicl==2.2.0`, checkpoint `tabicl-classifier-v2-20260212.ckpt` (HF `jingang/TabICL`), default inference (8 estimators), `random_state=0` |
| Data | 27 datasets from this repo's nanoprior ([prior.py](prior.py)): grid n_features ∈ {4, 8, 16} × n_classes ∈ {2, 4, 8} × 3 seeds; numerical features only; 1024 rows, stratified 512 context / 512 holdout; resampled until every class has ≥16 rows (1 dataset needed 2 attempts; one d8/k8 dataset ended with 7 classes) |
| Observation model | Per cell, independently: replicate count n ~ U{1..10}; replicates = x + 0.5·sd(col)·ε. Noise models (easy → hard): **gauss** ε~N(0,1); **student_t3** ε~t₃; **censored** Gaussian then clipped at the column's clean 10th/90th percentile (reported *at* the limit, no indicator); **cauchy** ε~Cauchy; **cauchy_hetero** Cauchy × per-row scale exp(N(0,1)) (heteroscedastic rows). Labels y stay clean; context *and* holdout X are observed through the same model |
| Feature arms | **clean** (reference); **mean** = `x_mean` only (what mean-imputation gives); **full** = `x_mean, x_median, x_sd, x_iqr, x_sem, x_logn` (sd/iqr/sem = 0 when n=1). 1 + 5 noise × 2 feature = **11 arms** |
| Fine-tuning | `tabicl.FinetunedTabICLClassifier` per dataset, fitted on the 512-row context only (internal 20% stratified val split, early stopping on val log-loss, never worse than pretrained on val). LR/epochs chosen on a disjoint **pilot pool** (64 prior datasets, different seeds), never on eval holdouts |
| Metrics | holdout accuracy, balanced acc, ROC-AUC (OvR macro), log-loss; all comparisons **paired over the 27 datasets**, ± = SE |

**Reproducibility (step 0).** Every dataset is seeded by `(seed, attempt)` for the prior and by
`(SEED_NOISE, seed, noise_model)` for the observation model, independent of process/worker order.
Regenerating everything from scratch reproduced all 654 stored arrays bit-for-bit (also with a different worker count).
TabICL inference with a fixed checkpoint + `random_state=0` is deterministic (re-inference max |Δp| = 0).
GPU fine-tuning is seeded but not guaranteed bitwise-deterministic; the resulting weights are stored instead.

```bash
cd experiments/obs_noise
python gen_data.py                    # env: requirements-data.txt  (~1 min, 16 workers)
python data_report.py                 # figures/datasets*.{csv,png}
python run_tabicl.py --stage pilot --lr 1e-5 --epochs 30     # env: requirements-tabicl.txt; repeat for the grid
python run_tabicl.py --stage eval --modes zeroshot
python run_tabicl.py --stage eval --modes finetuned --lr <chosen> --epochs <chosen>
python analyze.py --stage eval
```

**Stored artifacts** (`experiments/obs_noise/artifacts/`, gitignored):
- `data/eval/<name>.npz`: `X_clean, y, train_idx, test_idx`, and per noise model `<noise>__replicates` (n×d×10, NaN-padded), `<noise>__counts`, `<noise>__full`, `<noise>__mean`. `data/manifest.json` has seeds and class counts. `data/pilot_pool.npz` holds the pilot datasets.
- `runs/eval/results_eval.jsonl`: one record per (dataset, arm, mode) with metrics, timing, the checkpoint used and fine-tuning info.
- `runs/eval/preds/eval__<dataset>__<arm>__<mode>.npz`: holdout probabilities, `test_idx`, `y_test`.
- `runs/eval/ckpts/<dataset>/<arm>.ckpt`: fine-tuned weights (`{config, state_dict}`), written only when fine-tuning actually changed them. Otherwise the record's `ckpt` field is the base checkpoint name.
- Re-infer any prediction with `TabICLClassifier(model_path=ckpt, random_state=0).fit(X_arm[train_idx], y[train_idx]).predict_proba(X_arm[test_idx])`. `X_arm` is `X_clean` or `<noise>__<feats>` from the dataset npz.

### Data

Per-dataset stats: [figures/datasets.csv](experiments/obs_noise/figures/datasets.csv).

![clean datasets](experiments/obs_noise/figures/datasets_clean.png)

The prior gives diverse tables: smooth manifolds, discretized/stair-step columns, near-point-mass columns with rare
outliers (e.g. eval_01, 13, 22, 25), and class balance from 50/50 down to 2% minority classes.

![observation models](experiments/obs_noise/figures/observation_models.png)
![signal retention](experiments/obs_noise/figures/signal_retention.png)
![error vs count](experiments/obs_noise/figures/error_vs_count.png)

Signal kept per column, as the median corr(observed stat, clean value) across all 27 datasets:

| noise | mean | median |
|---|---|---|
| gauss | 0.96 | 0.96 |
| student_t3 | 0.91 | 0.93 |
| censored | ≈0.92 (wide: 0.65–0.98) | ≈0.91 |
| cauchy | **0.05** | 0.22 |
| cauchy_hetero | **0.03** | 0.15 |

- Under Cauchy noise the mean carries essentially no information, and more replicates don't help it (error-vs-n is flat). The median recovers steadily with n.
- Under Gaussian noise the mean is slightly better than the median, as expected (efficiency).
- **Caveat:** noise scale is 0.5·sd(col). For point-mass-plus-outlier columns, the outliers dominate the sd, so the noise is large relative to the bulk. In eval_01, censoring at the 10th/90th percentile clips ~100% of replicates because q10 ≈ q90.

### Results — zero-shot (steps 2, 6, 8)

Mean over 27 datasets; Δ = arm − clean, paired, ± SE.

| arm | acc | ROC-AUC | log-loss | Δacc | ΔAUC | Δlog-loss |
|---|---|---|---|---|---|---|
| **clean** | **0.742** | **0.848** | **0.635** | | | |
| gauss / mean | 0.673 | 0.789 | 0.780 | −0.069 ± 0.021 | −0.059 ± 0.014 | +0.145 ± 0.045 |
| gauss / full | 0.669 | 0.786 | 0.788 | −0.073 ± 0.021 | −0.061 ± 0.014 | +0.152 ± 0.046 |
| student_t3 / mean | 0.657 | 0.776 | 0.821 | −0.085 ± 0.024 | −0.072 ± 0.016 | +0.186 ± 0.053 |
| student_t3 / full | 0.660 | 0.780 | 0.815 | −0.082 ± 0.023 | −0.067 ± 0.014 | +0.180 ± 0.050 |
| censored / mean | 0.669 | 0.781 | 0.793 | −0.073 ± 0.022 | −0.067 ± 0.018 | +0.158 ± 0.047 |
| censored / full | 0.666 | 0.776 | 0.802 | −0.076 ± 0.023 | −0.071 ± 0.016 | +0.167 ± 0.047 |
| cauchy / mean | 0.581 | 0.699 | 0.987 | −0.161 ± 0.033 | −0.149 ± 0.020 | +0.352 ± 0.072 |
| cauchy / full | 0.628 | 0.749 | 0.885 | −0.114 ± 0.026 | −0.099 ± 0.018 | +0.249 ± 0.061 |
| cauchy_hetero / mean | 0.570 | 0.683 | 1.021 | −0.172 ± 0.032 | −0.165 ± 0.019 | +0.386 ± 0.074 |
| cauchy_hetero / full | 0.617 | 0.735 | 0.920 | −0.125 ± 0.027 | −0.113 ± 0.017 | +0.284 ± 0.061 |

Full summary stats − mean only, paired. "wins" = datasets where full is better:

| noise | Δacc | ΔAUC | Δlog-loss | wins (AUC) |
|---|---|---|---|---|
| gauss | −0.004 ± 0.002 | −0.002 ± 0.002 | +0.008 ± 0.002 | 7/27 |
| student_t3 | +0.003 ± 0.002 | +0.005 ± 0.003 | −0.006 ± 0.004 | 18/27 |
| censored | −0.004 ± 0.002 | −0.004 ± 0.003 | +0.009 ± 0.003 | 6/27 |
| cauchy | **+0.047 ± 0.010** | **+0.050 ± 0.006** | **−0.103 ± 0.023** | 26/27 |
| cauchy_hetero | **+0.047 ± 0.009** | **+0.052 ± 0.005** | **−0.102 ± 0.022** | 26/27 |

**Observations (zero-shot):**
1. **Even mild replicate noise is expensive.** Gaussian noise, where the per-cell mean keeps r ≈ 0.96 with the clean value, costs about 7 points of accuracy and 0.06 AUC. Pretrained TabICL is fragile to measurement noise in X: a column that is 96% correlated with the truth gives far less than 96% of the accuracy. This is the gap ObsPFN targets.
2. **The ladder is not monotone in the middle.** student_t3 (−8.5 pts) is a bit worse than gauss or censored (−7 pts). Censoring hurts little *on average* but has the widest spread: it destroys datasets whose signal sits in the tails or in spiky columns.
3. **Under heavy tails, summary stats help, but only via robust location.** With Cauchy noise, the full arm recovers about 30% of the mean-arm gap (−16 → −11 pts), consistently: AUC improves on 26/27 datasets. The table and figures point to the `median` column as the source; the spread columns can't be informative because the noise doesn't depend on y. A full − mean − median ablation would confirm this.
4. **With light noise, extra columns slightly hurt.** For gauss and censored, 6× the columns is a small but consistent loss (for gauss, full beats mean on AUC in only 7/27 datasets). TabICL gets no use out of sd/iqr/sem/logn as reliability signals; they just dilute. So "beats summary stats" (NOTES.md Stage 1) is a low bar for light noise, where the baseline is effectively `x_mean`. For heavy tails, the bar is the median.
5. **Heteroscedastic rows cost a little extra on top of Cauchy** (−1 pt acc, −0.015 AUC). Zero-shot TabICL has no way to down-weight unreliable rows. This is the motivation for NOTES.md Stage 2.

### Additional observation models (zero-shot TabICL, added during Exp 2)

Four more models were appended to `NOISE_MODELS` in gen_data.py. All 648 earlier arrays were verified bit-identical after regeneration.
- **`gauss_valhet`, `cauchy_valhet`:** value-dependent (one-directional) heteroscedasticity. Scale = 0.5·sd·2·rank(x)/n within the column, so the smallest values are nearly exact and the largest get 2× the base noise (average 1×).
- **`censored_left`:** Gaussian noise with a lower detection limit at the clean 30th percentile. Values below are reported *at* the limit, with no indicator.
- **`censored_left_heavy`:** t₂ noise with the limit at the 60th percentile. Most replicates sit at or below the border, and a few outliers land far above it.

| arm | acc | ROC-AUC | Δacc vs clean | ΔAUC vs clean |
|---|---|---|---|---|
| gauss_valhet / mean | 0.684 | 0.797 | −0.058 ± 0.020 | −0.050 ± 0.013 |
| gauss_valhet / full | 0.679 | 0.800 | −0.063 ± 0.021 | −0.048 ± 0.011 |
| cauchy_valhet / mean | 0.597 | 0.717 | −0.145 ± 0.030 | −0.131 ± 0.019 |
| cauchy_valhet / full | 0.638 | 0.757 | −0.104 ± 0.026 | −0.091 ± 0.015 |
| censored_left / mean | 0.668 | 0.781 | −0.074 ± 0.021 | −0.067 ± 0.015 |
| censored_left / full | 0.662 | 0.775 | −0.080 ± 0.022 | −0.072 ± 0.016 |
| censored_left_heavy / mean | 0.616 | 0.735 | −0.126 ± 0.029 | −0.113 ± 0.018 |
| censored_left_heavy / full | 0.630 | 0.743 | −0.112 ± 0.027 | −0.105 ± 0.018 |

Full − mean, paired (Δacc, ΔAUC, datasets where full is better):

| noise | Δacc | ΔAUC |
|---|---|---|
| gauss_valhet | −0.005 ± 0.003 (7/27) | +0.002 ± 0.006 (9/27) |
| cauchy_valhet | +0.042 ± 0.008 (24/27) | +0.040 ± 0.007 (23/27) |
| censored_left | −0.006 ± 0.002 (7/27) | −0.005 ± 0.002 (5/27) |
| censored_left_heavy | +0.014 ± 0.004 (20/27) | +0.008 ± 0.006 (20/27) |

The same pattern as the original five models:
- Light noise costs 6–7 points whether it's value-dependent or left-censored, and extra summary columns slightly hurt.
- Heavy tails cost 11–15 points, and the summary columns (via the median) recover 1–4 points.
- Value-dependent noise costs *less* than homoscedastic noise of the same average scale (gauss_valhet −5.8 vs gauss −6.9 points). Low-value cells are almost exact.
- Fine-tuning was not run on these arms, since it had no effect on any of the original arms.

### Results — fine-tuned (steps 3, 7, 8)

**Picking hyperparameters (pilot pool, 12 datasets × {clean, gauss/full, cauchy_hetero/full}).** Grid: lr ∈ {1e-5, 1e-4} × epochs ∈ {30, 100}.
- Every setting changed holdout metrics by at most ±0.007 log-loss and ±0.006 AUC.
- lr 1e-4 slightly hurt gauss log-loss (+0.007).
- Early stopping usually ended a run within a few seconds.
- I chose lr 1e-5 with 100 epochs: neutral everywhere, and marginally best on hetero.
- Note that with 512 context rows one epoch is a single optimizer step, so "100 epochs" means at most 100 steps.

**Eval (27 datasets × 11 arms, 297 fine-tunes).**
- Weights actually changed in 177 of 297 runs. In the rest, the "never worse on validation" safety net kept the pretrained weights.
- Median fine-tuning time was 9 s per dataset.
- Full tables are in [figures/results.md](experiments/obs_noise/figures/results.md), the plot in [figures/gap.png](experiments/obs_noise/figures/gap.png).

| arm | Δacc (FT − ZS) | ΔAUC | Δlog-loss | noisy − clean gap, ZS → FT (acc) |
|---|---|---|---|---|
| clean | +0.000 ± 0.001 | −0.000 ± 0.000 | −0.000 ± 0.000 | — |
| gauss / mean | −0.001 ± 0.001 | −0.001 ± 0.001 | +0.001 ± 0.001 | −0.069 → −0.070 |
| gauss / full | +0.001 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | −0.073 → −0.073 |
| student_t3 / mean | −0.000 ± 0.001 | −0.000 ± 0.000 | +0.001 ± 0.001 | −0.085 → −0.086 |
| student_t3 / full | −0.000 ± 0.001 | −0.001 ± 0.001 | +0.000 ± 0.001 | −0.082 → −0.083 |
| censored / mean | −0.001 ± 0.001 | −0.001 ± 0.001 | +0.001 ± 0.001 | −0.073 → −0.074 |
| censored / full | +0.000 ± 0.000 | +0.000 ± 0.001 | −0.000 ± 0.001 | −0.076 → −0.077 |
| cauchy / mean | +0.000 ± 0.001 | +0.001 ± 0.001 | −0.002 ± 0.002 | −0.161 → −0.161 |
| cauchy / full | +0.003 ± 0.001 | +0.000 ± 0.000 | −0.003 ± 0.002 | −0.114 → −0.111 |
| cauchy_hetero / mean | +0.001 ± 0.001 | +0.001 ± 0.000 | −0.002 ± 0.001 | −0.172 → −0.171 |
| cauchy_hetero / full | −0.001 ± 0.001 | +0.001 ± 0.001 | −0.003 ± 0.001 | −0.125 → −0.126 |

**Observations (fine-tuning):**
1. **Per-dataset fine-tuning recovers nothing** (step 8). The noise gap is unchanged in every arm: at most 0.3 points of accuracy recovered, i.e. between −3% and +2% of the gap closed. On clean data it is also a no-op, consistent with TabICL already being near its in-context optimum on these prior datasets.
2. **The information isn't there to recover.** With 512 noisy rows, one dataset can't teach the model a new observation model. The only lever that worked was giving it a better *input* (the median inside the `full` arm, +4.7 points under Cauchy). That points to pretraining over observation models (Exp 2), not per-dataset adaptation.
3. Fine-tuning helps log-loss slightly under heavy tails (−0.002 to −0.003, 14–17 of 27 datasets improve). This is calibration, not discrimination.

---

## Exp 2 — Replicate-level cell attention (set encoder) on nanoTabICL (2026-10-18)

**Question.** Can a small set encoder over each cell's replicates (NOTES.md Stage 1), added to a pretrained
nanoTabICL, recover the noise gap? Does it keep exact stock behaviour when n = 1? And does it beat the two
baselines: mean imputation, and summary-stat columns?

### Architecture ([obs_model.py](obs_model.py))

`ObsNanoTabICL(NanoTabICLv2)`, following the NOTES.md spec:
- **Input.** Replicates (B, R, C, K=10) plus a validity mask. The backbone receives one value per cell (the cell mean, as the spec says).
- **Encoder, per cell with n ≥ 2.**
  - Standardize with the training rows' column statistics.
  - φ: MLP(z, z − z̄) → d_s = 32.
  - One pre-LN masked self-attention block over the replicates.
  - Attention pooling: one learned seed query, plus a learned null key/value.
  - Concatenate log n.
- **Grouping.** Cell vectors are grouped at column offsets (0, 1, 3), the same as the backbone.
- **Injection.** A **zero-initialized, bias-free** linear map U adds the result to the backbone's cell token. This needed a 2-line hook in [model.py](model.py) (`extra_emb`, default None = unchanged).
- **Guarantees, verified to 0.0 max |Δ| in `python obs_model.py`:**
  - (i) At step 0 the model equals the stock model.
  - (ii) Cells with n = 1 contribute exactly zero at *any* weights. The encoder only runs on n ≥ 2 cells (gather/scatter), and U has no bias.

  So with one sample everywhere, the model *is* the stock model.
- **Size.** The replicate branch is 16k parameters on a 0.97M backbone.
- **Options added while iterating** (all keep guarantee (ii)):
  - `center`: the per-cell value the backbone sees.
    - `'mean'`: the spec.
    - `'median'`: the per-cell lower median.
    - `'attn'`: a learned location Σ_k softmax(s_k)·x_k, with per-replicate scores s_k from a zero-initialized head on the encoder's replicate states. It starts as the mean, so step 0 equals stock-on-means to 3e-7.
    - For `median` and `attn`, the replicates are standardized with the cell-*median* column statistics.
  - `med_feat`: φ also gets z − median(z) as a robust reference.
  - `use_sab`, `use_null`, `d_s`: ablations of the spec's components.

### Training protocol ([obs_train.py](experiments/obs_noise/obs_train.py))

- **Backbone.** ~1M-parameter nanoTabICLv2: embed 32, col/row/icl blocks = 2/2/6, 4 heads, 32 inducing points, 10 classes. This matches the NOTES.md budget.
- **Bank** ([gen_bank.py](experiments/obs_noise/gen_bank.py)). 2000 blocks × 64 = 128k clean nanoprior classification tables.
  - 1024 rows each; 2–16 numerical features per block; 2–10 classes.
  - Seeds are disjoint from Exp 1's eval and pilot sets.
  - Blocks 0–15 are held out for validation.
- **Batches.** 32 tables. Random 256–1024 rows, random 30–80% context.
- **Observation model.** Sampled on the GPU per table.
  - 25% of tables are clean (n = 1).
  - Otherwise one of the 5 Exp 1 noise models, scale ~ logU(0.1, 1)·sd(col), counts ~ U{1..10} per cell.
  - Censoring limits are random quantiles (0–0.2 / 0.8–1).
- **Optimizer.** AdamW (β = 0.9/0.95, wd 0.01), warmup then cosine to 10%, grad-clip 1, bf16 autocast. CE loss on test rows whose class appears in the context.
- **Phase 1 (stock).** 12k steps on clean tables, lr 1e-3.
- **Phase 2.** Every arm starts from the stock checkpoint with the *same* budget, data and noise stream, lr 3e-4:
  - `p2_clean`: continue on clean data. The fair 1-sample reference.
  - `p2_mean`: backbone on noisy cell means (= imputation).
  - `p2_stats`: backbone on 6 summary columns per feature (mean, median, sd, IQR, SEM, log n), i.e. 6× the columns.
  - `p2_obs*`: the replicate encoder.
- **Evaluation** ([obs_eval.py](experiments/obs_noise/obs_eval.py)). On Exp 1's 27 eval datasets, using the *stored* replicates (scale 0.5, censoring at 10/90%): same splits, one forward pass, no ensembling.

**Reproducibility.**
- Deterministic CUDA kernels (`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`).
- The complete bank is required (the block count is asserted).
- Checkpoints carry the numpy and torch RNG states, so a crash-and-resume continues the exact same stream.
- **Verified:** two identical 150-step runs give bit-identical weights, for both the stock and obs models. A run that crashed at step 100 and resumed is also bit-identical to the uninterrupted run.
- The final stock checkpoint `artifacts/pretrain/stock/last.ckpt` came from one uninterrupted run. Its weights sha256 is `e99ed0213349caf02d686dfdf87edd4bbc5d0475eaff6c0eff9dd89659a916f6`.
- Determinism holds on this GPU/driver/library stack (RTX 5070 Ti, torch 2.10+cu128); other hardware may differ.
- `torch.nanmedian` has no deterministic CUDA kernel, so median/IQR use a sort-based quantile that matches numpy's (|Δ| ≤ 3e-14).
- An earlier *preliminary* stock run, trained while the bank was still being generated and restarted once, is kept as `stock_prelim`. It was used **only for architecture exploration** below.

```bash
cd experiments/obs_noise
python gen_bank.py                                      # ~30 min on 14 cores
./supervise.sh stock --mode clean --steps 12000 --batch_size 32 --micro_batch 16 --lr 1e-3 --val_every 1000
./phase2.sh p2_clean p2_mean p2_stats p2_obs ...        # STEPS=4000 by default; supervise.sh = auto exact-resume
python obs_eval.py stock=artifacts/pretrain/stock/last.ckpt p2_obs=artifacts/pretrain/p2_obs/last.ckpt ...
```

### Exploration (from `stock_prelim`, 3000 steps, original 5-model noise mixture)

Used only to choose architectures; the final comparison below is re-run from the reproducible stock checkpoint.
Evaluated on the 27 Exp 1 datasets. The 4 new observation models (below) were **unseen** in this training, so their
columns measure generalization. Accuracy, paired Δ vs `x_p2_stats` (± SE, datasets better of 27):

| model | clean | gauss | t3 | cens | cauchy | c_het | g_valhet* | c_valhet* | cens_L* | cens_L_hvy* |
|---|---|---|---|---|---|---|---|---|---|---|
| stock_prelim (no phase 2) | +0.009±0.003 | −0.004±0.004 | −0.009±0.003 | −0.002±0.002 | −0.075±0.016 | −0.069±0.015 | +0.002±0.004 | −0.072±0.015 | −0.001±0.003 | −0.009±0.005 |
| x_p2_mean | +0.005±0.004 | +0.000±0.004 | −0.004±0.002 | +0.003±0.002 | −0.052±0.011 | −0.048±0.011 | +0.007±0.004 | −0.048±0.011 | +0.005±0.004 | −0.006±0.004 |
| **x_p2_stats** (abs.) | 0.700 | 0.654 | 0.643 | 0.649 | 0.613 | 0.600 | 0.652 | 0.615 | 0.643 | 0.605 |
| x_p2_obs (spec, mean-centered) | +0.005±0.003 | +0.006±0.003 | +0.005±0.003 | +0.006±0.003 | −0.009±0.004 | −0.012±0.004 | +0.007±0.003 | −0.006±0.005 | +0.006±0.003 | +0.009±0.004 |
| x_p2_obs_med (median-centered) | +0.006±0.003 | +0.002±0.003 | +0.005±0.003 | −0.003±0.002 | +0.007±0.004 | +0.001±0.002 | +0.004±0.003 | +0.001±0.003 | +0.001±0.004 | +0.005±0.003 |

\* not in the training mixture for these runs.

**What exploration showed:**
- **Training on noise helps everyone a bit, but means can't fix heavy tails.** `x_p2_mean` gains 2–3 points on Cauchy over stock but stays 5 points behind summary stats. The information is not in the mean.
- **The spec encoder (mean-centered) wins on light noise and loses on Cauchy.**
  - It beats summary stats by 0.5–0.9 points on gauss, t3, censoring and all three light *unseen* models, including `censored_left_heavy` (+0.9, AUC +0.010, 21/27).
  - It trails stats by about 1 point on Cauchy.
  - Diagnosis: the backbone still embeds the garbage cell mean, which the added encoder vector has to cancel. Also, replicates are standardized with cell-mean statistics, and heavy tails inflate the column sd, squashing every z toward 0.
- **The median-centered encoder fixes Cauchy but gives back the light-noise edge.**
  - Feeding the backbone the per-cell median, and standardizing with median-column statistics, makes it ≥ stats on Cauchy (+0.7 acc, AUC +0.008, 18/27) and ties elsewhere.
  - At *step 0* (encoder still a no-op) this variant already matches what the spec encoder reached after 3000 steps. Most of the heavy-tail gain is the robust center itself.
- **Summary stats are a strong baseline under heavy tails**, mostly via the median column. They cost 6× the columns: 1.3× the training-step time and 1.4× the memory of the SAB encoder, 4.7× stock (see the cost table below). They also slightly hurt clean accuracy (−0.6 to −0.9 vs the other arms).
- Every encoder variant keeps n = 1 behaviour exact by construction. On clean data its accuracy is within noise of `x_p2_mean` (0.705–0.707).

### Final comparison (from the reproducible `stock`, 4000 steps, 9-model noise mixture)

All arms start from the same stock checkpoint and see the same data/noise stream. They are evaluated on the 27 Exp 1
datasets, using the stored replicates (scale 0.5, the 9 observation models).
- Full tables (accuracy, ROC-AUC, log-loss; absolute values and paired Δ): [figures/pretrain_results.md](experiments/obs_noise/figures/pretrain_results.md).
- Raw per-dataset records: `artifacts/pretrain_eval_final.jsonl`.

Accuracy, mean over 27 datasets (best noisy-setting value in bold):

| model | clean | gauss | t3 | cens | cauchy | c_het | g_valhet | c_valhet | cens_L | cens_L_hvy |
|---|---|---|---|---|---|---|---|---|---|---|
| stock (no phase 2) | 0.703 | 0.652 | 0.630 | 0.644 | 0.531 | 0.534 | 0.648 | 0.550 | 0.642 | 0.592 |
| p2_clean (clean only) | 0.709 | 0.649 | 0.633 | 0.642 | 0.539 | 0.537 | 0.650 | 0.560 | 0.640 | 0.597 |
| p2_mean (imputation) | 0.701 | 0.656 | 0.638 | 0.648 | 0.560 | 0.551 | 0.657 | 0.574 | 0.647 | 0.600 |
| p2_stats (6 summary cols) | 0.701 | 0.653 | 0.644 | 0.648 | 0.612 | 0.599 | 0.650 | 0.620 | 0.646 | 0.614 |
| p2_obs (NOTES.md spec) | 0.703 | 0.656 | 0.647 | 0.648 | 0.610 | 0.590 | 0.658 | 0.618 | 0.646 | 0.619 |
| p2_obs_med | 0.697 | 0.656 | 0.648 | 0.646 | 0.617 | 0.597 | **0.660** | 0.624 | **0.648** | 0.619 |
| p2_obs_med_mf | 0.704 | 0.656 | 0.647 | **0.651** | 0.618 | 0.599 | 0.656 | 0.629 | 0.644 | 0.622 |
| p2_obs_attn | 0.704 | **0.657** | 0.646 | 0.649 | 0.621 | 0.598 | 0.657 | 0.629 | 0.646 | 0.623 |
| **p2_obs_attn_mf** | 0.703 | **0.657** | 0.646 | 0.649 | **0.626** | 0.598 | 0.657 | 0.629 | 0.646 | 0.622 |
| p2_obs_attn_mf_nosab | 0.702 | 0.655 | 0.646 | 0.650 | 0.615 | 0.596 | 0.658 | 0.621 | 0.646 | 0.619 |
| p2_obs_attn_mf_lr10 | 0.702 | 0.656 | **0.649** | 0.649 | **0.626** | **0.600** | **0.660** | **0.632** | 0.645 | **0.624** |

Paired Δ vs `p2_stats` for the main arms. Each cell is accuracy (points) / log-loss, ± SE; in parentheses, datasets where the model is better, of 27:

| model | gauss | t3 | cauchy | c_het | g_valhet | c_valhet | cens_L_hvy | clean |
|---|---|---|---|---|---|---|---|---|
| p2_mean | +0.4 / −0.005 | −0.5 / +0.012 | −5.1±1.2 / +0.123 (2) | −4.8 / +0.120 | +0.6 / −0.005 | −4.6 / +0.112 | −1.4 / +0.023 | +0.0 / +0.002 |
| p2_obs (spec) | +0.3 / −0.009 | +0.4 / −0.004 | −0.2 / +0.011 | −0.9±0.4 / +0.018 | +0.8 / −0.008 | −0.3 / +0.010 | +0.5 / −0.010 | +0.2 / −0.003 |
| p2_obs_med | +0.3 / −0.004 | +0.4 / −0.006 | +0.5 / −0.009 | −0.2 / +0.002 | +1.0 / −0.010 | +0.4 / −0.002 | +0.6 / −0.013 | −0.5 / +0.005 |
| **p2_obs_attn_mf** | +0.5 / −0.008 | +0.3 / −0.010 | **+1.5±0.4 / −0.021±0.007 (21)** | −0.1 / −0.005 | +0.7 / −0.009 | +0.9 / −0.018 (22) | +0.8 / −0.019 (22) | +0.2 / −0.006 |
| p2_obs_attn_mf_lr10 | +0.3 / −0.007 | +0.5 / −0.009 | +1.4±0.3 / −0.022 (20) | +0.1 / −0.005 | +1.0 / −0.010 | +1.1 / −0.022 (22) | +1.0 / −0.021 | +0.1 / −0.001 |

**1-sample check (clean data, n = 1 everywhere).** The encoder is the stock backbone by construction, so differences on
clean data come only from continued backbone training.

| | acc | ROC-AUC | log-loss |
|---|---|---|---|
| stock | 0.703 | 0.816 | 0.729 |
| p2_clean (4000 more clean-only steps) | 0.709 | 0.819 | 0.715 |
| p2_obs_attn_mf | 0.703 | 0.818 | 0.723 |
| p2_stats / p2_mean | 0.701 / 0.701 | 0.817 / 0.817 | 0.729 / 0.731 |

- `p2_obs_attn_mf` vs stock on clean: +0.000 acc, +0.002 AUC, −0.006 log-loss.
- Vs `p2_clean`: −0.6 ± 0.4 points accuracy and −0.001 AUC. That is the same small cost every noise-trained arm pays, because only 25% of their phase-2 tables are clean.
- **One-sample behaviour = stock nanoTabICL. ✔**

**Cost** ([bench_cost.py](experiments/obs_noise/bench_cost.py); RTX 5070 Ti; bf16; 16 features; GPU otherwise idle):

| architecture | train ms/step (8 tables × 1024 rows) | train peak GiB | inference ms / table (1024 rows) | inference peak GiB |
|---|---|---|---|---|
| stock / mean (C tokens) | 41 | 1.57 | 6.2 | 0.06 |
| summary stats (6C tokens) | 194 | 8.39 | 12.2 | 0.48 |
| encoder, spec (mean, SAB) | 145 | 6.09 | 9.9 | 0.13 |
| **encoder, attn + med_feat (SAB)** | 146 | 6.16 | 10.4 | 0.13 |
| encoder, attn + med_feat, no SAB | 92 | 5.10 | 8.4 | 0.10 |

### Summary — what worked

1. **Learned attention-weighted location (`center='attn'`), plus SAB, plus a median reference feature (`p2_obs_attn_mf`).** This is the best arm overall.
   - It beats summary stats on 7 of 9 noisy settings in accuracy (the other two are ties: −0.1 and −0.0 points) and on all 10 settings (clean included) in log-loss.
   - Its clearest gains are on heavy tails: Cauchy +1.5 ± 0.4 points (21/27), `censored_left_heavy` +0.8 (22/27 on log-loss), `cauchy_valhet` +0.9.
   - It ties on row-heteroscedastic Cauchy and left-censoring.
   - It closes about 55% of stock's Cauchy gap: 0.531 → 0.626, against a clean ceiling of 0.703.
   - It needs only the 16k-parameter branch on the 1M backbone, keeps C tokens, and costs 0.75× the training time, 0.73× the memory and 0.85× the inference time of summary stats.
   - With lr 10× on the branch (`_lr10`) results are equal within noise, marginally better on value-dependent noise.
2. **The zero-init + bias-free injection + gating design from NOTES.md works exactly as intended.** Step 0 equals stock, n = 1 equals stock at any weights (bit-exact), and clean performance is unchanged after training. It also made every variant a safe drop-in: none hurt clean data beyond the shared cost of mixing in noisy training data.
3. **The self-attention block over replicates is worth keeping.** Without it the encoder loses 1.2 ± 0.2 points on Cauchy (better on only 5/27) and learns more slowly, while light noise is unchanged. Replicates need to see each other to identify which one is the outlier.
4. **A robust reference matters under heavy tails.** The median feature adds about +0.6 on Cauchy over `attn` alone. Median-based standardization keeps z informative where mean-based standardization collapses it.
5. **Pretraining over observation models (ObsPFN's premise) is what moves the needle.** TabICL per-dataset fine-tuning recovered nothing (Exp 1). Continued pretraining with noise helped every arm, and the architecture determines how much of the heavy-tail information is usable.

### Summary — what did not work (or not enough)

- **The spec as written (mean-centered backbone input + additive encoder vector)** only ties summary stats. It wins slightly on light noise and loses on Cauchy (−0.2) and row-heteroscedastic Cauchy (−0.9 ± 0.4). The backbone keeps embedding a garbage mean that the additive vector has to cancel, and mean-based standardization squashes the replicates.
- **Median-centered input (`center='median'`)** fixes heavy tails at step 0, but costs Gaussian efficiency and clean accuracy (−1.2 vs `p2_clean`, the worst of all arms). The learned location gets the best of both.
- **Mean imputation, even with noise-aware training (`p2_mean`),** stays about 5 points behind on heavy tails. The information isn't in the mean.
- **Per-dataset fine-tuning of TabICL (Exp 1)** had no effect anywhere.
- **Row-level heteroscedasticity (`cauchy_hetero`) is not solved by any arm.** All of them are within ±1 point of summary stats. The cell encoder sees one cell at a time, and nothing tells the model that a whole row is unreliable. This is NOTES.md Stage 2 territory (per-row weights / row-level pooling of the encoder's spread signal).
- **The branch remains the main cost.** It runs on every replicate of every n ≥ 2 cell: about 3.5× stock training time at K = 10. `d_s = 16` and the no-SAB variant are cheaper (no-SAB 92 vs 146 ms) but lose heavy-tail accuracy. `d_s = 16` was not run in the final set.
- **Operational lessons.**
  - Running more than two GPU jobs on 16 GB OOM'd repeatedly. With the current NVML driver/library mismatch, an OOM surfaces as an `NVML_SUCCESS ... INTERNAL ASSERT FAILED` error. Rebooting would fix the mismatch.
  - `torch.nanmedian` breaks deterministic mode.
  - Full-block validation pinned 10 GB until it was chunked.

---

## Exp 3 — Median-only, Bayes-oracle reference, calibration, grouped noise, many replicates (2026-10-18)

### New evaluation pieces

- **Median-only arms.**
  - TabICL zero-shot on `x_median` columns (`tabicl_median`).
  - nanoTabICL continued on cell medians (`p2_median`, median imputation). Same protocol as `p2_mean`: 4000 steps from `stock`, 9-model noise mixture.
  - Old eval arrays were re-verified bit-identical after adding the median columns (1080 identical, 243 new).
- **Bayes-oracle per-cell denoiser** ([bayes.py](experiments/obs_noise/bayes.py)). Posterior mean E[x_ij | replicates] under the **true** observation model:
  - Noise family and scale, censoring limits (a CDF likelihood for values reported at a limit), and the value-dependent scale rule are all known.
  - The latent row scale of `cauchy_hetero` is integrated out jointly over the row's cells (24-point Gauss–Hermite).
  - The prior is each column's true empirical distribution.
  - It ignores cross-column structure and y, so it is the Bayes-optimal *per-cell* denoiser, not a Bayes-optimal classifier. It is a strong oracle reference.
  - Denoised tables are fed to `stock`, `p2_mean` and TabICL (`+bayes`, `tabicl_bayes`).
- **Grouped heteroscedasticity** ([gen_ext.py](experiments/obs_noise/gen_ext.py)). A visible, exact group column (4 groups × 256 rows, independent of y). Each group has its own observation model, with counts ~ U{1..10}:
  - g0: light Gaussian (0.25 sd);
  - g1: Gaussian with scale growing with the value (average 0.5 sd);
  - g2: Cauchy with scale growing with the value (average 0.25 sd);
  - g3: Cauchy with a lognormal per-row scale (median 0.5 sd).
- **Many replicates.** Every cell gets exactly n = 20 or n = 50 replicates, for all 9 noise models. Training never saw more than 10.
- **Calibration** ([calib.py](experiments/obs_noise/calib.py)). Top-label ECE, signed overconfidence (mean confidence − accuracy), Brier and reliability diagrams, from saved holdout probabilities (`artifacts/probs/<model>/`).

Raw results: `artifacts/exp3_eval.jsonl` (nanoTabICL) and `artifacts/tabicl_eval.jsonl` (TabICL), with per-dataset probabilities in `artifacts/probs/`.

### 1. Median-only

Accuracy, mean over 27 datasets (K ≤ 10 settings):

| model | clean | gauss | t3 | cens | cauchy | c_het | g_valhet | c_valhet | cens_L | cens_L_hvy | grouped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tabicl_mean | 0.742 | 0.673 | 0.657 | 0.669 | 0.581 | 0.570 | 0.684 | 0.597 | 0.668 | 0.616 | 0.641 |
| **tabicl_median** | 0.742 | 0.669 | 0.661 | 0.668 | 0.633 | 0.618 | 0.679 | 0.642 | 0.665 | 0.631 | 0.664 |
| tabicl_full | 0.742 | 0.669 | 0.660 | 0.666 | 0.628 | 0.617 | 0.679 | 0.638 | 0.662 | 0.630 | 0.659 |
| p2_mean | 0.701 | 0.656 | 0.638 | 0.648 | 0.560 | 0.551 | 0.657 | 0.574 | 0.647 | 0.600 | 0.615 |
| **p2_median** | 0.700 | 0.655 | 0.647 | 0.647 | 0.611 | 0.599 | 0.653 | 0.614 | 0.643 | 0.613 | 0.641 |
| p2_stats | 0.701 | 0.653 | 0.644 | 0.648 | 0.612 | 0.599 | 0.650 | 0.620 | 0.646 | 0.614 | 0.636 |
| p2_obs_attn_mf | 0.703 | 0.657 | 0.646 | 0.649 | 0.626 | 0.598 | 0.657 | 0.629 | 0.646 | 0.622 | 0.647 |

- **For both model families, median-only is as good as the 6 summary columns, or better.**
  - TabICL: full − median is −0.5 ± 0.2 points on Cauchy and never positive at K ≤ 10.
  - nanoTabICL: `p2_stats` − `p2_median` is within ±0.6 points at K ≤ 10, and −0.3 to −1.1 at n = 20–50.
  - The summary-stat baseline's heavy-tail gain is the median column. sd/IQR/SEM/log n add nothing, and at large n they slightly hurt.
  - The one exception is **censoring with many replicates**: TabICL full beats median by +1.1 (`cens_L_n50`) and +1.8 (`cens_L_hvy_n50`). The spread columns reveal the pile-up at the limit.
- **Median vs mean** follows the noise. The mean is better for Gaussian and value-dependent Gaussian noise (+0.1 to +0.5); the median is better for t3 (+0.4 to +0.9) and much better for all heavy tails (+1.3 to +5.3).
- **The learned-location encoder beats median imputation** on heavy tails: Cauchy +1.5 ± 0.4 (19/27), `cauchy_valhet` +1.5 ± 0.6, `cens_L_hvy` +0.8 ± 0.3, grouped +0.6 ± 0.3. It ties elsewhere, including row-heteroscedastic Cauchy (−0.0).

### 2. How close to the Bayes oracle?

**Prediction.** The oracle-denoised input is the reference: the best a single-value-per-cell model can do with a perfect per-cell denoiser. Accuracy:

| setting | p2_median | **p2_obs_attn_mf** | p2_mean + Bayes | share of median → oracle gap closed | clean ceiling (p2_mean) |
|---|---|---|---|---|---|
| cauchy | 0.611 | **0.626** | 0.633 | 68% | 0.701 |
| cauchy_hetero | 0.599 | 0.598 | 0.614 | 0% | 0.701 |
| cauchy_valhet | 0.614 | **0.629** | 0.647 | 45% (lr10: 55%) | 0.701 |
| censored_left_heavy | 0.613 | **0.622** | 0.625 | 67% | 0.701 |
| grouped | 0.641 | **0.647** | 0.656 | 40% | 0.701 |
| censored_left_heavy, n = 50 | 0.650 | **0.673** | 0.675 | 92% | 0.701 |
| censored_left, n = 50 | 0.674 | **0.686** | 0.687 | 92% | 0.701 |
| cauchy, n = 50 | **0.684** | 0.678 | 0.687 | < 0 | 0.701 |

Measured from `p2_mean` (mean imputation) instead of the median, the encoder closes 90% (Cauchy), 88% (`cens_L_hvy`), 75% (`c_het`) and 78% (grouped) of the gap.

Even a perfect per-cell denoiser stays 7–9 points below clean data on heavy tails (0.633 vs 0.701). That residual uncertainty about each cell's true value can't be removed.

**Estimation** ([loc_eff.py](experiments/obs_noise/loc_eff.py), [figures/loc_eff.md](experiments/obs_noise/figures/loc_eff.md)). Error of each per-cell location estimate vs the clean value, in units of the column sd, mean over 27 datasets:

| setting | mean | median | learned (attn_mf) | learned (lr10) | Bayes oracle |
|---|---|---|---|---|---|
| gauss — median abs error | **0.153** | 0.175 | 0.179 | 0.166 | 0.096 |
| cauchy — median abs error | 0.499 | **0.244** | 0.245 | 0.245 | 0.143 |
| cauchy — RMSE (clipped at 3 sd) | 1.293 | 0.722 | 0.638 | **0.634** | 0.385 |
| grouped — RMSE (clipped) | 0.878 | 0.571 | 0.498 | **0.492** | 0.291 |
| censored_left_heavy — median abs error | 0.459 | **0.335** | 0.338 | 0.337 | 0.177 |
| cauchy, n = 50 — median abs error | 0.501 | **0.074** | 0.133 | 0.111 | 0.044 |
| gauss, n = 50 — median abs error | **0.048** | 0.059 | 0.104 | 0.069 | 0.031 |

- The Bayes oracle has about 40–60% lower error than everything else. This is its oracle knowledge (noise model, limits, value distribution) at work: it shrinks toward plausible values, and it treats a value at the limit as "somewhere below".
- Within the training range, the learned location matches the median on typical cells and makes far fewer large errors. On Cauchy its clipped RMSE is 0.638 vs 0.722, and on grouped 0.498 vs 0.571. A likely reason is that the median of n = 2 is just their mean.
- **It does not extrapolate as a location estimator to n = 20–50.** It is worse than the median, and even worse than the mean under Gaussian noise. The lr10 variant extrapolates better but still trails. Classification at n = 50 shows the same thing (Cauchy: encoder 0.678 vs median 0.684), except under censoring, where the encoder uses the pile-up at the limit (below).

### 3. Calibration

ECE for every model and setting is 0.03–0.045. The ECE of a *perfectly calibrated* predictor with 512 test rows (labels resampled from its own probabilities) is already 0.030–0.034, so all models are close to calibrated; differences sit at that floor. Reliability diagrams: [figures/calib_nanotabicl.png](experiments/obs_noise/figures/calib_nanotabicl.png), [figures/calib_tabicl.png](experiments/obs_noise/figures/calib_tabicl.png).

Signed overconfidence (mean confidence − accuracy, ± SE over 27 datasets):

| model | clean | gauss | cauchy | cauchy_hetero | cens_L_hvy | grouped | cauchy_n50 |
|---|---|---|---|---|---|---|---|
| stock (nanoTabICL, clean-trained) | −0.004±0.003 | +0.001±0.004 | **+0.016±0.004** | **+0.013±0.004** | +0.011±0.005 | **+0.013±0.004** | **+0.016±0.005** |
| p2_mean | −0.003±0.003 | +0.001±0.004 | −0.005±0.004 | +0.004±0.004 | +0.004±0.004 | −0.003±0.003 | +0.002±0.005 |
| p2_median | +0.000±0.003 | +0.001±0.004 | +0.002±0.004 | +0.002±0.003 | −0.002±0.003 | +0.000±0.003 | −0.003±0.004 |
| p2_stats | −0.000±0.004 | −0.002±0.004 | −0.000±0.003 | +0.001±0.003 | +0.001±0.004 | +0.002±0.002 | −0.008±0.004 |
| p2_obs_attn_mf | +0.004±0.003 | +0.002±0.004 | −0.008±0.004 | +0.008±0.003 | −0.000±0.004 | +0.003±0.003 | −0.007±0.004 |
| tabicl_mean | +0.014±0.004 | +0.020±0.004 | +0.014±0.004 | +0.018±0.003 | +0.020±0.004 | +0.014±0.004 | +0.017±0.004 |
| tabicl_median | +0.014±0.004 | +0.021±0.004 | +0.014±0.004 | +0.016±0.004 | +0.014±0.004 | +0.011±0.004 | +0.014±0.004 |

- **A clean-trained model becomes overconfident under heavy-tailed noise.** Stock nanoTabICL goes from −0.4 points on clean data to +1.3 to +1.6 on Cauchy, row-heteroscedastic and grouped noise. It treats noisy inputs as exact.
- **Training on noise removes this, whatever the architecture.** All phase-2 arms are within ±0.8 points of zero. The encoder is slightly *under*confident on Cauchy (−0.8 ± 0.4) and slightly overconfident on row-heteroscedastic Cauchy (+0.8 ± 0.3).
- **TabICL is uniformly overconfident by 1.4–2.2 points, including on clean data.** That is mostly its default inference temperature (`softmax_temperature=0.9`), not the noise, which adds about 0.5 points (Gaussian). No TabICL input arm changes it.
- Brier score follows accuracy. Log-loss for the encoder is better than summary stats on all 10 K ≤ 10 settings (Exp 2).

### 4. Grouped heteroscedasticity

Overall accuracy: stock 0.597, p2_mean 0.615, p2_stats 0.636, p2_median 0.641, **p2_obs_attn_mf 0.647**, oracle reference 0.656, clean (`grouped_clean`) 0.70. For TabICL: mean 0.641, full 0.659, median 0.664, oracle 0.678.

Per group (test rows of each group):

| model | g0 light Gaussian | g1 Gaussian, value-dependent | g2 Cauchy, value-dependent | g3 Cauchy, row-heteroscedastic |
|---|---|---|---|---|
| stock | 0.641 | 0.637 | 0.585 | 0.527 |
| p2_mean | 0.647 | 0.648 | 0.605 | 0.558 |
| p2_median | 0.661 | 0.655 | 0.648 | 0.600 |
| p2_stats | 0.655 | 0.653 | 0.642 | 0.596 |
| **p2_obs_attn_mf** | **0.663** | 0.653 | **0.663** | **0.610** |
| p2_mean + Bayes oracle | 0.665 | 0.672 | 0.669 | 0.618 |
| clean rows (any model) | ≈ 0.70 | ≈ 0.70 | ≈ 0.71 | ≈ 0.70 |

- The encoder is best on the heavy-tailed groups and within 0.6–0.8 points of the oracle there (g2, g3). Median imputation and summary stats trail it by 1–2 points.
- **Noise in other groups costs every model about 4 points even on the nearly clean group g0** (0.66 vs 0.70, oracle included). Rows are predicted in context, so unreliable context rows degrade every prediction.
- I did not ablate the group column, so whether any model *uses* it to learn each group's reliability is untested.

### 5. Many replicates (n = 20, 50; training saw at most 10)

| setting | stock | p2_mean | p2_median | p2_stats | p2_obs (spec) | p2_obs_attn_mf | lr10 | oracle (p2_mean + Bayes) | TabICL median | TabICL full |
|---|---|---|---|---|---|---|---|---|---|---|
| gauss n = 50 | 0.683 | 0.689 | 0.687 | 0.684 | 0.688 | 0.686 | 0.685 | 0.689 | 0.709 | 0.707 |
| cauchy n = 20 | 0.537 | 0.557 | 0.667 | 0.662 | 0.651 | 0.671 | **0.674** | 0.676 | 0.687 | 0.685 |
| cauchy n = 50 | 0.533 | 0.558 | **0.684** | 0.673 | 0.661 | 0.678 | 0.681 | 0.687 | 0.705 | 0.704 |
| cauchy_hetero n = 50 | 0.527 | 0.548 | **0.668** | 0.660 | 0.634 | 0.660 | 0.667 | 0.673 | 0.691 | 0.686 |
| censored_left n = 50 | 0.681 | 0.684 | 0.674 | 0.669 | 0.685 | **0.686** | 0.682 | 0.687 | 0.691 | 0.702 |
| censored_left_heavy n = 20 | 0.621 | 0.630 | 0.645 | 0.641 | 0.657 | **0.662** | 0.658 | 0.663 | — | — |
| censored_left_heavy n = 50 | 0.635 | 0.643 | 0.650 | 0.646 | 0.666 | **0.673** | 0.668 | 0.675 | 0.665 | 0.683 |

- **Mean-based models never benefit from more replicates under Cauchy noise.** Stock and `p2_mean` stay at about 0.55 at n = 20 and 50, because the mean of Cauchy draws doesn't converge. Every robust input approaches the clean ceiling: 0.68 at n = 50 vs 0.70.
- **Under censoring, the encoder uses extra replicates almost optimally.**
  - At n = 50 it is within 0.1–0.2 points of the oracle (0.686 vs 0.687; 0.673 vs 0.675).
  - It beats median imputation by +1.2 ± 0.3 and +2.3 ± 0.5 points (22/27).
  - Median and summary stats stall, because a median can't tell "at the limit" from "below it". The encoder appears to read the fraction of replicates piled at the limit.
  - TabICL shows the same through its spread columns (full > median by +1.1 / +1.8).
- **Under Cauchy noise with n ≥ 20 the encoder's advantage disappears.** Plain median imputation is as good or better at n = 50 (0.684 vs 0.678; lr10 0.681). The learned location under-uses replicates beyond the training range (see the estimation table). Training with K up to 50, or scaling the encoder's attention by n, would be the fix. Not tried.

### Summary — what's new from Exp 3

- **Median-only ≈ summary stats**, for both TabICL and nanoTabICL. The summary-stat baseline's value is the median column. The bar for a replicate-aware model is median imputation (on heavy tails) and mean imputation (on light noise).
- **Against the Bayes oracle**, the learned-location encoder recovers about 2/3 of the median → oracle gap on Cauchy and heavy censoring, about 40–55% on value-dependent and grouped noise, and nothing on row-level heteroscedasticity. Under censoring with many replicates it is essentially oracle-level.
- **Calibration** is near the sampling floor for every model. The real effect is that a clean-trained model is overconfident under heavy-tailed noise, and noise-aware training fixes it. TabICL's ~+1.5-point overconfidence comes from its default temperature.
- **Open problems:**
  - Row-level reliability: the row-heteroscedastic group and `cauchy_hetero` are 0% recovered. This is NOTES.md Stage 2.
  - Extrapolation to n > 10 under heavy tails.
  - Whether the group column is used as a reliability cue: needs an ablation.

---

## Exp 4 — Do more replicates shrink uncertainty? Mixed single-sample / replicated tables (2026-10-18)

**Settings** ([gen_ext2.py](experiments/obs_noise/gen_ext2.py)), on the same 27 eval datasets, for gauss, t3, cauchy, cauchy_hetero and censored_left_heavy:
- **Fixed n:** every cell has exactly n ∈ {1, 2, 3, 5, 10} replicates. With Exp 3's n = 20 and 50 this gives a 1 → 50 curve; training saw n ≤ 10.
- **`mix`:** per **row**, all cells share n. Single sample with probability 0.5, else n ~ U{2..10}.
- **`ctx10_q1`:** context rows have n = 10, query rows n = 1 ("well-replicated training data, one new measurement").
- **`ctx1_q10`:** the reverse.

The mixed designs keep the first n of one 10-replicate draw, so the replicated rows are exact supersets of the single-sample rows. Bayes oracle: `bayes.py ext2`.

Analysis: [unc.py](experiments/obs_noise/unc.py) and [cond_calib.py](experiments/obs_noise/cond_calib.py). Full tables: [figures/uncertainty_nano.md](experiments/obs_noise/figures/uncertainty_nano.md), [figures/uncertainty_tabicl.md](experiments/obs_noise/figures/uncertainty_tabicl.md), [figures/cond_calib_nano.md](experiments/obs_noise/figures/cond_calib_nano.md), [figures/cond_calib_tabicl.md](experiments/obs_noise/figures/cond_calib_tabicl.md).

![uncertainty vs n](experiments/obs_noise/figures/uncertainty_vs_n.png)

### (A) Uncertainty vs replicates per cell, the same n everywhere in the table

Accuracy / mean confidence (max probability), mean over 27 datasets:

| model, noise | n=1 | n=2 | n=3 | n=5 | n=10 | n=20 | n=50 | clean |
|---|---|---|---|---|---|---|---|---|
| p2_mean, cauchy | .561/.561 | .553/.556 | .565/.566 | .552/.557 | .562/.560 | .557/.553 | .558/.560 | .701/.698 |
| p2_median, cauchy | .553/.552 | .548/.551 | .605/.608 | .632/.628 | .657/.652 | .667/.668 | .684/.681 | .700/.700 |
| p2_obs_attn_mf, cauchy | .549/.556 | **.573/.576** | .614/.614 | .635/.632 | .658/.654 | .671/.669 | .678/.671 | .703/.707 |
| p2_mean + Bayes, cauchy | .570/.574 | .599/.601 | .618/.623 | .638/.640 | .663/.663 | .676/.676 | .687/.686 | .701/.698 |
| p2_obs_attn_mf, gauss | .620/.627 | .644/.642 | .650/.651 | .658/.662 | .673/.675 | .684/.683 | .686/.687 | .703/.707 |
| p2_obs_attn_mf, cens_L_hvy | .565/.582 | .596/.603 | .612/.613 | .630/.629 | .647/.642 | .662/.655 | .673/.667 | .703/.707 |
| tabicl_median, cauchy | .576/— | .576 | .625 | .650 | .675 | .687 | .705 | .742 |

- **Yes, at the table level.** Confidence rises with n in step with accuracy for every noise-trained model and every noise model. Signed overconfidence stays within about ±1 point of zero from n = 1 to 50.
- **Where more replicates *don't* help, confidence correctly doesn't rise.** Mean-based inputs under Cauchy stay at about 0.55–0.56 in both accuracy and confidence, because the mean of Cauchy draws never settles.
- **Exceptions:**
  - Stock (clean-trained) stays overconfident under row-heteroscedastic Cauchy at every n (+1.4 to +1.9 points).
  - The encoder is overconfident on single censored samples (cens_L_hvy n = 1: +1.7 ± 0.4) and slightly underconfident at n = 50 (−0.6), consistent with it under-using replicates beyond n = 10 (Exp 3).
- **n = 2 is where the encoder clearly beats median imputation** on heavy tails (Cauchy .573 vs .548; cens_L_hvy .596 vs .590). The median of two values is their mean. From n = 3 on, median and encoder track each other until the encoder falls slightly behind at n ≥ 20.

### (B) Mixed single-sample and replicated rows

Accuracy, mean over 27 datasets. In `mix`, single and replicated query rows are scored separately:

| model, noise | all single (k1) | mix: single rows | ctx n=10 → query n=1 | all n=10 (k10) | mix: replicated rows | ctx n=1 → query n=10 |
|---|---|---|---|---|---|---|
| p2_median, cauchy | .553 | .561 | .539 | .657 | .619 | .610 |
| p2_stats, cauchy | .552 | .556 | .499 | .648 | .606 | **.448** |
| p2_obs_attn_mf, cauchy | .549 | .552 | .537 | .658 | .617 | .588 |
| p2_mean + Bayes, cauchy | .570 | .577 | .551 | .663 | .630 | .610 |
| p2_stats, gauss | .621 | .622 | .581 | .664 | .656 | **.379** |
| p2_obs_attn_mf, cens_L_hvy | .565 | .572 | .566 | .647 | .611 | .605 |
| tabicl_full, cens_L_hvy | .582 | .582 | **.505** | .656 | .623 | .612 |
| tabicl_median, cauchy | .576 | .587 | .549 | .675 | .638 | .629 |

Overconfidence (confidence − accuracy, ± SE) on the same groups:

| model, noise | mix: single rows | mix: replicated rows | ctx n=10 → query n=1 | ctx n=1 → query n=10 |
|---|---|---|---|---|
| p2_median, cauchy | +0.017±0.008 | −0.015±0.007 | +0.048±0.013 | −0.027±0.010 |
| p2_stats, cauchy | +0.018±0.007 | −0.009±0.006 | +0.049±0.016 | −0.036±0.024 |
| p2_obs_attn_mf, cauchy | +0.024±0.008 | −0.010±0.008 | +0.051±0.013 | −0.018±0.008 |
| p2_mean + Bayes, cauchy | +0.012±0.006 | −0.013±0.007 | +0.071±0.019 | −0.032±0.008 |
| p2_obs_attn_mf, gauss | +0.017±0.005 | −0.002±0.005 | +0.031±0.009 | −0.013±0.006 |
| p2_obs_attn_mf, cens_L_hvy | +0.018±0.006 | −0.004±0.006 | +0.039±0.011 | −0.038±0.010 |
| tabicl_median, cauchy | +0.015±0.007 | +0.004±0.007 | +0.077±0.018 | −0.005±0.008 |
| tabicl_full, cens_L_hvy | +0.023±0.007 | +0.019±0.006 | +0.081±0.020 | −0.027±0.012 |

**Findings:**
1. **Within a table, uncertainty doesn't shrink enough with a row's own replicate count.** In `mix`, every model that can use replicates (TabICL, all nanoTabICL arms and the oracle-denoised input) is overconfident on single-sample rows (+0.4 to +3.7 points) and neutral-to-underconfident on replicated rows (−1.5 to +2.0). Mean-imputation arms under Cauchy noise are the exception: there replicates carry no usable information and both groups are about equal. Confidence differs between the two kinds of row less than accuracy does. The models calibrate to the table's *average* measurement quality.
   - Even the arms that see per-cell reliability (summary stats: sd/SEM/log n; encoder: log n, spread) are no better.
   - Two likely causes:
     - (i) The training mixture drew counts independently per *cell*, never row-coherently, so there was little row-level reliability signal to learn.
     - (ii) For the encoder, n = 1 cells contribute exactly zero by design, so a single noisy sample looks identical to an exact value. This is the price of the "1 sample = stock" guarantee.
2. **Replicated rows don't hurt or help single-sample rows.** Single rows in `mix` score the same as in an all-single table (within about ±1 point).
3. **When context and query precision differ, calibration and accuracy suffer** — a realistic failure mode:
   - **Well-replicated context, single-sample queries:** overconfident by +2.3 to +8.9 points for median imputation, the encoders, the oracle-denoised input and TabICL. Mean-based inputs under Cauchy noise are unaffected, and `p2_stats` is erratic (−1.3 to +4.9, SE about 2). Accuracy also drops below the all-single table (e.g. Cauchy median −1.4, TabICL full on heavy censoring −7.7 points). The model infers the noise level from the precise context and applies it to noisy queries.
   - **Single-sample context, replicated queries:** underconfident by up to −6.9 points. Accuracy is limited by the noisy context, below the all-n = 10 table (0.59–0.61 vs 0.66 on Cauchy).
4. **Summary-stat columns break under a replicate-count shift.** Trained on context rows whose sd/IQR/SEM/log n are all 0, `p2_stats` treats those columns as signal and collapses on n = 10 queries: 0.38–0.47 accuracy, below stock (0.53–0.64). It is also the worst arm on n = 1 queries with n = 10 context for 4 of the 5 noise models.
   - TabICL with full stats degrades less: 0–8 points worse than median-only in the shifted designs, with the largest loss under heavy censoring.
   - Median imputation and the encoder are immune to this particular failure. The encoder gates n = 1 cells exactly, and log n only enters through the gated branch.

**Implications for ObsPFN (NOTES.md):**
- Stage 1 should train with **row-coherent and context/query-mismatched replicate counts**, not only i.i.d. per-cell counts. Otherwise per-row uncertainty is under-resolved.
- The n = 1 gate trades away the ability to flag single samples as noisy. A learned "n = 1" embedding, applied only when the table contains replicated cells, would keep exact stock behaviour on all-single tables while letting mixed tables express it.
- Per-row reliability is the remaining gap across Exps 2–4 (row-heteroscedastic noise, mixed tables, context/query mismatch). That is Stage 2's per-row weighting.

---

## Exp 5 — Robust standardization, robust replicate features, count regimes, single-sample embedding, regression + PIT (2026-10-18/19)

### 5a. Median/MAD standardization and robust per-replicate features (classification)

**Diagnostic that motivated it.** The encoder standardized replicates with the mean/sd of the context cell centers. Under heavy tails that sd is dominated by a few extreme centers: sd ÷ (1.4826·MAD) has median 3.6 (Cauchy), 4.2 (value-dependent Cauchy), 6.1 (per-row Cauchy) and 9.5 (heavy left-censoring). The ratio exceeds 3 in 59–93% of columns, so replicate z-values were squashed by about 4–10×, and inconsistently across columns. For Gaussian/t3 noise the ratio is about 1.3.

**Changes** ([obs_model.py](obs_model.py), all keeping the exact n = 1 → stock guarantee and step 0 = stock-on-means):
- `std='medmad'`: μ_j = median of the context cell centers; σ_j = max(1.4826·MAD, IQR/1.349, 10⁻³·sd).
- `zclip=10`: soft clipping z → 10·tanh(z/10).
- `robust_feats`: φ additionally gets:
  - the robust residual (z − z̃)/(MAD_cell + 0.05), soft-clipped;
  - the within-cell rank (k − 1)/(n − 1);
  - a limit-tie flag: the replicate equals the column's extreme context value, e.g. a censoring limit.

  So φ's input is [z, z − z̄, z − z̃, r, rank, tie].

Arms (from `stock`, 4000 steps, same protocol as Exp 2): `p2_obs_attn_mf_rstd` (standardization + clip) and `p2_obs_attn_mf_rstd_rf` (+ robust features).

**Accuracy, paired Δ vs `p2_obs_attn_mf`** (± SE; datasets better of 27):

| model | clean | gauss | t3 | cens | cauchy | c_het | g_valhet | c_valhet | cens_L | cens_L_hvy | grouped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p2_median | −0.3 | −0.3 | +0.1 | −0.1 | −1.5±0.4 (6) | +0.0 | −0.3 | −1.5±0.6 | −0.4 | −0.8±0.3 | −0.6 |
| p2_stats | −0.2 | −0.5 | −0.3 | −0.1 | −1.5±0.4 (5) | +0.1 | −0.7 | −0.9 | +0.0 | −0.8±0.3 | −1.1±0.4 |
| rstd | +0.1 | −0.1 | +0.3 | +0.3 | +0.1 | **+0.8±0.3 (17)** | +0.4 | +0.4 | +0.0 | +0.1 | −0.1 |
| **rstd_rf** | **+0.4±0.2** | +0.2 | +0.3 | +0.4 | +0.1 | **+1.0±0.3 (18)** | +0.2 | +0.3 | **+0.5±0.2** | +0.3 | +0.3 |
| p2_mean + Bayes oracle | −0.2 | +0.4 | +0.8 | +0.6 | +0.7 | +1.6±0.4 | +0.9 | +1.7±0.4 | +0.6 | +0.4 | +0.9 |

(Values in accuracy points; absolute `attn_mf`: 0.703 / 0.657 / 0.646 / 0.649 / 0.626 / 0.598 / 0.657 / 0.629 / 0.646 / 0.622 / 0.647.)

- **rstd_rf improves on attn_mf on all 11 settings in accuracy and all 11 in log-loss** (e.g. Cauchy log-loss −0.011 ± 0.005, 20/27). The gains are modest, at most 1 point.
- **First real progress on per-row Cauchy** (+1.0 ± 0.3), the setting every earlier encoder failed on. It has the largest squashing factor (6.1×), consistent with the diagnosis.
- **It fixes the many-replicate weakness:** Cauchy n = 50 0.683 (attn_mf 0.678, median 0.684, oracle 0.687); per-row Cauchy n = 50 0.666 (attn_mf 0.660).
- **The robust features mostly help censoring and clean data** (cens_L +0.5, clean +0.4), as intended. Heavy-tail gains come from the standardization.
- Against median imputation / summary stats, the lead on heavy tails is now +1.6 to +2.5 points.
- **Not fixed:** per-row calibration in mixed tables (single-sample rows still +1.1 to +2.1 overconfident; precise context → single-sample queries still +2.4 to +4.6). That's the target of 5b.

**Decision:** `rstd_rf` (`{"center": "attn", "med_feat": true, "std": "medmad", "zclip": 10, "robust_feats": true}`) is the encoder base for the next steps.

---

## Exp 6 — Literature-based measurement modes, median-breaking failures, and a pruned core suite (2026-10-19)

### Measurement / censoring modes in replicated experimental data (short literature survey)

| mode | where it shows up | how it's usually (mis)handled | implemented as |
|---|---|---|---|
| Left-censoring at LOD/LOQ | immunoassays, environmental, bioassay cohorts | substitution (LOD/2) or deletion: biased, variance too small; Tobit / multiple imputation better [1–3] | `censored_left`, `censored_left_heavy` (Exp 2) |
| Upper limit of quantification / detector saturation | assays with a ceiling, HTS dose-response | treated as censored ("only the bound is known") [6]; plateaus make potency unidentifiable [7] | **`saturation`**: right-censoring at the clean 75th pct, Gaussian 0.5 sd |
| Intensity-dependent missingness | label-free proteomics, qPCR non-detects | detection probability is **logit-linear in intensity**, "neither random nor censored" [4]; qPCR software substitutes Ct = 40 for undetermined values [5] | **`lod_logistic`**: detected w.p. sigmoid((r − θ)/s), θ = clean 35th pct, s = 0.3 sd; undetected → substitution floor (5th pct) |
| Non-detect substitution / failed reads | qPCR (Ct = 40), failed wells | majority-non-detect replicate sets summarized as the floor value [5] | **`fail_floor`**: each replicate fails w.p. p_cell ~ U(0, 0.8) → column floor (min − 1 sd) |
| Contamination / second mode | sample mix-ups, systematic error | — | **`fail_shift`**: failed replicates at x + 3 sd (bimodal), p_cell ~ U(0, 0.8) |
| Constant-CV (multiplicative) error | ELISA, most concentration assays | log-transform; a constant CV is more realistic than a constant variance [8, 9] | **`cv_lognormal`**: r = m + (x − m)·exp(0.35 ε − 0.35²/2), m = min − 0.25 sd |
| Plate / run (batch) effects | ELISA inter-assay CV, HTS plates (row/column/edge) [9, 10] | normalization to controls, LOESS | **`plate_effect`**: replicate k comes from run k, shared by the whole table: offset b_k ~ N(0, 0.5²)·sd (all columns) + c_kj ~ N(0, 0.2²)·sd, plus 0.25 sd noise. Replicates are **aligned across features**. |

Sources: [1] [PMC2592244](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2592244/), [2] [arXiv 2512.10212](https://arxiv.org/pdf/2512.10212), [3] [CDC stacks 201356](https://stacks.cdc.gov/view/cdc/201356/cdc_201356_DS1.pdf), [4] Li & Smyth 2023 [PMC10174703](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10174703/), [5] McCall et al. 2014 [qPCR non-detects](https://gene-quantification.com/mccall-qpcr-non-detects-2014.pdf), [6] [Monolix LOQ](https://monolix.lixoft.com/?p=16626), [7] [arXiv 2608.26538](https://arxiv.org/pdf/2608.26538), [8] [SLU report 2004:2](https://www.slu.se/contentassets/5116f4fbceaa42e7817a7a4acb4a24df/report2004_2.pdf), [9] [Enzo: CV in ELISA](https://www.enzo.com/note/cv-in-elisa-how-to-reduce-them-and-why-theyre-important/), [10] [Mpindi et al. 2015](https://research.utu.fi/converis/portal/detail/Publication/18048396?lang=en_GB).

**Median-breaking modes.** In `fail_floor` and `fail_shift`, about 37% of cells have a failure rate above 50%, so the median picks the failure value. On one dataset the median is off by more than 1 sd in 42–44% of cells (Cauchy: 10%). The Bayes oracle reduces the median absolute error from 0.30 (median) to 0.13 (`fail_floor`) and 0.08 (`fail_shift`), so the modes are well-posed.

**Bayes oracle for the new modes** ([bayes.py](experiments/obs_noise/bayes.py) `special_loglik`):
- mixtures with the failure rate integrated over a 16-point grid;
- a log-normal likelihood for constant CV;
- a logit-linear detection model, with the logistic-normal approximation for non-detects;
- right-censoring for saturation.

For `plate_effect` the oracle is **approximate**: a per-cell Gaussian with the total variance, ignoring the run correlation across rows. Checks: the GPU training versions match numpy statistically (cell-median error within 0.01, and within the table-to-table spread for `plate_effect`); old eval arrays are bit-identical after appending (1593 per task).

### Pruned core evaluation suite

Exps 1–5 showed several settings behaving like a sibling in every comparison (same ranking, same gaps). The **core suite** (`SUITE=core`, now the default in [obs_eval.py](experiments/obs_noise/obs_eval.py)) drops them:

| dropped | behaves like |
|---|---|
| `student_t3`, `censored` (two-sided), `gauss_valhet`, `censored_left` | `gauss` (light noise: ±0.5 points across all arms) |
| `cauchy_valhet` | `cauchy` |
| n-curve and mixed designs for `student_t3`, `cauchy_hetero` | curves track `gauss` / `cauchy` |

**Core suite:**
- **main** (counts U{1..10}): clean, gauss, cauchy, cauchy_hetero, censored_left_heavy, fail_floor, fail_shift, cv_lognormal, lod_logistic, saturation, plate_effect;
- **n curve** n ∈ {1, 2, 3, 5, 10, 20, 50} and **mixed designs** (mix, ctx10_q1, ctx1_q10) for gauss, cauchy and censored_left_heavy;
- **grouped.**

That's 43 settings instead of 70+. The dropped settings remain on disk (`SUITE=full`).

### 5b. Count regimes + learned single-sample embedding e₁ (classification): calibration not fixed

`p3_obs_e1` = rstd_rf encoder + e₁ (`single_emb='scaled'`: zero-init MLP(log column replicate spread) for n = 1 cells, only in columns whose context has replicated cells), trained from `stock` with count regimes (40% per-cell, 20% row-coherent, 20% column-coherent, 20% context ≠ query) and 20% of batches with up to 50 replicates.

- Accuracy on the core suite is about 0.4 points below `rstd_rf` (Cauchy 0.624 vs 0.628). The training budget is split across regimes.
- **Mixed-table calibration is essentially unchanged.**

  | setting | rstd_rf | p3_obs_e1 |
  |---|---|---|
  | `mix`, single-sample rows | +1.1 to +2.1 | +1.4 to +1.9 |
  | context n = 10 → query n = 1 | +2.4 to +4.6 | +2.7 to +3.4 |

  (Overconfidence, confidence − accuracy, in points.)
- So the model now *has* the single-sample information (e₁ marks it), but ordinary training doesn't turn it into per-row confidence.

### 5c. Regression baseline (stock nanoTabICL, 99-quantile head)

`stock_reg`: 12k deterministic steps on a separate regression bank (128k tables). Standardized targets are soft-clipped to ±30 sd *in training only*: 0.2% of prior tables have test targets beyond 100 sd (max 1.5×10⁵), which made the unclipped train loss 120 instead of about 0.2. Weights sha256 `aed0746360815f43ef18c2cfbe5b7d26594a78b5f312c2cce909fd3ad63c32dc`.

Core-suite main settings, mean over 27 regression datasets:

| | clean | gauss | cauchy | c_het | cens_L_hvy | fail_floor | fail_shift | cv_ln | lod_lgt | satur | plate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CRPS, stock_reg | 0.160 | 0.257 | 0.433 | 0.435 | 0.349 | 0.420 | 0.376 | 0.288 | 0.250 | 0.276 | 0.238 |
| CRPS, stock_reg + Bayes-denoised input | 0.160 | 0.250 | **0.299** | **0.322** | 0.308 | **0.255** | **0.236** | 0.282 | 0.237 | 0.267 | 0.230 |
| PIT calibration error, stock_reg | 0.062 | 0.033 | 0.029 | 0.030 | 0.028 | 0.028 | 0.029 | 0.033 | 0.035 | 0.035 | 0.035 |
| 90% interval coverage, stock_reg | 0.919 | 0.906 | 0.910 | 0.913 | 0.895 | 0.907 | 0.906 | 0.903 | 0.899 | 0.904 | 0.906 |

- **Stock regression is already well calibrated on average under noise** (PIT calibration error about 0.03, 90% coverage 0.90–0.91). Noisy features make the target look noisier in context, so the model widens its intervals.
- It is slightly *under*-confident on clean data (coverage 0.919, PIT error 0.062).
- The regression target is **sharpness at equal calibration**. The oracle shows the room: CRPS 0.43 → 0.30 (Cauchy), 0.42 → 0.26 (`fail_floor`), 0.38 → 0.24 (`fail_shift`).

### 5d. Next: probabilistic cell embedding + per-row temperature head

Plain training didn't turn reliability information into per-row confidence, so two mechanisms that act on it directly ([obs_model.py](obs_model.py)):
- **`dist_emb`: fixed, non-learned probabilistic descriptors** for every cell, including n = 1, in columns whose context has replicated cells:
  - **Empirical-Bayes Gaussian posterior.** Pooled column variance s²_j = median of (1.4826·MAD_cell)² over replicated context cells; shrunk cell variance v = (3·s²_j + (n−1)·s²_cell)/(3 + n − 1); posterior sd s_post = √(v/n). A single-sample cell gets s_post = s_j, the column's noise level, instead of nothing.
  - **Kernel mean embedding.** Mean over replicates of cos(w_m·z + b_m), with 8 fixed random Fourier features.
  - Both enter the cell vector through a zero-initialized, bias-free map.
- **`temp_head`: a per-query-row temperature** T = exp(W₂·tanh(W₁·f)), bias-free, W₂ zero-initialized. The features f are:
  - the row's mean log n, its share of single-sample cells in replicated columns, mean log(1 + within-cell MAD) and mean log s_post;
  - the context means of three of these.

  Classification: logits ÷ T. Regression: quantiles spread around the median × T, computed as out + (out − med)·(eᵗ − 1), which is exact at t = 0.
- **Guarantees, verified:** all features and descriptors are exactly 0 for all-single tables, so all-single tables equal stock at *random* weights (max |Δ| = 0.0) for both tasks. Step 0 equals stock-on-means (to 3e-7).

### 5e. Regression results: encoder vs median imputation vs oracle (27 regression datasets, core suite)

All regression arms start from `stock_reg`, with the same 4000-step protocol, count regimes and target clipping. `r3_obs_e1` = rstd_rf encoder + e₁; `r3_median` = median imputation; `stock_reg + Bayes` = stock fed oracle-denoised cells.

**CRPS** (lower is better):

| model | clean | gauss | cauchy | c_het | cens_L_hvy | fail_floor* | fail_shift* | cv_ln* | lod_lgt* | satur* | plate* | grouped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stock_reg | 0.160 | 0.257 | 0.433 | 0.435 | 0.349 | 0.420 | 0.376 | 0.288 | 0.250 | 0.276 | 0.238 | 0.338 |
| r3_median | 0.161 | 0.256 | 0.323 | 0.344 | 0.322 | **0.379** | **0.346** | 0.289 | 0.251 | 0.282 | 0.237 | 0.275 |
| **r3_obs_e1** | 0.161 | **0.251** | **0.306** | **0.328** | **0.304** | 0.390 | 0.364 | **0.285** | **0.244** | **0.273** | **0.231** | **0.262** |
| stock_reg + Bayes oracle | 0.160 | 0.250 | 0.299 | 0.322 | 0.308 | 0.255 | 0.236 | 0.282 | 0.237 | 0.267 | 0.230 | 0.258 |

\* not in the training mixture of these runs (Exp 6 modes).

**Paired Δ CRPS, encoder − median imputation** (± SE, datasets where the encoder is better, of 27):
- per-row Cauchy **−0.015 ± 0.003 (27)**; Cauchy −0.016 ± 0.004 (26); heavy censoring −0.018 ± 0.003 (25); grouped −0.012 ± 0.002 (21); detection dropout −0.007 ± 0.001 (25); saturation −0.010 (23); Gaussian −0.005 (23).
- **Worse on the unseen failure modes:** `fail_floor` +0.011 ± 0.004 (6), `fail_shift` +0.018 ± 0.004 (4).

**PIT calibration error / 90% coverage:**
- Encoder: 0.025–0.055 / 0.89–0.91.
- Median imputation: 0.024–0.047 (marginally lower, within about 1 SE on most settings).
- Stock: 0.028–0.062; stock overcovers clean data (0.919).
- The oracle-denoised input is *worse* calibrated than the encoder on 10 of 12 settings: it is a sharp point input, with no information about how sure the denoising is.

**Takeaways:**
1. **In regression, the encoder is close to the oracle denoiser on heavy tails and censoring** (Cauchy 0.306 vs 0.299; heavy censoring 0.304 vs 0.308, i.e. better), at equal or better calibration. That makes it a larger relative gain than in classification.
2. **The failure modes it never saw remain the gap** (`fail_floor` 0.390 vs oracle 0.255). Median imputation degrades less there because it ignores the failed minority automatically when failures are rare.
3. **Next:** train with the Exp 6 modes in the mixture (`--noise_set v2`), for both the encoder and the median baseline (`r4_*`).

### 5f. Probabilistic cell embedding + per-row temperature: fixes mixed-table calibration (classification)

`p3_obs_dist_temp` = rstd_rf encoder + `dist_emb` (empirical-Bayes posterior sd + kernel mean embedding) + `temp_head`, trained like `p3_obs_e1` (from stock, count regimes, 4000 steps). Evaluated on the core suite (27 datasets).

**Conditional calibration, overconfidence (confidence − accuracy) ± SE:**

| noise | model | mix: single-sample rows | mix: replicated rows | ctx n=10 → query n=1 | ctx n=1 → query n=10 |
|---|---|---|---|---|---|
| gauss | rstd_rf | +0.011±0.006 | −0.006±0.005 | +0.024±0.008 | −0.018±0.005 |
| gauss | p3_obs_e1 | +0.014±0.004 | −0.003±0.005 | +0.027±0.008 | −0.013±0.006 |
| gauss | p3_median | +0.017±0.005 | −0.004±0.005 | +0.033±0.008 | −0.015±0.007 |
| gauss | **p3_obs_dist_temp** | **−0.002±0.005** | **+0.007±0.006** | **+0.000±0.008** | **+0.004±0.006** |
| cauchy | rstd_rf | +0.017±0.009 | −0.011±0.007 | +0.046±0.014 | −0.025±0.009 |
| cauchy | p3_obs_e1 | +0.019±0.008 | −0.008±0.007 | +0.034±0.012 | −0.024±0.010 |
| cauchy | p3_median | +0.012±0.008 | −0.011±0.007 | +0.036±0.012 | −0.020±0.009 |
| cauchy | **p3_obs_dist_temp** | **+0.001±0.008** | **−0.002±0.006** | **+0.015±0.013** | **−0.003±0.009** |
| cens_L_hvy | rstd_rf | +0.019±0.006 | −0.008±0.006 | +0.044±0.014 | −0.046±0.012 |
| cens_L_hvy | p3_obs_e1 | +0.018±0.005 | −0.010±0.007 | +0.041±0.010 | −0.028±0.009 |
| cens_L_hvy | p3_median | +0.013±0.005 | −0.004±0.004 | +0.039±0.010 | −0.018±0.007 |
| cens_L_hvy | **p3_obs_dist_temp** | **+0.004±0.006** | **+0.002±0.007** | **+0.025±0.012** | **−0.004±0.008** |

- **Per-row miscalibration in mixed tables is essentially gone:** single vs replicated rows are both within ±0.7 points of zero, for all three noise models.
- **The hardest case (precise context, single-sample queries) drops from +3.3 to +4.6 to 0.0 / +1.5 / +2.5.** It is not fully solved under heavy censoring.
- **Learned e₁ alone (`p3_obs_e1`) did not fix this; neither did training under count regimes.** The difference is an explicit uncertainty quantity (posterior sd, defined for n = 1 cells) plus a mechanism that acts on output confidence per row. An ablation with the temperature head alone (`p3_obs_temp`) is running to separate the two.

**Accuracy / log-loss, paired Δ vs median imputation (`p3_median`, regime-trained):**
- **Accuracy better on 9 of 11 noisy settings:** constant-CV +1.6 ± 0.4 (21/27), Cauchy +1.0, per-row Cauchy +1.0, grouped +1.0, heavy censoring +0.8, saturation +0.7, plate +0.7, detection dropout +0.5, Gaussian +0.4.
- **Log-loss better on 10 of 12:** Cauchy −0.028 (25/27), heavy censoring −0.027.
- **Worse on the unseen failure modes** (`fail_floor` −0.6, `fail_shift` −0.5 accuracy; +0.01 log-loss), the same as in regression.
- **Clean is unchanged:** −0.1 accuracy, −0.004 log-loss.

**Decision:** `rstd_rf + dist_emb + temp_head` is the current best configuration. Next, it trains with the Exp 6 modes in the mixture (`--noise_set v2`) for both tasks (`r4_*`, `p4_*`), each with a median-imputation baseline under the same mixture.

**Ablation, temperature head without the cell-level descriptors (`p3_obs_temp`):**
- Its row features still include the empirical-Bayes posterior sd (row mean log s_post and its context mean). Only the cell-level injection of descriptors (`dist_emb`) is removed.
- **Accuracy and log-loss: indistinguishable from `p3_obs_dist_temp`** (all 12 core settings within ±0.3 points accuracy and ±0.007 log-loss).
- **Calibration is almost the same.** Single-sample-row overconfidence is −0.2 / +0.5 / +1.0 (gauss / cauchy / heavy censoring), vs −0.2 / +0.1 / +0.4 with `dist_emb`. The precise-context → single-query case is the same (−0.2 / +1.3 / +2.6).
- **So the posterior-sd uncertainty used at the row/output level does the work.** Injecting the probabilistic descriptors into each cell adds only a small calibration gain under heavy censoring, at about 2× activation memory (micro-batch 4 instead of 8).
- **Recommendation:** `temp_head` (with posterior-sd row features) is the simple default; `dist_emb` is optional where censored single samples matter.

**Regression summary-stats baseline (`r3_stats`):** ties median imputation everywhere (|ΔCRPS| ≤ 0.005) except **`fail_floor`, where it is better (−0.035 ± 0.007, 24/27)**: the spread columns reveal a cell split between real readings and floor values. The encoder still leads on all heavy-tail, censoring and grouped settings.

**Ablation, Vec2Summ-style pooling (`p3_obs_dist_temp_v2s`, `pool='v2s'`).** Following Vec2Summ ([Li, Conrad & Gagnon-Bartsch 2025](https://arxiv.org/abs/2508.07017v1)), which summarizes a collection by a Gaussian (mean vector plus spread) in embedding space: the cell vector = LN(W·[mean_k h_k, log(var_k h_k + ε)]) replaces attention pooling with a learned seed query and null key. Everything else is identical to `p3_obs_dist_temp`.
- **On par, no gain:** accuracy within ±0.4 points on all 12 core settings; log-loss marginally worse on 5 (+0.003 to +0.007), better on `fail_floor` (−0.008); calibration identical in every mixed design.
- The moment summary of the replicate embeddings carries the same information as learned attention pooling. **Attention pooling is kept** (it has the null key, which shrinks sparse cells toward a prior). v2s is a fine cheaper alternative.

### 5g. Training with the Exp 6 modes (`--noise_set v2`), regression

`r4_*`: from `stock_reg`, count regimes, the 15-mode noise mixture (original 9 + `fail_floor`, `fail_shift`, `cv_lognormal`, `lod_logistic`, `saturation`, `plate_effect`). The encoder arm uses the best configuration (rstd_rf + dist_emb + temp_head); the baseline is median imputation under the same mixture.

**CRPS:**

| model | clean | gauss | cauchy | c_het | cens_L_hvy | fail_floor | fail_shift | cv_ln | lod_lgt | satur | plate | grouped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r4_median (v2) | 0.161 | 0.256 | 0.324 | 0.348 | 0.322 | 0.362 | 0.338 | 0.289 | 0.248 | 0.282 | 0.236 | 0.276 |
| r3_obs_e1 (v1 training) | 0.161 | 0.251 | 0.306 | 0.328 | 0.304 | 0.390 | 0.364 | 0.285 | 0.244 | 0.273 | 0.231 | 0.262 |
| **r4_obs_dist_temp (v2)** | 0.163 | **0.249** | 0.311 | 0.334 | 0.306 | **0.312** | **0.299** | 0.287 | **0.241** | 0.273 | 0.232 | 0.268 |
| stock_reg + Bayes oracle | 0.160 | 0.250 | 0.299 | 0.322 | 0.308 | 0.255 | 0.236 | 0.282 | 0.237 | 0.267 | 0.230 | 0.258 |

**Paired Δ CRPS, encoder (v2) − median (v2)** (± SE; datasets better of 27): **better on all 11 noisy settings.**
- `fail_floor` −0.050 ± 0.007 (24); `fail_shift` −0.039 ± 0.005 (**27**); heavy censoring −0.017 (24); per-row Cauchy −0.014 (24); Cauchy −0.013 (24).
- Gaussian −0.007 (23); detection dropout −0.007 (21); saturation −0.008 (19); grouped −0.007 (20); plate −0.005 (19); constant-CV −0.002 (15).
- Clean +0.002 (16), a tie.
- **PIT calibration: equal** (Δ PIT error −0.001 to +0.004; 90% coverage 0.89–0.91 for both).

- **Training on the failure modes fixed the regression failure-mode gap:** `fail_floor` 0.390 → 0.312 and `fail_shift` 0.364 → 0.299, now clearly ahead of median (0.362 / 0.338) and summary stats (0.344 / 0.346).
- The cost is about +0.005 CRPS on Cauchy / per-row Cauchy vs the v1-trained encoder (the budget is spread over 15 modes instead of 9).
- The oracle remains ahead on the failure modes (0.255 / 0.236): a larger model or budget, or a dedicated failure-aware location head, could narrow this.

### 5h. Summary-stat columns under regime training (classification, `p3_stats`)

- **Still collapses on single-sample context → replicated queries** (Cauchy 0.474, Gaussian 0.475 vs median 0.609 / 0.646 and the encoder 0.608 / 0.650), **even though the training mixture now contains exactly that regime.**
- **Cause, structural:** with an all-single context, the sd / IQR / SEM / log n columns are constant (0) in the context. The backbone standardizes each column with its context sd (≈ 1e-8 for a constant column), so any non-zero query value becomes a huge standardized input. No amount of training fixes this. Median imputation and the encoder are immune: the encoder gates n = 1 cells instead of featurizing them, and spread enters only through the gated branch.
- Elsewhere `p3_stats` ≈ `p3_median`, except it is better on `fail_floor` (+0.9) and constant-CV (+0.9), where spread signals a failure or a large CV. The calibrated encoder (`p3_obs_dist_temp`) beats it on 9/12 core settings (+0.5 to +1.1 points on the noisy ones), ties on clean, and trails only on the failure modes (−1.5 / −0.7), which are not in v1 training; `p4_*` adds them.

### 5i. TabICLv2 regression baselines (27M-parameter backbone, zero-shot)

TabICLRegressor (`tabicl==2.2.0`, default inference, 99 quantile levels), fed mean / median / summary-stat / Bayes-oracle cells:

| CRPS | clean | gauss | cauchy | c_het | cens_L_hvy | fail_floor | fail_shift | cv_ln | lod_lgt | satur | plate | grouped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tabicl_mean | 0.125 | 0.229 | 0.394 | 0.395 | 0.328 | 0.364 | 0.321 | 0.250 | 0.205 | 0.227 | 0.186 | 0.290 |
| tabicl_median | 0.125 | 0.235 | 0.303 | 0.324 | 0.306 | 0.321 | 0.298 | 0.278 | 0.232 | 0.264 | 0.215 | 0.253 |
| tabicl_full | 0.125 | 0.231 | 0.310 | 0.327 | 0.302 | 0.279 | 0.279 | 0.274 | 0.226 | 0.256 | 0.206 | 0.255 |
| tabicl_bayes (oracle) | 0.125 | 0.226 | 0.275 | 0.300 | 0.284 | 0.230 | 0.210 | 0.266 | 0.212 | 0.242 | 0.208 | 0.233 |
| nano: r4_median (1M) | 0.161 | 0.256 | 0.324 | 0.348 | 0.322 | 0.362 | 0.338 | 0.289 | 0.248 | 0.282 | 0.236 | 0.276 |
| nano: **r4_obs_dist_temp** (1M) | 0.163 | 0.249 | 0.311 | 0.334 | 0.306 | 0.312 | 0.299 | 0.287 | 0.241 | 0.273 | 0.232 | 0.268 |

- **The backbone gap dominates cross-family comparisons** (clean CRPS 0.125 vs 0.163). The meaningful comparisons are within a backbone.
- **Within TabICL**, robust inputs matter as much as for nanoTabICL: median beats mean on heavy tails (Cauchy 0.303 vs 0.394), summary stats help on failures (0.279 vs 0.321 on `fail_floor`), and the oracle is best everywhere. Interestingly, the *mean* is best for TabICL on constant-CV, detection dropout, saturation and plate effects. Robust inputs cost there.
- **Within nanoTabICL**, the replicate encoder closes 30–110% of the median → oracle gap depending on the setting (e.g. heavy censoring: 0.322 → 0.306 vs oracle 0.308).
- **Calibration:** TabICL's PIT error is 0.020–0.030 on noisy data, but it overcovers clean data (0.937 for a nominal 90%, PIT error 0.055). Ours is 0.027–0.035 on noisy data, 0.912 / 0.051 on clean.
- **Next step this suggests:** attach the replicate encoder (zero-initialized, so step 0 = TabICLv2 exactly) to the pretrained TabICLv2 checkpoint and continue training with the observation models. TabICL shares nanoTabICL's architecture (nanoTabICL is its minimal re-implementation), so the encoder plugs into the same cell-embedding slot.

### 5j. Training with the Exp 6 modes, classification (`p4_*`, fair comparison)

Both arms trained identically: from `stock`, count regimes, the 15-mode mixture, 4000 steps. **Paired Δ, encoder (rstd_rf + dist_emb + temp_head) − median imputation** (± SE; datasets better of 27):

| | clean | gauss | cauchy | c_het | cens_L_hvy | fail_floor | fail_shift | cv_ln | lod_lgt | satur | plate | grouped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| accuracy (points) | +0.3 | +0.3 | **+0.9±0.4 (20)** | +0.2 | +0.4 | **+2.2±0.5 (21)** | **+0.9±0.4 (19)** | +0.4 | **+0.8±0.3 (16)** | +0.1 | +0.4 | +0.6 |
| log-loss | −0.003 | −0.005 | **−0.019 (23)** | −0.008 | **−0.019 (20)** | **−0.050 (25)** | **−0.027 (24)** | −0.004 | −0.014 | −0.011 | −0.006 | −0.011 |

**Better than median imputation on all 12 settings in both accuracy and log-loss.**

**Mixed-table calibration** (overconfidence; median v2 → encoder v2):

| | mix: single rows | mix: replicated rows | ctx n=10 → query n=1 | ctx n=1 → query n=10 |
|---|---|---|---|---|
| gauss | +0.016 → −0.005 | −0.007 → +0.004 | +0.029 → −0.002 | −0.017 → −0.009 |
| cauchy | +0.017 → −0.001 | −0.008 → −0.005 | +0.051 → +0.022 | −0.019 → −0.008 |
| cens_L_hvy | +0.017 → +0.006 | −0.006 → +0.003 | +0.042 → +0.030 | −0.016 → −0.007 |

- **Training on the failure modes turns them from losses into the largest gains** (`fail_floor` +4.2 points vs the v1-trained median baseline; +2.2 vs the v2-trained one).
- **The trade-off:** relative to the v1-trained encoder, the heavy-tail gains shrink (Cauchy +1.0 → +0.9 vs the matched median; per-row Cauchy +1.0 → +0.2), because the same 4000-step budget covers 15 modes. 8000-step runs (`p5_*`, `r5_*`) are testing whether this is purely budget.

### 5k. PIT histograms under heavy tails and failures (regression)

![PIT regression](experiments/obs_noise/figures/pit_regression.png)

Pooled PIT values over the 27 regression datasets' test rows (PIT err = calibration error of the pooled histogram; per-dataset averages are in the tables above).
- **No model shows a U shape (overconfident) or a hump (underconfident)** on Cauchy, per-row Cauchy, heavy censoring, `fail_floor`, `fail_shift` or grouped noise. Regression calibration is good across the board.
- **The remaining deviations are mild downward tilts** (more mass at low PIT): targets fall below the predictive distribution slightly more often than nominal, a small upward bias.
- TabICL with median input is flattest (pooled error 0.003–0.008). Our v2 encoder is at 0.009–0.017 (slight tilt on `fail_shift` and grouped), median imputation 0.006–0.015, stock 0.012–0.020.
- **The oracle-denoised input has the clearest tilt** (0.016–0.031). The Bayes posterior shrinks each cell toward the column's typical values, and the downstream model treats those shrunken inputs as exact, which biases its predictions. A sharp *point* estimate, however good, loses calibration information that the replicate-aware model keeps.

### 5l. Longer training (8000 steps, v2 mixture), regression: `r5_obs_dist_temp`

| CRPS | clean | gauss | cauchy | c_het | cens_L_hvy | fail_floor | fail_shift | cv_ln | lod_lgt | satur | plate | grouped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r4_median (v2, 4000) | 0.161 | 0.256 | 0.324 | 0.348 | 0.322 | 0.362 | 0.338 | 0.289 | 0.248 | 0.282 | 0.236 | 0.276 |
| r3_obs_e1 (v1, 4000) | 0.161 | 0.251 | 0.306 | 0.328 | 0.304 | 0.390 | 0.364 | 0.285 | 0.244 | 0.273 | 0.231 | 0.262 |
| r4_obs_dist_temp (v2, 4000) | 0.163 | 0.249 | 0.311 | 0.334 | 0.306 | 0.312 | 0.299 | 0.287 | 0.241 | 0.273 | 0.232 | 0.268 |
| **r5_obs_dist_temp (v2, 8000)** | **0.160** | **0.247** | **0.305** | **0.328** | **0.302** | **0.300** | **0.293** | **0.283** | **0.237** | **0.269** | **0.229** | **0.265** |
| stock_reg + Bayes oracle | 0.160 | 0.250 | 0.299 | 0.322 | 0.308 | 0.255 | 0.236 | 0.282 | 0.237 | 0.267 | 0.230 | 0.258 |

- **The budget hypothesis is confirmed.** Doubling the steps recovers the heavy-tail sharpness of the v1-trained encoder (Cauchy 0.305 vs 0.306; per-row Cauchy 0.328 = 0.328) *and* keeps and extends the failure-mode gains. It is the best nanoTabICL model on every setting.
- **Paired vs median imputation:** better on all 11 noisy settings, e.g. `fail_floor` −0.062 ± 0.008 (26/27), `fail_shift` −0.045 (26), heavy censoring −0.020 (25), per-row Cauchy −0.020 (25), Cauchy −0.019 (**27/27**). Clean ties (−0.000).
- **It matches or beats the Bayes-oracle-denoised input** on Gaussian (0.247 vs 0.250), heavy censoring (0.302 vs 0.308), detection dropout (0.237 = 0.237) and plate effects (0.229 vs 0.230). It is within 0.006 on Cauchy and per-row Cauchy; the remaining oracle gap is the failure modes (0.300 vs 0.255).
- **Calibration:** PIT error 0.025–0.034 on noisy settings. On clean data it is 0.044 with 90% coverage 0.901, the closest to nominal of any model, TabICL included (0.937).
- **Matched-budget check (`r5_median`, 8000 steps, v2):** the median baseline barely moves with the extra steps (CRPS 0.160 / 0.254 / 0.321 / 0.343 / 0.320 / 0.356 / 0.333 / 0.287 / 0.248 / 0.281 / 0.234 / 0.273). **Paired, the encoder is still better on all 11 noisy settings:** `fail_floor` −0.056 ± 0.008 (26/27), `fail_shift` −0.041 (**27/27**), heavy censoring −0.018 (25), Cauchy −0.016 (26), per-row Cauchy −0.015 (**27/27**), detection dropout −0.011 (24), saturation −0.012 (23), grouped −0.008 (21), Gaussian −0.007 (23), plate −0.005, constant-CV −0.005. Clean −0.000 (tie). PIT calibration error is equal on noisy settings (Δ within ±0.006) and better on clean (−0.013 ± 0.005).

### 5m. Longer training (8000 steps, v2 mixture), classification: `p5_obs_dist_temp`

| accuracy | clean | gauss | cauchy | c_het | cens_L_hvy | fail_floor | fail_shift | cv_ln | lod_lgt | satur | plate | grouped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p4_median (v2, 4000) | 0.700 | 0.652 | 0.608 | 0.597 | 0.614 | 0.589 | 0.599 | 0.630 | 0.654 | 0.644 | 0.661 | 0.639 |
| p3_obs_dist_temp (v1, 4000) | 0.701 | 0.657 | 0.624 | **0.610** | 0.623 | 0.563 | 0.582 | 0.639 | 0.658 | 0.647 | **0.667** | **0.649** |
| p4_obs_dist_temp (v2, 4000) | **0.704** | 0.655 | 0.617 | 0.599 | 0.619 | 0.611 | 0.608 | 0.634 | 0.662 | 0.645 | 0.665 | 0.645 |
| **p5_obs_dist_temp (v2, 8000)** | 0.703 | 0.657 | **0.624** | 0.603 | 0.622 | **0.631** | **0.616** | **0.641** | **0.662** | **0.650** | 0.666 | 0.645 |
| p2_mean + Bayes oracle | 0.701 | 0.661 | 0.633 | 0.614 | 0.625 | 0.651 | 0.669 | 0.643 | 0.666 | 0.653 | 0.670 | 0.656 |

**Paired Δ vs median imputation (v2):**
- **Accuracy better on all 12 settings:** `fail_floor` **+4.2 ± 1.0 (22/27)**, `fail_shift` +1.7 ± 0.5, Cauchy +1.6 ± 0.4, constant-CV +1.1, detection dropout +0.8, heavy censoring +0.7, per-row Cauchy +0.6, grouped +0.6, saturation +0.6, plate +0.5, Gaussian +0.5, clean +0.2.
- **Log-loss better on all 12:** `fail_floor` −0.099 (26/27), `fail_shift` −0.051 (26), Cauchy −0.029 (23), heavy censoring −0.027 (24).

**Mixed-table calibration (best so far):**

| | mix: single rows | mix: replicated rows | ctx n=10 → query n=1 | ctx n=1 → query n=10 |
|---|---|---|---|---|
| gauss | −0.003 | +0.006 | −0.003 | −0.007 |
| cauchy | +0.002 | −0.005 | +0.013 | −0.005 |
| cens_L_hvy | +0.000 | −0.001 | +0.023 | −0.006 |

(Median imputation, v2: +0.016 to +0.017 on single rows, +0.029 to +0.051 for ctx n=10 → query n=1.)

- **As in regression, the extra budget recovers the heavy-tail accuracy** (Cauchy 0.624, equal to the v1-trained encoder) **while keeping the failure-mode gains.**
- **Remaining gaps:** per-row Cauchy (0.603 vs 0.610 for the v1 encoder, 0.614 for the oracle); the failure modes vs the oracle (0.631 / 0.616 vs 0.651 / 0.669); precise context → single queries under heavy censoring (+2.3).

**Matched-budget check, classification (`p5_median`, 8000 steps, v2).**
- The median baseline barely moves with the extra budget (accuracy 0.701 / 0.653 / 0.613 / 0.596 / 0.615 / 0.586 / 0.602 / 0.633 / 0.654 / 0.642 / 0.662 / 0.643).
- **Paired, the encoder is better in accuracy on all 12 settings:** `fail_floor` +4.5 ± 1.1 (21/27), `fail_shift` +1.4, Cauchy +1.1, saturation +0.9, constant-CV +0.8, detection dropout +0.8, per-row Cauchy +0.7, heavy censoring +0.6, Gaussian +0.4, plate +0.4, grouped +0.2, clean +0.1.
- **Log-loss is better on all 11 noisy settings** (clean tie): `fail_floor` −0.099 (26/27), `fail_shift` −0.044 (26), heavy censoring −0.024, Cauchy −0.021 (24).
- **Mixed-table calibration of the 8000-step median baseline is still off** (single-sample rows +1.2 to +1.4; context n=10 → query n=1 +3.0 to +4.3). The encoder's is about 0 and +1.3 to +2.3 on the same cases.

---

## Status summary (2026-10-19)

**Current best model, both tasks:** `ObsNanoTabICL` with `{"center": "attn", "med_feat": true, "std": "medmad", "zclip": 10, "robust_feats": true, "dist_emb": true, "temp_head": true}`, continued from the stock checkpoint for 8000 steps with count regimes and the 15-mode observation mixture (`p5_obs_dist_temp`, `r5_obs_dist_temp`). Components:
1. **Replicate set encoder:** φ over [z, z − z̄, z − z̃, robust residual, within-cell rank, limit-tie flag], one masked self-attention block, attention pooling with a learned null key, plus log n. Gated: n = 1 cells contribute exactly 0.
2. **Learned attention-weighted location** fed to the backbone instead of the cell mean (zero-init → starts as the mean).
3. **Median/MAD column standardization** of replicates with soft clipping (fixes 4–10× squashing under heavy tails).
4. **Probabilistic cell descriptors:** empirical-Bayes posterior sd (defined for n = 1 cells) plus a kernel mean embedding, in replicated columns only.
5. **Per-row temperature / quantile-spread head** from row reliability features, including the row's posterior sd vs the context's.

Guarantees, verified to 0.0 max |Δ|: all-single tables equal stock nanoTabICL at any weights, and step 0 equals stock. Training is bit-reproducible, including across crash/resume.

**Against matched baselines** (same backbone, same 8000-step v2 training, 27 datasets each):

| | regression (CRPS) | classification (accuracy / log-loss) |
|---|---|---|
| vs median imputation | better on **11/11** noisy settings (−0.005 to −0.056); clean tie | accuracy better on **12/12** (+0.1 to +4.5 points); log-loss better on 11/11 noisy |
| vs summary-stat columns | better on all heavy-tail / censoring / failure settings (v1 comparison); stats also collapse under replicate-count shift | same; stats collapse to 0.47 accuracy when context and query counts differ (structural, Exp 5h) |
| vs Bayes-oracle-denoised input | matches or beats it on 4 settings (Gaussian, heavy censoring, detection dropout, plate); within 0.001 on constant-CV and 0.006 on Cauchy / per-row Cauchy; behind on failure modes (0.300 vs 0.255) | behind by 0.4–1 point on most settings, 2–5 points on failure modes |
| clean data | CRPS 0.160 (= stock) | accuracy 0.703 (= stock 0.703) |

**Calibration:**
- **Regression:** PIT calibration error 0.025–0.034 on noisy data, 90% coverage 0.89–0.90. Clean coverage 0.901, the best of all models; TabICL overcovers at 0.937. PIT histograms are flat (no U shape or hump) under heavy tails and failures.
- **Classification:** ECE at the sampling floor. Mixed single-sample / replicated tables are calibrated per row (single rows −0.3 to +0.2, where median imputation is +1.2 to +1.4). Precise context with single-sample queries is +1.3 / +2.3 under Cauchy / heavy censoring, vs +4.3 / +4.2 for the median.

**Open problems:**
1. **Failure modes vs the oracle.** The encoder now beats every baseline there, but stays 0.045 CRPS / 2–5 points behind the oracle-denoised input.
2. **Per-row heteroscedastic Cauchy:** +0.7 points over median, 1.1 behind the oracle in classification. The row-reliability embedding (`row_emb`, 5n) brings regression to within 0.002 CRPS of the oracle but helps classification only marginally; explicit per-row attention weights (Stage 2 proper) remain to be tried.
3. **Precise context → single-sample queries under heavy censoring** (+2.3 overconfidence).
4. **Backbone capacity:** TabICLv2 (27M) with median input beats our 1M encoder on most settings. Next: attach the zero-initialized replicate branch to the pretrained TabICLv2 checkpoint and continue training with the observation mixture.

### 5n. Row-reliability embedding (`row_emb`, NOTES.md Stage 2 in implicit form)

The temperature head's 7 row features (mean log n, single-cell share in replicated columns, mean log(1 + MAD), mean log posterior sd, plus context means) go through a zero-initialized, bias-free MLP into each row's ICL token, via a second 2-line hook in [model.py](model.py) (`extra_row_emb`). The aim is to let in-context attention discount unreliable *context* rows, not only rescale query confidence. All-single tables still equal stock exactly at random weights (verified for both tasks). Trained like p5/r5 (8000 steps, v2 mixture).

| paired Δ vs p5 / r5 (no row_emb) | target: per-row Cauchy | grouped | failure modes | other settings | calibration |
|---|---|---|---|---|---|
| classification (`p6`) | acc +0.3 ± 0.2 (14/27), log-loss −0.005 ± 0.002 (19) | +0.1 | `fail_floor` −0.6 acc, +0.011 log-loss | ties (±0.3) | similar; reverse-mismatch case slightly worse |
| regression (`r6`) | **CRPS −0.004 ± 0.001 (22/27)**, now 0.324 vs oracle 0.322 | −0.003 (21) | −0.004 (20) / −0.004 (19) | ties; heavy censoring +0.002 | unchanged |

- **Small, consistent gains in regression where intended** (per-row Cauchy, grouped, failure modes). Neutral to marginally positive in classification.
- Row-level reliability is now worth only about 0.3–0.5% of CRPS on top of the per-row temperature head, which already captured most of it. An explicit per-row attention-logit bias in the ICL stage (NOTES.md Stage 2 proper, log w in the attention logits) would be the stronger form; it needs a change to the backbone's attention.
- **Recommendation:** include `row_emb` for regression (cheap, small gain); optional for classification.

---

## Exp 7 (planned) — Replicate branch on the pretrained TabICLv2 checkpoint

**Why.** TabICLv2 (27M) with median input beats our 1M encoder on clean data (0.742 vs 0.703 accuracy) and on most noisy settings. Training it from scratch is out of reach (about 24.5 H100-days). Adding our zero-initialized branch to the released checkpoint and continuing training is cheap.

### Feasibility check (done, 2026-10-19)

- **Same architecture.** `NanoTabICLv2` at its default config (embed 128, col/row/icl = 3/3/12 blocks, 8 heads, 128 inducing vectors, 4 CLS tokens, group size 3, RoPE base 1e5, QASSMax) *is* TabICLv2. Both have 27.55M parameters.
- **The whole checkpoint loads, not only the col/row weights.**
  - After renaming keys, `tabicl-classifier-v2-20260212.ckpt` loads with 0 missing and 0 unexpected keys.
  - Only two tensors need a reshape: `ind_vectors` (128, 128) → (1, 128, 128), and `cls_tokens` (4, 128) → (1, 1, 4, 128).
  - TabICL's stored `rope.freqs` buffer equals our `inv_freq`, and both use the same half-split rotation layout.
- **Outputs match the original model.** On the same pre-standardized inputs, `tabicl._model.tabicl.TabICL` (which runs in fp16 autocast) and the converted nano model (fp32) agree to within 0.026 in the logits (typical logit size 6.9). Probabilities differ by ≤ 0.003, and the predicted class agrees on 99.4% of rows; the rest look like near-ties.

| TabICLv2 key | nano key |
|---|---|
| `col_embedder.in_linear`, `col_embedder.y_encoder` | `x_embed`, `y_embed_in` |
| `col_embedder.tf_col.blocks.i.ind_vectors` | `col_blocks.i.inducing_vectors` (unsqueeze 0) |
| `col_embedder.tf_col.blocks.i.multihead_attn{1,2}` | `col_blocks.i.tfm{1,2}` |
| `row_interactor.tf_row.blocks.i`, `.cls_tokens`, `.out_ln` | `row_blocks.i`, `row_cls_tokens` (→ 1×1×4×128), `row_ln` |
| `icl_predictor.tf_icl.blocks.i`, `.y_encoder`, `.ln`, `.decoder.{0,2}` | `icl_blocks.i`, `y_embed_icl`, `out_ln`, `out_mlp.{0,2}` |
| within blocks: `attn.{in_proj_*, out_proj, ssmax_layer}`, `norm{1,2}`, `linear{1,2}` | `{in_proj_*, out_proj, ssmax_layer}`, `ln_attn` / `ln_mlp`, `mlp.0` / `mlp.2` |
| `row_interactor.tf_row.rope.freqs` | dropped (recomputed as `inv_freq`) |

### Plan

1. **Converter.** Add `tabicl_convert.py`, which maps the checkpoint to a nano `state_dict` and checks the output match above as a test. Add `--init_tabicl <ckpt>` to [obs_train.py](experiments/obs_noise/obs_train.py), plus `--size tabicl` for the 27M config. `ObsNanoTabICL` wraps the backbone unchanged.
   - Because the branch is zero-initialized, step 0 equals TabICLv2 exactly.
   - All-single-sample tables stay TabICLv2 at any weights (the guarantee from Exp 2).
2. **Stage A: frozen backbone.** Train only the replicate branch for about 3k steps on the v2 observation mixture with count regimes and K ≤ 50 batches, at lr 3e-4. This cannot hurt clean performance.
3. **Stage B (optional): unfreeze the backbone.** Use lr about 1–2e-5 on the backbone and keep the branch's own rate (`--rep_lr_mult`), for about 5k steps.
   - Keep the 25% clean / all-single tables in every batch.
   - Stop early if clean validation accuracy falls below stock TabICLv2.
   - **Risk:** our nanoprior differs from TabICLv2's training prior, so continued training could cost TabICLv2 its generality. Item P5 below (mixing in `tabicl.prior`) is the mitigation.
4. **Matched baseline.** Continue TabICLv2 + median imputation with exactly the same steps and data, as `p5_median` did at 1M.
5. **Regression.** The regression checkpoint has a 999-quantile head. Either keep it and evaluate at every 10th quantile (TAUS k/100), or slice `out_mlp.2` to our 99 quantiles. Prefer keeping 999 so step 0 equals TabICLv2.
6. **Evaluation.** Use the core suite (27 datasets per task) and feed the raw `TabICL` module, as for the nano models. TabICLv2 sklearn numbers come from 8-estimator ensembles with their own preprocessing, so also report 1-pass numbers for a fair comparison (see A3).
7. **Cost (RTX 5070 Ti).** About 0.58 s/step in the benchmark (about 1.5 s in a real run) at 2.9 GiB with micro-batching. That is about 3.5 h for 8k steps plus about 2.5 h for the median baseline, per task, plus about 2–4 h of porting.

---

## Backlog — improvements to try on the 1M model first (2026-10-19)

1M runs are cheap: stock pretraining takes 42 min for 12k steps, an 8k-step encoder run about 2 h, and an 8k-step median baseline 0.5 h. Anything that works here carries over to the TabICLv2 branch unchanged. Ordered by expected value per GPU-hour.

**Facts from the logs that shape this list:**
- **Stock pretraining had not converged.** Clean validation accuracy was still rising at 12k steps: 0.782 / 0.787 / 0.790 / 0.791 at 6 / 8 / 10 / 12k. Training covered only 384k table draws over a 128k-table bank, i.e. each table was seen 3×.
- **The noise is drawn once per table.** [obs_train.py:39-41](experiments/obs_noise/obs_train.py#L39-L41) draws one noise model and one scale for *all* columns of a table. Real assays mix precise and noisy columns and different failure behavior per column. The encoder can currently infer the noise family from all columns at once, which real data won't allow.
- **Ensembling.** Our models are evaluated with 1 forward pass; TabICLv2 with 8-estimator ensembles. Part of the 4-point clean gap is ensembling, not the model.

### Evaluation (do first; no training)
- **A3. Test-time ensembling for nano models.** Average over 8 feature shuffles × class-label shifts, as TabICL does; optionally also over posterior draws of cell values.
  - This makes the TabICLv2 comparison fair, and probably narrows the clean gap at no training cost.
  - Also report TabICLv2 with `n_estimators=1`.
  - **Cost:** about 1 h of evaluation.

### Training
- **T1. Auxiliary denoising loss.** Training has `X_clean`, so supervise the encoder directly:
  - pinball loss of a small head's quantiles of x_clean given the replicates (or MSE of the attention-weighted center);
  - optionally the posterior sd.

  Costs nothing at inference. Aimed squarely at the largest remaining gap, the failure modes (CRPS 0.300 vs oracle 0.255; accuracy −2 to −5 points). The encoder currently learns denoising only through the task loss on 512-row tables. **Cost:** 1 run per task.
- **T2. Longer stock pretraining on fresh data.** Generate the bank on the fly or 4× larger, and pretrain 2–3× longer (1.5–2 h), then redo p5 / median from the new stock. Raises clean accuracy and the noisy baselines alike.
  - Also try joint training from scratch on the observation mixture, to test whether stock-then-continue loses anything.
- **T3. EMA of weights** (decay about 0.999) for evaluation checkpoints. Typically a free, small gain; a 5-line change.

### Prior
- **P1. Per-column noise heterogeneity.**
  - Draw the noise family and scale per column, not per table.
  - Keep some columns precise (n = 1 always, or scale ≈ 0), like covariates or metadata.
  - Allow per-column replicate-count designs (e.g. one assay run in triplicate, others singly).

  Closer to real panels; also a harder eval: add a `percol` setting to the core suite. **Cost:** 1 run + 1 eval set.
- **P2. Paired replicates / sample-level effects.** In real experiments replicate k of a row is often the *same physical sample* measured on every feature, so prep, dilution or batch errors are shared across features: x_rck = x_rc + a_rk + ε_rck, or multiplicative.
  - Today all replicates are independent across features.
  - This is the prerequisite for cross-feature replicate attention (asked earlier): without pairing in the prior, that attention has nothing to learn.
- **P3. Censoring indicators and missing cells (NOTES.md Stages 3–4).**
  - Many assays report "<LOD" / ">ULOQ" flags instead of a value at the limit. Add a per-replicate flag channel to φ.
  - Allow n = 0 cells (MAR, and MNAR via detection failure).
  - This also addresses precise context → single-sample queries under heavy censoring (+2.3 overconfidence): flagged values are unambiguous.
- **P4. Replicated or noisy targets.** Regression y is often itself a replicated assay readout, and classification labels can be noisy. Train on noisy y while evaluating on clean y (or the other way round). This mainly matters for calibration claims.
- **P5. Richer clean prior.**
  - Mix the TabICL prior (`tabicl.prior`, installed: SCM + tree-based, categorical features, more features) with the nanoprior.
  - Add a small set of real OpenML tables with simulated replicates to the eval suite.

  Needed anyway for Exp 7 Stage B, to keep TabICLv2 inside its training distribution.

### Architecture
- **A1. Column-context (two-pass) cell encoder.** Pass 1 encodes cells; their mean over the column's context rows becomes a column token; pass 2 re-encodes each cell conditioned on that token.
  - Failure floors, LOD and saturation values, and noise scale are column-level facts, which the per-cell encoder now re-infers from ≤ 10 replicates.
  - This is a learned empirical Bayes at the column level. The current `dist_emb` does a fixed version of it for the posterior sd only.
  - Targets the failure-mode oracle gap and pairs naturally with P1. **Cost:** about 30 lines; 1 run per task.
- **A2. Per-row reliability bias in ICL attention logits** (NOTES.md Stage 2 proper). Add a zero-initialized log w_j to the logits for context row j. `row_emb` (5n) could only add to the token. Needs an additive `attn_mask` in `TransformerBlock.attn`. Targets per-row Cauchy in classification.
- **A4. Wider backbone (3–4M, embed 48–64).** Only worthwhile if Exp 7 fails or is blocked: the TabICLv2 port answers the capacity question directly.
- **A5. Multi-seed (k = 2) pooling** for bimodal cells: one good-mode and one failure-mode summary. Likely subsumed by T1 + A1; try only if failure modes still lag.

### Suggested order
1. A3 (evaluation only).
2. T1 and P1 + `percol` eval, in parallel.
3. A1.
4. T2 + T3, re-running the winning configuration from a better stock.
5. P2 / P3 with new evals.
6. Exp 7 with the winning branch configuration and P5 mixed into the prior.

Items that change the prior (P1–P5) need matched median / stats baselines retrained on the same prior.

---

## Exp 7 — Ensembling, auxiliary denoising loss, heterogeneous noise within a table (2026-10-19, in progress)

Goal: beat matched median imputation by a clear margin, and match TabICLv2 (27M) with the 1M model, before porting the branch to TabICLv2 (see the plan above).

**Noise modes covered so far (check).**
- **Bimodal / median-breaking:** yes. `fail_shift` (failed replicates at x + 3 sd) and `fail_floor` (failed replicates at a column floor), with per-cell failure rate U(0, 0.8).
- **Intensity-dependent non-detects:** yes (`lod_logistic`), but substituted by a floor value, not missing.
- **Missing cells (n = 0, MCAR / MAR / MNAR):** **not yet.** Every cell has ≥ 1 replicate. That is backlog item P3 and needs missing-value handling in the backbone.

### 7a. Test-time ensembling (no training)

[obs_eval.py](experiments/obs_noise/obs_eval.py) `--n_ens N`: member i uses a random column permutation (i = 0: identity) and class labels shifted cyclically by i. Probabilities (classification) or sorted quantiles (regression) are averaged. `--n_ens 1` reproduces the earlier results exactly (max |Δ log-loss| = 0.0 over 297 records). TabICLv2 was re-run with `n_estimators=1` (`N_EST=1 tabicl_eval.py`, tag `tabicl_median_e1`).

| | clean acc, 1 → 8 | main noisy acc, 1 → 8 | clean CRPS, 1 → 8 | main noisy CRPS, 1 → 8 |
|---|---|---|---|---|
| stock nanoTabICL | 0.703 → 0.714 | 0.601 → 0.611 | 0.160 → 0.156 | 0.332 → 0.329 |
| median imputation (p5 / r5) | 0.701 → 0.711 | 0.626 → 0.633 | 0.160 → 0.158 | 0.298 → 0.296 |
| ours (p5 / r5) | 0.703 → 0.713 | 0.637 → 0.644 | 0.160 → 0.158 | 0.279 → 0.276 |
| TabICLv2 27M, median input | 0.742 → 0.742 | 0.645 → 0.646 | 0.125 → 0.125 | 0.278 → 0.278 |

- **Ensembling is worth about +1 point of accuracy (and −0.002 to −0.004 CRPS) for the 1M models.**
- **It is worth nothing for TabICLv2.** Its 1-estimator and 8-estimator results are the same, so its lead is not ensembling.
- **With 8 members, our model matches TabICLv2 on the noisy core settings:**
  - classification 0.644 vs 0.646 (paired, TabICLv2 is +0.003);
  - regression CRPS 0.276 vs 0.278, i.e. better.
- **The remaining gap is clean-data quality of the 1M backbone:** 0.713 vs 0.742 accuracy; CRPS 0.158 vs 0.125. That is a backbone or pretraining question (backlog T2), not a replicate-handling one.
- **Calibration.** Ensembling makes the nano models slightly underconfident: overconfidence −0.018, ECE 0.035 → 0.041, cov90 0.90 → 0.92. TabICLv2 is overconfident (+0.017).

### 7b. Heterogeneous noise within a table: new evaluation settings

The training prior so far drew **one** noise model and scale per table ([obs_train.py](experiments/obs_noise/obs_train.py) `observe`). Real panels mix precise and noisy assays, and data from several labs / batches / sites.

New settings ([gen_ext3.py](experiments/obs_noise/gen_ext3.py), `artifacts/data_ext3`, source `ext3`):

| setting | design |
|---|---|
| `percol` | own core mode + scale (0.5 sd × logU(0.5, 2)) per column; 20% exact columns (n = 1) |
| `pergroup` | 3 hidden row groups, each with its own mode, scale and (w.p. 0.5) a replicate cap U{1..10} |
| `pergroup_vis` | the same draws + the exact group index as a column |
| `colgroup_vis` | own mode + scale per (group, column); group column visible; 20% exact columns |

The Bayes oracle (`bayes.py ext3`) knows each cell's model, scale and group. It reduces the median-imputation error (median |error| / sd) from 0.15 / 0.26 / 0.26 / 0.13 to 0.09 / 0.18 / 0.18 / 0.08.

Existing models (trained on the per-table prior), mean accuracy / CRPS over the 4 settings:

| | het acc, 1 pass | het acc, ens 8 | het CRPS, 1 pass | het CRPS, ens 8 |
|---|---|---|---|---|
| stock | 0.606 | 0.618 | 0.306 | 0.303 |
| median imputation (p5 / r5) | 0.626 | 0.635 | 0.285 | 0.283 |
| ours (p5 / r5) | 0.627 | 0.637 | 0.277 | 0.274 |
| nano + Bayes-oracle input | 0.636 | | 0.278 | |

**The per-table-trained encoder barely beats median imputation here** in classification (`percol` 0.660 vs 0.662 for one pass) and only modestly in regression (−0.006 CRPS). This is the prior gap that 7d addresses.

### 7c / 7d. Auxiliary denoising loss and heterogeneous-noise training (running)

- **Auxiliary loss** (`rep_cfg aux`, `--aux_w 0.3`). Training knows the clean value, so replicated cells get extra targets:
  - a linear head on the cell vector predicts the 10/25/50/75/90% quantiles of the clean z (pinball loss);
  - the learned attention-weighted center is pulled toward the clean z (absolute error).

  Both are in the encoder's standardized, soft-clipped units. The head is unused at inference, so both guarantees are unchanged (re-verified). Weight 0.3: at 1.0 the auxiliary term (about 1.1) was about twice the task loss.
- **Noise layouts** (`--layouts '[0.25,0.25,0.25,0.25]'` over `table, col, group, colgroup`, one layout per batch):
  - `col`: own model + scale per column, 20% exact columns;
  - `group`: 2–4 row groups with their own model, scale and (w.p. 0.5) replicate cap;
  - `colgroup`: per (group, column);
  - in half of the group batches the exact group index is appended as a column (`--group_vis_prob`).

  The default (no `--layouts`) reproduces the old random stream bit-for-bit (checked against the previous `observe` for v1 / v2, with and without count regimes).
- **Runs** ([phase4.sh](experiments/obs_noise/phase4.sh); same init, budget and mixture as p5 / r5: 8000 steps from stock, v2 modes, count regimes, 20% K ≤ 50 batches):
  - `p7_obs_dist_temp_aux` / `r7_…`: aux only;
  - `p7L_median`, `p7L_obs_dist_temp`, `p7L_obs_dist_temp_aux` (and `r7L_…`): het-trained, with a matched median baseline.
- **Bank extended** to 8000 blocks per task for backlog T2 (longer stock pretraining on fresh data). Blocks are seeded by index, and `Bank` now uses the first `--n_blocks` files, so earlier runs are unchanged.

Summary script: [compare_exp7.py](experiments/obs_noise/compare_exp7.py).

---

## Real-data evaluation (planned, 2026-10-19)

**Why.** Every result so far uses simulated observation models on top of clean tables. Before the TabICLv2 port (Exp 7 plan), check that the gains appear on **real** measurement noise: real replicates, real non-detects and failures, and per-column noise that nobody chose. On real data there is no `X_clean` and no Bayes oracle, so the comparison is on task metrics only.

**Datasets** (downloaded to `experiments/obs_noise/artifacts/real_raw/`, which is gitignored). All three are standard benchmarks with published baselines.

| dataset | used in | rows × columns | replicates per cell | real noise | label / split |
|---|---|---|---|---|---|
| **Tox21 qHTS** (`tox21/`) | MoleculeNet / TDC Tox21 (the 12 tasks), Tox21 Data Challenge 2014 | ≈ 10k substances × 12 assays (AC50, efficacy, curve class) | 3 independent runs (`SAMPLE_DATA_TYPE` = `set1/2/3`); 2–5% of substances have extra samples or controls per run | inactive = no AC50 (~48%), i.e. right-censored at the top tested concentration (≈ 90–120 µM, varies by sample); curve-fit failures; inconclusive calls; a cytotoxicity counter-screen per run (`viability1/2/3`) | held-out MoleculeNet task (e.g. predict SR-p53 from the other 11 assays' replicates); `moleculenet_tox21.csv.gz` gives the benchmark labels |
| **LINCS Cell Painting + L1000** (`lincs/`) | Way et al. 2022 (Cell Systems), MoA prediction; JUMP / Cell Painting Gallery cpg0004 | 1,558 compounds × 6 doses × 745 morphology features (CP); 1,403 compounds × 978 genes (L1000) | CP: 5 wells (2–4 after QC); L1000: 3 (1–2 after QC) | well/plate position and batch effects, failed wells, L1000 replicates that are known to reproduce poorly | MoA class, with the paper's compound split (`split_moas_cpds.csv`); MODZ level-5 consensus (`modz_*.csv`) is the community's replicate-weighted baseline |
| **Gas sensor array drift** (`gas_drift/`, UCI 224 / OpenML `gas-drift`) | standard drift benchmark (Vergara et al. 2012) | 13,910 measurements × 128 features | 16 sensors = 4 types × 4 physical units, so each (type, feature) has 4 replicate units | unit-to-unit bias, 3-year sensor drift (heteroscedastic, non-stationary) | 6 gas classes; batches 1–10 in time order (train early batches, test later ones). This copy has no concentration column, so classification only |

**Files.**
- `tox21/<protocol>/<protocol>.txt`: one row per (sample, run, channel) with the 15-point curve, AC50, efficacy, curve class and SMILES. Parse with `sep='\t', index_col=False` (rows have a trailing tab). `<protocol>.aggregrated.txt` is NCATS's own collapsed call, for reference. Protocols map to MoleculeNet as: NR-AR `ar-mda-kb2-luc-agonist-p1`, NR-AR-LBD `ar-bla-agonist-p1`, NR-AhR `ahr-p1`, NR-Aromatase `aromatase-p1`, NR-ER `er-luc-bg1-4e2-agonist-p2`, NR-ER-LBD `er-bla-agonist-p2`, NR-PPAR-gamma `pparg-bla-agonist-p1`, SR-ARE `are-bla-p1`, SR-ATAD5 `elg1-luc-agonist-p1`, SR-HSE `hse-bla-p1`, SR-MMP `mitotox-p1`, SR-p53 `p53-bla-p1`. Source: `https://opendata.ncats.nih.gov/public/tox21/assay/data/<protocol>.zip`.
- `lincs/cp_level4_cpd_replicates{,_nonspherized}.csv.gz`, `lincs/L1000_level4_cpd_replicates.csv.gz`: one row per replicate well, keyed by compound × `Metadata_dose_recode` (CP) or `pert_id` × `dose` (L1000). Source: Git LFS of `broadinstitute/lincs-profiling-complementarity`.
- `gas_drift/Dataset/batch{1..10}.dat`: libsvm format, `class idx:value …`.

**To do.**
1. **Loaders** that write the existing eval format (`<setting>__replicates` n × d × K, NaN-padded, `__counts`, `train_idx` / `test_idx`):
   - Tox21: features = per-run log AC50 and efficacy per assay, with inactive runs reported at the top tested concentration (as in `censored`); also try an explicit censoring flag (backlog P3).
   - LINCS: reduce 745 / 978 features to ≤ 100 (PCA fitted on DMSO wells, or feature selection), one row per compound × dose, or per compound with doses as columns. Many MoA classes: keep the top-k MoAs, or several k-way subtasks, as TabICL's class limit requires.
   - Gas drift: 32 columns (8 features × 4 sensor types), K = 4 units. **Check** the sensor-index → type mapping in Vergara et al. 2012 before grouping.
2. **Arms**: stock + mean, stock + median, summary stats, MODZ (LINCS), p5 / p7 median baselines, our encoder, TabICLv2 + median; 1-pass and 8-member ensembles.
3. **Replicate-subsampling curve** (as in Exp 4): give every cell 1, 2, … K of its real replicates. If our gain over median grows with K, the model is using real replicate structure.
4. **Real noise vs. simulated**: fit our observation-mode mixture to each dataset (tail index, censoring fraction, failure rate per column) and compare to the training prior; report which modes are actually present.

### 7e. Follow-ups to the auxiliary loss (queued, [phase5.sh](experiments/obs_noise/phase5.sh))

**Early validation of the aux runs** (8 noise models of the training validation set, step 3000 classification / 2000 regression):

| | `fail_floor` | `fail_shift` | other settings |
|---|---|---|---|
| p5 → p7 (+aux) accuracy | +3.4 | −1.8 | within ±0.5 (Cauchy +0.5) |
| r5 → r7 (+aux) CRPS | −0.037 | +0.029 | within ±0.003 |

The pattern was the same at step 1000.

**Hypothesis.** The center term forces each cell to pick a mode.
- Floor failures are identifiable within a cell: they are tied at the column minimum.
- For upward-shifted failures, a cell whose replicates are mostly failures looks like a cell at a higher value. Only the rest of the column shows the shift.

**Two arms** (from stock, same budget and mixture as p5 / r5; queued after the running aux runs, followed by the het-trained aux runs):
- `obs_dist_temp_auxq`: aux with **only the quantile head** (`aux_center: false`), i.e. no pull on the learned center.
- `obs_dist_temp_col`: **column-context encoder** (`col_ctx`, backlog A1). A separate per-replicate DeepSets MLP (`phi_col`, same features as φ) is averaged over each cell's replicates, then over the column's replicated **context** cells. This gives a column token. LayerNorm → zero-init linear (`rep_col`) adds it to every replicate embedding of the column's replicated cells, before the self-attention block.
  - This is a learned, column-level empirical Bayes: failure floors and shifts, limits, noise shape.
  - +5k parameters.
  - All-single tables still equal stock exactly, with every zero-init map randomized (`python obs_model.py --col_ctx`); step 0 equals the p5 starting point (3e-7, as for `center: attn`).

### 7c result. Auxiliary loss, classification (`p7_obs_dist_temp_aux`, 27 eval datasets)

The validation trend held: a **wash overall**.

| paired Δ vs p5 (1 pass) | `fail_floor` | `fail_shift` | Cauchy / per-row Cauchy | heavy censoring | het settings (4) | clean |
|---|---|---|---|---|---|---|
| accuracy (points) | **+1.5 ± 0.5** | **−1.9 ± 0.5** | +0.2 / +0.3 | −0.3 | +0.3 to +0.4 | 0.0 |
| log-loss | −0.034 | +0.039 | −0.009 / −0.005 | 0.000 | −0.005 to −0.008 | |

- **Means over the 10 main settings:** 0.636 vs 0.637 for one pass, 0.644 = 0.644 with 8 members.
- **Calibration:** unchanged (ECE 0.036; overconfidence −0.002).
- **Interpretation:** supervising the denoised value per cell trades floor-type failures, which are identifiable within a cell, against shift-type failures, which are not. This supports the column-context arm (7e), now running.

### 7c/7d results. Regression aux; het-trained encoder (27 eval datasets)

**Regression aux (`r7_obs_dist_temp_aux`): slightly worse than r5 everywhere.**

| | main CRPS | het CRPS | cov90 noisy |
|---|---|---|---|
| r5, 8 passes | 0.276 | 0.274 | 0.906 |
| r7 (+aux), 8 passes | 0.278 | 0.277 | 0.901 |

Clean is unchanged (0.158). With the classification wash (7c), **aux with the center term is dropped.** The quantile-only variant (`auxq`) is running.

**Het-trained encoder (`p7L_obs_dist_temp`, layouts table / col / group / colgroup at 25% each):**

| | main (10) | het (4) |
|---|---|---|
| p7L encoder vs p7L median, 8 passes | **+0.7 ± 0.2 (20/27)** | **+0.7 ± 0.2 (16/27)** |
| p7L encoder vs p5 encoder, 8 passes | −0.4 | **+0.4** |

Per het setting, vs p7L median: `percol` +0.9, `pergroup` +0.5, `pergroup_vis` +0.5, `colgroup_vis` +0.7.

- **It works as intended, but the margin is small.** On the heterogeneous settings it is the best nano model (0.641 with 8 passes vs 0.637 for p5), and its margin over its matched median baseline is +0.7 points instead of +0.2 for p5. It costs 0.4 points on the single-noise-per-table settings (only 25% of its training batches use that layout).
- **TabICLv2 is still +1.5 points ahead on het.** Its clean-backbone advantage matters more when some columns are exact (0.656 vs 0.641). Nano + Bayes-oracle input reaches only 0.636 there, so the het gap to TabICLv2 is not a replicate-handling gap.

### 7e result (early stop). Aux with only the quantile head (`auxq`)

Stopped at step 3000 (classification) / 2000 (regression). Validation Δ vs p5 / r5 at the same step:
- **Classification:** within ±0.005 on all 11 settings (`fail_shift` −0.005, `fail_floor` −0.002).
- **Regression:** slightly worse everywhere (`fail_floor` +0.009, `fail_shift` +0.006, others +0.000 to +0.002 CRPS).

Without the center term, the auxiliary loss does nothing useful. **Conclusion for aux:** denoising supervision of the cell vector or center does not help on top of the task loss. **Dropped.**

### 7f. Running: column context + longer stock pretraining (T2), with early stopping

- **`p7_obs_dist_temp_col` / `r7_obs_dist_temp_col`:** the column-context encoder (7e), from stock, same budget as p5 / r5.
  - **Stop rule:** at step ≥ 3000, stop if the failure modes and Cauchy are not better than p5 / r5 at the same step (≥ +0.5 points / −0.005 CRPS on at least one, no regression elsewhere).
- **`stock2` / `stock2_reg`:** the stock recipe (clean, lr 1e-3, warmup 500, cosine to 10%), but **30k steps** (2.5×) on the **8000-block bank**: 512k tables, each seen ≈ 1.9×, vs 128k tables seen 3× for `stock`.
  - **Stop rule:** at 20k steps, stop if clean validation accuracy is not above `stock`'s final value (0.791 classification; R² 0.701 regression). The longer cosine schedule makes equal-step comparisons before that point unfair.
  - **If it helps:** redo the best arm and the median baseline from `stock2`, then re-compare with TabICLv2.
- **`r7L_obs_dist_temp`** (het-trained regression encoder): paused at step 0 to free GPU memory. It resumes exactly (`resume_r7L.sh`) after the first col run finishes.

### Real-data check: which observation modes are in LINCS? ([real_noise_modes.py](experiments/obs_noise/real_noise_modes.py))

**Method.** Leave-one-out residuals within (compound, dose) cells with ≥ 3 replicate wells: e_k = r_k − median of the other replicates, standardized per feature by 1.4826 × median |e|. 300 random features per dataset, controls excluded.

| | L1000 (25k replicate rows, 7.5M residuals) | Cell Painting, non-spherized (48k rows, 14M residuals) | Gaussian |
|---|---|---|---|
| P(\|e\| > 3) | 5.0e-2 | 6.8e-2 | 2.7e-3 |
| P(\|e\| > 5) | 2.3e-2 | 3.2e-2 | 5.7e-7 |
| P(\|e\| > 8) | 1.1e-2 | 1.7e-2 | 1e-15 |
| \|e\| density beyond 3 | monotone decreasing | monotone decreasing | |
| sign of extremes (\|e\| > 5), median \|share positive − 0.5\| | 0.12 (null 0.05) | 0.11 (null 0.02) | |
| features whose extremes are > 90% one sign | 1.7% | 0.0% | |
| values tied at the column minimum | 4e-5 | 2e-5 | |
| replicates extreme in > 10× the usual share of features (well failures) | 0.2% (hold 2% of extremes) | 0.5% (hold 6% of extremes) | |

- **Heavy tails dominate.**
  - 2–3% of replicates are more than 5 robust sd from their siblings, and 1–1.7% more than 8 sd.
  - The tail ratio P(> 3) / P(> 8) ≈ 4–5 sits between Cauchy (≈ 2.6) and t₃ (≈ 14), roughly t₂. That is the `censored_left_heavy` noise family, and between the `cauchy` and `student_t3` modes.
- **No `fail_shift` signature.**
  - The \|e\| histogram is monotone, with no bump at a fixed offset.
  - Extremes are only mildly sign-asymmetric. Almost no feature has one-sided extremes (0–1.7%, vs 100% under `fail_shift`).
- **No floor / non-detect substitution.** Level-4 profiles are normalized and continuous. Censoring matters for Tox21 instead: ≈ 48% of runs are inactive, i.e. right-censored at the top concentration (see the dataset table).
- **Whole-well failures exist but are rare.** 0.2–0.5% of wells, holding 2–6% of extremes. These are failures **paired across features** (backlog P2), which the per-cell-independent prior does not model.
- Caveat: the Cell Painting grouping kept a few large control-like groups (sizes 1632 / 3264) that the DMSO filter missed. They are well-defined replicate groups and do not change the qualitative picture.

**Implications.**
1. **Do not tune for `fail_shift`.** It stays as a robustness check, not a target. The aux loss's `fail_shift` cost (7c) is therefore less important than it looked, but aux is still dropped: it gave no gain on the heavy-tail modes either.
2. **Judge the column-context encoder (7f) on heavy tails, censoring and het settings**, not on `fail_shift`.
3. **Prior weighting.** Real noise looks like t₂–Cauchy tails at a 2–3% outlier rate, plus rare paired well failures. The training mixture should weight heavy-tail modes more, and add paired (well-level) failures (P2).

### 7f result. Longer stock pretraining (`stock2`, 30k steps on the 8000-block bank), classification

| | val clean acc / loss | eval clean acc, 1 pass | eval clean acc, 8 passes |
|---|---|---|---|
| `stock` (12k steps, 2000 blocks) | 0.791 / 0.503 | 0.703 | 0.714 |
| **`stock2`** (30k steps, 8000 blocks) | **0.800 / 0.484** | **0.718** | **0.725** |
| TabICLv2 27M | | 0.742 | 0.742 |

- **Paired over the 27 datasets:** `stock2` vs `stock` is +1.4 ± 0.7 (1 pass) and +1.1 ± 0.5 (8 passes). The gap to TabICLv2 shrinks from 3.9 / 2.8 to **2.4 ± 0.9 / 1.7 ± 0.8** points.
- **Training had not converged.** Validation accuracy kept rising to the end: 0.794 / 0.796 / 0.800 at 18k / 20k / 30k. More steps, or a slightly larger backbone, should close more of the gap.
- **Noisy settings (stock, no replicate handling):** small gains, e.g. Gaussian 0.652 → 0.654, plate 0.662 → 0.672, `percol` 0.653 → 0.666. Cauchy is unchanged (0.531 → 0.545). Replicate handling is still needed for heavy tails.
- **Next:**
  - `p8_median` and `p8_obs_dist_temp_col` from `stock2` (queued, `phase7.sh`);
  - `stock2_reg` (running).

**`stock2_reg` (regression, same recipe): stopped at 20k / 30k.**

| | clean validation R² | clean validation CRPS |
|---|---|---|
| `stock2_reg` at 20k | 0.694 | 0.182 |
| `stock_reg` final | **0.701** | 0.183 |

Mixed, and below the stop rule (R²). The R² trajectory was flat from 17k (0.693), so the longer schedule buys the regression backbone little at this size. The classification gain (+1.1 to +1.4 points clean) does not carry over to regression.

### 7f result. Column-context encoder (`p7_obs_dist_temp_col`, `r7_obs_dist_temp_col`), 27 eval datasets

| | main (10) | het (4) | Δ vs matched median, main (8 passes) | Δ vs p5 / r5, main (8 passes) |
|---|---|---|---|---|
| classification, 8 passes | **0.646** acc | 0.637 | **+1.3 ± 0.2 (24/27)** | +0.2 |
| regression, 8 passes | **0.269** CRPS | 0.275 | **−0.026 ± 0.003 (27/27)** | −0.007 |

- **The best 1M model in both tasks.**
- **Classification:** it equals TabICLv2 + median on the main settings (0.646 = 0.646).
- **Regression:**
  - main settings: it beats TabICLv2 + median (0.269 vs 0.278) and matches nano with Bayes-oracle input (0.268);
  - PIT error: 0.028 / 0.030 (1 / 8 passes), the best of the nano models;
  - clean: unchanged.
- Unlike aux, it helps both failure modes and the heavy-tail settings. **`col_ctx` is adopted** for the TabICLv2 port.

---

## Exp 8 — Replicate branch on the pretrained TabICLv2 (27.55M) backbone (2026-10-19, running)

**Port ([tabicl_port.py](tabicl_port.py)).**
- `load_tabicl(task)` maps the released checkpoints onto `NanoTabICLv2` / `ObsNanoTabICL`.
- Two differences were found and fixed while verifying:
  1. **Feature grouping offset.** TabICL groups feature i with i + 1, i + 2, i + 4; nano with i, i + 1, i + 3. The groups are the same, but the column order is shifted by one, which changes the row-RoPE positions. New `NanoTabICLv2(group_shift=…)`: 1 = nano (default, existing checkpoints unchanged), 0 = TabICL.
  2. **Bias-free LayerNorms** in the regression checkpoint: loaded as zero biases (exactly equivalent).
- **Verified:** nano equals `TabICL._train_forward` to ≤ 2.6e-4 (fp32, 3–16 features, both tasks; `python tabicl_port.py`).
- TabICL's own eval-mode path differs from its training path by 0.01–0.07, through a separate column-embedder inference code path. An earlier "match" (fp16, 4e-3 in probabilities) was hiding the grouping offset.

**Pipeline.**
- [obs_train.py](experiments/obs_noise/obs_train.py) `--arch tabicl`: 27.55M config, released weights as init.
- Regression head: 999 quantiles (levels k / 1000) for training. [obs_eval.py](experiments/obs_noise/obs_eval.py) uses every 10th, i.e. levels k / 100, matching nano.
- `--freeze_backbone` now sets `requires_grad=False` on the backbone.
- Cheap frequent validation: `--val_blocks 4`, 7 noise settings, every 250 steps; training loss every 25 steps (`--log_every`).

**Reference: TabICLv2 + median through our pipeline** (raw module, nano's column standardization, 1 pass), vs the sklearn wrapper with 1 estimator (which adds its own preprocessing):
- Equal within ±0.4 on most settings.
- Lower on clean (−1.1 ± 0.8), heavy left-censoring (**−3.5**), detection dropout (−1.0) and `fail_floor` (−0.7). The wrapper's preprocessing helps with point masses at a limit.
- Matched comparisons use our pipeline throughout. The wrapper numbers are reported alongside.

**Stage A (running, [phase_tx.sh](experiments/obs_noise/phase_tx.sh)).**
- Frozen backbone, branch only (p7 column-context config).
- 6000 steps, lr 3e-4, v2 mixture with count regimes, both tasks.
- About 1.3 s/step with two runs sharing the GPU.
- Monitored every 250 steps against the TabICLv2 + median reference ([monitor_tx.py](experiments/obs_noise/monitor_tx.py)).
- Step 0 (= TabICLv2 on cell means): noisy validation accuracy −1.6 points vs median (Cauchy −6.1, censoring +2.3).

**Median start (`center: 'attn_med'`).**
- The center fed to the backbone is median + 10·α·(attention center − median), with α zero-initialized. The median is numpy's linear median, the same arithmetic as the baseline's `masked_quantile`.
- **Step 0 equals TabICLv2 + median imputation exactly:** max |Δ| = 0.0 on logits and quantiles, both tasks, with all other branch maps at their zero init. Every validation point is therefore a direct "gain over median" readout.
- α gets a gradient immediately, and the attention scores learn once it moves. All-single tables still equal TabICLv2 exactly with every map randomized.
- **Why not a saturated softmax on −|z − median|:** it also gives the exact median, but it passes almost no gradient to the learned scores.
- **Why not Huber / trimmed mean:** a better neutral start in principle, but it would need its own matched imputation baseline. On LINCS-like noise (t₂–Cauchy tails, 2–3% outliers) it behaves like the median anyway.
- **Runs:** `tx_obs_colmed_A[_reg]`, same budget as stage A, queued after it ([phase_tx2.sh](experiments/obs_noise/phase_tx2.sh)). Stage B will use the median start in any case, so that early backbone gradients are not spent adapting to mean-imputed inputs.

**Optimizer for stage B: Muon, to match TabICLv2's pretraining** ([muon.py](muon.py), vendored from `tabicl.train._muon`, BSD-3).
- TabICL's training code puts all parameters in one Muon group: non-2D ones reshaped, `matched_adamw_rms=0.2` (lr on the AdamW scale), Nesterov, momentum β₁ = 0.9, weight decay 0 by default. Its config notes that the released checkpoints were trained with Muon's cautious-WD option off, i.e. with Muon.
- `obs_train.py --optim muon`:
  - backbone: that Muon group, `--muon_wd` 0;
  - replicate branch: Muon's internal AdamW (β = 0.9 / 0.95, wd 0.01), as in stage A.
  - CPU-checked: loss decreases, the gate moves, and the optimizer state round-trips through `state_dict`.
- **Stage A stays on AdamW:** the backbone is frozen, so there is no mismatch, and the branch is small (mostly vectors, LayerNorms and the scalar gate).
- **Stage B ([phase_txB.sh](experiments/obs_noise/phase_txB.sh)):**
  - start from the better stage-A run;
  - backbone lr 1.5e-5 (warmup 200, cosine to 10%); branch lr 20× that = 3e-4;
  - 4000 steps;
  - matched baseline `tx_median_B`: TabICLv2 + median imputation, continued with the same optimizer, steps and data.

**Interim 27-dataset eval of stage A (mid-training; frozen TabICLv2 + branch, mean start, 1 pass)**, vs the sklearn wrapper:

| classification (accuracy) | clean | main (10) | het (4) | log-loss main | ECE main | Δ main vs wrapper 1-est. | Δ het vs wrapper |
|---|---|---|---|---|---|---|---|
| TabICLv2 wrapper, 1 est. / 8 est. | 0.742 / 0.742 | 0.645 / 0.646 | 0.656 / 0.656 | 0.846 | 0.040 | | |
| TabICLv2 + median, our pipeline | 0.730 | 0.640 | 0.652 | 0.860 | 0.043 | −0.5 | −0.4 |
| **stage A, step 2750** | 0.730 | **0.658** | 0.656 | **0.820** | **0.035** | **+1.2 ± 0.2 (25/27)** | +0.0 |
| TabICLv2 + Bayes-oracle input, wrapper | 0.742 | 0.667 | 0.661 | 0.801 | 0.040 | +2.2 | +0.5 |

| regression (CRPS) | clean | main (10) | het (4) | PIT err main | cov90 main | Δ main vs wrapper 1-est. | Δ het vs wrapper |
|---|---|---|---|---|---|---|---|
| TabICLv2 wrapper, 1 est. / 8 est. | 0.125 | 0.278 | 0.262 | 0.024 | 0.900 / 0.903 | | |
| TabICLv2 + median, our pipeline | 0.128 | 0.279 | 0.263 | 0.026 | 0.887 | +0.001 | +0.000 |
| **stage A, step 2500** | 0.128 | **0.259** | **0.254** | 0.028 | 0.888 | **−0.019 ± 0.002 (26/27)** | **−0.008 ± 0.002 (23/27)** |
| TabICLv2 + Bayes-oracle input, wrapper | 0.125 | 0.245 | 0.253 | 0.024 | 0.906 | −0.033 | −0.009 |

- **Halfway through stage A, the frozen-TabICLv2 branch beats the TabICLv2 sklearn wrapper on the main noisy settings in both tasks.**
- It already captures about 60% of the Bayes-oracle headroom: (0.658 − 0.645) / (0.667 − 0.645) in classification, (0.278 − 0.259) / (0.278 − 0.245) in regression.
- **Regression het:** −0.008, close to the oracle's −0.009.
- **Calibration:**
  - classification is better (ECE 0.035 vs 0.040; log-loss −0.026);
  - regression PIT error is slightly worse (0.028 vs 0.024), with 90% coverage 0.888, i.e. a little narrow.
- **The remaining clean gap is our pipeline, not the branch.**
  - Clean data: 0.730 vs 0.742 (CRPS 0.128 vs 0.125). The wrapper's preprocessing is worth that much on TabICLv2 + median as well.
  - The branch cannot touch n = 1 data.
  - The fix: port the wrapper's column preprocessing into the pipeline, or let stage B learn it.

**Target changed to the off-the-shelf wrapper → ported its input preprocessing** (`NanoTabICLv2(preproc='tabicl')`, [model.py](model.py) `tabicl_preprocess`; on for every new TabICLv2-arch run via `tabicl_port.nano_config`).
- **Which preprocessing.** The wrapper with `n_estimators=1` takes the first ensemble configuration, which uses normalization `'none'`; the Yeo-Johnson branch only enters 8-estimator ensembles. Its per-column pipeline, fitted on the context rows:
  1. constant-column filter;
  2. z-score with population sd + 1e-6, clipped to ±100;
  3. two-stage 4-sd outlier soft clip;
  4. one random feature / class permutation.
- **Port:** steps 2–3 in torch, applied to the per-cell value the backbone sees (median, mean or the learned center), fitted on context rows; the replicate encoder keeps its own robust standardization.
  - **Verified:** equal to `tabicl._sklearn.preprocessing.PreprocessingPipeline('none')` to **2.4e-14**, on the 27 eval datasets × {clean, heavy censoring, Cauchy, `fail_floor`} medians.
  - **Not ported:** the constant-column filter. Those columns become all-zero instead, and no eval context has one.
- **Old runs are unaffected.** Checkpoints without `preproc` in their `model_cfg` keep the plain z-score: `tx_median0`, the running mean-start stage A.
- **Queue ([lanes_preproc.sh](experiments/obs_noise/lanes_preproc.sh)), two lanes, each starting when a running stage-A run frees the GPU:**
  - **lane 1:** reference `tx_median0p` (TabICLv2 + median with the preprocessing; validation + 27-dataset eval) → median-start stage A `tx_obs_colmed_A` → stage B branch `tx_obs_B` → `tx_obs_B_reg`;
  - **lane 2:** `tx_median0p_reg` → `tx_median_B` → `tx_obs_colmed_A_reg` → `tx_median_B_reg`.
  - Stage B starts from the median-start stage A, which now differs from the mean-start run in its preprocessing too, so it is not a pure start-point comparison any more.

**Precision was the remaining gap: bf16 autocast degrades the released TabICLv2.**
- Diagnosis on the 27 eval datasets, after the preprocessing port:
  - the TabICL module's eval path, its training path and our port all reproduce the wrapper in fp32 (heavy censoring 0.629–0.630, clean 0.742);
  - **our model under bf16 autocast: 0.591 / 0.720** (CPU check).
- TabICL trains with autocast dtype float32 and TF32 matmuls (`tabicl.train` defaults), and runs inference in fp16.
- **Fix:** `obs_train.py --precision` defaults to fp32 + TF32 for `--arch tabicl` (bf16 kept for the nano runs, which were trained that way). `obs_eval.py` evaluates TabICLv2-arch checkpoints in fp32 + TF32.
- **Reference, TabICLv2 + median through our pipeline (fp32 + preprocessing, `tx_median0p`) now equals the wrapper 1-est.:**
  - classification: clean 0.742, main 0.645, het 0.656, heavy censoring 0.630 (wrapper 0.629);
  - regression: clean 0.126 (0.125), main 0.278, het 0.262;
  - validation clean accuracy 0.801, vs 0.773 under bf16.
- **Restarted:** the median-start stage A and `tx_median_B` had started under bf16; they were stopped and restarted in fp32. Their bf16 logs are kept as `train_*_bf16.log`.

**Mean-start stage A (`tx_obs_col_A[_reg]`; trained in bf16, no preprocessing), evaluated in fp32, 1 pass, vs the off-the-shelf wrapper:**

| | clean | main (10) | het (4) | log-loss / ECE main | Δ main vs wrapper 1-est. | Δ het |
|---|---|---|---|---|---|---|
| classification | 0.742 (= wrapper) | **0.661** (wrapper 0.645, oracle 0.667) | 0.660 (0.656) | **0.813 / 0.034** (0.846 / 0.040) | **+1.5 ± 0.3 (25/27)** | +0.3 ± 0.2 (16/27) |
| regression (CRPS) | 0.125 (= wrapper) | **0.255** (wrapper 0.278, oracle 0.245) | **0.251** (0.262; oracle 0.253) | PIT err 0.023 (0.024), cov90 0.903 | **−0.024 ± 0.003 (27/27)** | **−0.011 ± 0.002 (24/27)** |

- **The frozen-backbone branch beats the off-the-shelf TabICLv2** on noisy data in both tasks, with no clean-data cost and better calibration.
- It captures about 70% of the Bayes-oracle headroom: classification 73%, regression 70%.
- Regression het slightly beats the oracle-denoised input.
- The fp32 median-start runs (with the wrapper preprocessing) and stage B follow.

Comparison script: [compare_tx.py](experiments/obs_noise/compare_tx.py).

**Two fixes found while restarting in fp32.**
1. **NaN gradients from the preprocessing** with the median start: the losses went non-finite from step 3. Cause: `tabicl_preprocess` differentiated through its own fitted statistics. A context column with a single repeated value (point masses and censoring limits are common in the prior) gives a second-stage sd of sqrt(0), whose gradient is 0·∞ = NaN. The median start made it bite, because it passes gradients into exactly those cells.
   - **Fix:** the fitted statistics are detached (constants, as in the wrapper, which fits them on the training rows); gradients flow through x only. The forward values are unchanged.
   - Re-checked: the same seed and data stream give finite gradients and parameters.
2. **Memory:** in fp32 each 27M job needs 7–9 GiB at micro-batch 4, so two jobs did not fit (`tx_median_B` hit OOM 3×).
   - Measured peaks at micro-batch 2: stage A obs 3.6 GiB, stage B obs 4.7 GiB (Muon), stage B median 3.8 GiB; K = 50 batches ≤ 4.5 GiB.
   - All TabICLv2 runs now use micro-batch 2; the effective batch is unchanged at 32 tables.
   - About 1.3 s/step with two jobs.

**Branch upgrades for the 27M backbone (queued as stage-A screens).** Options in [obs_model.py](obs_model.py); all keep both guarantees, verified with every zero-init map randomized (`python obs_model.py --col_tok`).

| idea | decision | implementation |
|---|---|---|
| capacity matched to the backbone | **screen** (`tx_scr_cap_A`) | `d_s` 64 (was 32) + `n_sab` 2 (a second self-attention block over replicates): 124k vs ~30k branch params |
| richer column token: include single cells, several modes, cross-attention | **screen** (`tx_scr_tok_A`) | `col_tok` 4. Every context cell (replicated or single) is summarized as [median z, log n, log(1 + MAD z), limit-tie share, n == 1] → MLP → pooled per column by 4 learned seed queries (several modes, e.g. a floor cluster plus the bulk). Replicates cross-attend to the 4 tokens; zero-init output. |
| target-aware column summary | flag ready (`col_tok_target`), screened only if `col_tok` helps | context rows' label embedding added to their cell summaries before pooling (leak-free) |
| cross-feature, same-replicate attention (whole-well failures) | **deferred** | needs paired replicates in the prior (backlog P2) plus matching eval data; without them there is nothing to learn |

- **Screens:** classification, frozen backbone, median start, fp32, 3000 steps each. Compared with `tx_obs_colmed_A` at the same steps.
- **Schedule:** lane 2 after `tx_median_B` ([lane2b.sh](experiments/obs_noise/lane2b.sh); configs in `screen_cfgs.json`, read at launch). The regression median-start stage A and `tx_median_B_reg` follow them.

**Global gate failed; per-cell gate (`center: 'attn_medg'`).**
- With the median start and one global gate α, the gate stayed at ≈ 0: 10α = −0.0075 / −0.0045 at steps 250 / 750. The center therefore stayed the median.
- `tx_obs_colmed_A` gained only +0.39 / +0.41 points over median at steps 500 / 750: only through the embedding paths, without location learning.
- **Cause:** one scalar cannot serve all noise families. Cauchy cells want the median (α ≤ 0), censored cells a mean-like or attention center (α > 0), so their gradients cancel.
- **Fix:** g = `rep_gate`(cell vector), zero-init linear head; center = median + g · (attention center − median), per cell.
  - Step 0 equals TabICLv2 + median exactly (0.0, TabICLv2 config with the preprocessing).
  - The gate head gets a gradient immediately (norm 7.7 at step 0). The attention scores learn once g ≠ 0.
- **Runs:** `tx_obs_colmed_A` stopped at about step 1000 (kept as a record). The stage A, screens and stage B now use `attn_medg` ([lanes_g.sh](experiments/obs_noise/lanes_g.sh)).
- **`tx_median_B`** (TabICLv2 + median continued with Muon, lr 1.5e-5): +0.17 points over its step-0 reference at step 1000; clean −0.1. A small bar for stage B to clear.

**`tx_median_B` (TabICLv2 + median imputation, backbone continued with Muon, lr 1.5e-5, 4000 steps; fp32 + wrapper preprocessing), 27 datasets, 1 pass:**
- clean **0.744** (wrapper 0.742);
- main 0.648, **+0.3 ± 0.1 vs the wrapper (20/27)**;
- het 0.657; log-loss 0.839; ECE 0.035.

**No forgetting:** continued training with Muon on our noise mixture keeps, and slightly improves, TabICLv2's clean accuracy. The noisy gain from imputation plus continued training alone is small (+0.3). On validation it plateaued at +0.22 from step 2000.

**Per-cell gate `tx_obs_colmedg_A`, validation vs median:** +0.76 / +0.93 / +1.11 / +1.23 at steps 1000 / 1500 / 2000 / 2500, still rising.
- `fail_floor` +4.2, heavy censoring +1.7, Cauchy / per-row Cauchy +0.6, clean 0.

**Screens and checks (stage A, classification).**
- **`tx_scr_gate10_A` (10× lr on the gate and location heads): stopped at 1500.** Identical to `tx_obs_colmedg_A` at the same steps (+0.75 / +0.93 / +0.92 vs +0.76 / +0.86 / +0.93). The location learning rate is not the bottleneck.
- **Mean vs median imputation under fp32 + preprocessing** (TabICLv2, step 0, validation points):

  | heavy censoring | Cauchy | per-row Cauchy | `fail_floor` | Gaussian / plate / detection dropout |
  |---|---|---|---|---|
  | −0.6 | −6.1 | −5.2 | −5.3 | +0.5 / +0.8 / +0.5 |

  - The mean-start run's large censoring gains (+5.5) were measured against the bf16 median reference, which bf16 degraded most exactly there.
  - Under fp32 the median is the stronger start everywhere that matters, and the per-cell gate's +1.8 on heavy censoring is a genuine gain.
- **`tx_obs_colmedg_A`:** +1.31 / +1.33 / +1.37 at steps 3500 / 3750 / 4000. Still rising slowly as the cosine schedule decays. The branch keeps training in stage B (branch lr 3e-4), so stage A is not extended.

**`tx_obs_colmedg_A` final (per-cell gate, fp32 + preprocessing, 6000 steps), 27 datasets vs the wrapper 1-est.:**

| | clean | main (10) | het (4) | log-loss main |
|---|---|---|---|---|
| 1 pass | 0.742 | 0.660, **+1.5 ± 0.3 (25/27)** | 0.661, +0.5 ± 0.2 | 0.815 |
| 8 passes | 0.741 | 0.661, +1.5 | 0.663, **+0.6 ± 0.2 (17/27)** | 0.813 |

- Same main-setting gain as the bf16-trained mean-start run (+1.5), slightly better on het (+0.5 / +0.6 vs +0.3).
- The median start mainly buys a clean step-0 readout and safety; ensembling adds nothing for a TabICLv2 backbone.
- Final validation: +1.45 over median at steps 5500–6000 (flattening as the cosine schedule decays).
- **Stage B (`tx_obs_B`) from it:** +1.33 points over `tx_median_B` at step 250, clean unchanged (0.8002 vs 0.7999).
- **Capacity screen `tx_scr_cap_A`:** +1.21 at step 1750, vs about +1.0 for `tx_obs_colmedg_A` at that step.

**Screens, final (stage A, classification; mean noisy validation gain over median, points):**

| step | base (`tx_obs_colmedg_A`, d_s 32) | capacity (`tx_scr_cap_A`, d_s 64 + 2 SAB) | column tokens (`tx_scr_tok_A`, 4 tokens) |
|---|---|---|---|
| 1000 | +0.76 | **+0.94** | +0.65 |
| 1500 | +0.93 | **+1.15** | +0.85 |
| 3000 | ≈ +1.27 | **+1.40** (3000-step schedule) | stopped at 1500 |
| 6000 | +1.45 | — | — |

- **Capacity wins.** It reaches in 3000 steps what the base reaches in 6000, mainly on `fail_floor` and censoring.
- **Pooled column tokens are slower** than the single average (`fail_floor` +2.2 vs +3.1 at step 1500); stopped. `col_tok_target` is therefore not screened.
- **Regression stage A (`tx_obs_colmedg_A_reg`) uses the capacity config.** The name is kept for the lane-1 dependency; the config is in `reg_cfg.json`, logged at launch.
- **Full-length capacity stage A for classification (`tx_obs_cap_A`, 6000 steps)** queued after stage B ([lane3.sh](experiments/obs_noise/lane3.sh)).
- **Stage B `tx_obs_B`:** +1.40 / +1.49 / +1.51 / +1.61 / +1.65 / +1.68 over `tx_median_B` at steps 750 … 2750. Clean equal to `tx_median_B` (0.800).

### Stage B result, classification (`tx_obs_B`: from `tx_obs_colmedg_A`, backbone unfrozen with Muon lr 1.5e-5, branch AdamW 3e-4, 4000 steps, fp32 + preprocessing). 27 datasets.

| | clean | main (10) | grouped | het (4) | log-loss main | ECE main | Δ main vs wrapper 1-est. | Δ het vs wrapper |
|---|---|---|---|---|---|---|---|---|
| TabICLv2 wrapper, 1 est. | 0.742 | 0.645 | 0.663 | 0.656 | 0.846 | 0.040 | | |
| `tx_median_B` (TabICLv2 + median, same continued training) | 0.744 | 0.648 | 0.667 | 0.657 | 0.839 | 0.035 | +0.3 ± 0.1 | +0.1 |
| `tx_obs_colmedg_A` (stage A) | 0.742 | 0.660 | 0.670 | 0.661 | 0.815 | 0.035 | +1.5 ± 0.3 | +0.5 |
| **`tx_obs_B`, 1 pass** | 0.742 | **0.664** | 0.674 | **0.664** | **0.805** | 0.035 | **+1.9 ± 0.4 (25/27)** | **+0.8 ± 0.3 (21/27)** |
| **`tx_obs_B`, 8 passes** | 0.742 | **0.666** | 0.676 | **0.666** | 0.803 | 0.035 | **+2.0 ± 0.4 (25/27)** | **+1.0 ± 0.3 (21/27)** |
| TabICLv2 + Bayes-oracle input, wrapper | 0.742 | 0.667 | 0.678 | 0.661 | 0.801 | 0.040 | +2.2 | +0.5 |

- **Main settings:** within 0.1–0.3 points of the Bayes-oracle-denoised input (about 90% of the oracle headroom).
- **Het settings:** above the oracle input (0.666 vs 0.661).
- **Log-loss:** about equal to the oracle's (0.803 vs 0.801).
- **No clean cost; better calibration than the wrapper.**
- **Stage B adds +0.4 / +0.3 (main / het)** over stage A; plain continued training of TabICLv2 + median adds only +0.3 over the wrapper.
- Validation vs `tx_median_B` at the same step: +1.33 (step 250) → +1.68 (2750), then flat to the end.

**Memory: capacity-sized jobs are serialized.**
- The capacity branch (d_s 64, 2 SAB) ran out of memory 4× next to the unfrozen stage B, and 3× next to a second capacity stage A.
- **Activation checkpointing** of the replicate encoder during training (`torch.utils.checkpoint`; identical loss and gradients, max |Δ| = 0.0) removed the 50-replicate attention-map peak, but two capacity jobs still do not fit together.
- **New order ([lane4.sh](experiments/obs_noise/lane4.sh)):**
  1. regression stage A (running);
  2. regression stage B branch + `tx_median_B_reg` in parallel;
  3. regression evaluation;
  4. classification capacity stage A (`tx_obs_cap_A`, 6000);
  5. a second classification stage B from it (`tx_obs_capB`; `phase_txB.sh` now takes `OBS_NAME`);
  6. its evaluation.

### Regression, TabICLv2 backbone (fp32 + preprocessing), 27 datasets, 1 pass (CRPS, lower is better)

| | clean | main (10) | het (4) | PIT err | cov90 | Δ main vs wrapper | Δ het vs wrapper |
|---|---|---|---|---|---|---|---|
| TabICLv2 wrapper, 1 est. | 0.125 | 0.278 | 0.262 | 0.024 | 0.900 | | |
| `tx_median_B_reg` (median + continued training, Muon) | 0.126 | 0.275 | 0.260 | 0.022 | 0.902 | −0.003 (27/27) | −0.002 |
| **`tx_obs_colmedg_A_reg`** (stage A: per-cell gate + capacity, frozen) | 0.126 | **0.254** | **0.251** | **0.022** | 0.904 | **−0.025 ± 0.003 (26/27)** | **−0.011 ± 0.002 (25/27)** |
| TabICLv2 + Bayes-oracle input | 0.125 | 0.245 | 0.253 | 0.024 | 0.906 | −0.033 | −0.009 |

- **Stage A alone:** about 75% of the oracle headroom on the main settings; better than the oracle input on het; better calibrated than the wrapper.
- Validation: −0.0172 vs median at step 6000, flat over the last 500 steps.
- **Stage B (`tx_obs_B_reg`), mid-run:** −0.0198 vs `tx_median_B_reg` at step 2250 (−0.0163 at step 500), i.e. about −0.023 vs TabICLv2 + median. Clean validation CRPS steady at 0.1775 (step 0: 0.1771).

**Paired test, stage B vs stage A (classification), 27 datasets.**

| | accuracy Δ (points) | log-loss Δ |
|---|---|---|
| main, 1 pass | **+0.41 ± 0.12 (19/27; t p = 0.002, Wilcoxon p = 0.0008)** | **−0.010 ± 0.002 (25/27; p = 0.0003)** |
| main, 8 passes | +0.51 ± 0.12 (23/27; p = 0.0002) | −0.010 (25/27) |
| het, 1 pass | +0.33 ± 0.14 (18/27; p = 0.03) | −0.006 (23/27; p < 0.001) |
| clean, 1 pass | −0.01 (p = 0.9) | +0.002 (p = 0.03; slight clean-calibration cost) |

- **The gain is concentrated in the failure modes:** `fail_shift` +1.9 ± 0.6, `fail_floor` +1.1 ± 0.3.
- **Heavy tails and censoring are ≈ unchanged:** Cauchy +0.4 ± 0.3, per-row Cauchy −0.1, heavy censoring −0.1. These are the modes LINCS actually shows (real-data check); the other settings are within +0.1 to +0.5.
- Stage B's validation gain plateaued: +1.68 at step 2750, +1.71 at step 4000. More steps of the same recipe are unlikely to help.
- Settings are not independent and several comparisons were made. The main / log-loss results survive a Bonferroni correction; het does not.

## Exp 9 — Real replicate data (LINCS) and a real-shaped synthetic prior (2026-10-20, running)

**Training data is synthetic only.**
- Every run trains and validates on the synthetic bank (`artifacts/bank*`: clean nanoprior tables from `gen_bank.py`), with replicate noise simulated at training time.
- No real sample enters training. The TabICLv2 weights are the released ones, pretrained by its authors on TabICL's synthetic prior.
- **Caveat: LINCS is now development data.** The v3 prior below (the `contam` shape, the mixture weights) was designed after looking at LINCS: the replicate noise check and the mean-vs-median result on the dose split. LINCS results for v3-prior runs are therefore not a clean held-out test; earlier runs' priors were fixed before LINCS.
- **Held-out real test sets:** Tox21 and gas-sensor drift. Their loaders come next. Nothing is tuned on them, and they are only looked at once the prior is frozen.

**Loader ([real_data.py](experiments/obs_noise/real_data.py)), same format as the simulated settings** (`replicates` n × d × K NaN-padded, `counts`, `y`, `train_idx`, `test_idx`):
- one row per (compound, dose); its replicate wells are the replicates (L1000 ≈ 3, Cell Painting ≈ 4.9);
- features: z-score + PCA to 16 components, both fitted on the DMSO negative-control wells only;
- 10-way MoA tasks: MoAs ranked by number of compounds, 5 tasks per dataset;
- single-MoA compounds only; positive-control groups (> 10 wells) excluded.
- **Two splits:**
  - `compound`: the paper's split, test compounds unseen;
  - `dose`: compounds in the context, a seeded third of each compound's dose rows held out; a row's wells stay together.
- Evaluation: [real_eval.py](experiments/obs_noise/real_eval.py) (our checkpoints), [real_tabicl.py](experiments/obs_noise/real_tabicl.py) (the off-the-shelf wrapper), [real_summary.py](experiments/obs_noise/real_summary.py).

**Compound split: near chance for every model.** Accuracy 0.18–0.20 (chance 0.10); log-loss 2.9–3.0 for the TabICLv2 models, worse than uniform (2.30). Replicate handling cannot matter there: our branch ±1 point, slightly worse log-loss.

**Dose split (10 tasks), all wells:**

| model | acc, 1 / 2 / 3 / all wells | log-loss | vs wrapper + mean, 8 est. |
|---|---|---|---|
| wrapper + **mean**, 8 est. | 0.551 / 0.593 / 0.641 / **0.657** | **1.006** | |
| TabICLv2 + mean, our pipeline | 0.548 / 0.593 / 0.633 / 0.654 | 1.026 | −0.2 ± 0.5 |
| wrapper + median | 0.546 / 0.592 / 0.551 / 0.577 | 1.218 | −8.0 ± 1.1 |
| stage A, mean start (v2 prior) | … / 0.631 | 1.089 | −2.5 ± 0.5 |
| stage A, per-cell median start (v2 prior) | … / 0.624 | 1.121 | −3.2 ± 0.7 |
| **stage B** (v2 prior) | 0.549 / 0.595 / 0.618 / **0.640** | 1.068 | **−1.6 ± 1.0**; vs median **+6.4 ± 1.1 (10/10)** |
| best 1M models | … / 0.36–0.37 | 1.79 | −29 |

- **On real LINCS replicates the mean beats the median by 8 points.** The noise is near-Gaussian with rare outliers and only 3–5 wells, so the mean is efficient. The median of 3 wells is even worse than that of 2 (which is their mean).
- **The branch recovers most of the median's loss** (+6.4 over median, 10/10) and improves monotonically with wells, but **trails plain mean imputation by 1.6 points.** The v2 prior (heavy tails and failures dominate) taught robustness the real data does not need.
- The TabICLv2 backbone is essential: the 1M models are about 30 points lower.

**v3 prior (dev-informed, synthetic).**
- New mode `contam`: Gaussian, plus per table p_out ~ U(0, 0.08) of the replicates replaced by outliers logU(3, 30)× wider. Tails: P(|e| > 3 / 8 robust sd) = 3.6% / 1.8%, vs LINCS 5.0–6.8% / 1.1–1.7%.
- Sampling weights: contam 6, Gaussian 3, Student-t₃ / plate / constant-CV 2, all other modes 1 (≈ 58% near-Gaussian; robustness to the rest kept).
- Earlier noise streams are unchanged.
- **Run `tx_obs_real_A`:** stage A, capacity config, per-cell median start, 6000 steps. Then a stage B from it (`tx_obs_realB`). Both evaluated on the LINCS splits and the 27 synthetic datasets ([lane5.sh](experiments/obs_noise/lane5.sh)).

**Do the v2-prior models transfer to real data? (prior fixed before any real data was seen)**

Accuracy, all replicates; Δ vs the wrapper with median imputation, 8 estimators (ours: 8 passes); paired over tasks.

| | LINCS dose split (dev, 10 tasks) | Gas drift (held-out test, 9 tasks: context batch b → test b + 1) |
|---|---|---|
| wrapper + mean, 8 est. | 0.657, **+8.1 ± 1.2 (10/10)** | 0.801, −1.4 ± 0.9 |
| wrapper + median, 8 est. | 0.576 | **0.816** |
| TabICLv2 + mean, our pipeline | 0.656, +8.0 | 0.797, −1.9 |
| TabICLv2 + median, our pipeline | 0.576, 0.0 | 0.809, −0.7 |
| stage A (v2 prior) | 0.625, +4.9 ± 0.8 (9/10) | 0.796, −2.0 ± 1.3 |
| **stage B (v2 prior)** | **0.642, +6.6 ± 1.2 (10/10)** | **0.786, −3.0 ± 1.0 (2/9)** |
| best 1M models (1 pass) | 0.36–0.37 | 0.71 |

- **The v2 prior does not transfer well enough.**
  - On LINCS the branch beats median imputation by a lot but trails plain mean imputation (−1.5).
  - On gas drift (held out, untuned) it brings nothing and stage B is slightly worse.
- **Gas drift's error is dominated by the batch-to-batch drift and by unit-to-unit sensor bias** (replicates are different physical sensors). The synthetic prior has neither.
  - Task accuracies range from 0.2 (batch 9 → 10) to 0.99.
  - Ensembling over feature orders matters there: the wrapper goes from 0.787 to 0.816 with 1 → 8 estimators.
  - Stage A's apparent 1-pass lead (0.804 vs 0.787) disappears with 8 passes.
- Our pipeline's lower gas log-loss (0.68 vs 0.87) comes from the pipeline (no 0.9 softmax sharpening), not from the branch.
- **Takeaway:** synthetic gains need a prior that resembles the real measurement process. The v3 prior (near-Gaussian-dominated) is the test: LINCS (dev) should close the gap to the mean; gas drift (held out) should at least not lose.

**Tox21 (held-out test).** Loader `real_data.py --datasets Tox21`:
- rows = substances screened in all 12 assays;
- columns = per-run log10 AC50 (inactive runs censored at the top tested concentration) and efficacy of 11 assays (22 columns);
- replicates = the 3–4 screening runs (independent across assays);
- target = the 12th assay's call (majority of runs active → 1, none → 0, otherwise dropped);
- 12 tasks, 1000 / 1000 stratified, 2–16% actives.

ROC-AUC (12 tasks):

| | 1 run | 2 | all | Δ vs wrapper + median, 1 est. |
|---|---|---|---|---|
| wrapper (mean / median, 1 / 8 est.) | 0.886 | 0.894–0.895 | 0.899–0.901 | ±0.2 |
| TabICLv2 + mean / median, our pipeline | 0.886 | 0.894 | 0.899 | 0.0 |
| stage A / B (v2 prior) | 0.886–0.887 | 0.892–0.895 | 0.900 / 0.901 | +0.1 / +0.2 ± 0.2 |
| best 1M | 0.873 | 0.881 | 0.888 | −1.1 |

- Replicate runs help everyone (+1.3 AUC points from 1 run to all), but **the branch adds nothing over simple imputation**: each run is already a fitted AC50, so run-to-run noise is small relative to the signal.

**Summary of real-data transfer (v2 prior):**
- the branch is never worse than median imputation;
- it is much better on LINCS (+6.6);
- it is on par with the best simple imputation on Tox21 and gas drift;
- it is below mean imputation on LINCS (−1.5).

The synthetic gains (+2 accuracy / −0.03 CRPS vs the wrapper) do not carry over to these real datasets as gains over the best simple baseline.

**Why the branch trails the mean on LINCS: the learned center is not noise-adaptive.**

Center position between the cell median (0) and the cell mean (1), median over replicated cells:

| | LINCS cells | synthetic Gaussian | synthetic Cauchy | synthetic heavy censoring | synthetic `fail_floor` |
|---|---|---|---|---|---|
| per-cell-gate stage A (v2) | 0.02 | +0.05 | +0.00 | +0.02 | +0.03 |
| stage B (v2) | −0.03 | | | | |
| v3 stage A, step 1500 | 0.03 | +0.06 | +0.01 | +0.09 | +0.01 |
| mean-start stage A (v2) | −0.05 | +0.13 | −0.05 | +0.01 | +0.02 |

- **Every model feeds the backbone a median-like value for every noise family**, even Gaussian, and even the mean-start run, which learned to move there. The synthetic gains come through the branch's embeddings, not through a better point estimate.
- **On LINCS that costs averaging efficiency.** Wrapper, 1 est., dose split: mean of wells 0.653, per-feature median 0.577, first well only 0.546, medoid well 0.455. So it is averaging (+11 points over one well) that matters, not well coherence; the medoid result rules out a pairing explanation.
- **Replicate sd / column sd:** L1000 0.81, Cell Painting 0.39, gas drift 0.25, Tox21 0.18 (prior: logU(0.1, 1), median 0.32). The low noise in gas drift and Tox21 explains why replicate handling does not matter there.
- **Cause in the prior:** mean vs median barely affects the loss in its Gaussian modes (moderate noise, 5.5 replicates on average), while the heavy-tail modes reward the median strongly. The v3 weights alone did not change this (center 0.03 at step 1500; LINCS dev −4.3 vs mean). **`tx_obs_real_A` was stopped at about step 1700.**

**v4 prior (synthetic only): add efficiency pressure.**
- `--scale_max 2` (noise scale logU(0.1, 2) × sd) and `--small_k_prob 0.5` (half of the tables with U{2..5} replicates, as in real assays), on top of the v3 mixture. Defaults reproduce the old streams.
- **Mean start with a per-cell gate (`center: 'attn_meang'`):** center = mean + g₁ (median − mean) + g₂ (attention center − mean), both gates zero-init.
  - Step 0 equals TabICLv2 + mean imputation exactly (0.0).
  - The median term breaks the zero-gradient deadlock: a single gate on (attention center − mean) gets no gradient at step 0, because zero-init scores put the attention center at the mean (grad norms 0.13 vs 6e-8).
  - **Rationale:** leaving the mean is triggered by outliers a cell can see in its own residuals; leaving the median toward the mean needs to know the noise is Gaussian, which 3 replicates cannot reveal.
- **Screens ([lane6.sh](experiments/obs_noise/lane6.sh)):** `tx_v4med_A` (median start) vs `tx_v4mean_A` (mean start). Base branch size (two jobs fit), 3000 steps. Then the LINCS dev evaluation, the center diagnostic and the 27-dataset synthetic evaluation. Gas drift and Tox21 stay untouched until a v4 configuration is chosen.

**v4 screens (3000 steps, base branch size), LINCS dose split (dev):**

| | acc, all wells | log-loss | vs wrapper + mean, 1 est. | center position on LINCS (median 0 → mean 1) |
|---|---|---|---|---|
| v2 stage A (per-cell gate, 6000 steps) | 0.624 | 1.121 | −2.9 ± 0.7 | 0.02 |
| v2 stage B | 0.640 | 1.068 | −1.3 ± 1.1 | −0.03 |
| **v4 median start** | 0.641 | 1.073 | −1.3 ± 0.9 | 0.01 |
| **v4 mean start** | 0.642 | 1.054 | −1.1 ± 0.6 | 0.24–0.28 |

- The v4 prior lifts stage A by +1.7, in half the steps, to the level of the old stage B. It still trails the mean.
- **Even the mean start drifts toward the median**, and adapts only weakly to the noise family (synthetic Gaussian 0.35 vs Cauchy 0.16).
- Synthetic validation shows the same: on Gaussian noise mean imputation beats median by +0.5, but the models reach +0.2 / +0.4.
- **Reason:** the gate is a linear head on a cell vector built from 2–5 replicates, which cannot tell Gaussian from heavy-tailed cells. Heavy-tailed cells dominate the gradient, so the model robustifies everywhere.

**Huber imputation with a column-pooled scale** ([huber.py](experiments/obs_noise/huber.py); torch `obs_model.huber_location`, equal to 9e-16).
- Scale: median over the column's replicated context cells of |replicate − cell median| × the Gaussian small-n factor for the cell size (simulated; odd n larger, because the median replicate has residual 0). IRLS with c = 1.345.
- Simulation (2–5 replicates), RMSE vs the clean value: Gaussian — mean 0.279, median 0.304, **Huber 0.281**; 5% outliers at 20× — 1.233, 0.863, **0.858**.
- **LINCS dose (dev), wrapper:** Huber 0.663 / 0.656 (1 / 8 est.), mean 0.653 / 0.657, median 0.577 / 0.576. **Huber − mean = +1.0 ± 0.5 (6/10) with 1 est.**
- **Huber start (`center: 'attn_huberg'`):** center = Huber + g₁ (median − Huber) + g₂ (attention center − Huber).
  - Step 0 equals TabICLv2 + Huber imputation exactly (0.0).
  - Both gates get gradient at step 0 (0.17 / 2.6).
- **Matched baseline:** imputation mode `huber` (`obs_train` / `obs_eval` / `real_eval`).
- **Runs ([lane7.sh](experiments/obs_noise/lane7.sh)):** `tx_huber0p` (reference) and `tx_v4huber_A` (v4 prior, 3000 steps).

**Caveat: the v4 runs' validation curves are confounded.** `--small_k_prob` also applied inside validation, so `tx_v4med_A`, `tx_v4mean_A` and `tx_v4huber_A` validated on different tables (half with U{2..5} replicates) than their step-0 references; `tx_v4huber_A` therefore shows −0.45 at step 0 instead of 0. Their 27-dataset and real-data evaluations use stored data and are not affected. **Fixed:** small-K tables are now drawn only in training batches (`regimes is not None`); runs started later validate on the fixed design.

**TabICLv2 + Huber imputation (`tx_huber0p`, no training):**
- **LINCS dose (dev):** 0.664, **+1.0 ± 0.5 vs wrapper + mean (6/10)**, the best real-data result so far; compound split 0.164 / 0.233 (L1000 / Cell Painting).
- **27 synthetic datasets:** main 0.640 vs 0.645 for median imputation. Heavy censoring is −5.7 (0.572 vs 0.629), because ties at the detection limit pull Huber down; the rest is about even.

**Huber-start result (`tx_v4huber_A`, v4 prior, 3000 steps), and what training does on real data.**

LINCS dose split (dev):

| | accuracy | Δ |
|---|---|---|
| TabICLv2 + Huber imputation (no training) | **0.664** | +1.0 vs wrapper + mean |
| Huber start, trained | 0.652 | **−1.2 ± 0.5 vs its own start (0/10)** |
| trained, learned center only (embedding maps and temperature head zeroed) | 0.646 | −1.8 |
| trained, embeddings only (center forced back to Huber) | 0.656 | −0.8 |

- Synthetic main settings: +0.8 vs the wrapper, like the other v4 runs.
- **Wherever the branch starts, synthetic training lands it at about 0.64–0.65 on LINCS:**
  - median start 0.576 → 0.624–0.641 (training helps);
  - mean start 0.654 → 0.642 (hurts);
  - Huber start 0.664 → 0.652 (hurts).
- Both learned parts move away from what works on LINCS, the center most. The synthetic gains sit in modes LINCS does not have (censoring +6.9, failures +3.8 over Huber on validation).

**Bug found on a held-out set: degenerate Huber scale.**
- On Tox21, Huber imputation first scored AUC 0.827 (−7.2 vs median imputation, 0/12). Efficacy is exactly 0 for inactive runs, so the pooled residual scale was 0; all IRLS weights vanished and the estimate collapsed to 0.
- **Fix** (numpy and torch, still equal to 9e-16): σ ≤ 1e-12 falls back to the cell median, the σ → 0 limit of the Huber estimate. A correctness fix for degenerate input, not tuned on Tox21.
- After the fix: Tox21 0.901, **+0.2 ± 0.1 (8/12)**; LINCS unchanged (no ties there).
- **`tx_v4huber_A` was trained with the buggy estimator.** The prior has point-mass columns, so its starting center collapsed in some batches; its results are partly confounded.

**Real-data scoreboard** (Δ vs the wrapper with median imputation, 1 est.; held-out sets untouched by any design choice except the bug fix above):

| | LINCS dose (dev), acc | Gas drift (test), acc | Tox21 (test), AUC |
|---|---|---|---|
| wrapper + mean | +7.6 | −0.4 | 0.0 |
| **TabICLv2 + Huber imputation (fixed estimator)** | **+8.7** | 0.0 | **+0.2 ± 0.1** |
| stage A, per-cell gate (v2) | +4.7 | **+1.7 ± 0.6 (1 pass)** | +0.1 |
| stage B (v2) | +6.3 | 0.0 | +0.2 |
| Huber start, trained (v4, buggy) | +7.5 | −0.4 | +0.1 |

- **The one transferable win is the estimator, not the learned branch.**
- Huber imputation with a column-pooled scale is best or tied on all three real datasets, and it costs nothing to apply.
- The synthetic-trained branch gives large synthetic gains (classification stage B +2.0 points, regression −0.032 CRPS vs the wrapper) but no consistent real-data gain.
- The gas-drift stage A lead (+1.7) disappears with 8 passes (see above).

**Change of direction (user, 2026-10-20): less hand-built bias in the model; invest in the prior and in user-supplied side information.**
- The Huber start, the per-cell gates and the column-statistics gate inputs were all hand-designed fixes for specific failure cases. They were dropped from the main line. The code stays, as options, for reference.
- **`tx_v5gen_A` (running, [lane9.sh](experiments/obs_noise/lane9.sh)):** the v5 prior with a generic branch, i.e. the plain learned center (`attn`: attention over the replicates, starting at the mean), with no Huber start and no gates. It tests whether the prior alone can teach adaptivity. LINCS dev is a kill switch only.
- **Next:**
  1. **A broad, continuous synthetic prior instead of a list of named modes.** Draw each column's (and row group's) noise from continuous families: tail index ν ~ logU(1, 50) (Gaussian → Cauchy), contamination rate and width, row and value heteroscedasticity strength, additive vs multiplicative (constant CV) mix, plate / well offsets shared across columns with random strength, left / right censoring with random limits, detection dropout, replicate-count design (mostly 2–5, sometimes up to 50). Fresh draws and a longer schedule.
  2. **User-specified side information as optional inputs** (with the other agent's censoring spec channels, `cens_model.py`): per-replicate flags (below LOD / above ULOQ / observed), and possibly a "replicates are paired across features" flag. In training the prior shows the flags with some probability, so the model learns both with and without them. Without flags the input equals the generic model.
  3. **Evaluation unchanged:** LINCS dev as a kill switch; gas drift and Tox21 held out until the end. Tox21 has natural censoring (inactive = above the top concentration), so it doubles as a real test of the censoring flags.

**Replicates as rows (LINCS dose split, dev; wrapper, 1 est.; no training).** Context = every well of every context row (label repeated); each test well predicted separately, probabilities averaged over the row's wells.

| | acc | L1000 | Cell Painting | log-loss | vs mean imputation |
|---|---|---|---|---|---|
| mean imputation (baseline) | 0.653 | 0.503 | 0.803 | **1.014** | |
| **wells as context rows, test wells averaged** | 0.648 | **0.467** | **0.829** | 1.181 | −0.5 ± 1.6 (5/10) |
| wells as context rows, mean-imputed test rows | 0.570 | 0.439 | 0.702 | 1.247 | −8.3 |
| mean-imputed context, test wells averaged | 0.548 | 0.425 | 0.671 | 1.414 | −10.5 |

- **The instance-level view has signal the per-cell summaries discard:** Cell Painting +2.6 (+9 on moa05: 0.898 vs 0.806).
- It **hurts the noisier L1000** (−3.6) and is **overconfident** (log-loss +0.17): the in-context learner treats a row's wells as independent examples.
- **Context and test representations must match** (the hybrids lose 8–10 points).
- **Implication:** a group-aware instance-level model (paired replicates; wells as tokens that know their row; learned pooling of evidence) is the first concrete source of real-data headroom beyond mean / Huber imputation.

**Revised priorities (2026-10-20):**
1. **Paired / instance-level replicate modeling,** promoted. Novel, matches how LINCS-like data is produced, and the only lever with demonstrated real-data headroom.
2. **A real replicate benchmark where replicate handling can matter,** with datasets chosen before looking at any model result.
   - Current sets: LINCS is near-Gaussian (mean ≈ optimal for location); gas drift and Tox21 have low replicate noise (1 → all replicates: +3 accuracy / +1.3 AUC points, vs +11 on LINCS).
   - The synthetic gains sit in censoring, non-detect and outlier modes these datasets lack. Candidates: qPCR with non-detects, label-free proteomics with intensity-dependent missingness, immunoassays with LOD / ULOQ, imaging screens with failed wells.
   - Screening criterion: a large 1 → all-replicate gain (headroom) and a known measurement process.
3. **A continuous synthetic prior,** including paired tables (well effects, within-group variability) and fresh draws.
4. **Stage A only; capacity by default,** with a base-size control in the first run.

**Training-free group-normalized attention (LINCS dose split, dev; TabICLv2, 1 est.; [model.py](model.py) `key_bias`).**
- **Mechanism.** `key_bias` (B, rows) is added to the attention logits wherever context rows are keys (column-stage inducing attention, ICL). The default is None = unchanged.
- **Group normalization.** Wells as context rows with bias −log(n_wells of the unit), so each unit carries total weight 1 in every attention over rows.
- **Exactness check.** With non-uniform duplication (10 rows ×3, bias −log 3) the output equals the original to 2e-5 when preprocessing is the identity. With the TabICL preprocessing the residual is 0.39, because the z-score / clip statistics count wells. Scratch scripts: `groupnorm_lincs.py`, `groupnorm_ablate.py`.

| test wells averaged per unit | L1000 | Cell Painting | avg |
|---|---|---|---|
| mean imputation (reference) | 50.6 | 80.3 | 65.4 |
| Huber imputation | 50.6 | 82.1 | **66.4** |
| wells as rows, no bias | 47.1 | **82.8** | 65.0 |
| + group bias, SSMax count = number of units | 36.1 | 53.9 | 45.0 |
| + group bias, SSMax count = real key count | 47.0 | 82.6 | 64.8 |
| + group bias in the column stage only | 45.1 | 81.6 | 63.4 |
| + group bias in ICL only, real SSMax count | 46.6 | 83.0 | 64.8 |

- **Group normalization is neutral (−0.2) once SSMax keeps the real key count.**
- The −20 came from SSMax: telling it there are fewer keys flattens attention over the ~3k real keys. `model.py` now always passes the real count.
- Reweighting units does not cure the L1000 loss or the overconfidence of wells-as-rows. The model's problem is treating wells as independent examples in its *predictions*, not their attention weight.
- **No training-free route beats Huber imputation on LINCS.** Instance-level gains need training, with paired replicate tables in the prior.

**`tx_v5gen_A` result** (v5 prior, generic branch: plain learned center, no Huber start, no gates; frozen backbone; 4000 steps; [lane9.sh](experiments/obs_noise/lane9.sh)).
- **LINCS dev.** Never degraded: steps 0 → 4000 went 65.4 → 66.0 / 65.5 / 64.9 / 66.0 / 65.8 / 65.6 / 65.7 / 65.9. The kill switch passed.
  - Final: L1000 50.2, Cell Painting 81.7, average **65.9**, vs mean imputation 65.4 and **Huber imputation 66.4**.
- **Synthetic, 27 datasets, 1 pass, vs the wrapper (1 est.):** main **+0.8 ± 0.2** (22/27), het 0.0, log-loss 0.828 (wrapper 0.846).
  - For comparison: stage B (v2) +1.9 (25/27), and Huber imputation −0.5 (it loses on the heavy censoring / failure modes).

| | synthetic main vs wrapper | LINCS dev |
|---|---|---|
| wrapper (median, 1 est.) | 0 | 57.6 (mean imputation: 65.4) |
| Huber imputation | −0.5 | **66.4** |
| `tx_v5gen_A` (generic, v5) | +0.8 | 65.9 |
| stage B (v2) | **+1.9** | 64.0 |

- **The generic branch no longer drifts away from real data:** a real-shaped prior stopped the LINCS degradation the v2/v4 branches showed.
- It is not yet better than a good fixed estimator on near-Gaussian real data.
- Its synthetic gain is less than half of stage B's, because the v5 prior gives the hard modes low weight.
- **Next:** the continuous prior (value-dependent missingness, intensity-dependent CV, paired well effects, realistic non-detect substitution, broad replicate-count designs) with the censoring flags; larger branch with a base-size control; the same kill switch.

**Matched ensembles (8 passes everywhere; LINCS dose dev).** Our checkpoints: 8 passes of column permutation + class shift. The wrapper: its own 8 estimators.

| | 1 pass / est. | 8 passes / est. |
|---|---|---|
| wrapper, mean-imputed input | 65.3 | 65.7 |
| wrapper, median-imputed input | 57.7 | 57.6 |
| our pipeline, mean imputation | 65.4 | 65.6 |
| Huber imputation | **66.4** | 66.1 |
| `tx_v5gen_A` | 65.9 | **66.2** |
| stage B (v2) | 64.0 | 64.2 |

- Synthetic, 27 sets: `tx_v5gen_A` 8 passes is +1.0 ± 0.2 vs the wrapper at 1 est. The wrapper's own 8 est. adds only +0.1, so the matched gap is about +0.9. Stage B at 8 passes: +2.0.
- **Rule from now on:** report both. The headline is 8 vs 8 against the off-the-shelf wrapper with mean-imputed input (what a user would feed it).

## Exp 10 — Continuous replicate prior (v6) + user side information (2026-10-20, running)

**Prior** (`observe_v6` in [obs_train.py](experiments/obs_noise/obs_train.py), `--noise_set v6`). No named noise models: every table draws its own observation process, and the components combine freely. P6 gives the probability that a table has each component; each strength is continuous.

| component | P(table) | draw |
|---|---|---|
| noise scale | always | logU(0.05, 2) × column sd; columns spread exp(U(0,1)·N) (p 0.7) |
| tails | 0.5 | eps = N / \|N'\|^a, a ∈ [0, 1] (Gaussian → Cauchy), per-column jitter |
| contamination | 0.35 | share U(0, 0.1) of replicates, logU(3, 30) wider, symmetric or one-sided (0.3) |
| row heteroscedasticity | 0.4 | exp(U(0, 0.8)·N) per row |
| value-dependent noise | 0.35 | exp(β·z(x)), β ~ U(−0.8, 0.8): intensity-dependent CV either way |
| paired well effect | 0.4 | N(0, ω²σ²) per (row, replicate k), shared by all columns (loadings U(0.3, 1.5)) |
| run / plate effect | 0.3 | per (column, k) + common per k, U(0, 0.8)·σ |
| failed wells | 0.2 | share U(0, 0.15); floor / large shift / lost; whole well (all columns) w.p. 0.5 |
| missing at random | 0.3 | replicates lost w.p. U(0, 0.3) |
| lower detection limit | 0.35 | column quantile U(0.02, 0.6); soft or sharp sigmoid on the measured value; undetected → absent (MNAR: informative counts, n = 0 cells) / at the limit (flagged '<' with the spec) / substituted floor (never flagged) |
| upper limit | 0.12 | quantile U(0.8, 0.99), reported at the limit, flagged '>' |
| spec shown | 0.5 | flags kept; otherwise all 0 |
| exact columns / tables | 0.3 / p_clean | covariates: n = 1, no noise |
| replicate design | always | n = 1 / 2 / 3 / 4 / 5 / U{6..K} per table / U{1..K} per cell = 5 / 20 / 30 / 15 / 10 / 10 / 10%; row variation (0.3); test rows n = 1 (0.1); K = 50 in 10% of batches |

- **Replicate index k is one physical well / run, shared across the columns of a row (paired tables).** Holes come from dropout, failures and non-detects.
- **Default draw statistics:** mean n 3.1; 3.4% n = 0 cells; 24% n = 1.

**Side information** ([obs_model.py](obs_model.py)). All new parameters are zero-init and bias-free, so a table without side information equals the model without it at any weights (self-test `python obs_model.py --side`: step 0 vs stock on mean imputation 2e-7; zero flags vs no flags 0.0).
- **`flag_emb`.** Per-replicate censoring flags {'<', '>'} (value reported at the limit, the convention of [CENSORING.md](CENSORING.md)). They go into the set encoder, and the cell's share of flagged replicates goes into the cell vector, which also reaches n = 1 cells.
- **`miss_emb`.** A learned vector for n = 0 cells. The backbone value of those cells is always the context column mean, the wrapper's NaN handling (`fill_missing`).

**Runs** ([lane10.sh](experiments/obs_noise/lane10.sh)).
- Generic branch (center `attn`, no gates), frozen TabICLv2, 6000 steps, the LINCS kill switch.
- `tx_v6cap_A`: d_s 64, 2 SAB. `tx_v6base_A`: base size, as the control.
- Validation: v6, v6_gauss, v6_heavy, v6_paired, v6_fail, v6_mnar, v6_lod_flag / v6_lod_noflag (the same draws, flags shown / hidden), plus 5 named modes.
- Evaluation at 1 and 8 passes.

**Exp 10 first result: both v6 runs hit the LINCS kill switch at step 500.**
- LINCS at step 500: `tx_v6cap_A` 64.1, `tx_v6base_A` 64.6 (step 0: 65.4, margin 0.5). Capacity is not the cause: the base-size control drops too.
- Validation at step 500 (cap / base, accuracy):
  - The named heavy modes improve strongly: Cauchy 0.608 → 0.661 / 0.657, fail_floor 0.600 → 0.636 / 0.618, censored_left_heavy 0.644 → 0.663 / 0.657, contam +1.5.
  - The v6 settings are flat: v6 0.703 → 0.703 / 0.702.
  - v6_gauss gets **worse**: 0.729 → 0.726 / 0.727, loss 0.655 → 0.662 / 0.660.
  - The flags are not used yet: v6_lod_flag = v6_lod_noflag to 0.001.
- **Headroom of fixed estimators on v6** (TabICLv2, no training; mean / median / Huber imputation):

  | | v6 | v6_gauss | v6_heavy | v6_fail | v6_mnar | v6_lod_flag |
  |---|---|---|---|---|---|---|
  | mean | 0.703 | 0.729 | 0.689 | 0.694 | 0.678 | 0.685 |
  | median | 0.706 | 0.727 | 0.695 | 0.699 | 0.678 | 0.684 |
  | Huber | 0.706 | 0.729 | 0.695 | 0.700 | 0.679 | 0.688 |

- **Reading.** Most v6 tables carry irreducible noise: large σ, few replicates, so location estimation gains little (Huber vs mean only +0.3). The gradient comes from the heavy-tail and failure tables. The branch buys robustness at the cost of Gaussian efficiency, and near-Gaussian LINCS punishes that.
  - In v6 only about a third of tables have neither heavy tails nor contamination (0.5 × 0.65); in v5 (which passed) about 70% of the mass was Gaussian-like.
- **Screen ([lane10s.sh](experiments/obs_noise/lane10s.sh), 500 steps each, the same 6000-step schedule, resumable).**
  - A: `p_tail` 0.5 → 0.2 (less tail mass).
  - B: no structured effects (`p_well`, `p_plate`, `p_valhet` = 0).
  - Training mixture via `--p6`; validation keeps the default P6, so all runs are compared on the same draws.

**Screen result (step 500, LINCS dev; step 0 = 65.4).**
- **A, `p_tail` 0.2: 65.4, passes.**
- B, no structured effects: 64.1, killed.
- **Tail mass is what costs LINCS.** The paired / plate / value-dependent effects do not.
- The early Gaussian-validation dip (v6_gauss loss 0.655 → 0.660) appears in every run, A included. `tx_v5gen_A` had the same dip at steps 250–1000 (gauss loss 0.689 → 0.692) and had recovered by step 3000, so it is a transient.
- **Continued ([lane11.sh](experiments/obs_noise/lane11.sh)):**
  - `tx_v6s_tail_A` (base size) resumed at step 500 → 6000.
  - `tx_v6tcap_A` (the same prior, d_s 64 + 2 SAB) starts from step 0.
  - Same kill switch.

**Both tail-reduced runs were killed too:** `tx_v6s_tail_A` 63.98 at step 1000 (65.38 at 500), `tx_v6tcap_A` 62.96 at 500. The screen's step-500 pass was within the noise of a single check, and the tail mass is not the explanation.

**Decomposition on LINCS dose dev** (1 pass; [scratchpad decomp_lincs.py]: zero the additive embedding → center only; feed the cell mean → embedding only).

| checkpoint | full | center only | embedding only | neither (= mean imputation) |
|---|---|---|---|---|
| `tx_v5gen_A` (v5, passed) | 65.9 | 61.3 | 64.9 | 65.4 |
| `tx_v6base_A` (killed @500) | 64.6 | 65.9 | 65.7 | 65.4 |
| `tx_v6s_tail_A` (killed @1000) | 64.0 | 64.1 | **65.9** | 65.4 |
| `tx_v6tcap_A` (killed @500) | 63.0 | 62.7 | **65.7** | 65.4 |

- **The learned center (attention over replicates) is what does not transfer.**
  - In the killed runs center-only ≈ full, and it is below mean imputation.
  - `tx_v5gen_A`'s center was even worse alone (61.3); its embedding had co-adapted to compensate.
- **The additive embedding alone beats mean imputation in every v6 run.**
- **Change (generic, less built-in bias):** center `mean`. The backbone sees the plain cell mean, as at step 0; all learned information goes through the zero-init embedding. Runs `tx_v6m_A` / `tx_v6mcap_A` ([lane12.sh](experiments/obs_noise/lane12.sh)) use the default v6 prior and the same kill switch.

**Mean-center runs, kill checks (LINCS dev; step 0 = 65.4).**
- `tx_v6m_A` (base size): 64.3 at step 500, **killed**.
- `tx_v6mcap_A` (d_s 64, 2 SAB): 65.5 at 500, 65.4 at 1000, **passed**. It continues to 6000 and is reported at its last checkpoint.
- **The mean center does not by itself fix the transfer.** The base run drops as before. Trained jointly, the embedding can shift the backbone's token as a center would, so the "embedding alone is fine" result from the attn runs does not carry over.
- The single-check noise is about ±0.5–0.7, so one pass vs one kill at this margin is weak evidence about capacity.
- **Synthetic validation barely improves on v6 in any run.** `tx_v6mcap_A` at step 1000: v6 loss 0.719 → 0.715 (Huber imputation without training: 0.709). The named heavy modes do improve: Cauchy 0.935 → 0.845, fail_floor 0.943 → 0.848.
  - The v6 prior gives the branch little learnable signal per table (large σ, few replicates).
  - It needs either a longer schedule or a prior with more cases where the replicates are informative.

**`tx_v6mcap_A` final** (v6 default prior, mean center, d_s 64 + 2 SAB, flags + n = 0 embeddings, 6000 steps).
- **LINCS dev** after the kill window: 64.9 / 65.5 / 65.8 / 65.0 / 66.7 / 65.3 / 66.2 / 65.3 / 65.4 / 65.4 (steps 1500–6000).

| | synthetic main vs wrapper (1 / 8 passes) | LINCS dev (1 / 8) | LINCS log-loss (8) |
|---|---|---|---|
| wrapper, mean input | 0 / +0.1 | 65.3 / 65.7 | 1.006 |
| Huber imputation | −0.5 / – | 66.4 / 66.1 | 1.018 |
| `tx_v5gen_A` (v5) | +0.8 / **+1.0** | 65.9 / **66.2** | 1.011 |
| `tx_v6mcap_A` (v6) | +0.3 / +0.5 | 65.4 / 65.9 | 1.032 |
| stage B (v2) | +1.9 / +2.0 | 64.0 / 64.2 | – |

- **Synthetic validation on its own prior:** v6 loss 0.719 → 0.705, vs Huber imputation 0.709 (it passed Huber only after about step 2500).
  - Flags help, but only a little: v6_lod_flag 0.737 vs v6_lod_noflag 0.739, on the same draws.
  - fail_floor 0.943 → 0.760; Cauchy 0.935 → 0.813; Gaussian unchanged.
- **Verdict: v6 is not better than v5 on anything we measure.** It is behind on the 27 synthetic sets (+0.5 vs +1.0, 8 vs 8), on LINCS (65.9 vs 66.2) and on log-loss.
  - The side-information channels work (exact guarantees, a small flag gain) but are not yet worth much.
  - The held-out real sets were **not** run: no dev evidence justifies spending them.

## Held-out real-data evaluation of `tx_v5gen_A` (2026-10-20; [heldout_eval.sh](experiments/obs_noise/heldout_eval.sh), [heldout_summary.py](experiments/obs_noise/heldout_summary.py))

- **Setup.** The first use of GasDrift, Tox21, UPS1_low and CPTAC for this model. All arms were re-run fresh on the same tasks (`artifacts/real_heldout.jsonl`), all replicates.
- **Reference:** the off-the-shelf wrapper, mean-imputed input, 8 estimators. Δ is paired per task, mean ± SE (n better).
- **Metric:** accuracy for LINCS and GasDrift; AUC for the binary Tox21, UPS1_low and CPTAC.

| dataset (tasks) | replicate headroom (wrapper, 1 → all) | Huber imputation, 8 | **`tx_v5gen_A`, 8** | stage B, 8 | log-loss: wrapper / v5gen |
|---|---|---|---|---|---|
| L1000 dose (5), dev | +8.9 | −0.5 ± 0.7 | +0.2 ± 1.2 | −3.5 ± 1.3 | 1.424 / 1.427 |
| Cell Painting dose (5), dev | +12.2 | **+1.4 ± 0.6** | +0.9 ± 0.6 | +0.7 ± 1.3 | 0.588 / 0.595 |
| GasDrift (9) | +2.0 | +0.4 ± 0.6 | −1.1 ± 0.7 | −1.5 ± 0.9 | 0.869 / 0.690 |
| Tox21 (12), AUC | +1.5 | +0.2 ± 0.2 | 0.0 ± 0.3 | +0.1 ± 0.2 | 0.170 / 0.167 |
| UPS1_low (10), AUC | +5.2 | −0.6 ± 0.2 | 0.0 ± 0.3 | −0.8 ± 0.4 | 0.069 / **0.060** |
| CPTAC (10), AUC | +0.7 | −0.8 ± 0.4 | **+1.3 ± 0.4 (9/10)** | +0.6 ± 0.7 | 0.082 / **0.075** |

- **`tx_v5gen_A` is the most consistent arm.** It is never clearly worse than the wrapper (GasDrift −1.1 ± 0.7 is the weakest).
  - It shows a clear held-out win on **CPTAC**: +1.3 AUC, 9 of 10 splits, log-loss −9%. CPTAC is the proteomics set with MNAR floor substitution, the replicate pattern the prior's censoring / floor modes model.
  - It is better calibrated on UPS1_low: log-loss 0.060 vs 0.069, −13%, at equal AUC.
- **Huber imputation does not generalize.** It is the best arm on Cell Painting and GasDrift, but it loses on both proteomics sets (−0.6, −0.8): non-detect floors break its location estimate.
- **Stage B (unfrozen backbone) is the least robust:** L1000 −3.5, UPS1_low −0.8. This confirms the stage-A-only decision.
- **GasDrift log-loss:** every arm of our pipeline (mean imputation included, 0.68) beats the wrapper's 0.87. That comes from the pipeline (preprocessing / ensembling), not the replicate branch.
- **Overall.** The gains are real but modest: +1 AUC on the one dataset with informative missingness, and calibration gains. Accuracy elsewhere is tied with the wrapper.
  - Where replicates carry a lot of information (LINCS +9 / +12 from 1 → all), the mean already captures most of it. Neither the branch nor Huber adds much beyond it.

## Exp 11 step 1 — real regression with replicates, training-free (2026-10-20)

**Tasks** ([real_data.py](experiments/obs_noise/real_data.py); designs fixed before any evaluation).
- **GasConc** (held-out): 6 gases. Target log10 ppmv (UCI 270, the concentration release). Replicates = the 4 sensor units per type.
- **Tox21Pot** (held-out): 12 assays. Target = median log10 AC50 of substances active in most runs (uncensored target). Features = the other 11 assays' per-run log AC50 (censored at the top concentration) + efficacy.
- **L1000 / CellPainting `_dose_reg`** (dev): target = dose level on the dose splits.
- Evaluation: [real_eval_reg.py](experiments/obs_noise/real_eval_reg.py), [reg_real_summary.py](experiments/obs_noise/reg_real_summary.py). CRPS is in context-sd units. Δ is paired vs the wrapper (mean input, 8 est.), negative = better.

| dataset | replicate headroom (wrapper CRPS, 1 → all) | wrapper R² | Huber, 8 | stage A `tx_obs_col_A_reg`, 8 | stage A `colmedg`, 8 | stage B, 8 |
|---|---|---|---|---|---|---|
| L1000 dose (dev) | −0.003 | −0.08 | +0.005 ± 0.005 | −0.000 ± 0.005 | −0.006 ± 0.005 | −0.005 ± 0.005 |
| CellPainting dose (dev) | **−0.074** | 0.36 | +0.000 ± 0.002 | +0.020 ± 0.006 (0/5) | +0.013 ± 0.004 (0/5) | +0.016 ± 0.003 (0/5) |
| GasConc | −0.002 | 0.93 | −0.000 | +0.001 ± 0.001 | +0.001 ± 0.001 | +0.001 ± 0.000 |
| Tox21Pot | −0.011 | 0.18 | −0.001 ± 0.001 | **−0.005 ± 0.001 (12/12)** | **−0.005 ± 0.001 (11/12)** | **−0.008 ± 0.002 (12/12)** |

- **The headroom is small.** The replicates matter much less for regression than for LINCS MoA classification:
  - GasConc is saturated (R² 0.93; one replicate is already as good as four).
  - L1000 dose is unpredictable (R² < 0), so it is useless as a dev set.
- **Tox21Pot is a consistent held-out gain for every trained branch** (11–12 of 12 assays), at −1.2% (stage A) to −1.8% (stage B) CRPS, i.e. 50–70% of the 1 → all replicate gain again on top of averaging. Huber imputation gives nothing. As with CPTAC in classification, the gain is on the dataset with censored replicate features.
- **Cell Painting dose (the only dev set with headroom): every trained branch loses** (+0.013 to +0.020, 0/5), about as much as median imputation (+0.023). These are old v2-prior checkpoints with median-start / learned centers, the transfer failure diagnosed in Exp 9 / 10.
- **Verdict on step 1: partial.** There is a real but small held-out gain where features are censored, and no clear gain elsewhere. The existing checkpoints carry the known center problem.

## Exp 11 step 2 — regression stage A on the v5 prior + learned context-row attention (2026-10-20, running)

- **`row_attn`** ([obs_model.py](obs_model.py)). The 7 row-reliability features (`row_features`; all exactly 0 for all-single tables) go through a bias-free MLP with a zero-init output, soft-clipped to ±5. The result is a logit bias per context row in every ICL attention ([model.py](model.py) `icl_key_bias`).
  - Self-test `python obs_model.py --row_attn`: step 0 vs stock 3e-7; n = 1 vs stock at random weights 0.0; the bias changes n > 1 outputs.
- **Regression kill switch.** `dev_real_eval` returns −100 × CRPS for regression, so a margin of 0.5 = 0.005 CRPS. The dev set is `CellPainting_dose_reg` only (L1000 dose is unpredictable). Step 0 = 0.428 = mean imputation.
- **Runs** ([lane13.sh](experiments/obs_noise/lane13.sh)). The v5 prior and settings of `tx_v5gen_A`, generic branch (center attn), d_s 64 + 2 SAB, frozen TabICLv2 regression backbone, 4000 steps.
  - `tx_v5reg_ra_A` (+ `row_attn`) vs `tx_v5reg_A` (control).
  - Then real regression (dev + GasConc, Tox21Pot) and the 27 synthetic sets, at 1 and 8 passes.

## Shape features: does the replicate set carry signal beyond its mean? (training-free, 2026-10-20)

- **Setup** ([shape_features.py](experiments/obs_noise/shape_features.py), [shape_summary.py](experiments/obs_noise/shape_summary.py)). The off-the-shelf wrapper, all replicates. Per cell it gets the mean (baseline), mean + sd, a sketch (mean, median, sd, q25, q75, min, max), or the sketch + log n. Δ is paired per task vs mean only.
- **8 estimators** (classification: points; regression: CRPS, lower is better):

| dataset | mean | + sd | sketch | best trained branch (`tx_v5gen_A` / stage A reg, 8) |
|---|---|---|---|---|
| L1000 dose, acc | 50.7 | **+1.0 ± 0.3 (4/5)** | +0.3 ± 1.2 | +0.2 ± 1.2 |
| Cell Painting dose, acc | 80.6 | **+1.5 ± 0.7 (4/5)** | −0.2 ± 1.5 | +0.9 ± 0.6 |
| GasDrift, acc | 80.1 | +2.9 ± 2.0 (4/9; 1 est.: +5.4 ± 2.3) | +0.6 ± 1.4 | −1.1 ± 0.7 |
| Tox21, AUC | 90.1 | +0.0 ± 0.1 | +0.3 ± 0.2 | 0.0 ± 0.3 |
| UPS1_low, AUC | 95.8 | +0.3 ± 0.1 | **+0.8 ± 0.3 (9/10)** | 0.0 ± 0.3 |
| CPTAC, AUC | 89.8 | **+4.0 ± 0.6 (10/10)** | +3.3 ± 0.4 (10/10) | +1.3 ± 0.4 |
| Cell Painting dose reg, CRPS | 0.432 | +0.021 (0/5, worse) | +0.030 (worse) | +0.013 (worse) |
| GasConc / Tox21Pot, CRPS | 0.038 / 0.433 | 0.000 / +0.002 | 0.000 / 0.000 | +0.001 / −0.005 |

- **In classification the replicate spread is label signal on most datasets**, and the hand feature beats every trained branch.
  - On CPTAC it is +4.0 AUC (10/10) vs the trained branch's +1.3. There, high within-cell spread marks censoring / floor substitution, which is far more frequent for the spiked UPS proteins (28% vs 6.5% of cells).
  - LINCS gains +1.0 / +1.5 with sd; UPS1_low gains +0.8 with the sketch.
- **Regression gets nothing, and Cell Painting dose regression gets worse.** The extra columns hurt the small dose-level tasks.
- **Implications.**
  1. "Mean + sd → wrapper" is a strong, free baseline that the replicate branch must beat; it currently does not. The branch has the spread, but it reaches the frozen backbone only through an additive embedding, apparently less usable than an explicit column.
  2. This supports the distribution-as-signal direction: shape matters even with 2–6 replicates.
  3. **Caveat:** this used the held-out sets for a training-free comparison. Any feature design chosen from it must be validated on new data.

**Exp 11 step 2 result: both regression runs were killed at step 500 on Cell Painting dose (dev).**
- CRPS: `tx_v5reg_ra_A` (+ row_attn) 0.428 → 0.453; the control `tx_v5reg_A` 0.428 → 0.464.
- Row attention is not the cause; it was slightly better than the control.
- The learned branch on the v5 prior does not transfer in regression either. The same holds for the hand shape features (+0.021 to +0.030), so the extra replicate information itself appears to confuse the dose-level regression.
- Step 3 (oversampling) is not started: step 2 did not pass.

## Shape learning (SHAPE.md), day 0 progress (2026-10-20)

**Prior headroom** ([shape_headroom.py](experiments/obs_noise/shape_headroom.py); TabICLv2 fp32, 4 held-out blocks). Each entry is the share of the mean → oracle loss gap closed with no training. Oracle accuracy is 0.801 in every setting.

| setting | mean acc | + sd | sketch |
|---|---|---|---|
| location | 0.736 | 1% | 7% |
| scale | 0.596 | 43% | 50% |
| mixture share | 0.627 | 12% | 27% |
| rare subpopulation | 0.569 | 11% | 13% |
| mode separation | 0.582 | 47% | 56% |
| tail | 0.580 | 19% | 19% |
| paired state | 0.680 | 2% | 5% |
| paired XOR | 0.665 | 4% | 8% |
| replicate-sized sets | 0.643 | 3% | 4% |
| population-sized sets | 0.688 | 17% | 28% |

- No setting is solved by the sketch, and none is pure noise, so the family shares are kept as designed.
- The paired settings are where hand features help least: that is the case for the joint path.

**New data.**
- HAR (dev, 5 subject-disjoint splits).
- Musk (final; 102 molecules).
- PhysioNet 2012 set-a (final; 4000 stays, 554 deaths, 41 columns, 26% n = 0 cells).
- Zenodo 6546964 cytometry: NK and MS verified; HVTN and preeclampsia re-downloading (checksum mismatch).

**HAR dev baselines (wrapper, 8 est.).**

| arm | accuracy | log-loss | overconfidence, high-spread rows | overconfidence, low-spread rows |
|---|---|---|---|---|
| mean | 73.1 | 0.851 | +0.149 | +0.056 |
| mean + sd | 82.6 | 0.567 | +0.171 | +0.043 |
| sketch | 86.5 | 0.439 | +0.127 | +0.038 |
| cluster shares | 83.7 | 0.539 | +0.097 | +0.076 |

- The wrapper is most overconfident on its most spread-out rows, which is the reliability failure to fix ([calib_rel.py](experiments/obs_noise/calib_rel.py), [calib_summary.py](experiments/obs_noise/calib_summary.py)).
- LINCS dose is underconfident or roughly calibrated in every bin.

**Model.**
- ISAB encoder (`isab`, linear in K).
- Per-cell replicate-sd columns (`sd_cols`; step 0 = the mean + sd input).
- Encoder chunked over cells with per-chunk checkpointing: identical to one call (outputs 3e-7, gradients 9e-7), and it fixes the population-size OOM.
- Self-tests pass.

**Screen pair 1 running** ([lane_shape1.sh](experiments/obs_noise/lane_shape1.sh)): V0 (mean) vs V1 (+ sd columns), with the dev kill on HAR + LINCS dose and the learning kill at step 1000.

**Robust location / scale pairs (wrapper, 8 est., CPU; paired vs the mean; [shape_features.py](experiments/obs_noise/shape_features.py) arms).**

| dataset | mean + sd | Huber + sd | median + 1.4826 MAD | Huber + MAD |
|---|---|---|---|---|
| HAR | +9.6 | **+10.5** | +8.6 | +9.1 |
| L1000 dose | +1.0 | **+1.1** | −7.5 | −0.2 |
| Cell Painting dose | **+1.5** | +1.4 | −7.7 | −0.9 |
| CPTAC, AUC | **+4.0** | +3.7 | +2.6 | +3.1 |
| UPS1_low, AUC | +0.3 | −0.0 | −2.9 | **+1.0 (10/10)** |
| Tox21, AUC | +0.0 | +0.1 | −0.2 | −0.2 |
| GasDrift | +2.9 | +2.8 | +0.8 | +2.1 |

- **Huber + sd is a slight overall improvement on mean + sd.**
  - It has the best log-loss on 4 of 7 sets (HAR 0.514 vs 0.567; L1000 1.378; Cell Painting 0.543; Tox21).
  - It has the best confidence-error AUROC on HAR (0.842).
  - It costs a little on proteomics.
- **The median-based pairs fail with 2–6 replicates,** the same lesson as median imputation.
- **No pair fixes HAR's overconfidence on high-spread rows** (about +0.15). That is left for the learned branch.
- **Candidate variant:** the backbone sees Huber + sd at step 0 (a different summary function, not a new mechanism), to try if V1 passes the screen.

**Screen V0 (shape prior, backbone sees the mean, ISAB marginal encoder) passed both kill switches.**
- Dev (HAR + LINCS dose, mean accuracy over 15 tasks): 67.7 → 71.9 (step 500) → 72.2 (step 1000).
- Learning check: shape-only settings' gap closed 0.040 → 0.277 (≥ +0.10 required).

Per dev dataset at step 1000, 1 pass; the wrapper at 1 est.:

| | V0 | wrapper mean | mean + sd | sketch |
|---|---|---|---|---|
| HAR acc / log-loss | **83.2 / 0.501** | 72.3 / 0.873 | 82.0 / 0.591 | 86.0 / 0.498 |
| HAR overconfidence, high-spread tertile | **+0.109** | +0.149 (8 est.) | +0.171 (8 est.) | +0.127 (8 est.) |
| L1000 dose acc | **52.3** | 50.3 | 51.6 (8 est.) | |
| Cell Painting dose acc | **81.1** | 80.3 | 82.1 (8 est.) | |

- **The first trained branch with a real gain on L1000:** +2.0 over the wrapper. The best earlier was `tx_v5gen_A` at 50.2.
- On HAR, the branch learned roughly what hand-built shape features give, from synthetic data alone, and it is the least overconfident arm on spread-out rows.
- Synthetic at step 1000 ≈ the sketch per family (scale 0.698 vs 0.699, mode separation 0.702 vs 0.705, mix 0.674 vs 0.675). Joint settings stay below the sketch (XOR 0.665 vs 0.676, state 0.675 vs 0.686); that is the target of V2.
- The screen continues sequentially (V1 → V2 → V3) after V1 hit OOM next to V0.

**Screen V1 (+ sd columns).**
- Dev 71.6 → 72.1 → 72.5.
- Learning check: 0.244 → 0.276, i.e. **+0.03 < 0.10 → fails the rule as stated.**
- On synthetic at step 1000, V1 = V0 (sh loss 0.740 vs 0.742; every family within ±0.004). The sd columns give V1's start what V0 learns; the learned part then adds little on top.

| dev, 1 pass, step 1000 | V0 | V1 |
|---|---|---|
| HAR acc / log-loss / confidence AUROC | 83.2 / 0.501 / 0.806 | **84.7 / 0.424 / 0.850** |
| HAR overconfidence, high-spread tertile | **+0.109** | +0.133 |
| L1000 acc / log-loss | **52.3** / 1.420 | 50.9 / 1.418 |
| Cell Painting acc / log-loss | 81.1 / 0.618 | **81.7 / 0.590** |

- **V1 has the best log-loss on all three dev sets.** Its HAR log-loss of 0.424 is better than the best hand-feature wrapper (sketch: 0.498 at 1 est.).
- V0 is better on L1000 accuracy and on HAR's high-spread calibration.
- **The learning rule is mis-specified for sd-column variants.** Their step 0 already closes 24% of the gap, so "+0.10 over step 0" measures the head start, not whether the architecture learns. V1's verdict is recorded as is.
- **For V2 / V3 (same start as V1):** learning is judged against V1's step-1000 value (0.276), not against their own step 0, stated before their results.

**Screen V2 (V1 + joint path, width 16).**
- Dev 71.6 → 72.0 → 72.3. Learning 0.244 → 0.272 (vs V1's 0.276: no gain).
- The joint settings are unchanged vs V1: sh_xor 0.663 / 0.799 vs 0.664 / 0.798; sh_state 0.673 vs 0.674.
- The joint path did train (output map norm 0 → 0.46). **The prior gave it little to learn.** The original XOR tables code a single column pair with one projection of x, and the other columns carry marginal signal, so sh_xor's mean → sketch → oracle is 0.665 → 0.676 → 0.801 (mostly ordinary noise).
- **Prior fix (option `xor_pairs`, validation `sh_xorp`).** Every column pair of an XOR table is XOR-coded: pair j carries column 2j's u through the share of anti-correlated instances, and every marginal is uninformative.
  - Headroom: mean 0.589, + sd 0.604, sketch 0.618 (10% of the gap), oracle 0.801. That leaves a large joint-only gap.
- **Matched pair queued after V3** ([lane_shape_xorp.sh](experiments/obs_noise/lane_shape_xorp.sh)): V1b (sd columns) vs V2b (+ joint), both on the xor_pairs prior. The learning check includes sh_xorp.

**Screen V3 (learned sd column).** Dev 72.4 / 72.2; learning 0.270 (V1: 0.276). No gain: all four marginal variants (V0–V3) end at the same synthetic level by step 1000.

**Final-evaluation cytometry sets built** ([real_data_sets.py](experiments/obs_noise/real_data_sets.py); sizes and labels only, no metrics). Each takes 1024 cells per sample, so rare responder populations are not lost; technical / scatter / viability channels are dropped. Splits are 20 seeded label-stratified halves, grouped by subject.

| dataset | samples | labels | groups | markers |
|---|---|---|---|---|
| NK CyTOF (CMV status) | 20 | 11 / 9 | 20 (one per donor) | 37 |
| HVTN flow (ENV vs GAG stimulation; 48 subjects × 2, grouped by subject) | 96 | 48 / 48 | 48 | 7 |
| Preeclampsia CyTOF (23 patients, mostly 2 samples, grouped) | 45 | 22 / 23 | 23 | 33 |

- MS scRNA-seq is dropped: 8 patients, decided on size alone.

**Matched pair on the xor_pairs prior (step 1000).**

| | V1b (sd columns) | V2b (+ joint path) |
|---|---|---|
| sh_xorp acc / loss (step 0: 0.620 / 0.892; sketch 0.618) | 0.624 / 0.884 | **0.644 / 0.837** |
| gap closed (learning check) | 0.298 | **0.324** |
| HAR acc / log-loss / overconfidence, high-spread tertile | 85.2 / 0.407 / +0.127 | **86.9 / 0.374 / +0.100** |
| L1000 dose acc: as evaluated (unpaired) / with `paired` | 50.5 / 50.5 | 49.2 / **51.5** |
| Cell Painting dose acc: unpaired / paired | **82.3** / 82.3 | 79.9 / 80.9 |
| dev (HAR + LINCS, kill switch) | 72.7 | 72.0 |

- **The joint path learns joint-only signal beyond the hand sketch** (sh_xorp 0.644 vs sketch 0.618). V2b is the best of all arms on HAR, wrappers included (sketch: 86.0 / 0.498), and the best calibrated on spread-out windows.
- The LINCS dose tasks were not marked paired, so the joint path was off there. **They are now marked paired** (`paired = True` added to the LINCS npz files: a well is measured on every feature). With the flag, V2b's L1000 is 51.5; Cell Painting remains 1.4 below V1b.
- Run-to-run variation on LINCS between identical architectures is about ±1 (V1 vs V1b).
- **Selection: V2b** (highest gap closed, the plan's criterion; best on paired dev).
  - Full run: resumed from step 1000 to 6000 ([lane_shape_full.sh](experiments/obs_noise/lane_shape_full.sh)).
  - **Note:** dev values from step 1500 on include paired LINCS, so they are not strictly comparable with the logged steps 0 / 500 / 1000.
  - The kill window has passed; dev is monitored by hand, and Cell Painting is the thing to watch.

**`tx_shV2b_A` final (6000 steps).**
- Dev 72.4–73.5 throughout steps 1500–6000 (73.1 re-scored at 1000, 72.9 at 6000): no drift.
- Synthetic at 6000: sh loss 0.735 (step 0: 0.749); sh_xorp 0.651 / 0.821 (sketch 0.618); modesep 0.699; mix 0.771.

Dev, 8 passes vs the wrapper at 8 est.:

| | V2b | mean | mean + sd | Huber + sd | sketch |
|---|---|---|---|---|---|
| HAR acc / log-loss / overconfidence, high-spread tertile | 86.1 / **0.388** / **+0.096** | 73.1 / 0.851 / +0.149 | 82.6 / 0.567 / +0.171 | 83.6 / 0.514 / +0.150 | **86.5** / 0.439 / +0.127 |
| L1000 acc / log-loss | **52.5** / 1.391 | 50.7 / 1.424 | 51.6 / 1.393 | 51.7 / **1.378** | 51.0 / 1.437 |
| Cell Painting acc / log-loss | **82.9** / 0.571 | 80.6 / 0.588 | 82.1 / 0.546 | 82.0 / **0.543** | 80.4 / 0.673 |

- V2b has the best accuracy on both LINCS sets, and the best log-loss and high-spread calibration on HAR.
- It is underconfident on LINCS (overconfidence −0.06 to −0.08), which costs it log-loss there.
- The final held-out evaluation is running ([final_shape_eval.sh](experiments/obs_noise/final_shape_eval.sh)).

## Shape learning: final evaluation of `tx_shV2b_A` (2026-10-21; SHAPE.md section 7, run once)

AUC, paired vs the mean + sd wrapper (8 est.), 8 passes:

| dataset | V2b | joint path off | Huber + sd | sketch | cluster shares |
|---|---|---|---|---|---|
| NK (20 donors) | +0.2 ± 2.3 | +1.0 | **+8.0 ± 2.5** | −1.0 | +2.5 |
| HVTN | +3.1 ± 0.9 (17/20) | +1.0 | **+8.4 ± 1.3** | +8.3 | −5.6 |
| Preeclampsia | +0.2 ± 1.5 | −0.9 | +1.0 | **+3.9** | +2.2 |
| Musk | **+1.2 ± 1.6** | +0.4 | −0.4 | −1.0 | −2.4 |
| PhysioNet | −0.5 ± 0.3 | −0.5 | 0.0 | **+0.4** | n/a |

- **The primary criterion is not met:** V2b beats mean + sd by more than 2 SE on 1 of 5 datasets (HVTN), and is not significantly worse on any.
- **Secondary bar (Huber + sd):** V2b is worse on NK −7.8 ± 2.7, HVTN −5.3 ± 1.2 and PhysioNet −0.5 ± 0.1; tied on PreE −0.8 ± 1.7 and Musk +1.6 ± 1.6.
- **vs cluster shares:** HVTN +8.7 ± 2.0, Musk +3.7 ± 1.0, NK −2.2 ± 2.7, PreE −1.9 ± 1.5.
- **Robust location is the missing signal on cytometry.** Mean + sd → Huber + sd is +8 AUC on NK and HVTN: markers are multimodal (cell types), so the mean is a poor summary. The backbone sees the mean, and the embedding does not carry robust location strongly enough. This is the explicit-column lesson again.
- **The joint path helps where expected:** vs off, HVTN +2.1, Musk +0.8, PreE +1.1; NK −0.8.
- **HVTN gains a lot from 8 passes** (87.1 vs 80.8 at 1 pass). That suggests a sensitivity to column order or a set-size mismatch: 64 instances in training vs 1024 in evaluation.
- **Calibration:**
  - NK: V2b is near-calibrated on high-spread rows (−0.006 vs mean + sd +0.044).
  - HVTN: underconfident (−0.09).
  - PreE: every arm is about +0.3 overconfident, at near-chance AUC (56–61).
  - PhysioNet: about calibrated.
- **These five datasets are now dev** (looked at). A later final test needs fresh held-out data.
- **Next** (user-approved): (1) the backbone sees Huber + sd columns (step 0 = the Huber + sd wrapper); (2) training on 256 instances per cell, with test-time subsample averaging.

## Shape learning, round 2 (dev now includes the five former final sets)

**V2b + test-time subsample averaging** (4 × 256 instances, no training). NK 67.8 → 66.0, HVTN 87.1 → 83.1, PreE 57.6 → 59.1; Musk and PhysioNet unchanged (sets ≤ 256). More instances help; averaging smaller subsamples loses the rare cells. **Dropped as a default.**

**V4a (backbone sees Huber + sd; joint path; 64 instances; 1000 steps).** Dev kill switch passed (72.5 → 72.9 → 72.4).

| dataset | V4a vs the Huber + sd wrapper (8 est.) | V2b → V4a |
|---|---|---|
| HAR acc | +0.1 ± 0.2 | 86.1 → 83.8 |
| L1000 acc | **+1.2 ± 0.6** | 52.5 → 53.0 |
| Cell Painting acc | +0.5 ± 0.4 | 82.9 → 82.5 |
| NK AUC | −3.0 ± 1.4 | 67.8 → 72.5 |
| HVTN AUC | 0.0 ± 0.3 | 87.1 → 92.5 |
| PreE AUC | −0.5 ± 1.0 | 57.6 → 57.9 |
| Musk AUC | +1.0 ± 1.0 | 85.2 → 84.6 |
| PhysioNet AUC | **+0.3 ± 0.1** | 84.2 → 85.0 |

- **V4a ≈ the Huber + sd wrapper.** It recovers the cytometry gap but loses V2b's learned HAR gain.
- **The branch added almost nothing in 1000 steps:** synthetic sh loss 0.742 → 0.743, sh_xorp 0.887 → 0.889, learning check 0.289 → 0.289.
  - The weights did train, with no skipped steps.
  - V2b's joint learning switched on between steps 500 and 750 (sh_xorp 0.888 → 0.852); V4a's has not by step 1000. This looks like a variable-onset learning phase.
- **Queued after V4b:** V4c, the V4 configuration warm-started from V2b's trained branch (lr 1e-4, 1000 steps), to keep the learned joint / HAR gains while the backbone sees Huber + sd. `--init` now loads checkpoints with `weights_only=False`.

## PLAN 2 — a replicate layer on frozen TabICLv2 (2026-10-21, in progress)

[PLAN_2.md](PLAN_2.md). Goal restated: beat the wrapper fed **mean + sd** *and* fed **Huber + sd**, and also the per-dataset-selected best hand summary (step 0c), on data never seen. Bitter-lesson line: no hand-picked estimators or columns in the shipped layer; the prior is where the investment goes.

### Protocol change: dev sets are split into VAL and TEST

[devsplit.py](experiments/obs_noise/devsplit.py) is now the single registry. Every dataset already looked at is DEV and is split; `real_eval.py` / `real_eval_reg.py` refuse TEST without `--allow_test`, and unknown names default to TEST.

- **VAL** (all decisions): HAR, L1000, CellPainting, HVTN (Beijing for regression).
- **TEST** (one final read, gated on `FULL_DONE` and `PROMOTE`): NK, PreE, Musk, PhysioNet, GasDrift/GasConc, Tox21/Tox21Pot, UPS1, CPTAC, PhysioNetLOS.
- **HELDOUT**: empty; fresh sets must be added here *before* any run.

**The bar is locked on disk before any candidate result** ([val_ref.jsonl](experiments/obs_noise/artifacts/val_ref.jsonl), wrapper, 8 estimators, VAL only):

| arm | HAR | L1000 | CellPainting | HVTN |
|---|---|---|---|---|
| mean | 0.731 | 0.507 | 0.806 | 0.686 |
| mean + sd | 0.826 | 0.516 | 0.821 | 0.756 |
| Huber + sd | 0.836 | 0.517 | 0.820 | **0.839** |
| sketch (7 cols) | **0.865** | 0.511 | 0.801 | **0.840** |

- **HVTN is the largest curated-imputation gain in the project:** +15.2 points over mean. Cytometry is where the hand summaries are strongest and where every learned branch has lost.
- **The sketch, not Huber + sd, is the best explicit block on the population-style sets** (HAR +13.4, HVTN +15.3), while it is clearly worse on CellPainting (0.801 vs 0.821: extra columns hurt with 3-5 wells).
- Confidence tracks error much better with an explicit block: confAUROC 0.628 → 0.802 on HVTN, 0.766 → 0.842 on HAR, but it *drops* to 0.874 for the sketch on CellPainting. Calibration claims must be per regime.

### Step 0a — the cytometry scale hypothesis is refuted

[cyto_scale.py](experiments/obs_noise/cyto_scale.py), raw `X` quantiles per marker (a data property: no labels, no metrics):

| set | share of negatives (median) | 1% quantile (median) | 50% | 99% |
|---|---|---|---|---|
| NK | 75.5% | −0.47 | −0.35 | 4.25 |
| PreE | 68.0% | −0.68 | −0.38 | 2.90 |
| HVTN | **0.01%** | +4.20 | +5.49 | 6.73 |

- NK and PreE are centred on zero with heavy negative mass; HVTN is positive-only and shifted up by 4. No single arcsinh(x/5) matches all three.
- **The planned `asinh_mean_sd` / `asinh_huber_sd` arms are dropped:** there is no transform that would make the cytometry gap a scale artifact. Huber + sd's cytometry gain is not a scale effect.

### v7 — a continuous observation prior, and the general head that failed to beat curation

**Prior** (`observe_v7` in [obs_train.py](experiments/obs_noise/obs_train.py), `--noise_set v7`). No named families: each column draws continuous latents (location slope, scale slope, tail index, mixture share, mode separation, rare mass), each a continuous function of `u = z(x)`, and instances come from one continuous location-scale-tailed mixture built from them. The continuous measurement process (`measure_layer`, shared with the shape prior) composes on top: plate / well offsets, failed instances, LOD / ULOQ censoring (flagged or absent), MAR dropout, all with continuous rates.

**Head** (`tx_headM_A`): `center='attn'`, median feature, robust standardization, `dist_emb`, `temp_head`, `col_ctx`, ISAB encoder, joint path, `flag_emb`, `miss_emb`. **No** `sd_cols`, no `backbone_loc`, no gates, no learned columns — the pure "no hand columns" arm of PLAN_2 section 8.

**Screen (1000 steps, shape_k 128, ~5 s/step).**

| step | dev_real (VAL, 1 pass) | val Δacc / Δloss | clean |
|---|---|---|---|
| 0 | 66.94 | — | 0.801 |
| 500 | **75.61 (+8.66)** | +1.11 / −0.027 | 0.801 |
| 1000 | 74.48 (+7.53) | **+1.64 / −0.040** | 0.801 |

- **It learns from a neutral start.** Starting 8 points below the hand arms' own start, it climbed 8 points and peaked mid-screen. Per-setting at step 250 the gains were broad, not tail-specific: v7_clean +1.48, v7 +1.15, v7_meas +0.92, v7_mnar +0.89, v7_well +0.83, v7_xorp +0.74, v7_cens +0.59.
- **`clean` stayed exactly 0.00** for 1000 steps: the no-clean-cost guarantee holds in a real run.

**8-pass VAL eval — not promoted.**

| arm | CellPainting | HAR | HVTN | L1000 |
|---|---|---|---|---|
| headM (v7) | **0.820** | 0.838 | 0.797 / 0.806 (sub) | 0.502 |
| Huber + sd | 0.820 | 0.836 | **0.839** | **0.517** |
| mean + sd | 0.821 | 0.826 | 0.756 | 0.516 |
| sketch | 0.801 | **0.865** | 0.840 | 0.511 |

- It **ties** Huber + sd on CellPainting and HAR and **loses 3.3 on HVTN and 1.5 on L1000.** It closes the gap to curated imputation from a neutral start but does not pass it.
- `dev_real` peaked at step 500 and sagged to step 1000, so the screen did not justify the 6000-step run. The checkpoint is kept as the v7 comparison arm.

### V4b / V4c killed: the Huber + sd start is a ceiling, not a platform

- **V4a** (Huber + sd input, 64 instances): val Δacc **+0.01**, dev_real 72.49 → 72.35. Flat.
- **V4b** (same, 256 instances): train loss falls (0.597 → 0.565) while val stays flat (Δacc −0.15 at step 250, −0.02 at 500) and dev_real is 72.45 at step 500. Killed at 625. **Instance count is not the lever.**
- **V4c** (warm start from V2b) was killed unlaunched: its ceiling is the same Huber + sd baseline.
- Reading: hand features saturate the task; with a frozen backbone there is nothing left for the branch to add, and the falling train loss is overfitting to the training draws.

### Step 1 — the learned column-aware estimator (`backbone_loc='learned_m'`)

**Implementation** ([obs_model.py](obs_model.py)). Weights `w_k = softplus(c0 + g(|r_k|/s_c, t_c, log n))`, `c0 = log(e-1)` so `w ≡ 1` at init and `g`'s output layer is zero-init; 3 differentiable IRLS iterations from the cell median. `s_c` is the pooled robust column scale ([`pooled_scale`](obs_model.py), the `huber_location` scale in raw units, a CONTEXT statistic and therefore not differentiated, as the wrapper's preprocessing is not); `t_c` is `col_ctx`'s `phi_col` pooled over **all** replicated cells (context and query, no labels). The backbone then reads **explicit columns**: the learned location plus the weighted scale, one pair per replicated feature. `rep_emb=False` drops the additive embedding entirely, so the learned information can only reach TabICL as columns.

**Guarantees** (`python obs_model.py --learned_m`). Step 0 **equals the mean + sd wrapper input exactly (0.0)**, `w ≡ 1` at init, all-`n=1` tables equal stock at random weights (0.0), and randomizing `g` changes `n>1` outputs.

**Two numerical bugs found and fixed before the screen could train** (both would have produced silently dead runs):

1. **Zero variance under censoring.** Censoring piles replicates on a limit, so a cell's variance is exactly 0 and `d√x` is infinite at 0: `SqrtBackward0` returned NaN, poisoning `loc_c0` / `loc_w` / `phi_col` on every `sh_cens` batch. Fixed with `var.clamp_min(1e-8)` **in both the learned and the plain sd path** (the plain path was a latent bug all along). All shape settings (`sh`, `sh_cens`, `sh_xorp`, `sh_rare`, `sh_tail`, `sh_pop`, `sh_rep`) now give finite forward and gradients.
2. **The branch-owned-parameter predicate.** `--freeze_backbone` and the optimizer groups keyed on the `rep_` prefix, so the new `loc_*` parameters of the learned estimator were frozen. Now `is_layer = rep_* or loc_*`, used for the init assert, the freeze, and the optimizer groups.

Also fixed in the lanes: `--ps` JSON passed through an unquoted-expanded variable is word-split by bash (the escaped quotes become literal characters), which crashes at `json.loads`. Pass it as its own quoted argument.

**M1 (learned columns only, no embedding) — the positive result.**

| step | dev_real (VAL, 1 pass) | val Δacc / Δloss |
|---|---|---|
| 0 | 72.77 | — |
| 500 | 74.70 (+1.93) | +0.19 / −0.005 |
| 1000 | **78.07 (+5.31)** | **+0.48 / −0.011** |

- **Its step 0 is the mean + sd wrapper by construction (72.77 ≈ 73.0), and it climbs 5.3 points above it**, monotonically, still rising at the screen end.
- 1.9 s/step: with `rep_emb=False` the replicate encoder runs forward-only (no gradient reaches it), so M1 learns the weight curve and nothing else.

**M1, 8-pass VAL eval (1000 steps):**

| arm | CellPainting | HAR | HVTN | L1000 | mean |
|---|---|---|---|---|---|
| **M1** | **0.829** | 0.820 | **0.842** | **0.522** | **0.753** |
| Huber + sd | 0.820 | 0.836 | 0.839 | 0.517 | 0.753 |
| mean + sd | 0.821 | 0.826 | 0.756 | 0.516 | 0.730 |
| per-set best hand (the 0c bar) | 0.821 | **0.865** | 0.840 | 0.517 | 0.761 |

- **The first arm in this project to beat the hand summaries on HVTN** (+0.3 over the best hand arm on the set where every previous branch lost), and it also wins L1000 (+0.5) and CellPainting (+0.8).
- **HAR is the one loss** (−1.6 vs Huber + sd, −4.5 vs sketch). HAR is a population set where the 7-column sketch dominates, and M1 exposes only 2 columns per feature.
- Log-loss on VAL: HAR 0.519, L1000 1.387, CellPainting 0.550, HVTN 0.366.

**M2 (M1 + the V2b branch, warm-started) — the embedding adds nothing, again.**

| step | dev_real |
|---|---|
| 0 | 72.36 |
| 500 | **72.08 (−0.28)** |

- 2.6 points **below** M1 at the same step, val negative from step 250 (−0.13), and 4.7× slower (9.0 s/step) because the embedding keeps the encoder in the backward.
- Killed at step 500 by a rule stated before the read. This is the **fourth** independent negative for the additive embedding channel (V3 `sd_learn`, V4a/V4b, the v7 head, M2), against two positives for explicit columns (the hand sd / sketch, M1).

### M3 (in progress) — does the slot count bind?

`tx_m3_A` = M1 + `sd_learn=2`: each replicated feature feeds TabICL 4 columns (`learned location`, `weighted scale`, `L1`, `L2`) instead of 2. It targets M1's single deficit: the sketch's only advantage over `mean + sd` is *more slots*.

- Step 0 dev_real 72.75 = M1's 72.77, so the guarantee holds and M3 starts exactly at M1's input.
- Screening at 1000 steps; M1's paused checkpoint (step 1000, verified) is the resume point if M3 loses.
- **Decision rule, stated before the result:** M3 wins if it is ≥0.5 better on the VAL mean *and* recovers the HAR deficit without giving back HVTN or L1000. A tie means slot count is not the constraint and HAR instead needs an explicitly distributional column. M4 (M3 + embedding) stays skipped unless M3 wins, since M2 already answered the embedding question.

### Method notes

- **Run hygiene.** Queued lanes gate on marker lines in `artifacts/logs/phase3.log`; a stale marker from a crashed launch let a regression arm start beside a classification run (two GPU jobs, OOM risk). Markers are now removed when a run is killed, and the final test read additionally requires a `FULL_DONE` file so it cannot fire while a screen is still being read.
- **Monitoring.** [monitor_shape.py](experiments/obs_noise/monitor_shape.py) (log-only: step, s/step, ETA to the next milestone, train-loss trend, dev kill readout, val deltas) and [track_shape.sh](experiments/obs_noise/track_shape.sh) (15-minute snapshots). `nvidia-smi` is unusable on this machine (NVML driver/library mismatch), so GPU state is read through torch.
