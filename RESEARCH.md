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
