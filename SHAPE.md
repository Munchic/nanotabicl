# PLAN: shape learning — cells as distributions (2026-10-20)

**Goal.** A sample's measurements are a set, and the label can depend on the set's shape: spread, a subpopulation share, bimodality, a tail, co-expression across features. The aim is to beat the off-the-shelf TabICLv2 wrapper, and the **mean + sd** wrapper baseline, on real data where shape carries signal. The backbone stays frozen. A table without sets equals TabICLv2 exactly.

**Why now (evidence).**
- The learned centre does not transfer (Exp 9/10 decomposition).
- Replicate denoising has little real headroom beyond the mean: LINCS gains +9 / +12 from averaging, and nothing reliable beyond it.
- A plain per-cell **sd column** fed to the wrapper gains +4.0 AUC on CPTAC (10/10), +1.0 / +1.5 on LINCS and +0.8 on UPS1_low (sketch). That beats every trained branch (RESEARCH.md, "Shape features").
- So shape is signal even with 2–6 replicates, and an explicit column reaches the frozen backbone better than our additive embedding does.

---

## 1. Input format

**One format for every dataset.** Replicates `(B, R, C, K)` + `mask`, as now, plus a per-table flag `paired`.

| set type | instance k is | examples | paired |
|---|---|---|---|
| marginal | one measurement of that cell only | Tox21 runs, UPS1 runs | no |
| joint | one physical instance measured on all features | cytometry cells, LINCS wells, Musk conformations, HAR time steps | yes |

- K runs from 1 to thousands. Training subsamples to `K ≤ 64` per row. Evaluation uses up to 512, averaged over subsample draws (which doubles as an ensemble).
- The true `n` is passed as a feature, so subsampling does not hide set size.
- n = 0 and n = 1 cells keep the existing handling (`fill_missing`, the n = 1 guarantee).

## 2. Model (frozen TabICLv2 + shape branch)

**Backbone input.**
- The per-cell mean (or the variant's step-0 columns, see V1).
- No learned centre; the decomposition showed it is what fails to transfer.

**Marginal path.** The existing per-cell set encoder, with two changes:
- Full self-attention over replicates is replaced by **induced set attention**: 16 inducing points, linear in K.
- Pooled output is added through the zero-init embedding, as now.

**Joint path (new, `paired` tables only).**
- Each instance is a vector over the C (anonymous) columns. A token per (instance, column) is φ(z) of the column-standardized value.
- One small transformer layer runs across columns within each instance; d = 16, the same structure as TabICL's row stage, so it handles any C.
- Tokens are then pooled over instances per column with induced attention, giving per-(row, column) vectors.
- These vectors are added to the marginal path's cell vector, so co-expression ("high in A and low in B") reaches every involved column's token.
- A row-level pooled vector also goes into the row token via the existing `extra_row_emb` hook.

**Guarantees (self-tests, as for `--side` / `--row_attn`).**
- Step 0 = the variant's baseline exactly.
- An all-n = 1 table equals stock at any weights.
- A table with `paired` = 0 never touches the joint path.

**Cost.** The joint path is R · K · C tokens. With R ≤ 1024, K ≤ 64 and C ≤ 32 in training, that is about 2M tokens at d = 16, with activation checkpointing (as the replicate encoder). If it does not fit beside a second run, runs are serialized.

## 3. Prior: "parameters → populations" (the core change)

The bank's SCM tables are reused: clean `X (R, C)` and labels `y`, classification and regression. **Each clean value x_rc becomes a parameter of a population, not its centre.** The labels depend on x, so they depend on the shape by construction.

**Marginal families.** Per column, one family f_c with random constants is drawn; x enters through a standardized u = z(x).

| family | instance distribution given u | mean informative? | share |
|---|---|---|---|
| location (today's prior) | N(u, σ²) + v5 noise modes | yes | 30% |
| scale | N(μ₀, (σ₀ e^{a·u})²) | no | 10% |
| mixture share | π N(μ₁, s²) + (1 − π) N(μ₂, s²), π = σ(a·u + b) | partly | 10% |
| rare subpopulation | share π = σ(a·u + b) ∈ [0.1%, 10%] at a distant location | barely | 10% |
| mode separation | ½ N(μ₀ − d, s²) + ½ N(μ₀ + d, s²), d = d₀ e^{a·u} | no (sd partly) | 5% |
| tail / skew | skew-t with skewness or tail index driven by u | no | 5% |

**Joint families (paired tables, 30% of tables).**
- Per table, T ∈ {3..8} instance *types* (cell types) are drawn, each a mean vector over the columns, shared by all rows. Per-row type proportions are softmax(W x_r + b). Instances are drawn from the row's type mixture.
- In **XOR designs** (a third of joint tables), the types are paired so that every column's marginal is identical whatever the proportions (types A = (+, −) and B = (−, +) trade against C = (+, +) and D = (−, −)). Only the joint distribution carries the label, so per-cell marginals and sketches cannot see it.
- In the other joint tables, per-type shifts also depend on x_r ("state"), and marginals and joints both carry signal.

**Measurement layer (on top of every family).**
- The v5 replicate noise mix: Gaussian-dominated, contamination, row heteroscedasticity, plate effects.
- v6's missingness / censoring at low weight (5%).
- The v5 mass is kept, because v6 showed that too much heavy-tail mass costs LINCS.

**Set-size design.**
- Replicate regime (40%): n ∈ {1..10}, mostly 2–5.
- Population regime (60%): n ~ logU(32, 1024), subsampled to 64.
- Exact columns (covariates, n = 1): 20% of columns in 30% of tables.

**Built-in sanity check.** For every family the training-free headroom is computed once:
- wrapper on the mean;
- wrapper on mean + sd;
- wrapper on the sketch;
- oracle (the clean x fed directly).

This is done before any training.
- A family where the sketch already reaches the oracle tests nothing learned.
- A family where nothing beats the mean is noise.
- The share weights above are adjusted from this table, and only from it (no real-data tuning).

## 4. Training

- **Scope.** Stage A only: frozen TabICLv2 classification backbone first, regression second (after classification passes).
- **Optimisation.**
  - AdamW 3e-4, warmup 200, cosine to 10%.
  - 6000 steps (screens: 1000), batch 32, micro-batch 2, fp32 + TF32, activation checkpointing.
  - Bank blocks 0..15 are held out for validation, as in all runs.
- **Synthetic validation.** Fixed seed, every 250 steps. Settings: one per marginal family, joint-XOR, joint-state, replicate-only (v5 gauss / contam) and clean.
  - Logged next to the precomputed reference arms (mean / mean + sd / sketch / oracle), as % of the oracle gap closed.

## 5. Kill switches (never used to pick a checkpoint)

1. **Real dev kill.** At steps 500 and 1000, stop if the dev score is more than the margin below step 0.
   - Dev set: LINCS dose (L1000 + Cell Painting, accuracy, margin 0.5 points) plus **UCI HAR** (accuracy, margin 0.5).
   - HAR is used as dev because its spread signal is large and well known, so a model that cannot exploit shape fails loudly there.
   - The rule is the same as Exp 9–11; the last checkpoint is reported.
2. **Learning kill.** At step 1000, stop if the shape-only synthetic settings (scale, rare, mode separation, joint-XOR) have closed less than 10% of their mean → oracle gap.
   - This means the architecture is not learning shape at all: stop and fix the design, rather than spend 6000 steps.
3. **Guarantee check at launch.** The self-tests must pass (step 0 = baseline, n = 1 = stock, paired = 0 never touches the joint path), or the run does not start.

## 6. Variants

**Screen**, 1000 steps each, two at a time.

| id | backbone input at step 0 | encoder | question |
|---|---|---|---|
| V0 | mean (step 0 = wrapper on the mean) | marginal path only (today's branch + induced attention) | does the new prior alone teach shape through the embedding? |
| **V1** | **mean + sd columns** (step 0 = the mean + sd wrapper, the strongest free baseline) | marginal path, embedding on top | does an explicit column + learned refinement beat the column alone? |
| V2 | as V1 | + joint path | does joint structure add over marginals (XOR settings, cytometry dev)? |
| V3 | as V1, but the sd column is replaced by S = 2 *learned* summary columns per feature (initialised to sd) | marginal path | do learned columns beat the hand sd? |

**Prior ablations** (on the screen winner):
- without joint tables;
- location share 30% vs 50%;
- population regime off (replicate sizes only).

Notes:
- V1 / V3 sd columns are emitted only for columns with replicated context cells, so all-n = 1 tables remain stock.
- **Selection.** Synthetic validation (oracle gap closed) among the variants that pass both kill switches. The winner gets the full 6000 steps.
- **Capacity.** d_s 64 + 2 blocks by default. The base size is reported as a control only for the final model.

## 7. Final evaluation

**Datasets, fixed now, before any model sees them.** Seeded splits; each is evaluated once, after the full run.

| dataset | domain | task | set type | split |
|---|---|---|---|---|
| NK-cell CyTOF (CMV status) | immunology | classification | joint (cells) | 20× repeated stratified halves (about 20 donors) |
| Preeclampsia CyTOF | immunology | classification (label per sample, to be confirmed on load) | joint | 10 seeded stratified splits |
| HVTN flow cytometry | vaccines | classification (stimulation / response, to be confirmed) | joint | 10 splits |
| MS scRNA-seq (PCA 16–32) | neurology | MS vs control per donor | joint | 20× repeated halves |
| Musk v2 | chemistry (multiple-instance learning) | classification | joint (conformations) | 10 splits |
| PhysioNet 2012 set-a | ICU | in-hospital mortality, AUC | marginal per variable (irregular times) | 10 splits |
| MISR aerosol (if a verified copy is found) | remote sensing | regression, CRPS | joint (pixels) | 10 splits |
| CPTAC, UPS1_low, Tox21, GasDrift | existing | as before | marginal / joint | as before; **flagged**: used in the shape-feature check |

**Arms** (all at 8 estimators / passes; 1 also reported).
- Wrapper on the mean; wrapper on mean + sd (**the reference**); wrapper on the sketch.
- **Cluster-frequency wrapper** (joint sets): k-means on pooled context instances → per-row cluster shares as features, the standard cytometry featurization.
- Our variants: final model; final with the joint path disabled; base size.

**Metrics.**
- Accuracy / AUC / log-loss, or CRPS.
- Paired per split vs the mean + sd wrapper: mean ± SE, n better / n.
- Per dataset first, then the number of datasets won by more than 2 SE.

**Success criterion, stated in advance.**
- Better than the mean + sd wrapper by more than 2 SE on at least half of the new held-out datasets, and never significantly worse on any.
- On the joint datasets, better than the cluster-frequency wrapper.
- Anything less is reported as is. The flagged existing datasets are reported but do not count toward the criterion.

## 8. Order of work

1. **Data and training-free baselines** (this is step 1 of the earlier plan; downloads running).
   - Loaders for the new sets into the replicate format.
   - Mean / mean + sd / sketch / cluster-frequency wrapper arms on every new set **except** the final-eval rows: for those, only checks that labels and sizes load correctly, no metrics.
   - HAR gets the full baseline table, since it is dev.
2. **Prior.** `observe_shape` in obs_train.py (families, joint types, set sizes) with the per-family headroom table. Adjust the shares from the headroom only.
3. **Model.** Induced attention, the joint path, sd / learned summary columns, self-tests.
4. **Screen** V0–V3 (1000 steps, kill switches).
5. **Full run** of the winner (6000 steps).
6. **Final evaluation** (once), write-up in RESEARCH.md.

## 9. Risks

- **TabICL's 10-class limit and small cohorts.** The NK set has about 20 donors, so repeated splits give wide intervals. Its results are read as direction, not significance.
- **Feature-count limits** for scRNA-seq: PCA to 16–32 components, fitted on context donors' cells only.
- **PhysioNet's instances are irregular in time.** It is treated as marginal sets per variable, with time-binned pairing only as a later variant.
- **The joint path's memory.** If it does not fit beside a second run, runs are serialized (about 2 h per 1000 screen steps).
- **The prior's family shares are a choice.** They are set from synthetic headroom only, never from dev or held-out data, to avoid repeating v6's tuning loop.

**Amendment (2026-10-21, before any final-evaluation result exists).** The wrapper on **Huber + sd** is added as a secondary bar: the best hand-built pair on dev, with the best log-loss on 4 of 7 sets (RESEARCH.md, "Robust location / scale pairs"). The primary criterion (vs mean + sd) is unchanged.
