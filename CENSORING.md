# PLAN: censoring spec channels on TabICLv2, a 4-day validation (2026-10-20 → 10-24)

Evaluates a proposal to pass the observation model to the network as extra input channels (the "spec"), and turns it
into a 4-day plan. Background: [RESEARCH.md](RESEARCH.md) (Exps 1–9), [NOTES.md](NOTES.md).

**Proposal, in one line.** Give the model a per-table spec (feature censoring type and bound; y censoring; row group;
replicate group), encode each structure with a zero-initialized mechanism, and fine-tune TabICLv2 on a prior where that
structure varies and is sometimes hidden ("spec dropout").

**Decision.** Build only the spec channels that real assay data actually records, and only where the spec adds
information the model cannot infer from the data:
1. **y censoring** (`<` / `>` qualifiers on the target, e.g. pIC50 "> 10 µM"): priority 1.
2. **Feature censoring flags**: priority 2, trained in the same run.
3. **Row groups** (random effects + a `same_group` attention bias): only if an inference-only check shows headroom.
4. **Replicates**: already built (Exps 2–8). Reuse, don't rebuild.

Compute is not the binding constraint: a 27M stage-B run is about 1.5–2 h (≈ 1.2–1.3 s/step), so 4 days is about 40
run slots on two lanes. Implementation and evaluation are. Spend the effort on baselines and real-data tests.

---

## Why this order (evidence from RESEARCH.md)

- **The model can't infer what replicates don't reveal.** On LINCS every replicate branch trails mean imputation
  (`tx_obs_B` −1.6, v4 screens −1.5/−1.6 vs wrapper + mean, 8 est.). With 2–5 replicates a cell cannot tell Gaussian
  from heavy-tailed noise, so the gate robustifies everywhere (Exp 9). The proposal's diagnosis ("the model averages over
  observation models because nothing tells it which applies") is right. But the fix only works for metadata that real
  data records. Censor qualifiers and batch IDs are recorded; noise family never is.
- **y censoring has never been modelled.** y is always clean in the prior. The usual fixes (set to bound, drop the
  row) are biased, and there is no cheap preprocessing equivalent of "loss on the latent y".
- **Feature censoring is mostly solved when bounds are column-constant.** The branch reads pile-ups at the column
  limit (limit-tie feature) and is near the oracle at n = 50 (Exp 3). `tx_v4huber_A` recovers heavy censoring from
  0.589 to 0.651 within 250 steps. A flag adds information only for **per-sample bounds** (Tox21: top tested
  concentration ≈ 90–120 µM, varying by sample) and for **censored n = 1 cells**, which the branch gates to zero by
  design. The evaluation must include both, or the flag shows no gain by construction.
- **Groups are untested and the baseline is strong.** No experiment has checked whether the model uses a visible group
  column (Exp 3, open). A one-hot group column or group aggregates may already capture random effects. When the query's
  group is absent from the context (gas drift: batch b → b + 1), a `same_group` bias cannot help.

## Proposal → status

| proposal item | status in the repo | action |
|---|---|---|
| replicate set encoder alongside the median | done, and stronger: SAB + per-cell gate + column context; ≈ 90% of the oracle headroom on synthetic data (`tx_obs_B`) | reuse `tx_obs_B[_reg]`, or the Huber start if it wins (below) |
| fine-tune from TabICLv2, zero-init | done (Exp 8); Muon stage B showed no forgetting | reuse the recipe |
| mix in the original prior | 25% clean tables; TabICL-prior mixing (P5) not needed so far | skip |
| heteroscedastic replicate noise | done (v2–v4 modes, layouts) | skip |
| **y censoring, loss on the latent y** | new | **build (P1)** |
| **feature censoring flag, per cell** | partial (value at the limit, no flag; limit-tie inference only) | **build (P2), same run** |
| spec dropout | new | build, per channel |
| group random effects + `same_group` bias | group layouts and a visible group column exist; no random effects; attention bias (backlog A2) never built | **gated on check G** |

---

## P1. y censoring (regression)

**Prior** ([obs_train.py](experiments/obs_noise/obs_train.py)). After the clean table is drawn, per table:
- with probability 0.6, censor y. Direction: left, right or both (1 : 1 : 0.5). Censored fraction U(0.05, 0.5) per side.
- bound: one constant per table (w.p. 0.5), otherwise per row (base quantile ± jitter, like a varying top concentration).
- the context shows the censored value (the bound) plus a flag in {observed, <, >}. **Query loss is the pinball loss on
  the latent y**, for every query row, censored or not.
- spec dropout: w.p. 0.3 the y flag is hidden (all "observed"); the context still shows the bound and the loss stays on
  the latent y. With no censoring drawn, latent = observed, so the old stream's behaviour is a special case.
- target standardization: fit the context mean / sd on **uncensored** context rows (fall back to all rows if fewer than 8).
  Same rule in training and evaluation.

**Model.** A flag embedding, zero-init and bias-free over a one-hot whose "observed" entry maps to the zero vector. It is
added wherever y enters:
- `y_embed_in` in the column embedder ([model.py:51](model.py#L51));
- `y_embed_icl` in the ICL stage ([model.py:68](model.py#L68)).

**Guarantees, to verify as for the replicate branch:** step 0 equals stock, and a table with no flags equals stock at
*random* weights (max |Δ| = 0.0).

**Output.** The quantile head predicts the latent y, so P(y beyond bound) for a censored query comes free.

## P2. Feature censoring flags (same run as P1)

- **Prior:** w.p. 0.5 per table, censor 20–60% of the columns. Each censored column gets a left, right or two-sided bound,
  constant or per row, as for y. Spec dropout 0.3, independent of y's.
- **Model:** a per-cell flag embedding {observed, <, >} through the existing `extra_emb` hook ([model.py:49](model.py#L49)),
  zero-init, observed → 0. It applies to **every cell, n = 1 included**. It lives outside the replicate branch, so the
  "n = 1 → stock" gate is unaffected and "no flags → stock" holds exactly.
- **With replicates (day 3 only):** also pass a per-replicate flag into φ (backlog P3). The cell flag is "any replicate
  censored".

## G. Group check (inference only; it decides whether to build the `same_group` bias)

Synthetic regression tables with group random effects (2–8 groups; offset ~ N(0, τ²) on y, optionally on features too,
τ ~ logU(0.1, 1) × sd). Queries come from groups seen in the context. TabICLv2 (fp32 + preprocessing), four arms:
- G0: no group information;
- G1: one-hot group columns;
- G2: leak-free group aggregates (context-row group means of the features, plus a group-mean target residual);
- G3: oracle, with the true offset subtracted.

**Gate:** build the `same_group` attention bias only if the best of G1 / G2 closes **< 50%** of the G0 → G3 gap.
Otherwise groups are a preprocessing problem; document it and stop.

---

## Baselines (all inference only, run on day 0)

TabICLv2, both through the wrapper and through our pipeline (fp32 + preprocessing, 1 pass).

| side | arm | what it does |
|---|---|---|
| y | B1 set-to-bound | censored context y = bound (what you get with no special handling) |
| y | B2 drop | remove censored context rows |
| y | B3 EM | 2 rounds: from TabICLv2's predictive quantiles for each censored context row, replace y with E[y \| y beyond bound]; refit |
| y | B4 oracle | the latent y shown in the context (ceiling) |
| X | F1 at-bound | value at the bound, no flag |
| X | F2 indicator | F1 + an indicator column per column with ≥ 1 censored **context** cell (avoids the constant-column blow-up of Exp 5h) |
| X | F3 oracle | latent feature values |
| replicates | median, mean, Huber | as in Exp 9 ([huber.py](experiments/obs_noise/huber.py)) |

## Training runs

Recipe from Exp 8 stage B: backbone unfrozen, Muon lr 1.5e-5, warmup 200, cosine to 10%. New maps on AdamW at 3e-4.
fp32 + TF32, micro-batch 2, batch 32, 25% clean tables. Validation every 250 steps on new y-censoring settings
(`ycens_r30`, `ycens_l30`, `ycens_rowbound`, `fcens_rowbound`, plus clean).

| run | init | prior | spec | steps | purpose |
|---|---|---|---|---|---|
| `tx_cens_B_reg` (R1) | TabICLv2 reg | clean features + P1/P2 censoring | dropout 0.3 | 4000 | the method; spec on vs off evaluated on the same weights |
| `tx_cens0_B_reg` (R0) | TabICLv2 reg | identical stream | always hidden | 4000 | matched control: training on censored data with the latent loss, minus the flags |
| `tx_obs_cens_B_reg` (R2, day 3) | best replicate base (below) | R1 prior + replicate noise | dropout 0.3 | 3000 | combined model for real assay data |
| `tx_obs_cens0_B_reg` (R2 control) | same | same | always hidden | 3000 | matched control for R2 |

R1 and R0 have no replicate branch, so both fit on the GPU together.

## Evaluation

**Synthetic** (27 regression datasets; new `gen_ext4.py` settings): censored fraction {0, 15, 30, 50%} × {left, right}
× {constant, per-row bound}, for y and for features. Scored per model, spec on and spec off:
- CRPS on the latent y, split by whether the query is censored;
- PIT calibration error and 90% coverage;
- **share of the B1 → B4 gap closed.**

**Real (held out; looked at only after the prior is frozen at R1's launch):**
- **Tox21 regression.** Target: one assay's log AC50, with inactive runs right-censored at that sample's top tested
  concentration. Features: the other 11 assays (log AC50 + efficacy). 12 tasks.
  - CRPS on uncensored queries; −log P(y beyond bound) on censored queries.
  - **Re-censoring test (ground truth on real data):** among actives, take a threshold t at the 70th / 50th percentile of
    potency. Show context actives beyond t as censored at t; score CRPS on query actives at their known values. This
    gives ground truth on real data.
- **Your assay data** (day 3–4), with the same two metrics and the same re-censoring test.

## Decision rules

| when | check | if it fails |
|---|---|---|
| R1/R0 step 0 | step 0 = TabICLv2 exactly; no flags = stock at random weights | fix before training |
| R1 step 1000 (≈ 30 min) | spec-on R1 beats R0 by ≥ 0.005 CRPS on validation at ≥ 30% censoring; clean within ±0.002 of stock | stop R1. The finding is that the latent loss alone is what matters; carry R0 forward |
| end of day 2 | on Tox21 re-censoring, the best of R1 / R0 beats the best of B1–B3 by > 2 SE (paired over tasks) | stop the censoring track; write up the negative result |
| end of day 2 | per-row-bound feature settings: spec on > spec off by > 2 SE | drop P2 flags from R2 |
| `tx_v4huber_A` done (≈ 14:00 today) | LINCS dose ≥ `tx_huber0p` (0.664) and synthetic ≥ `tx_v4med_A` | keep `tx_obs_B_reg` as the replicate base for R2 |

**Replicate track.** `tx_v4huber_A` is the last centering screen. If it wins, run a regression Huber stage A
(`tx_v4huber_A_reg`, about 1.5 h) on day 2 as R2's base. Either way, stop screening replicate centers after it.

---

## Schedule

| day | lane 1 | lane 2 | off-GPU work |
|---|---|---|---|
| **0** (10-20, rest of day) | `tx_v4huber_A` (running) → its eval | inference only: baselines B1–B4 and F1–F3 on synthetic data; group check G | implement P1/P2 prior, flag embeddings and guarantee tests; `gen_ext4.py` settings; Tox21 regression loader |
| **1** (10-21) | R1 `tx_cens_B_reg` | R0 `tx_cens0_B_reg` | evaluation scripts: spec on/off, censored metrics, re-censoring |
| **2** (10-22) | synthetic eval of R1 / R0 → Tox21 eval | `tx_v4huber_A_reg` if Huber won | write up; go / no-go |
| **3** (10-23) | R2 `tx_obs_cens_B_reg` | R2 control | load your assay data and baselines |
| **4** (10-24) | evaluate on your assay data | buffer, or the `same_group` bias if G passed its gate | RESEARCH.md write-up |

## Out of scope this round

- A noise-family channel in the spec: real data never records one.
- Classification with censored y: pIC50 is regression. Add it later if R1 works.
- Rebuilding the replicate encoder; TabICL-prior mixing (P5); paired replicates (P2 backlog).
- The `same_group` bias, unless check G passes its gate.

## Pitfalls

- **Per-row bounds and n = 1 censored cells must appear in training and in evaluation.** Otherwise the flags have
  nothing to add over limit-tie inference.
- **Leakage.** Flags, group aggregates and the EM baseline may use context rows only.
- **Censored y in the column embedder.** y enters the column-embedding stage too. Without the flag at `y_embed_in`,
  that stage keeps the bound-valued y.
- **Freeze.** The prior is frozen at R1's launch: no tuning on Tox21 or your assay data after that. LINCS stays dev-only.
- **Memory.** Two 27M stage-B jobs fit at micro-batch 2; a third has OOM'd before. Capacity-size replicate jobs run
  alone.

---

## Progress log

### Day 0 (2026-10-20): implementation, baselines, first runs

**Code (new files only; the shared `model.py` / `obs_model.py` / `obs_train.py` are untouched):**
- [cens_model.py](cens_model.py): `CensNanoTabICL`. Flags {0 no spec / observed, 1 `<`, 2 `>`, 3 known observed (`known=True`)}.
  - Feature flags enter through the existing `extra_emb` hook; target flags through forward hooks on `y_embed_in` and
    `y_embed_icl`.
  - Optional gated row adapter (`row_adapter=h`): a zero-init MLP on [row token after `row_ln`, flag, observed y],
    applied to censored context rows only.
  - `python cens_model.py` verifies: step 0 = backbone, and no spec / unflagged = backbone at random weights
    (max |Δ| = 0.0), for all variants.
- [cens_train.py](experiments/obs_noise/cens_train.py): the P1/P2 prior (censoring on y and X, constant or per-row
  bounds), latent-y pinball loss, Muon / frozen training, fixed validation settings with the spec on and off.
- [cens_baselines.py](experiments/obs_noise/cens_baselines.py): B1 bound, B2 drop, B3 impute, F2 indicator columns, on
  the identical validation draws.
- [cens_real.py](experiments/obs_noise/cens_real.py): Tox21 as censored regression (natural 30/50% and re-censoring
  designs), not run yet.
- [tabarena_iid.py](experiments/obs_noise/tabarena_iid.py) + [tabarena_wrapper.py](experiments/obs_noise/tabarena_wrapper.py):
  the TabArena IID guard (below).
- Lanes: `lane_cens.sh` (R1 → R1b → R1c), `lane_censF.sh` (F2, F1). Decisions are logged in `artifacts/logs/cens.log`.

**Validation, CRPS in latent-y sd units (4 held-out bank blocks; latent-context oracle B4 = 0.141):**

| setting | B1 bound | B2 drop | B3 impute | R1 unfrozen, step 1000, spec on | F2 frozen + adapter, step 750, spec on |
|---|---|---|---|---|---|
| y right 30% | 0.278 | 0.293 | 0.262 | 0.221 | 0.223 |
| y left 30% | 0.251 | 0.265 | 0.229 | 0.207 | 0.209 |
| y right 30%, per-row bounds | 0.267 | 0.248 | 0.240 | 0.191 | 0.197 |
| y right 50% | 0.430 | 0.435 | 0.389 | 0.305 | 0.318 |
| X right 30%, per-row bounds | 0.162 | | 0.158 (indicators) | 0.169 | 0.161 |
| **clean** | 0.141 | | | **0.148 (+5%)** | **0.141 (= stock, exact)** |

**What happened to the plan's runs:**
1. **R1 (unfrozen, spec dropout 0.3).** The spec works: spec on beats spec off by about 0.04 at step 1000, far past the
   0.005 kill bar. But clean CRPS rose +5% (0.141 → 0.148), breaking the ±0.002 rule. Stopped at step 1000.
2. **R1b (`--known`).** Tested the hypothesis that "spec hidden" and "nothing censored" being the same input makes the
   model hedge. Clean drifted the same way (0.1471 at step 500 with a known spec). Hypothesis rejected; stopped.
3. **R1c (`--max_tail 3 --lat_clip 10`).** Tested target-scale inflation: censoring a heavy prior tail leaves latent
   targets at up to 1e7 observed sd. The drift was unchanged at step 250 (0.1446 = R1). Stopped.
4. **The cause is the censoring prior on an unfrozen backbone, not continued training.** Under the same metric, the
   other lane's `tx_median_B_reg` (same Muon recipe, no censoring) moves clean only from 0.1412 to 0.1417. `obs_train`'s
   validation metric (standardized units) is dominated by a few heavy-tailed tables and hid this; use latent-sd units.
5. **The TabArena IID guard confirms the damage on real data.** Unfrozen censoring runs lose 4–11% RMSE vs stock on
   wine_quality and airfoil_self_noise; `tx_median_B_reg` does not.
6. **Change of design: freeze the backbone.** No spec then equals TabICLv2 bit for bit, at any weights, so IID
   performance is stock by construction.
   - With the gated row adapter, the frozen model (F2) matches the unfrozen R1 on every censoring setting at equal steps.
   - Spec dropout and the "known" state are moot when frozen (no spec = stock), so F runs show flags on every censored
     table.
   - The R0 control (same prior, flags hidden) is meaningless for a frozen backbone. The comparison is stock plus the
     B1–B3 inference baselines.

### TabArena IID guard

**Selection.** TabArena leaderboard per-dataset results (HF space `TabArena/leaderboard`, `per_dataset_explorer.html`;
51 datasets × 88 methods, all random IID splits), cached in `artifacts/tabarena/frontier.csv`. Datasets were ranked by
the spread between frontier models (TabICLv2, TabFM, TabPFN-2.6, TabPFN-3, RealTabPFN-2.5, LimiX-2, Mitra-v2) divided by
the per-split SE, i.e. the ability to detect a regression. Filters: |target skew| < 1, mostly numeric features.

| dataset | rows × feats | skew | spread / SE | TabArena RMSE: TabICLv2 / TabFM / TabPFN-2.6 / TabPFN-3 |
|---|---|---|---|---|
| houses | 20,640 × 8 | −0.17 | 9.1 | 0.195 / 0.182 / 0.196 / 0.189 |
| physiochemical_protein | 45,730 × 9 | 0.57 | 7.3 | 2.94 / 2.89 / 3.01 / 2.89 |
| superconductivity | 21,263 × 81 | 0.86 | 2.8 | 8.94 / 8.94 / 9.17 / 9.01 |
| wine_quality | 6,497 × 12 | 0.19 | 2.5 | 0.594 / 0.586 / 0.593 / 0.590 |
| airfoil_self_noise | 1,503 × 5 | −0.42 | 2.8 | 1.078 / 1.072 / 1.097 / 0.990 |
| QSAR-TID-11 (pIC50, optional) | 5,742 × 1024 | 0.59 | 2.1 | 0.808 / 0.736 / 0.756 / 0.744 |

**Excluded:**
- skewed targets: diamonds, miami_housing, healthcare_insurance;
- no model separation (spread / SE ≤ 0.5): concrete, QSAR_fish_toxicity;
- 30% categorical: Food_Delivery_Time (optional).

**Protocol.** Official OpenML splits, repeat 0, 3 folds. Our pipeline is the raw module in fp32 + TF32 with the
wrapper's preprocessing, 1 pass. The ICL stage is O(n_context²) in memory, so the context is a seeded 4,096-row
subsample of each training fold; airfoil uses its full folds. The wrapper anchor (`tabarena_wrapper.py`, uv env) gets the
identical subsample.

**Results.** Mean RMSE over 3 folds (repeat 0). Context capped at 4,096 rows (superconductivity 2,048), except where
"full context" is noted:

| | airfoil | wine | houses | protein | superconductivity |
|---|---|---|---|---|---|
| wrapper, 8 estimators | 1.069 | 0.598 | 0.210 | 3.701 | (OOM at the memory cap) |
| wrapper, 1 estimator | 1.165 | 0.598 | 0.213 | 3.724 | |
| stock, our pipeline (1 pass) | 1.165 | 0.598 | 0.213 | 3.724 | 11.78 |
| `tx_median_B_reg` (continued training, no censoring) | 1.108 | 0.598 | 0.215 (+0.7%) | 3.756 (+0.9%) | 11.86 (+0.6%) |
| R1 step 1000, no spec (full context) | 1.237 (+6%) | 0.661 (+11%) | | | |
| R1b step 500, no spec / known spec (full context) | 1.239 / 1.236 | 0.627 / 0.617 | | | |
| **F2 (frozen + adapter), no spec** | **1.165** | **0.598** | **0.213** | | |

- Our stock pipeline reproduces the 1-estimator wrapper to 3–4 decimals. For comparisons with published TabArena numbers,
  match the ensemble size (airfoil: 1.165 with 1 estimator vs 1.069 with 8).
- **F2 without a spec is identical to stock on every fold**, as guaranteed.
- Even plain continued training (`tx_median_B_reg`) drifts +0.6–0.9% on the three larger datasets. That is relevant to
  every unfrozen stage B, including the replicate lane's.

### F2 final (step 3000) and Tox21

**Validation (synthetic), spec on, vs the best baseline:**

| setting | F2 | best baseline | share of the B1 → oracle gap closed by F2 |
|---|---|---|---|
| y right 30% | 0.215 | 0.262 (impute) | 46% |
| y left 30% | 0.199 | 0.229 | 48% |
| y right 30%, per-row bounds | 0.189 | 0.240 | 62% |
| y right 50% | 0.300 | 0.389 | 45% |
| X right 30%, per-row bounds | 0.160 | 0.158 (indicators) | 10% |
| clean | 0.1412 | | = stock exactly |

F1 (linear maps only, no adapter) reached 0.254 on y right 30% at step 1250, vs F2's 0.219: the adapter does the work.

**Tox21 censored regression (held out; 12 target assays, paired), F2 spec on minus the best simple baseline:**

| design | metric | F2 − best | tasks better |
|---|---|---|---|
| re-censored at p50 (ground truth known) | CRPS | −0.021 ± 0.014 vs impute | 7/12 |
| re-censored at p70 | CRPS | +0.009 ± 0.004 vs impute | 2/12 |
| natural 30% | proper censored CRPS, all queries | +0.005 ± 0.001 vs bound | 1/12 |
| natural 50% | proper censored CRPS, all queries | +0.014 ± 0.003 vs bound | 0/12 |
| natural 30% / 50% | NLL of censored queries | −0.17 / −0.22 vs impute | 10 / 11 of 12 |
| natural 30% / 50% | CRPS of observed queries | +0.053 / +0.121 vs impute | 1/12 |

The proper score is the CRPS of min(y, c) against the prediction censored at c, with c = the task's median bound
(about 1.96 log µM). Scoring censored and observed queries separately is not proper: it rewards knowing in advance which
queries will be censored.

**Reading.** The synthetic gain does not transfer to Tox21.
- F2 puts more mass beyond the bound. That is right for censored queries and wrong for observed ones. On the proper score
  it is slightly worse than leaving censored targets at the bound.
- On re-censored data, the clean "impute through censoring" test, it is mixed: better at 50% censoring, worse at 30%.
- Likely cause: prior mismatch. Tox21 censoring is not "a quantile of a smooth latent":
  - observed log AC50 piles up just below the bound (the assay's dynamic range);
  - inactives (censored) are a distinct population, informative of the features, whose latent lies far beyond;
  - the prior extrapolates a smooth tail instead.
- With spec off, F2 equals bound (and stock) exactly, so nothing is lost by shipping it opt-in. But there is no
  evidence yet that turning it on helps on real assay data.

### Next (proposed)
1. **Make the prior's censoring assay-shaped**, synthetic only, with Tox21 still held out (it has now been looked at
   once, so treat it as dev from here and keep the user's assay data as the test):
   - the bound sits at the top of a bounded dynamic range, with an observed pile-up just below it;
   - a share of rows is a separate "inactive" population, beyond the bound and feature-dependent (mixture prior);
   - re-run F2 frozen (about 1 h) and re-check the re-censoring and natural designs.
2. **Test on the user's assay data** (pIC50 with `>` qualifiers) using the same three metrics; this needs the data path.
3. **Defer feature-censoring flags:** at most 10% of the gap on synthetic data, and indicator columns already get most of it.
4. **Replicate lane (other agent):** any unfrozen stage B should carry the TabArena guard as a gate (+0.6–0.9% drift
   measured above).
