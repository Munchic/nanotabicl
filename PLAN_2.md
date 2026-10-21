# PLAN 2: a replicate layer that beats the best hand-picked summary (2026-10-21)

**Goal.** A layer on top of a frozen TabICLv2 that takes raw replicate sets and beats the wrapper fed **mean + sd** and the wrapper fed **Huber + sd** on data it has never seen. The backbone stays frozen. A table without replicates equals TabICLv2.

**Bar, stated now.** The layer must beat both wrappers, and also the **CV-selected best of the hand summaries** (step 0c). Choosing between hand summaries per dataset is a useful tool, not a learned result.

---

## 1. Evidence this plan is built on

From RESEARCH.md and SHAPE.md:
- **No single hand summary wins.** Huber + sd leads on cytometry (NK +8.0, HVTN +8.4 AUC vs mean + sd). Mean + sd leads on CPTAC (Huber + sd 0.3 lower). The learned V2b leads on HAR log-loss and LINCS accuracy.
- **Explicit columns reach the frozen backbone; the additive embedding carries less.** sd column +4.0 on CPTAC vs the branch's +1.3. V4a ends equal to its own Huber + sd start.
- **The learned attention centre does not transfer** (Exp 9/10 decomposition), even with the column token: the v6 runs and V2b all had `col_ctx`.
- **Free-form learned summary columns added nothing** over the sd column at matched replicate counts (V3, `sd_learn`).
- **Per-dataset fine-tuning recovers nothing** (Exp 1: at most 0.3 points).
- **Real headroom is small on most sets.** Run-to-run noise on LINCS is about ±1 point.

**Corrections to earlier assumptions (checked 2026-10-21).**
- **The off-the-shelf wrapper drops context-constant columns** (`UniqueFeatureFilter` in tabicl `_sklearn/preprocessing.py`; checked on current main, not at the 2.2.0 tag).
  - The Exp 5h collapse of summary columns under count shift (0.38 to 0.47 accuracy) was in our own pipeline, which has no such filter.
  - With the wrapper, an all-n = 1 context drops the sd column and falls back to mean only.
  - The remaining exposure: mixed-n contexts, and replicated context with single-measurement queries. There sd = 0 means both "n = 1" and "low spread". The size of the effect is unknown.
- **The cytometry scale is unverified.** `build_cyto` reads `X` from the Zenodo "preprocessed" h5ad files without a transform. Whether `X` is on an arcsinh scale has not been checked. If it is not, part of Huber + sd's cytometry gain is a scale effect.

## 2. Rules

- **devsplit.py.** Decisions on VAL only: HAR, L1000, CellPainting, HVTN (Beijing for regression). TEST sets are read once, at the end, with `--allow_test`.
- **Held-out data.** Fresh datasets go into `HELDOUT` before any run in this plan (candidates in section 6).
- **Comparisons.** Paired per task, mean ± SE, n better / n. Headline is 8 vs 8 passes / estimators; 1 vs 1 is also reported.
- **Matched compute.** An arm that runs V views reports both V × 8 passes and V × ⌈8 / V⌉ passes, against the wrappers at 8.

## 3. Step 0: training-free (about one day; evaluation only)

### 0a. Cytometry scale check (minutes)

- Print per-marker 1 / 50 / 99% quantiles and the share of negative values of `X` for NK, HVTN and PreE.
  - This is a data property: no labels or metrics, so it is allowed for test sets.
- If `X` is not on an arcsinh scale (CyTOF arcsinh(x / 5) is roughly −1 to 8):
  - add arms `asinh_mean_sd` and `asinh_huber_sd` to shape_features.py: arcsinh(x / 5) on instances, then summarize;
  - score them on HVTN only.
- **Decision.** If asinh mean + sd closes most of the Huber + sd gap on HVTN, the gap is a scale effect.
  - The transform goes into the loader for every arm.
  - Cytometry then no longer justifies step 1.

### 0b. Count shift on real data (hours)

- **New option `--q_ks`** for real_eval.py and shape_features.py: subsample **query rows only** to k ∈ {1, 2} replicates; the context keeps all of them.
  - The existing `subsample` (real_eval.py:32) draws for all rows, so this is a few lines.
- **Second protocol `--mixed`:** each context row reduced to n = 1 with probability 0.5 (seeded); queries unchanged.
- **Arms** (VAL only):
  - wrapper with mean, mean + sd, Huber + sd;
  - V2b, V4a / V4c;
  - `tx_headM_A`, if finished.
- **Decision.** Step 2 runs only if mean + sd or Huber + sd lose by more than 2 SE vs mean-only, or vs a learned arm, under query shift or mixed n, on at least two VAL sets. Otherwise step 2 is dropped.

### 0c. In-context CV selection over views (the deployable floor, and the new bar)

- **Views:**
  - mean + sd and Huber + sd (plus the asinh variants if 0a calls for them);
  - the trained branch: V2b or V4c, whichever is better on VAL.
- **Scoring:** for each view, 5-fold CV within the context (stratified, grouped where the task has groups), as the log-loss on held-out folds.
- **Combination:** weights ∝ exp(−CV log-loss / τ). Three pre-registered variants:
  - argmin;
  - uniform average;
  - softmax, with one τ chosen on VAL for all datasets.
- **Code:** new script `select_views.py`. It reuses the shape_features.py summaries and the real_eval.py checkpoint loading.
- **Expected:** close to the best fixed view per dataset, and therefore better than each fixed view on average.
- **Risk:** small contexts (HVTN about 48 rows) make CV noisy. The soft variants are the guard.
- **Output:** the CV selector replaces the two wrappers as the bar for steps 1 and 2.

## 4. Step 1: a learned, column-aware estimator writing explicit columns

**Idea.**
- The backbone reads explicit columns well, and the right estimator differs by column.
- So the layer writes the location and scale columns itself: a weighted mean and a weighted sd of each cell's replicates.
- The weights depend on each replicate's standardized residual and on a column token.

**Parametrization.**
- **Weights:** w_k = softplus(c0 + g(|r_k| / s_c, t_c, log n)).
  - c0 is set so that w ≡ 1 at init, and g's output layer is zero-init.
  - So step 0 is exactly the mean + sd wrapper input.
- **Residuals:** r_k is the residual from the cell's current estimate, over 3 differentiable IRLS iterations from the cell median, as huber.py does.
- **Column scale:** s_c is the pooled robust column scale (`huber.pooled_scale`, ported to torch).
- **Column token:** t_c is the existing `col_ctx` `phi_col`, pooled over **all** replicated cells of the column: context and query, no labels.
- **Reachability:** Huber's weight min(1, 1.345 s_c / |r|) is in the family, so the Huber + sd input is reachable. The mean is the start.
- **Scale column:** the weighted sd with the same weights; no new parameters.
- **n = 1 cells:** value unchanged, scale column 0 (as `sd_cols`). All-single tables equal stock.

**How it differs from what failed.**
- **The learned centre** was free attention over per-replicate features, and it was the backbone's main location. Here the weights are a one-dimensional function of residual size per column, conditioned on thousands of residuals, starting from the mean.
- **V3's learned columns** were readouts of the cell vector. Here the column is an estimator with a known start and a known reachable alternative.

**Implementation.**
- `backbone_loc: 'learned_m'` in obs_model.py, next to `'huber'`.
- Self-tests, as for `--side`:
  - step 0 equals the mean + sd wrapper input;
  - n = 1 equals stock;
  - a per-column weight-curve plot.

**Runs.** 1000 steps each, with the V2b settings: shape prior with `xor_pairs`, frozen backbone, dev kill on VAL at margin 0.5, checked at steps 500 and 1000.

| id | backbone columns | branch | question |
|---|---|---|---|
| M1 | learned_m location + scale | off (no additive embedding) | do learned explicit columns alone beat the CV selector? |
| M2 | as M1 | V2b branch (joint path, embedding), warm-started from `tx_shV2b_A` | does the learned branch add on top? |

**Diagnostic, read before accuracy.** Plot w(|r| / s) per column on VAL.
- **Expected:** near flat on L1000 and Cell Painting columns (near-Gaussian), and down-weighted tails on HVTN markers if 0a left a gap.
- **If every column learns the same curve,** the column token is not being used. Fix that before reading accuracy.

**Success on VAL.**
- M1 or M2 beats the CV selector (0c) on the mean over VAL tasks by more than 2 SE.
- It is not worse than the best wrapper by more than 2 SE on any VAL set.
- The winner gets the full 6000 steps.

## 5. Step 2: a count-aware uncertainty column (only if 0b triggers it)

- **Column:** the scale column becomes scale + h(log n, t_c, context count statistics), with h zero-init.
  - Step 0 equals step 1's winner.
  - At n = 1 the column can express "unknown spread" instead of 0.
- **Prior:** add `count_regimes` (obs_train.py: `row`, `ctxq`) to the shape prior's replicate regime, at 30% of tables. One flag.
- **Evaluation:** the 0b protocols (query shift, mixed n) plus the standard VAL eval.
- **Success:** beats step 1's winner under shift by more than 2 SE on the sets where 0b found a wrapper loss, and ties it at matched counts.

## 6. Final evaluation (run once)

- **Freeze:** the winner (step 1 or 2) and the 0c selector configuration.
- **Before the read:** add 2 to 3 never-evaluated replicate datasets to `HELDOUT`, chosen for a known measurement process and a large 1 → all replicate headroom. Candidates:
  - qPCR with non-detects;
  - immunoassays with LOD / ULOQ flags;
  - a further cytometry cohort.
- **The read:** one `--allow_test` read of TEST + HELDOUT, all arms: mean, mean + sd, Huber + sd, CV selector, winner.
- **Primary criterion (stated now):** the winner beats both mean + sd and Huber + sd by more than 2 SE on at least half of the HELDOUT + TEST datasets, and is never significantly worse.
- **Secondary:** it beats the CV selector on more datasets than it loses.

## 7. Order and cost

| step | cost | GPU | gates |
|---|---|---|---|
| 0a scale check | minutes | no | the cytometry arms of 0b / 0c |
| 0b count shift | hours | eval only | step 2 |
| 0c CV selector | hours | eval only | the bar for steps 1 and 2 |
| 1 M1, M2 screens | 2 × about 2 h | yes | the 6000-step run |
| 2 (conditional) | 1 screen + eval | yes | |
| 6 final | hours | eval only | |

## 8. Relation to the queued v7 head (lane_head*.sh)

- **Role:** it is the pure "no hand columns" arm and stays as the comparison.
- **Two risks from earlier results:**
  - it uses `center: attn`, the component the Exp 10 decomposition found does not transfer;
  - it has no explicit columns, the channel that works.
  - Consider `center: mean` for its screen.
- **Scheduling:** its checkpoint enters 0b / 0c as a view when it finishes. Steps 0a to 0c do not need the GPU and can run while it trains.

## 9. Not in this plan, and why

- **Unfreezing / LoRA.** Stage B lost on L1000 (−3.5) and UPS1_low (−0.8), and it breaks the exact guarantees. Revisit only if step 1 shows the explicit-column gap is still open.
- **A two-level SCM prior.** The v7 continuous prior already tests the "no named families" question.
- **Test-time fine-tuning of the layer.** Exp 1 recovered nothing.
- **More hand columns.** The sketch loses to mean + sd on small sets (Cell Painting −0.2 vs +1.5).
- **Clean-teacher distillation.** An add-on if M1 / M2 start learning late (`sh_xorp` still flat at step 750); not a step of its own.
