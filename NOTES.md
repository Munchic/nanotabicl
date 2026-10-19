ObsPFN — telling a tabular foundation model how its data was measured
Problem description
Tabular foundation models (TFMs) (TabPFN-2.5, TabICLv2) match or outperform tuned GBDT (XGBoost, CatBoost) models on small and medium tabular data in fields such as finance and health records [1,2]. Experimental datasets in biology, chemistry, and physics are different: measurements are replicated, heterogeneous in reliability, and censored at assay limits. A TFM amortizes inference with respect to a prior fixed at pretraining [3], so the observation model — replication, reliability, censoring, missingness, and the likelihood they shape — cannot be specified at inference. Existing TFMs cannot represent replicates at all. Reliability has no mechanism either: lacking weights, PFNs resort to discarding data or adjusting the decision threshold [4]. Censoring is addressed only by pretraining dedicated models, one observation model per checkpoint [5,6,7]. We propose making the observation model a conditioning input at inference by pretraining over a distribution of observation models and demonstrate it on replicate-aware cells and per-row reliability weighting.
Technical proposal
We build on NanoTabICL [8] with an open prior, at ~1M parameters and introduce three main changes. Stage 0: absent information must be absent. No TFM is shown to marginalize: TabICLv2 mean-imputes, so the model cannot tell "no value here" from "average value here," while TabPFN-3's missingness signal lets it condition on absence rather than integrate over it. Stage 1: replicates are evidence, not rows. Expanding replicates into rows lengthens the context and breaks row independence. Instead, feature replicates pool inside the cell, and label replicates weight the loss by their count (a row can have three feature replicates and five label replicates). Stage 2: a weight is a precision. Declaring weight w asserts that a row carries the information of w observations. Specified below but outside of the proposal scope, Stages 3 and 4 extend the same conditioning interface to missingness mechanisms and censoring.
#
What
Scope
Implementation
Evaluation
0
Marginalize
Premise
Absent-column embedding, masked out of row attn
p(y|x,z) = p(y|x) for uninformative z
1
Replicates per cell
Primary goal
Attn pooling over padded set, contaminated replicates in prior, label count weights the loss
Beats summary stats (mean, median, IQR, SEM, n)
2
Per-row reliability
Stretch goal
log(w) in ICL attn logits, heteroscedastic rows in prior
w=0 reproduces a fit without that row
3
Missing semantics
Future follow-up
MAR/MNAR mechanisms in the prior, declared at inference
MNAR recovers structure MAR misses
4
Censoring
Future follow-up
Censoring indicator in the prior with a censored NLL
Censored, replicated data unseen in training


Compute request: ~1M parameters, 16 GB VRAM; roughly 80 GPU-hours for the full sweep on L4 ($0.80/GPU hr), under $200 including re-runs.
Related problems
No TFM accepts replicates in a cell. The nearest mechanism is weighted in-context learning for LLMs, which biases attention logits toward better demonstrations but infers the weights from a proxy metric [9]. No TFM accepts per-observation weights; the only substitute, duplicating rows, backfires: oversampling drops worst-class accuracy to 0.259 vs 0.446 uncorrected [4].

Missingness is handled inconsistently: TabPFN-3 passes an explicit missingness signal [2], TabICLv2 mean-imputes [3]; neither accepts a user-specified mechanism at inference. Censoring is handled only by pretraining: SurvPFN adds a survival prior and a censored negative log-likelihood to a NanoTabPFN backbone [7].
Sources
[1] TabArena: A Living Benchmark for ML on Tabular Data, NeurIPS 2025 — TFMs lead on small data; GBDTs remain competitive
[2] TabPFN-2.5, arXiv:2511.08667 — scale limits
[3] TabICLv2, arXiv:2602.11139 — SCM prior fixed at pretraining; mean-imputes
[4] Correcting Class Imbalance in PFNs, arXiv:2605.21742 — no loss reweighting under ICL
[5] SurvivalPFN, arXiv:2605.15488 — right-censored prior
[6] Survival In-Context, arXiv:2603.29475 — dedicated survival prior
[7] SurvPFN, arXiv:2606.04564 — TFMs can't natively handle censoring
[8] NanoTabICL — github.com/soda-inria/nanotabicl
[9] Reweighting Demonstration Examples for ICL, EMNLP 2023, arXiv:2310.08309 — attention-logit weighting
