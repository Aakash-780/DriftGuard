# Experimental protocol

Everything here is implemented in `driftguard/experiments/` and executed by
`python -m driftguard.experiments.run_all`. All randomness is derived from integer seeds; rerunning with the same
library versions reproduces the CSV files bit-for-bit (verified by `tests/test_experiments.py` for the detection run).

## Data and splits

Breast Cancer Wisconsin (Diagnostic), 569 rows, 30 numeric features, from `sklearn.datasets`. Positive class =
malignant (label 1). For each seed `s` the data are split (stratified, `random_state=s`) into

| split | share | rows | role |
|---|---|---|---|
| train | 30 % | 170 | fits the baseline model (and is re-used in candidate fits) |
| reference | 20 % | 114 | drift reference **and** fixed "anchor" validation set; never trained on |
| stream | 35 % | 199 | pool that simulated incoming batches are drawn from |
| test | 15 % | 86 | never used by any pipeline decision; used only for post-hoc recovery measurement |

Each seed changes the split, the simulation and the model, so seed-to-seed variance includes data-split variance.

## Scenarios (see `data/simulation.py`)

Batches contain 100 rows drawn **with replacement** from the stream pool; batch `i` is a pure function of `(seed, i)`.
Drift starts at batch 4 (`drift_start`); 12 batches per run.

| scenario | what changes | P(x) | P(y\|x) |
|---|---|---|---|
| baseline | nothing | same | same |
| covariate | rows re-sampled with exponential tilt (β=1.5) on a random 40 % of features; every row keeps its own label | **changes** | preserved |
| sudden | step: random 40 % of features translated by ±1.5 reference-σ (clipped at 0) | **changes** | rows can cross the boundary |
| gradual | same translation, ramped over 4 batches (intensity .25, .5, .75, 1) | **changes** | as above |
| concept | labels flipped (p = 0.8) where `worst perimeter` is above its reference mean | same | **changes** |

Note: the covariate tilt favours rows with high feature scores, which are disproportionately malignant, so this scenario also shifts the class prior (malignant share ≈ 0.38 → 0.73 in the committed run). It is covariate shift with P(y|x) preserved, but not a *pure* P(x) change at constant prevalence; F1 is prevalence-dependent.

*Intensity* ∈ [0,1] is 0 before onset. "Full intensity" batches are those with intensity = 1; for every scenario that
is batch index ≥ `drift_start + ramp_batches − 1` = 7 (the "post" window, 5 batches).

## Experiment 1 — detector comparison (static model)

Detectors are fitted on the reference split and applied to every batch. Detectors never receive labels.
The deployed model is never retrained, so its per-batch metrics show what drift does to a model nobody updates.

Detectors and default (a-priori, configurable) thresholds:

* **ks** — two-sample KS per feature; Benjamini–Hochberg at α = 0.05; dataset drift if ≥ 20 % of features rejected.
* **psi** — 5 reference-quantile bins per feature; ε = 1e-4; feature drifted if PSI ≥ 0.25; dataset drift if ≥ 20 % of features.
* **domain_classifier** — 5-fold cross-validated ROC-AUC of a standardised, L2-regularised (C = 0.1) logistic regression
  separating reference rows from batch rows; dataset drift if AUC ≥ 0.65.
* **prediction** — KS test (α = 0.01) on predicted P(malignant), reference vs batch. Reported as a monitor of the *model's
  outputs*, not as a feature-drift detector; it is not part of the retraining vote.
* **evidently** (optional, `--with-evidently`) — Evidently `DataDriftPreset`, share 0.5.

### Metric definitions

Let a *flag* be a detector's dataset-level decision for one batch.

* **False-positive rate (FPR)** = flagged batches ÷ batches with intensity 0 (no drift present), pooled over all scenarios
  (pre-onset batches of drift scenarios are distribution-identical to baseline batches). Wilson 95 % interval; batches are
  treated as independent draws, which is approximate because they share a reference sample.
* **Detection rate (post-onset)** = flagged ÷ batches with index ≥ onset (includes partial-intensity ramp batches).
* **Detection rate (full intensity)** = flagged ÷ batches with intensity = 1.
* **Run-level detection rate** = share of runs (seeds) with ≥ 1 flagged batch at or after onset.
* **Detection delay** = (first flagged batch index ≥ onset) − onset, in batches, averaged over runs that detected. Undetected
  runs are counted (`n_detected_runs / n_runs`) but not imputed into the mean.
* **Performance before / after** = mean batch F1 over batches < onset vs batches ≥ 7; paired per seed; mean Δ with a
  t-based 95 % CI over seeds.

## Experiment 2 — retraining arms (full pipeline)

Each run builds a fresh artifact directory, trains the baseline, streams the 12 batches through `DriftGuardPipeline` and
records all events. Arms see identical batches for a given (seed, scenario) because batch generation does not depend on the
model, so arm-vs-control differences are paired.

| arm | policy | candidate fitted on |
|---|---|---|
| never | no retraining (control) | – |
| performance_only | F1 drop ≥ 0.05 below the deployed model's baseline for 2 consecutive labeled batches | train + window |
| drift_only | ≥ 2 detectors flag for 2 consecutive batches | train + window |
| drift_and_performance | both signals | train + window |
| drift_or_performance | either signal (the default policy) | train + window |
| drift_or_performance+window_only | as above | **recent window only** (ablation: stale-data hypothesis) |

Common rules: cooldown 3 batches after an attempt; ≥ 100 labeled rows in a window of the last 4 batches; the window is split
by a hash of row identity into candidate-train (70 %) and recent-holdout (30 %) so no row id is on both sides.

Promotion gate (all must hold, evaluated on the recent holdout unless stated): holdout ≥ 30 rows with ≥ 5 per class;
candidate F1 ≥ deployed F1 + 0.01; candidate F1 ≥ 0.70; candidate recall ≥ deployed recall − 0.10; candidate F1 on the
**anchor (reference) split** ≥ deployed F1 − 0.10. On promotion the drift reference is replaced by the training window and the
performance baseline by the candidate's holdout metrics.

### Metric definitions

* **Retraining attempts / promoted / rejected** — counts of retrain events with status promoted/rejected per run (skipped events
  are counted separately). In the baseline scenario every attempt is a false trigger.
* **Acceptance rate** = Σ promoted ÷ Σ attempts.
* **Trigger delay** = first retraining batch index − onset (runs with a trigger only).
* **instream_gain_vs_never** = per-seed difference in mean batch F1 over batches ≥ 7, arm minus `never`; mean and 95 % CI.
  **Optimistic by construction**: batches are bootstrap draws from the same 199-row pool whose "A-partition" rows trained the
  candidate, so later batches contain rows the candidate has seen. The holdout used for promotion is clean (rows are partitioned by
  id), and `test_delta_f1` is clean (separate split); this in-stream number is not.
* **test_delta_f1 (recovery; the primary, leak-free metric)** = F1(final deployed model) − F1(initial baseline model), both evaluated on the same 300-row
  sample drawn with replacement from the **test** split under fully developed drift.

## What the experiments cannot show

* The drift is **synthetic and known**; real production drift is messier, unlabeled, and rarely has a clean onset.
* Batches are bootstrap draws from 199 pool rows, so repeated rows occur and effective sample sizes are smaller than 100 per batch.
* One small tabular dataset (569 rows). Absolute numbers are not transferable; the *qualitative* behaviours (drift without
  degradation, degradation without drift, gate rejections) are the point.
* Thresholds were fixed a priori and not tuned on these results; different thresholds shift the FPR/detection trade-off.
* Promotion decisions rest on a holdout of ~100–130 rows: a 0.01 F1 margin is well inside its sampling noise.

## Experiment 3 — sensitivity to drift magnitude (static model)

Experiment 1 saturates at the default magnitude (every feature detector flags ~100 % of drifted batches), which cannot rank the
detectors. This sweep reuses the `sudden` scenario with `shift_magnitude` ∈ {0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5} reference-σ
(same 40 % of features, same seeds, 20 seeds), run twice: with **random** ± signs on the shifted features (the default scenario) and with **coherent** signs (all shifted features move upward). Random signs can push near-duplicate features (radius vs perimeter) in opposite directions, a physically implausible joint configuration that a multivariate detector can see even when each marginal moves little; the coherent variant checks how much of any multivariate advantage is due to that and reports, per magnitude: each detector's full-intensity detection rate, the pooled
no-drift FPR, and the static model's F1 before vs after. It shows how small a shift each detector can see and whether detection
precedes measurable harm.
