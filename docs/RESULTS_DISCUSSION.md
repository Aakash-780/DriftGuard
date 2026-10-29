# Results and discussion

**Scope.** This document interprets the *committed* run in `artifacts/experiments/` (20 seeds for Experiments 1 and 3, 10 seeds for
Experiment 2; library versions in `config.json`). All numbers below are copied from `RESULTS.md` / the `*_summary.csv` files. If you
re-run with different library versions the figures may shift slightly; if they shift a lot, trust your run and update this text. The synthetic
nature of the drift limits what can be concluded (see §6).

## 1. RQ1 — How do the detectors compare?

**False positives.** Over the 560 batches per detector in which no drift was present (all scenarios pooled):

| detector | false positives | FPR | Wilson 95 % |
|---|---|---|---|
| KS (BH-corrected) | 4 | 0.7 % | 0.3 – 1.8 % |
| PSI | 2 | 0.4 % | 0.1 – 1.3 % |
| domain classifier | 7 | 1.3 % | 0.6 – 2.6 % |
| prediction drift (α = 0.01) | 10 | 1.8 % | 1.0 – 3.3 % |

A 2-of-3 vote over the feature detectors was wrong in 2 of 560 batches (0.4 %). The one lone false alarm that surfaced in the test suite
(KS and PSI both fired on one unremarkable bootstrap batch) is consistent with this: highly correlated features mean one unusual batch moves
many features at once. The prediction-drift check, which tests the model's output distribution at α = 0.01, was the least specific; it is reported
as a monitor, not used in the retraining vote.

**Detection at the default magnitude saturates and cannot rank the detectors.** With a 1.5σ shift of 40 % of the features (`sudden`, `gradual`) or the
covariate tilt, KS and PSI flagged 100 % of full-intensity batches in every run, the domain classifier 97.5–100 %, with mean delay of 0 batches
(first affected batch). In the `gradual` scenario the first batch has only 25 % intensity and was still flagged. Delay therefore tells us little here.
Prediction drift lagged slightly on `gradual` (mean delay 0.75 batches, 90.6 % of post-onset batches).

**The magnitude sweep (Experiment 3) is what separates them.** Detection rate on full-intensity batches, 20 seeds:

| shift (σ) | KS | PSI | domain classifier | static-model ΔF1 [95 % CI] |
|---|---|---|---|---|
| *random signs* 0.1 | 5.0 % | 1.9 % | **85 %** | −0.011 [−0.024, 0.002] |
| 0.2 | 12.5 % | 14.4 % | **100 %** | −0.012 [−0.024, 0.001] |
| 0.3 | 78 % | 67 % | 100 % | −0.013 [−0.026, 0.001] |
| 0.5 | 100 % | 100 % | 100 % | −0.017 [−0.029, −0.004] |
| *coherent signs* 0.1 | 3.1 % | 2.5 % | **42 %** | −0.009 [−0.022, 0.004] |
| 0.2 | 19 % | 23 % | **100 %** | −0.016 [−0.030, −0.002] |
| 0.3 | 73 % | 71 % | 100 % | −0.023 [−0.036, −0.010] |
| 1.0 | 100 % | 100 % | 100 % | −0.134 [−0.184, −0.084] |

* The multivariate domain classifier detects much smaller shifts than the per-feature tests at similar FPR (≈ 1.3 % vs 0 % in this sweep — KS/PSI had no false
  positives at these magnitudes, so the classifier buys sensitivity with a small FPR cost). KS and PSI behave similarly (both marginal); KS was somewhat more sensitive at 0.3σ (78 % vs 67 %, random signs; 73 % vs 71 % coherent), and the
  difference is small relative to the gap to the domain classifier.
* **Part of that advantage was a simulator artefact, which I checked rather than assumed.** With random signs, near-duplicate features (radius vs perimeter) can be
  pushed in opposite directions, creating combinations that never occur in the data; that is easy to see jointly but barely visible marginally. With coherent
  (all-upward) shifts the classifier's 0.1σ detection rate roughly halves (85 % → 42 %). It still beats KS/PSI by a wide margin at 0.1–0.2σ, so a real multivariate
  advantage remains, but the size of the gap depends on how realistic the shift's joint structure is.
* **Detection precedes harm by a wide margin.** At 0.2–0.3σ the domain classifier (and, at 0.3σ, KS/PSI) flags nearly every batch while the static model's F1
  change is ≈ −0.01 to −0.02 and its confidence interval is at or near zero. Whether early detection is a benefit (early warning) or a cost (alarm without harm) depends
  on the operating context; the data cannot decide that.
* The domain classifier is **linear**; a unit test pins that it does *not* detect a covariance-only change with unchanged means.

## 2. RQ2 — Does drift coincide with performance loss?

Static (never-retrained) model, mean batch F1 before drift vs after drift reached full intensity, paired over 20 seeds:

| scenario | F1 before → after | Δ [95 % CI] | feature detectors | seeds with ΔF1 ≤ −0.05 |
|---|---|---|---|---|
| baseline | 0.931 → 0.923 | −0.008 [−0.020, 0.004] | quiet | 5 % |
| covariate | 0.931 → 0.971 | **+0.039** [0.026, 0.053] | flag ~100 % | 0 % |
| sudden / gradual | 0.931 → 0.874 | −0.057 [−0.082, −0.032] | flag 100 % | 50 % |
| concept | 0.931 → 0.373 | **−0.559** [−0.586, −0.531] | silent | 100 % |

* **Drift without degradation:** under covariate drift every feature detector fires and F1 does not fall.
  *Caveat:* the tilt also raises the malignant share of batches from 0.38 to 0.73 (checked directly on the simulator), so this scenario mixes covariate shift with
  a class-prior shift, and F1 depends on prevalence. "F1 rose" should not be read as "the model got better"; "no degradation" is the supported claim.
* **Drift with variable harm:** under sudden/gradual drift all three detectors fire in every run, but only half of the seeds lose ≥ 0.05 F1.
* **Degradation without drift:** under concept drift F1 collapses in every seed while KS, PSI and the domain classifier run-level detect in 15 %, 10 % and 5 % of runs —
  consistent with the false-positive floor (a single false flag in 8 batches happens by chance 5–10 % of the time). Prediction drift fired in 4 of 20 runs.
  Feature-input monitoring cannot protect against this failure mode; only labeled performance monitoring does.
* The `sudden` and `gradual` rows are identical by construction (both reach the same intensity-1 state before the comparison window).

## 3. RQ3 — Which retraining policy?

Mean over 10 seeds; "attempts" = candidates trained and judged by the gate. `test ΔF1` is the leak-free recovery metric (final deployed vs initial model on a post-drift
sample from the untouched test split); the in-stream gain in `RESULTS.md` is optimistic (later batches re-sample rows the candidate trained on) and is not used for conclusions.

| scenario | arm | runs that retrained | attempts (promoted / rejected) | trigger delay | test ΔF1 [95 % CI] |
|---|---|---|---|---|---|
| baseline (no drift) | performance_only = drift_or_performance | 3 / 10 | 3 (1 / 2) | – | −0.002 [−0.007, 0.003] |
| baseline | drift_only, drift_and_performance | 0 / 10 | 0 | – | 0 |
| covariate | drift_only = drift_or_performance | 10 / 10 | 25 (6 / 19) | 1.0 | +0.008 [−0.005, 0.020] |
| covariate | performance_only, drift_and_performance | 0 / 10 | 0 | – | 0 |
| sudden | drift_only = drift_or_performance | 10 / 10 | 21 (15 / 6) | 1.0 | +0.046 [−0.005, 0.097] |
| sudden | performance_only = drift_and_performance | 5 / 10 | 6 (4 / 2) | 3.4 | +0.033 [−0.017, 0.082] |
| gradual | drift_only = drift_or_performance | 10 / 10 | 26 (12 / 14) | 1.0 | +0.049 [0.001, 0.096] |
| concept | performance_only = drift_or_performance | 10 / 10 | 30 (2 / 28) | 1.0 | −0.011 [−0.028, 0.006] |
| concept | drift_only, drift_and_performance | 0 / 10 | 0 | – | 0 |

(Several arms coincide in these scenarios because the performance signal here implies the drift signal or never fires; overlapping lines in
`fig_retraining_f1_timeline.png` are identical trajectories, not plotting errors.)

What the data support:

* **The performance rule false-triggered in 3 of 10 no-drift runs** (batch F1 noise of ~0.03 vs a 0.05 threshold with 2-batch persistence). One of those
  candidates was promoted despite no drift (harmless here: test ΔF1 −0.002). The two-detector drift vote never false-triggered in these 10 runs.
* **Under feature drift, drift-based triggers act earlier** (1 batch after onset — the 2-batch persistence rule — vs 3.4 for the performance rule) and produce larger gains on the
  clean test sample, but with 10 seeds the 95 % CIs of `test ΔF1` mostly include zero (only `gradual` drift-based barely excludes it). **A significant recovery effect is not
  established**; the point estimates are positive (+0.03 to +0.05 F1).
* **Under covariate drift, drift triggers were largely wasted effort**: 25 attempts, 19 rejected, no demonstrable test gain, because the static model was not hurt. The
  performance-based policies correctly did nothing. A side effect: when a candidate is rejected the drift reference is not refreshed, so persistent drift re-triggers every cooldown period.
* **No policy recovers from concept drift.** The performance rule fires in every run (3 attempts each — the cooldown-limited maximum) but only 2 of 30 candidates were promoted.
  **Both promoted candidates were worse on the clean test sample** (seed 2: 0.476 → 0.420; seed 5: 0.324 → 0.269). The gate admitted two harmful updates, so it did not make
  retraining safe under this drift.
* **The "stale training data" hypothesis was tested and not supported.** Fitting candidates on the recent window only (`window_only`) promoted 0 of 30 candidates under concept
  drift. A plausible but *untested* explanation: the window straddles both regimes (trigger fires after two degraded batches, so about half the window is pre-drift) and 80 % label
  flips in half of feature space leave little learnable signal in ~100 post-drift rows. A change-point-aware window would be the next thing to try.

No policy dominates. If one must be named: performance-gated policies avoid wasted retraining under pure covariate drift but react late and false-trigger on noise; drift-gated
policies react early and avoid noise triggers here but retrain needlessly when drift is harmless; neither helps under concept drift. The default (`drift_or_performance`) combines the
strengths *and* weaknesses of both.

## 4. RQ4 — Does the promotion gate prevent harmful updates?

Partly. It rejected 19 of 25 candidates under covariate drift (no benefit), 28 of 30 under concept drift, and 6 of 21 under sudden drift, i.e. it declined most
non-beneficial candidates and the registry shows the deployed model unchanged in those cases (verified by tests). But it also let through the two concept-drift candidates that made the
clean-test F1 worse, and one no-drift candidate. The gate judges candidates on ~100–130 held-out rows with a 0.01 F1 margin — well inside that sample's sampling noise — so it is a
*noisy* safeguard, not a guarantee. A paired bootstrap/permutation test on the holdout, a larger margin, or a delayed shadow-deployment phase are the obvious hardening steps.

## 5. Summary of what is and is not supported

| Claim | Status |
|---|---|
| Feature drift and concept drift are different failure modes; feature detectors cannot see concept drift | Supported (concept: F1 −0.56, detectors at false-positive level) |
| Detected drift does not imply degraded accuracy | Supported (covariate: no loss; sudden: half the seeds lose < 0.05) |
| Multivariate domain classifier is more sensitive than KS/PSI | Supported for this simulator, ~half of the apparent size is a joint-structure artefact; classifier is linear |
| KS and PSI differ materially | Not supported (similar; KS slightly more sensitive at 0.3σ) |
| Drift-triggered retraining recovers performance under feature drift | Positive point estimates, **not statistically established** at 10 seeds |
| Retraining recovers concept drift | **Not supported** (0 benefit; 2 harmful promotions) |
| Promotion gate prevents harmful updates | **Partly supported** (most rejected; 2 harmful promotions in concept drift) |
| Chosen thresholds are optimal | Not claimed — fixed a priori and untuned |

## 6. Threats to validity

* Synthetic, known-onset drift on one small dataset; effect sizes do not transfer.
* Batches are bootstrap draws from 199 rows (effective sample sizes below 100); batch-level flags are not fully independent, so Wilson intervals are approximate.
* In-stream post-retraining metrics are optimistic by construction (pool-row reuse); only test-split metrics and the id-partitioned promotion holdout are leak-free. The performance
  baseline after promotion (holdout F1) may also be slightly pessimistic relative to later in-stream batches for the same reason.
* Ten seeds for Experiment 2 give wide intervals; more seeds would be needed to decide the `test ΔF1` questions above.
* Detector thresholds, window length, cooldown and gate margins were set once and not swept; conclusions are conditional on them.
