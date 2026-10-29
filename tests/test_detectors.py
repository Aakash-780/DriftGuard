import numpy as np
import pandas as pd
import pytest

from driftguard.core.config import DriftConfig
from driftguard.data.simulation import DriftSimulator, ScenarioConfig
from driftguard.monitoring.detectors import (
    DomainClassifierDetector, KSDetector, PSIDetector, bh_reject, bin_proportions, build_detectors,
    prediction_drift, psi_from_proportions, quantile_edges,
)


def frame(rng, n, shift=0.0, d=6):
    return pd.DataFrame(rng.normal(shift, 1.0, size=(n, d)), columns=[f"f{i}" for i in range(d)])


def test_psi_known_value():
    # (0.8-0.5)ln(0.8/0.5) + (0.2-0.5)ln(0.2/0.5) = 0.4159...
    assert psi_from_proportions([0.5, 0.5], [0.8, 0.2]) == pytest.approx(0.41589, abs=1e-4)
    assert psi_from_proportions([0.3, 0.7], [0.3, 0.7]) == 0.0


def test_psi_empty_bin_is_finite():
    assert np.isfinite(psi_from_proportions([0.5, 0.5], [1.0, 0.0], eps=1e-4))


def test_quantile_bins_cover_all_values():
    x = np.arange(100.0)
    edges = quantile_edges(x, 5)
    p = bin_proportions(np.r_[x, [-1e9, 1e9]], edges)
    assert p.sum() == pytest.approx(1.0) and len(p) == len(edges) + 1


def test_bh_reject_known_case():
    # thresholds alpha*k/m = .01 .02 .03 .04 .05 -> only the first two p-values clear their threshold
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.2])
    assert bh_reject(p, 0.05).tolist() == [True, True, False, False, False]
    assert not bh_reject(np.array([0.2, 0.3, 0.9]), 0.05).any()


def test_bh_is_step_up_and_less_strict_than_bonferroni():
    # m=3: thresholds .0167 .033 .05. 0.02 fails its own threshold but the larger ranks pass -> all rejected.
    p = np.array([0.02, 0.03, 0.04])
    assert bh_reject(p, 0.05).all()
    assert not (p < 0.05 / 3).any()  # Bonferroni would reject nothing


@pytest.mark.parametrize("cls", [KSDetector, PSIDetector, DomainClassifierDetector])
def test_no_drift_not_flagged_and_shift_flagged(cls):
    rng = np.random.default_rng(0)
    ref, same, shifted = frame(rng, 400), frame(rng, 200), frame(rng, 200, shift=1.5)
    det = cls(DriftConfig()).fit(ref)
    assert det.detect(same).drifted is False
    res = det.detect(shifted)
    assert res.drifted is True and res.detector == det.name


def test_ks_reports_per_feature_results():
    rng = np.random.default_rng(1)
    ref = frame(rng, 300)
    cur = ref.sample(200, random_state=0).copy()
    cur["f0"] += 3.0
    res = KSDetector(DriftConfig(ks_min_drifted_share=0.1)).fit(ref).detect(cur)
    assert res.feature_drifted["f0"] and res.n_drifted_features >= 1
    assert res.feature_scores["f0"] == max(res.feature_scores.values())


def test_domain_classifier_detects_coordinated_multivariate_shift():
    rng = np.random.default_rng(2)
    cov = 0.8 * np.ones((10, 10)) + 0.2 * np.eye(10)
    ref = pd.DataFrame(rng.multivariate_normal(np.zeros(10), cov, 400))
    shift = np.r_[np.ones(5), -np.ones(5)] * 0.5  # opposite-sign shifts along a low-variance direction
    cur = pd.DataFrame(rng.multivariate_normal(shift, cov, 200))
    ref.columns = cur.columns = [f"f{i}" for i in range(10)]
    assert DomainClassifierDetector(DriftConfig()).fit(ref).detect(cur).drifted


def test_linear_domain_classifier_limitation_covariance_only_change():
    # DOCUMENTED LIMITATION: the domain classifier is linear, so a change in correlation structure with
    # unchanged means/marginals is not detected (a non-linear classifier would be needed).
    rng = np.random.default_rng(2)
    ref = pd.DataFrame(rng.multivariate_normal([0, 0], [[1, .9], [.9, 1]], 400), columns=["a", "b"])
    cur = pd.DataFrame(rng.multivariate_normal([0, 0], [[1, -.9], [-.9, 1]], 200), columns=["a", "b"])
    assert not DomainClassifierDetector(DriftConfig()).fit(ref).detect(cur).drifted


def test_detectors_deterministic(splits):
    b = DriftSimulator(splits, ScenarioConfig(name="sudden", seed=1), 100).batch(6)
    dets = build_detectors(("ks", "psi", "domain_classifier"), DriftConfig(), seed=0)
    first = [d.fit(splits.X_ref).detect(b.X).score for d in dets]
    second = [d.fit(splits.X_ref).detect(b.X).score for d in dets]
    assert first == second


def test_concept_drift_invisible_to_feature_detectors(splits):
    sim = DriftSimulator(splits, ScenarioConfig(name="concept", seed=1), 100)
    dets = build_detectors(("ks", "psi", "domain_classifier"), DriftConfig(), seed=0)
    for d in dets:
        d.fit(splits.X_ref)
    flags = [d.detect(sim.batch(i).X).drifted for d in dets for i in range(5, 10)]
    assert not any(flags)


def test_prediction_drift_detects_distribution_change():
    rng = np.random.default_rng(0)
    assert not prediction_drift(rng.uniform(0, 1, 500), rng.uniform(0, 1, 500), 0.01)["drifted"]
    assert prediction_drift(rng.uniform(0, 1, 500), rng.uniform(0.5, 1, 500), 0.01)["drifted"]


def test_unknown_detector_name():
    with pytest.raises(ValueError):
        build_detectors(("nope",), DriftConfig())
