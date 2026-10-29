import numpy as np
import pandas as pd
import pytest

from driftguard.experiments.common import ExperimentConfig, t_ci, wilson
from driftguard.experiments.metrics import first_detection_delay, summarize_detection, summarize_fpr
from driftguard.experiments.run_detection import run_detection_one


def test_first_detection_delay():
    idx = np.arange(8)
    assert first_detection_delay(idx, np.array([0, 0, 1, 0, 0, 1, 1, 1]), onset=4) == 1  # early FP at 2 ignored
    assert np.isnan(first_detection_delay(idx, np.array([0, 0, 1, 0, 0, 0, 0, 0]), onset=4))


def hand_made():
    rows = []
    flags = {0: [0, 0, 1, 0, 0, 1, 1, 1], 1: [0, 0, 0, 0, 0, 0, 0, 0]}  # seed 0 detects with delay 1; seed 1 never
    for seed, f in flags.items():
        for i, v in enumerate(f):
            rows.append({"seed": seed, "scenario": "sudden", "batch_index": i,
                         "intensity": 0.0 if i < 4 else 1.0, "ks_drifted": v})
    return pd.DataFrame(rows)


def test_detection_summary_on_hand_computed_example():
    cfg = ExperimentConfig(scenarios=("sudden",), drift_start=4, ramp_batches=1)
    row = summarize_detection(hand_made(), ["ks"], cfg).iloc[0]
    assert row["fpr_no_drift_batches"] == pytest.approx(1 / 8)  # 1 flag in 8 pre-onset batches
    assert row["detection_rate_post_onset"] == pytest.approx(3 / 8)
    assert row["run_level_detection_rate"] == 0.5 and row["n_detected_runs"] == 1
    assert row["mean_delay_batches"] == 1.0


def test_pooled_fpr_with_wilson_interval():
    out = summarize_fpr(hand_made(), ["ks"]).iloc[0]
    assert out["false_positives"] == 1 and out["no_drift_batches"] == 8
    assert out["wilson95_lo"] < out["fpr"] < out["wilson95_hi"]


def test_stat_helpers():
    lo, hi = wilson(0, 50)
    assert lo == 0.0 and 0 < hi < 0.1
    ci = t_ci([0.1, 0.2, 0.3])
    assert ci["mean"] == pytest.approx(0.2) and ci["lo"] < 0.2 < ci["hi"] and ci["n"] == 3


def test_detection_run_is_deterministic_and_well_formed():
    cfg = ExperimentConfig(n_batches=6)
    a = pd.DataFrame(run_detection_one(0, "sudden", cfg))
    b = pd.DataFrame(run_detection_one(0, "sudden", cfg))
    assert a.equals(b) and len(a) == 6
    assert {"ks_drifted", "psi_drifted", "domain_classifier_drifted", "prediction_drifted", "f1"} <= set(a.columns)
    assert a.loc[a.batch_index >= 4, "ks_drifted"].all() and not a.loc[a.batch_index < 4, "ks_drifted"].any()


def test_sensitivity_summary_hand_example():
    from driftguard.experiments.metrics import summarize_sensitivity

    rows = []
    for seed in (0, 1):
        for i in range(8):
            rows.append({"seed": seed, "sign_mode": "random", "shift_magnitude": 0.5, "batch_index": i, "intensity": 0.0 if i < 4 else 1.0,
                         "ks_drifted": int(i >= 6), "f1": 0.9 if i < 4 else 0.8})
    cfg = ExperimentConfig(drift_start=4, ramp_batches=1)
    out = summarize_sensitivity(pd.DataFrame(rows), ["ks"], cfg).iloc[0]
    assert out["ks_detection_rate"] == 0.5 and out["ks_fpr"] == 0.0
    assert out["static_f1_delta"] == pytest.approx(-0.1)
