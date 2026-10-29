import numpy as np
import pytest

from driftguard.data.partition import holdout_mask
from driftguard.data.simulation import SCENARIOS, DriftSimulator, ScenarioConfig


def sim(splits, name, seed=3, **kw):
    return DriftSimulator(splits, ScenarioConfig(name=name, seed=seed, **kw), batch_size=100)


def test_same_seed_same_batch(splits):
    a, b = sim(splits, "sudden").batch(6), sim(splits, "sudden").batch(6)
    assert a.X.equals(b.X) and a.y.equals(b.y) and a.fingerprint() == b.fingerprint()


def test_different_seed_or_index_differs(splits):
    assert sim(splits, "baseline", 1).batch(0).fingerprint() != sim(splits, "baseline", 2).batch(0).fingerprint()
    assert sim(splits, "baseline").batch(0).fingerprint() != sim(splits, "baseline").batch(1).fingerprint()


def test_intensity_profiles():
    step = ScenarioConfig(name="sudden", drift_start=4)
    assert [step.intensity(i) for i in (0, 3, 4, 9)] == [0, 0, 1, 1]
    ramp = ScenarioConfig(name="gradual", drift_start=4, ramp_batches=4)
    vals = [ramp.intensity(i) for i in range(4, 9)]
    assert vals == [0.25, 0.5, 0.75, 1.0, 1.0]
    assert all(ScenarioConfig(name="baseline").intensity(i) == 0 for i in range(10))


def test_pre_onset_batches_identical_across_scenarios(splits):
    ref = sim(splits, "baseline").batch(2)
    for name in SCENARIOS:
        assert sim(splits, name).batch(2).X.equals(ref.X)


def test_concept_drift_changes_labels_not_features(splits):
    base, conc = sim(splits, "baseline").batch(8), sim(splits, "concept").batch(8)
    assert base.X.equals(conc.X)  # P(x) untouched
    assert (base.y != conc.y).sum() > 0  # P(y|x) changed
    assert base.row_ids.tolist() == conc.row_ids.tolist()


def test_sudden_shift_moves_selected_features_only(splits):
    s = sim(splits, "sudden")
    base, shifted = sim(splits, "baseline").batch(8), s.batch(8)
    moved = np.where(~np.isclose(base.X.to_numpy(), shifted.X.to_numpy()).all(axis=0))[0]
    assert set(moved) <= set(s.sel.tolist()) and len(moved) > 0
    assert (shifted.X.to_numpy() >= 0).all()


def test_covariate_shift_preserves_row_labels(splits):
    b = sim(splits, "covariate").batch(8)
    pool_y = splits.y_stream
    ids = [int(r[1:]) for r in b.row_ids]
    assert (pool_y.loc[ids].to_numpy() == b.y.to_numpy()).all()  # each row keeps its own label


def test_regime_batch_uses_test_pool(splits):
    r = sim(splits, "sudden").regime_batch(1.0, 50)
    assert all(str(x).startswith("t") for x in r.row_ids) and len(r.X) == 50


def test_unknown_scenario():
    with pytest.raises(ValueError):
        ScenarioConfig(name="nope")


def test_holdout_partition_is_deterministic_and_balanced():
    ids = [f"s{i}" for i in range(4000)]
    m1, m2 = holdout_mask(ids, 17, 0.3), holdout_mask(ids, 17, 0.3)
    assert (m1 == m2).all() and abs(m1.mean() - 0.3) < 0.03
    assert (holdout_mask(["s5", "s5"], 17, 0.3) == holdout_mask(["s5"], 17, 0.3)[0]).all()


def test_coherent_signs_shift_every_selected_feature_upward(splits):
    s = DriftSimulator(splits, ScenarioConfig(name="sudden", seed=3, coherent_signs=True), 100)
    assert (s.signs == 1.0).all()
    base, shifted = sim(splits, "baseline").batch(8), s.batch(8)
    assert (shifted.X.to_numpy()[:, s.sel] >= base.X.to_numpy()[:, s.sel]).all()
    r = DriftSimulator(splits, ScenarioConfig(name="sudden", seed=3), 100)
    assert (r.sel == s.sel).all()  # same features either way; only the signs differ
