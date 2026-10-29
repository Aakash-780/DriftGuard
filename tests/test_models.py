import numpy as np
import pytest

from driftguard.core.config import ModelConfig, PromotionConfig
from driftguard.models.evaluation import compute_metrics, evaluate_model
from driftguard.models.promotion import evaluate_candidate
from driftguard.models.registry import ModelRegistry
from driftguard.models.training import get_baseline_model, train_model


def test_metrics_known_values():
    m = compute_metrics([1, 1, 0, 0], [1, 0, 0, 0], [0.9, 0.4, 0.2, 0.1])
    assert m["accuracy"] == 0.75 and m["precision"] == 1.0 and m["recall"] == 0.5
    assert m["f1"] == pytest.approx(2 / 3) and m["roc_auc"] == 1.0


def test_roc_auc_none_for_single_class():
    assert compute_metrics([1, 1, 1], [1, 1, 1], [0.9, 0.8, 0.7])["roc_auc"] is None


def test_baseline_training_is_deterministic_and_good(splits):
    from driftguard.core.config import DataConfig

    m1 = get_baseline_model(DataConfig(seed=0), ModelConfig())
    m2 = train_model(splits.X_train, splits.y_train, ModelConfig(), 0)
    assert (m1.predict_proba(splits.X_test) == m2.predict_proba(splits.X_test)).all()
    assert evaluate_model(m1, splits.X_test, splits.y_test)["accuracy"] > 0.85


def test_logistic_regression_option(splits):
    m = train_model(splits.X_train, splits.y_train, ModelConfig(model_type="logistic_regression"), 0)
    assert evaluate_model(m, splits.X_test, splits.y_test)["roc_auc"] > 0.9


def test_no_overlap_between_splits(splits):
    sets = [set(x.index) for x in (splits.X_train, splits.X_ref, splits.X_stream, splits.X_test)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert sets[i].isdisjoint(sets[j])
    assert sum(len(s) for s in sets) == 569


# ---------------------------------------------------------------- promotion gate
def _models(splits):
    good = train_model(splits.X_train, splits.y_train, ModelConfig(), 0)
    bad = train_model(splits.X_train, 1 - splits.y_train, ModelConfig(), 0)  # label-flipped: deliberately terrible
    return good, bad


def test_better_candidate_is_promoted(splits):
    good, bad = _models(splits)
    d = evaluate_candidate(bad, good, (splits.X_stream, splits.y_stream), (splits.X_ref, splits.y_ref), PromotionConfig())
    assert d.promote and all(c["passed"] for c in d.checks.values())


def test_worse_candidate_is_rejected(splits):
    good, bad = _models(splits)
    d = evaluate_candidate(good, bad, (splits.X_stream, splits.y_stream), (splits.X_ref, splits.y_ref), PromotionConfig())
    assert not d.promote and not d.checks["f1_improvement"]["passed"]


def test_equal_candidate_is_rejected_for_no_improvement(splits):
    good, _ = _models(splits)
    d = evaluate_candidate(good, good, (splits.X_stream, splits.y_stream), (splits.X_ref, splits.y_ref), PromotionConfig())
    assert not d.promote and d.checks["f1_improvement"]["value"] == 0


def test_small_or_single_class_holdout_is_rejected(splits):
    good, bad = _models(splits)
    anchor = (splits.X_ref, splits.y_ref)
    small = evaluate_candidate(bad, good, (splits.X_stream.head(10), splits.y_stream.head(10)), anchor, PromotionConfig())
    assert not small.promote and not small.checks["holdout_size"]["passed"]
    pos = splits.y_stream == 1
    single = evaluate_candidate(bad, good, (splits.X_stream[pos], splits.y_stream[pos]), anchor, PromotionConfig())
    assert not single.promote


def test_reference_regression_check_can_block(splits):
    good, bad = _models(splits)
    cfg = PromotionConfig(max_reference_f1_drop=-0.5)  # demands +0.5 F1 on the anchor: impossible for equal models
    d = evaluate_candidate(good, good, (splits.X_stream, splits.y_stream), (splits.X_ref, splits.y_ref), cfg)
    assert not d.checks["reference_regression"]["passed"]


# ---------------------------------------------------------------- registry
def test_registry_versions_and_pointer(tmp_path, splits):
    reg = ModelRegistry(tmp_path / "reg")
    assert reg.current_version() is None
    model, _ = _models(splits)
    r1 = reg.register(model, notes="a")
    assert r1.version == 1 and r1.status == "candidate" and reg.current_version() is None
    reg.promote(1)
    r2 = reg.register(model)
    assert r2.version == 2 and r2.parent_version == 1 and reg.current_version() == 1  # registering does not deploy
    reg.reject(2, "no good")
    assert reg.get(2).status == "rejected" and reg.current_version() == 1
    reg.promote(2)
    assert reg.get(1).status == "archived" and reg.current_version() == 2
    loaded, rec = reg.load_current()
    assert rec.version == 2 and np.array_equal(loaded.predict(splits.X_test), model.predict(splits.X_test))


def test_registry_unknown_version(tmp_path):
    with pytest.raises(KeyError):
        ModelRegistry(tmp_path / "r").promote(7)
