from dataclasses import replace

import numpy as np
import pytest

from driftguard.core.config import PromotionConfig, RetrainConfig
from driftguard.core.tracking import Tracker
from driftguard.data.simulation import Batch, DriftSimulator, ScenarioConfig
from driftguard.pipelines.pipeline import BatchValidationError, DriftGuardPipeline
from driftguard.pipelines.reporting import monitoring_frame, retrain_frame


def make_pipe(settings, **retrain_kw):
    s = replace(settings, retrain=replace(settings.retrain, **retrain_kw)) if retrain_kw else settings
    p = DriftGuardPipeline(s, tracker=Tracker.disabled())
    p.ensure_baseline()
    return p


def stream(p, name, n, seed=0, **kw):
    return p.run_scenario(ScenarioConfig(name=name, seed=seed, **kw), n)


def test_baseline_is_registered_and_persisted(settings):
    p = make_pipe(settings)
    rec = p.current_record()
    assert rec.version == 1 and rec.status == "production" and rec.reference_metrics["f1"] > 0.8
    again = DriftGuardPipeline(settings, tracker=Tracker.disabled())  # new process, same artifacts
    assert again.current_record().version == 1 and again.ensure_baseline().version == 1


def test_predict_and_validation(pipeline, splits):
    out = pipeline.predict(splits.X_test.head(5))
    assert len(out["predictions"]) == 5 and all(0 <= p <= 1 for p in out["probabilities"])
    with pytest.raises(BatchValidationError):
        pipeline.predict(splits.X_test.head(5).drop(columns=["mean radius"]))


def test_invalid_batch_is_recorded_and_stops(pipeline, splits):
    X = splits.X_stream.head(50).copy()
    X.iloc[0, 0] = np.nan
    ev = pipeline.process_batch(Batch(X=X, y=splits.y_stream.head(50)))
    assert ev["status"] == "invalid" and pipeline.state.batch_counter == 0 and len(pipeline.buffer) == 0
    assert pipeline.history.all()[-1]["status"] == "invalid"


def test_duplicate_batch_is_skipped_without_state_change(pipeline):
    sim = DriftSimulator(pipeline.splits, ScenarioConfig(name="sudden", seed=0), 100)
    first = pipeline.process_batch(sim.batch(0))
    n_events, counter, rows = len(pipeline.history.all()), pipeline.state.batch_counter, len(pipeline.buffer)
    second = pipeline.process_batch(sim.batch(0))
    assert first["status"] == "ok" and second["status"] == "duplicate_skipped"
    assert (len(pipeline.history.all()), pipeline.state.batch_counter, len(pipeline.buffer)) == (n_events, counter, rows)


def test_unlabeled_batch_is_monitored_but_not_buffered(pipeline, splits):
    ev = pipeline.process_batch(Batch(X=splits.X_stream.head(60)))
    assert ev["status"] == "ok" and ev["performance"] is None and len(pipeline.buffer) == 0


def test_no_drift_no_retraining(settings):
    p = make_pipe(settings)
    stream(p, "baseline", 8)
    assert p.state.retrain_attempts == 0 and p.current_record().version == 1


def test_sudden_drift_is_detected_and_retrained(settings):
    p = make_pipe(settings)
    evs = stream(p, "sudden", 12)
    pre, post = evs[:4], evs[4:]
    assert not any(e["drift"]["dataset_drift"] for e in pre)
    assert all(e["drift"]["dataset_drift"] for e in post[:3])
    assert p.state.retrain_attempts >= 1
    assert any(e["action"] in ("retrain_promoted", "retrain_rejected") for e in post)
    # drift must persist min_consecutive batches before triggering
    first = next(i for i, e in enumerate(evs) if e["action"].startswith("retrain_"))
    assert first >= 4 + p.settings.retrain.min_consecutive - 1


def test_concept_drift_degrades_performance_without_feature_drift(settings):
    p = make_pipe(settings, policy="performance_only")
    evs = stream(p, "concept", 10)
    # Features look normal. The detectors have a small non-zero false-positive rate (measured by the
    # experiments: ~0.4% for a 2-of-3 vote), so a lone flagged batch is possible; sustained drift is not.
    assert sum(e["drift"]["dataset_drift"] for e in evs) <= 1
    assert max(e["streaks"]["drift"] for e in evs) < p.settings.retrain.min_consecutive
    f1 = [e["performance"]["f1"] for e in evs]
    assert np.mean(f1[:4]) - np.mean(f1[5:]) > 0.2  # but the model got clearly worse
    assert any(e["performance_degraded"] for e in evs)
    assert p.state.retrain_attempts >= 1  # the performance rule fired although no drift detector did


def test_covariate_drift_can_flag_drift_without_hurting_f1(settings):
    p = make_pipe(settings, policy="never")
    evs = stream(p, "covariate", 12)
    assert sum(e["drift"]["dataset_drift"] for e in evs[4:]) >= 6
    f1 = [e["performance"]["f1"] for e in evs]
    assert np.mean(f1[6:]) > np.mean(f1[:4]) - 0.05  # drift flagged, accuracy did not fall


def test_rejected_candidate_never_replaces_deployed_model(settings):
    impossible = replace(settings, promotion=PromotionConfig(min_improvement=5.0))
    p = make_pipe(impossible, policy="drift_only")
    stream(p, "sudden", 12)
    assert p.state.retrain_attempts >= 1
    assert p.current_record().version == 1 and p.registry.current_version() == 1
    rejected = [r for r in p.registry.list() if r.status == "rejected"]
    assert rejected and all(r.version != 1 for r in rejected)
    ev = [e for e in p.history.all() if e["event_type"] == "retrain"][0]
    assert ev["status"] == "rejected" and ev["deployed_version_after"] == 1


def test_promotion_updates_deployed_model_and_reference(settings):
    p = make_pipe(settings, policy="drift_only")
    before = p.reference.copy()
    stream(p, "sudden", 12)
    promoted = [e for e in p.history.all() if e["event_type"] == "retrain" and e["status"] == "promoted"]
    assert promoted, "expected at least one promotion for sudden drift with default settings"
    assert p.current_record().version == promoted[-1]["candidate_version"] > 1
    assert p.registry.get(1).status == "archived"
    assert not p.reference.equals(before) and promoted[-1]["reference_refreshed"]
    assert p.paths.reference_file.exists()


def test_forced_retrain_and_no_duplicate_on_same_window(settings):
    p = make_pipe(settings, policy="never")
    assert p.retrain(force=True)["status"] == "skipped"  # nothing buffered yet
    stream(p, "sudden", 8)
    first = p.retrain(force=True, reason="test")
    assert first["status"] in ("promoted", "rejected") and first["forced"]
    again = p.retrain(force=True)
    assert again["status"] == "skipped" and "identical" in again["message"]


def test_unforced_retrain_respects_rules(pipeline):
    assert pipeline.retrain()["status"] == "not_triggered"


def test_state_persists_across_restart(settings):
    p = make_pipe(settings)
    stream(p, "sudden", 6)
    q = DriftGuardPipeline(settings, tracker=Tracker.disabled())
    assert q.state.batch_counter == 6 and len(q.history.all()) == len(p.history.all())
    assert q.history.has_batch(p.history.all()[0]["batch_fingerprint"])


def test_candidate_train_and_holdout_rows_are_disjoint(settings):
    p = make_pipe(settings, policy="never")
    stream(p, "sudden", 8)
    from driftguard.data.partition import holdout_mask

    w = p.buffer.window(p.settings.retrain.window_batches)
    hold = holdout_mask(w["row_id"], p.settings.retrain.partition_salt, p.settings.retrain.holdout_fraction)
    assert set(w[hold]["row_id"]).isdisjoint(set(w[~hold]["row_id"]))
    assert 0 < hold.sum() < len(w)


def test_reporting_frames(settings):
    p = make_pipe(settings)
    stream(p, "sudden", 8)
    mf, rf = monitoring_frame(p.history.all()), retrain_frame(p.history.all())
    assert len(mf) == 8 and {"flag_ks", "score_psi", "f1", "action"} <= set(mf.columns)
    assert list(rf.columns).count("status") == 1


def test_reset_restores_fresh_baseline(settings):
    p = make_pipe(settings)
    stream(p, "sudden", 8)
    rec = p.reset()
    assert rec.version == 1 and p.state.batch_counter == 0 and not p.history.all() and len(p.buffer) == 0


def test_metrics_summary_counts(settings):
    p = make_pipe(settings)
    stream(p, "sudden", 8)
    m = p.metrics_summary()
    assert m["counts"]["monitoring_runs"] == 8 and m["latest"]["scenario"] == "sudden"
