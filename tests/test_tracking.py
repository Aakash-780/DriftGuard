import importlib.util

import pytest

from driftguard.core.tracking import Tracker, flatten

HAS_MLFLOW = importlib.util.find_spec("mlflow") is not None


def test_disabled_tracker_is_noop():
    t = Tracker.disabled()
    assert t.log_run("x", params={"a": 1}) is None and t.runs_frame() is None


def test_flatten():
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}


@pytest.mark.skipif(HAS_MLFLOW, reason="only meaningful when mlflow is missing")
def test_missing_mlflow_degrades_gracefully(tmp_path):
    t = Tracker(True, f"sqlite:///{tmp_path}/m.db", "x", tmp_path / "a")
    assert not t.enabled and t.log_run("r") is None


@pytest.mark.skipif(not HAS_MLFLOW, reason="mlflow not installed")
def test_mlflow_roundtrip(tmp_path):
    t = Tracker(True, f"sqlite:///{(tmp_path / 'm.db').as_posix()}", "driftguard-test", tmp_path / "art", strict=True)
    run_id = t.log_run("unit", tags={"k": "v"}, params={"p": {"q": 1}}, metrics={"m": 0.5, "bad": float("nan")},
                       dicts={"d.json": {"x": 1}})
    assert run_id
    runs = t.runs_frame()
    assert run_id in set(runs["run_id"]) and runs.loc[runs["run_id"] == run_id, "metrics.m"].iloc[0] == 0.5


@pytest.mark.skipif(not HAS_MLFLOW, reason="mlflow not installed")
def test_pipeline_logs_to_mlflow(settings, tmp_path):
    from dataclasses import replace

    from driftguard.data.simulation import ScenarioConfig
    from driftguard.pipelines.pipeline import DriftGuardPipeline

    s = replace(settings, mlflow_mode="true", mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'p.db').as_posix()}")
    p = DriftGuardPipeline(s)
    p.ensure_baseline()
    evs = p.run_scenario(ScenarioConfig(name="sudden", seed=0), 8)
    assert p.current_record().mlflow_run_id
    assert all(e["mlflow_run_id"] for e in evs if e["status"] == "ok")
