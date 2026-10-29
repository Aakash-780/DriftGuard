import pytest

from driftguard.core.config import RetrainConfig
from driftguard.pipelines.rules import evaluate_rules


def run(policy, drift=0, perf=0, rows=400, since=None, **kw):
    cfg = RetrainConfig(policy=policy, **kw)
    return evaluate_rules(cfg, drift_streak=drift, perf_streak=perf, labeled_rows=rows, batches_since_retrain=since)


@pytest.mark.parametrize("policy,drift,perf,expected", [
    ("never", 5, 5, False),
    ("performance_only", 5, 0, False), ("performance_only", 0, 2, True),
    ("drift_only", 2, 0, True), ("drift_only", 1, 5, False),
    ("drift_and_performance", 2, 0, False), ("drift_and_performance", 0, 2, False), ("drift_and_performance", 2, 2, True),
    ("drift_or_performance", 2, 0, True), ("drift_or_performance", 0, 2, True), ("drift_or_performance", 1, 1, False),
])
def test_policy_truth_table(policy, drift, perf, expected):
    assert run(policy, drift, perf).trigger is expected


def test_cooldown_blocks_then_releases():
    blocked = run("drift_only", drift=3, since=1, cooldown_batches=3)
    assert not blocked.trigger and "cooldown" in blocked.blocked_by[0]
    assert run("drift_only", drift=3, since=3, cooldown_batches=3).trigger


def test_insufficient_labeled_rows_blocks():
    d = run("drift_only", drift=3, rows=50, min_labeled_rows=100)
    assert not d.trigger and "insufficient" in d.blocked_by[0]


def test_reasons_mention_policy_and_signal():
    d = run("drift_or_performance", drift=2, perf=2)
    assert len(d.reasons) == 2 and all("drift_or_performance" in r for r in d.reasons)


def test_min_consecutive_is_configurable():
    assert not run("drift_only", drift=2, min_consecutive=3).trigger
    assert run("drift_only", drift=3, min_consecutive=3).trigger
