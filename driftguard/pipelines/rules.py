"""Configurable retraining trigger rules.

Two independent signals are formed from consecutive-batch persistence:

* ``drift_signal``  -- at least ``min_detectors_drifted`` detectors flagged the batch, for
  ``min_consecutive`` consecutive batches.
* ``perf_signal``   -- the labeled batch metric fell at least ``max_metric_drop`` below the deployed
  model's baseline, for ``min_consecutive`` consecutive batches.

A *policy* combines them. Guards (cooldown, labeled-data availability) can block a trigger.
Drift alone never proves degradation; ``drift_only`` exists for comparison, not as a recommendation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from driftguard.core.config import RetrainConfig


@dataclass
class TriggerDecision:
    trigger: bool
    reasons: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    drift_signal: bool = False
    perf_signal: bool = False

    def to_dict(self) -> dict:
        return {
            "trigger": self.trigger, "reasons": self.reasons, "blocked_by": self.blocked_by,
            "drift_signal": self.drift_signal, "perf_signal": self.perf_signal,
        }


def evaluate_rules(
    cfg: RetrainConfig,
    *,
    drift_streak: int,
    perf_streak: int,
    labeled_rows: int,
    batches_since_retrain: int | None,
    drift_detail: str = "",
    perf_detail: str = "",
) -> TriggerDecision:
    drift_signal = drift_streak >= cfg.min_consecutive
    perf_signal = perf_streak >= cfg.min_consecutive
    policy = cfg.policy

    if policy == "never":
        want = False
    elif policy == "performance_only":
        want = perf_signal
    elif policy == "drift_only":
        want = drift_signal
    elif policy == "drift_and_performance":
        want = drift_signal and perf_signal
    elif policy == "drift_or_performance":
        want = drift_signal or perf_signal
    else:  # pragma: no cover - guarded by Settings
        raise ValueError(f"unknown policy {policy!r}")

    reasons: list[str] = []
    if want:
        if drift_signal:
            reasons.append(f"data drift persisted for {drift_streak} batch(es). {drift_detail}".strip())
        if perf_signal:
            reasons.append(f"performance degradation persisted for {perf_streak} batch(es). {perf_detail}".strip())
        reasons = [f"policy '{policy}': " + r for r in reasons]

    blocked: list[str] = []
    if want:
        if batches_since_retrain is not None and batches_since_retrain < cfg.cooldown_batches:
            blocked.append(f"cooldown: {batches_since_retrain} < {cfg.cooldown_batches} batches since last retrain attempt")
        if labeled_rows < cfg.min_labeled_rows:
            blocked.append(f"insufficient labeled data: {labeled_rows} < {cfg.min_labeled_rows} rows")
    return TriggerDecision(
        trigger=bool(want and not blocked), reasons=reasons, blocked_by=blocked,
        drift_signal=drift_signal, perf_signal=perf_signal,
    )
