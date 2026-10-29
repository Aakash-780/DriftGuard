"""Candidate-vs-current promotion gate.

A candidate is promoted only if ALL checks pass. Everything is evaluated on data the candidate
never trained on: a recent-window holdout (row ids partitioned away from the training rows) and the
fixed reference ("anchor") split.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from driftguard.core.config import PromotionConfig
from driftguard.models.evaluation import evaluate_model


@dataclass
class PromotionDecision:
    promote: bool
    reasons: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    candidate_metrics: dict = field(default_factory=dict)
    current_metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "promote": self.promote,
            "reasons": self.reasons,
            "checks": self.checks,
            "candidate_metrics": self.candidate_metrics,
            "current_metrics": self.current_metrics,
        }


def evaluate_candidate(current, candidate, holdout, anchor, cfg: PromotionConfig) -> PromotionDecision:
    """``holdout`` and ``anchor`` are ``(X, y)`` tuples."""
    Xh, yh = holdout
    Xa, ya = anchor
    yh = np.asarray(yh)
    n = len(yh)
    n_pos = int(yh.sum())
    n_neg = n - n_pos
    checks: dict = {}

    size_ok = n >= cfg.min_holdout_rows and min(n_pos, n_neg) >= cfg.min_class_rows
    checks["holdout_size"] = {
        "passed": bool(size_ok), "value": {"rows": n, "positives": n_pos, "negatives": n_neg},
        "threshold": {"min_rows": cfg.min_holdout_rows, "min_per_class": cfg.min_class_rows},
    }
    if not size_ok:
        return PromotionDecision(
            False,
            [f"holdout too small or single-class (rows={n}, pos={n_pos}, neg={n_neg}); cannot validate"],
            checks,
        )

    cur_h = evaluate_model(current, Xh, yh)
    cand_h = evaluate_model(candidate, Xh, yh)
    cur_a = evaluate_model(current, Xa, ya)
    cand_a = evaluate_model(candidate, Xa, ya)

    improvement = cand_h["f1"] - cur_h["f1"]
    checks["f1_improvement"] = {
        "passed": bool(improvement >= cfg.min_improvement - 1e-12),
        "value": improvement, "threshold": cfg.min_improvement,
    }
    checks["min_candidate_f1"] = {
        "passed": bool(cand_h["f1"] >= cfg.min_candidate_f1),
        "value": cand_h["f1"], "threshold": cfg.min_candidate_f1,
    }
    recall_delta = cand_h["recall"] - cur_h["recall"]
    checks["recall_safety"] = {
        "passed": bool(recall_delta >= -cfg.max_recall_drop - 1e-12),
        "value": recall_delta, "threshold": -cfg.max_recall_drop,
    }
    ref_delta = cand_a["f1"] - cur_a["f1"]
    checks["reference_regression"] = {
        "passed": bool(ref_delta >= -cfg.max_reference_f1_drop - 1e-12),
        "value": ref_delta, "threshold": -cfg.max_reference_f1_drop,
    }
    failed = [k for k, v in checks.items() if not v["passed"]]
    reasons = (
        ["all promotion checks passed"] if not failed
        else [f"failed check '{k}': value={checks[k]['value']:.4f} vs threshold={checks[k]['threshold']}" for k in failed]
    )
    return PromotionDecision(
        promote=not failed, reasons=reasons, checks=checks,
        candidate_metrics={"holdout": cand_h, "reference": cand_a},
        current_metrics={"holdout": cur_h, "reference": cur_a},
    )
