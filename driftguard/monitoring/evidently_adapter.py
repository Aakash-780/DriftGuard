"""Optional Evidently AI adapter (legacy 0.4.x API, pinned to ``evidently==0.4.40``).

IMPORTANT: this module was written against the documented 0.4.x API
(``evidently.report.Report``, ``evidently.metric_preset.DataDriftPreset``, ``Report.run``,
``Report.as_dict``, ``Report.save_html``) but could NOT be executed in the environment where the
project was authored (no network). It is therefore opt-in (``DRIFTGUARD_DETECTORS=ks,psi,domain_classifier,evidently``)
and parses the result defensively, raising a clear error if the structure differs.
Evidently's numeric default for small samples is a KS-based test, so expect it to behave much like ``ks``;
the research experiment can quantify the agreement.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from driftguard.core.config import DriftConfig
from driftguard.monitoring.detectors import DriftDetector, DriftResult


def _import():
    try:
        from evidently.metric_preset import DataDriftPreset  # type: ignore
        from evidently.report import Report  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "detector 'evidently' requested but evidently (0.4.x legacy API) is not installed; "
            "run `pip install evidently==0.4.40` or remove it from DRIFTGUARD_DETECTORS"
        ) from exc
    return Report, DataDriftPreset


def _run_report(reference: pd.DataFrame, current: pd.DataFrame, drift_share: float):
    Report, DataDriftPreset = _import()
    report = Report(metrics=[DataDriftPreset(drift_share=drift_share)])
    report.run(reference_data=reference, current_data=current)
    return report


class EvidentlyDetector(DriftDetector):
    name = "evidently"

    def __init__(self, cfg: DriftConfig):
        _import()
        self.cfg = cfg
        self._ref: pd.DataFrame | None = None

    def fit(self, reference: pd.DataFrame) -> "EvidentlyDetector":
        self._ref = reference.reset_index(drop=True).copy()
        return self

    def detect(self, current: pd.DataFrame) -> DriftResult:
        report = _run_report(self._ref, current.reset_index(drop=True), self.cfg.evidently_drift_share)
        as_dict = report.as_dict()
        dataset, table = None, None
        for m in as_dict.get("metrics", []):
            res = m.get("result", {})
            if "dataset_drift" in res and "share_of_drifted_columns" in res:
                dataset = res
            if "drift_by_columns" in res:
                table = res["drift_by_columns"]
        if dataset is None:
            raise RuntimeError("unexpected Evidently report structure: dataset-drift result not found")
        feature_scores = feature_flags = None
        if table:
            feature_scores = {c: float(v.get("drift_score", float("nan"))) for c, v in table.items()}
            feature_flags = {c: bool(v.get("drift_detected", False)) for c, v in table.items()}
        share = float(dataset["share_of_drifted_columns"])
        return DriftResult(
            detector=self.name, drifted=bool(dataset["dataset_drift"]), score=share,
            threshold=float(self.cfg.evidently_drift_share), share_drifted=share,
            n_drifted_features=int(dataset.get("number_of_drifted_columns", 0)),
            feature_scores=feature_scores, feature_drifted=feature_flags,
            details={"source": "evidently DataDriftPreset (per-column drift_score = p-value or distance)"},
        )

    def config(self) -> dict:
        return {"drift_share": self.cfg.evidently_drift_share, "preset": "DataDriftPreset (library defaults)"}


def save_html_report(reference: pd.DataFrame, current: pd.DataFrame, path: Path, drift_share: float = 0.5) -> Path:
    """Write a full Evidently HTML report to ``path`` (a path chosen by the application, never by API users)."""
    report = _run_report(reference.reset_index(drop=True), current.reset_index(drop=True), drift_share)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(path))
    return path
