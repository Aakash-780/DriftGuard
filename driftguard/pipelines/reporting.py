"""Turn raw history events into tidy DataFrames (shared by the dashboard, experiments and tests)."""
from __future__ import annotations

import pandas as pd

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def monitoring_frame(events: list[dict]) -> pd.DataFrame:
    rows = []
    for e in events:
        if e.get("event_type") != "monitoring" or e.get("status") != "ok":
            continue
        row = {
            "event_id": e.get("event_id"), "timestamp": e.get("timestamp"), "batch_seq": e["batch_seq"],
            "scenario": e["scenario"], "batch_index": e["batch_index"], "intensity": e["intensity"],
            "model_version": e["model_version"], "n_flagged": e["drift"]["n_flagged"],
            "dataset_drift": e["drift"]["dataset_drift"],
            "prediction_drift": e["prediction_drift"]["drifted"],
            "performance_drop": e.get("performance_drop"), "degraded": e.get("performance_degraded"),
            "action": e["action"], "labeled": e["labeled"],
        }
        for d in e["drift"]["detectors"]:
            if "error" in d:
                continue
            row[f"flag_{d['detector']}"] = d["drifted"]
            row[f"score_{d['detector']}"] = d["score"]
        for m in METRICS:
            row[m] = (e.get("performance") or {}).get(m)
        rows.append(row)
    return pd.DataFrame(rows)


def retrain_frame(events: list[dict]) -> pd.DataFrame:
    rows = []
    for e in events:
        if e.get("event_type") != "retrain" or e.get("status") == "not_triggered":
            continue
        cm = (e.get("candidate_metrics") or {}).get("holdout") or {}
        um = (e.get("current_metrics") or {}).get("holdout") or {}
        rows.append({
            "event_id": e.get("event_id"), "timestamp": e.get("timestamp"), "status": e["status"],
            "forced": e.get("forced"), "batch_seq": e.get("batch_seq"),
            "previous_version": e.get("previous_version"), "candidate_version": e.get("candidate_version"),
            "trigger_reasons": " | ".join(e.get("trigger_reasons", [])),
            "current_holdout_f1": um.get("f1"), "candidate_holdout_f1": cm.get("f1"),
            "current_holdout_recall": um.get("recall"), "candidate_holdout_recall": cm.get("recall"),
            "train_rows": e.get("train_rows"), "holdout_rows": e.get("holdout_rows"),
            "decision": "; ".join((e.get("decision") or {}).get("reasons", [])) or e.get("message", ""),
            "mlflow_run_id": e.get("mlflow_run_id"),
        })
    return pd.DataFrame(rows)


def detector_frame(event: dict) -> pd.DataFrame:
    rows = []
    for d in event["drift"]["detectors"]:
        if "error" in d:
            rows.append({"detector": d["detector"], "drifted": None, "score": None, "threshold": None,
                         "features_flagged": None, "note": d["error"]})
            continue
        rows.append({"detector": d["detector"], "drifted": d["drifted"], "score": d["score"],
                     "threshold": d["threshold"], "features_flagged": d.get("n_drifted_features"), "note": ""})
    return pd.DataFrame(rows)


def feature_frame(event: dict) -> pd.DataFrame:
    """Per-feature scores/flags for every detector that reports them (KS statistic, PSI, ...)."""
    out: dict[str, dict] = {}
    for d in event["drift"]["detectors"]:
        if "error" in d or not d.get("feature_scores"):
            continue
        for f, v in d["feature_scores"].items():
            out.setdefault(f, {})[f"{d['detector']}_score"] = v
            out[f][f"{d['detector']}_flag"] = (d.get("feature_drifted") or {}).get(f)
    return pd.DataFrame.from_dict(out, orient="index").rename_axis("feature").reset_index()
