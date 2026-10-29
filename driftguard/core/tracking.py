"""MLflow tracking wrapper.

MLflow is *optional at runtime*: if it is missing or fails, tracking silently degrades to a no-op
(a warning is logged) and the pipeline keeps working -- the local registry and ``history.jsonl`` remain
the source of truth. The default backend is a local SQLite file, so no server is needed; run
``mlflow ui --backend-store-uri <uri>`` only to browse it.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from driftguard.core.logging import get_logger
from driftguard.core.utils import to_jsonable

log = get_logger("tracking")


def flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _clean_params(params: dict) -> dict[str, str]:
    return {k: str(v)[:250] for k, v in flatten(params).items()}


def _clean_metrics(metrics: dict) -> dict[str, float]:
    out = {}
    for k, v in flatten(metrics).items():
        if isinstance(v, bool):
            out[k] = float(v)
        elif isinstance(v, (int, float)) and math.isfinite(float(v)):
            out[k] = float(v)
    return out


class Tracker:
    def __init__(self, enabled: bool, tracking_uri: str | None = None,
                 experiment_name: str = "driftguard", artifact_root: Path | None = None,
                 strict: bool = False):
        self.enabled = False
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._mlflow = None
        if not enabled:
            return
        try:
            import mlflow  # noqa: WPS433

            self._mlflow = mlflow
            if artifact_root is not None:
                Path(artifact_root).mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(tracking_uri)
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            if client.get_experiment_by_name(experiment_name) is None:
                loc = Path(artifact_root).resolve().as_uri() if artifact_root is not None else None
                client.create_experiment(experiment_name, artifact_location=loc)
            mlflow.set_experiment(experiment_name)
            self.enabled = True
        except Exception as exc:
            log.warning("MLflow tracking disabled (%s: %s)", type(exc).__name__, exc)
            if strict:
                raise

    # ------------------------------------------------------------ constructors
    @classmethod
    def disabled(cls) -> "Tracker":
        return cls(False)

    @classmethod
    def from_settings(cls, settings) -> "Tracker":
        if settings.mlflow_mode == "false":
            return cls.disabled()
        if settings.mlflow_mode == "auto":
            try:
                import mlflow  # noqa: F401
            except ImportError:
                log.info("mlflow not installed; experiment tracking disabled")
                return cls.disabled()
        return cls(True, settings.mlflow_uri, settings.mlflow_experiment,
                   settings.paths.mlflow_dir / "artifacts")

    # ------------------------------------------------------------ generic run
    def log_run(self, run_name: str, *, tags: dict | None = None, params: dict | None = None,
                metrics: dict | None = None, dicts: dict | None = None, model=None,
                register: bool = False) -> str | None:
        if not self.enabled:
            return None
        mlflow = self._mlflow
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            with mlflow.start_run(run_name=run_name) as run:
                if tags:
                    mlflow.set_tags({k: str(v) for k, v in tags.items()})
                if params:
                    mlflow.log_params(_clean_params(params))
                if metrics:
                    mlflow.log_metrics(_clean_metrics(metrics))
                for fname, payload in (dicts or {}).items():
                    mlflow.log_dict(to_jsonable(payload), fname)
                if model is not None:
                    import mlflow.sklearn  # noqa: WPS433

                    mlflow.sklearn.log_model(
                        model, artifact_path="model",
                        registered_model_name="driftguard-classifier" if register else None,
                    )
                return run.info.run_id
        except Exception as exc:
            log.warning("MLflow logging failed (%s: %s); continuing without tracking", type(exc).__name__, exc)
            return None

    # ------------------------------------------------------------ domain helpers
    def log_baseline(self, record, model, settings_dict: dict) -> str | None:
        return self.log_run(
            f"baseline-v{record.version}",
            tags={"driftguard.kind": "baseline", "driftguard.version": record.version, "driftguard.decision": "promoted"},
            params={"model": record.params, "data_fingerprint": record.data_fingerprint,
                    "train_rows": record.train_rows, "config": settings_dict},
            metrics={"reference": record.metrics.get("reference", {}), "test": record.metrics.get("test", {})},
            model=model, register=True,
        )

    def log_monitoring(self, event: dict, detector_configs: dict) -> str | None:
        det = {d["detector"]: d for d in event.get("drift", {}).get("detectors", []) if "error" not in d}
        metrics: dict[str, Any] = {"n_flagged": event["drift"]["n_flagged"], "dataset_drift": event["drift"]["dataset_drift"]}
        for name, d in det.items():
            metrics[f"drift_score_{name}"] = d["score"]
            metrics[f"drifted_{name}"] = d["drifted"]
        pd_ = event.get("prediction_drift")
        if pd_:
            metrics["prediction_drift_p"] = pd_["p_value"]
        for k, v in (event.get("performance") or {}).items():
            if k not in ("n", "positive_rate"):
                metrics[f"perf_{k}"] = v
        return self.log_run(
            f"monitor-batch-{event['batch_seq']}",
            tags={"driftguard.kind": "monitoring", "driftguard.scenario": event["scenario"],
                  "driftguard.action": event["action"], "driftguard.model_version": event["model_version"],
                  "driftguard.batch_fingerprint": event["batch_fingerprint"]},
            params={"detectors": detector_configs}, metrics=metrics,
            dicts={"trigger.json": event.get("trigger", {})},
        )

    def log_retrain(self, event: dict, candidate, rules_cfg: dict, promotion_cfg: dict) -> str | None:
        dec = event.get("decision") or {}
        cm = event.get("candidate_metrics") or {}
        metrics = {f"candidate_holdout_{k}": v for k, v in (cm.get("holdout") or {}).items() if k != "n"}
        metrics.update({f"candidate_reference_{k}": v for k, v in (cm.get("reference") or {}).items() if k != "n"})
        metrics.update({f"current_holdout_{k}": v for k, v in ((event.get("current_metrics") or {}).get("holdout") or {}).items() if k != "n"})
        metrics.update({"train_rows": event.get("train_rows"), "holdout_rows": event.get("holdout_rows")})
        return self.log_run(
            f"retrain-v{event.get('candidate_version')}",
            tags={"driftguard.kind": "retrain", "driftguard.decision": event["status"],
                  "driftguard.version": event.get("candidate_version"), "driftguard.forced": event.get("forced"),
                  "driftguard.trigger_reasons": " | ".join(event.get("trigger_reasons", []))[:500]},
            params={"model": event.get("params", {}), "rules": rules_cfg, "promotion": promotion_cfg},
            metrics=metrics, dicts={"decision.json": dec},
            model=candidate, register=(event["status"] == "promoted"),
        )

    def runs_frame(self, max_results: int = 200):
        """MLflow runs for this experiment as a DataFrame, or ``None`` when unavailable."""
        if not self.enabled:
            return None
        try:
            self._mlflow.set_tracking_uri(self.tracking_uri)
            return self._mlflow.search_runs(
                experiment_names=[self.experiment_name], max_results=max_results, output_format="pandas"
            )
        except Exception as exc:  # pragma: no cover
            log.warning("could not read MLflow runs: %s", exc)
            return None
