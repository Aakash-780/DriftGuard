"""End-to-end monitoring and adaptive-retraining orchestration.

For every incoming batch (``process_batch``):

 1. validate schema / values (invalid batches are recorded and stopped here)
 2. skip batches that were already processed (content fingerprint)  -> no duplicate retraining
 3. run the configured drift detectors and the prediction-drift check
 4. record drift results
 5. evaluate the deployed model if labels are present
 6. apply the configurable retraining rules
 7-11. if triggered: train a candidate on (train split + older-window rows), validate it on a held-out
       slice of the recent window and on the fixed anchor split, compare with the deployed model,
       promote only if every gate passes, otherwise keep the deployed model untouched
12. append everything to the event history (and MLflow when available)

Training / validation / test separation
 * ``train``     baseline fit data (re-used in candidate fits)
 * ``reference`` fixed anchor validation set + initial drift reference (never trained on)
 * window rows   partitioned by row-id hash into candidate-train vs recent-holdout (disjoint by id)
 * ``test``      never used by any decision; only the experiments read it
"""
from __future__ import annotations

import hashlib
import shutil
import threading
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from driftguard.core.config import Settings
from driftguard.core.logging import get_logger
from driftguard.core.tracking import Tracker
from driftguard.data.loader import make_splits
from driftguard.data.partition import content_row_ids, holdout_mask
from driftguard.data.simulation import Batch, DriftSimulator, ScenarioConfig
from driftguard.data.validation import FeatureSchema, align_columns, validate_batch, validate_features
from driftguard.models.evaluation import compute_metrics, evaluate_model, positive_proba
from driftguard.models.promotion import evaluate_candidate
from driftguard.models.registry import ModelRecord, ModelRegistry
from driftguard.models.training import get_baseline_model, model_params, train_model
from driftguard.monitoring.detectors import build_detectors, prediction_drift
from driftguard.pipelines.rules import evaluate_rules
from driftguard.pipelines.store import BufferStore, HistoryStore, StateStore

log = get_logger("pipeline")


class BatchValidationError(ValueError):
    def __init__(self, report):
        super().__init__("; ".join(report.errors))
        self.report = report


class NoModelError(RuntimeError):
    pass


class InsufficientDataError(RuntimeError):
    pass


class DriftGuardPipeline:
    def __init__(self, settings: Settings | None = None, *, tracker: Tracker | None = None):
        self.settings = settings or Settings.from_env()
        s = self.settings
        self.paths = s.paths
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.splits = make_splits(s.data)
        self.schema = FeatureSchema.from_reference(self.splits.X_ref)
        self.features = list(self.schema.feature_names)
        self.registry = ModelRegistry(self.paths.registry_dir)
        self.history = HistoryStore(self.paths.history_file)
        self.state_store = StateStore(self.paths.state_file)
        self.buffer = BufferStore(self.paths.buffer_file, self.features, s.retrain.max_buffer_rows)
        self.tracker = tracker if tracker is not None else Tracker.from_settings(s)
        self._lock = threading.RLock()
        self._model = None
        self._record: ModelRecord | None = None
        self._detectors: list = []
        self._pred_ref: np.ndarray | None = None
        self._last_batch: tuple[pd.DataFrame, pd.Series | None, dict] | None = None
        self._reference = self._load_reference()
        if self.registry.current_version() is not None:
            self._load_current()
            self._refit_monitors()

    def reload(self) -> None:
        """Sync pipeline state with disk artifacts (history, state, deployed model, reference)."""
        with self._lock:
            self.history = HistoryStore(self.paths.history_file)
            self.state_store = StateStore(self.paths.state_file)
            cur_v = self.registry.current_version()
            if cur_v is not None and (self._record is None or self._record.version != cur_v):
                self._load_current()
                self._reference = self._load_reference()
                self._refit_monitors()

    # ------------------------------------------------------------------ properties
    @property
    def state(self):
        return self.state_store.state

    @property
    def reference(self) -> pd.DataFrame:
        return self._reference

    def current_model(self):
        self._require_model()
        return self._model

    def current_record(self) -> ModelRecord:
        self._require_model()
        return self._record  # type: ignore[return-value]

    def _require_model(self) -> None:
        if self._model is None:
            raise NoModelError("no model is deployed; train a baseline first (ensure_baseline)")

    # ------------------------------------------------------------------ setup
    def _load_reference(self) -> pd.DataFrame:
        f = self.paths.reference_file
        if f.exists():
            try:
                df = pd.read_csv(f)
                if list(df.columns) == self.features:
                    return df
            except Exception:  # pragma: no cover
                log.warning("could not read %s; using initial reference split", f)
        return self.splits.X_ref.reset_index(drop=True)

    def _load_current(self) -> None:
        self._model, self._record = self.registry.load_current()

    def _refit_monitors(self) -> None:
        self._detectors = build_detectors(self.settings.detectors, self.settings.drift, seed=self.settings.data.seed)
        for d in self._detectors:
            d.fit(self._reference)
        self._pred_ref = positive_proba(self._model, self._reference)

    def ensure_baseline(self) -> ModelRecord:
        """Train, register and deploy the baseline model if nothing is deployed yet."""
        with self._lock:
            if self.registry.current_version() is not None:
                if self._model is None:
                    self._load_current()
                    self._refit_monitors()
                return self._record  # type: ignore[return-value]
            s = self.settings
            model = get_baseline_model(s.data, s.model)
            ref_m = evaluate_model(model, self.splits.X_ref, self.splits.y_ref)
            test_m = evaluate_model(model, self.splits.X_test, self.splits.y_test)
            rec = self.registry.register(
                model, params=model_params(s.model, s.data.seed),
                metrics={"reference": ref_m, "test": test_m}, reference_metrics=ref_m,
                train_rows=len(self.splits.X_train), data_fingerprint=self.splits.train_fingerprint(),
                trigger_reason="initial baseline", notes="baseline trained on the train split",
            )
            rec = self.registry.promote(rec.version)
            run_id = self.tracker.log_baseline(rec, model, s.to_dict())
            if run_id:
                rec = self.registry.update(rec.version, mlflow_run_id=run_id)
            self._load_current()
            self._refit_monitors()
            log.info("baseline model v%d deployed (reference F1=%.3f)", rec.version, ref_m["f1"])
            return rec

    # ------------------------------------------------------------------ prediction
    def predict(self, X: pd.DataFrame) -> dict:
        with self._lock:
            self._require_model()
            rep = validate_features(X, self.schema, min_rows=1)
            if not rep.valid:
                raise BatchValidationError(rep)
            Xa = align_columns(X, self.schema)
            proba = positive_proba(self._model, Xa)
            pred = self._model.predict(Xa)
            return {
                "model_version": self._record.version,  # type: ignore[union-attr]
                "predictions": [int(p) for p in pred],
                "probabilities": [float(p) for p in proba],
                "labels": ["malignant" if p == 1 else "benign" for p in pred],
            }

    # ------------------------------------------------------------------ monitoring
    def process_batch(self, batch: Batch, allow_retrain: bool = True) -> dict:
        with self._lock:
            self._require_model()
            t0 = time.perf_counter()
            s, rcfg = self.settings, self.settings.retrain
            v_before = self._record.version  # type: ignore[union-attr]
            report = validate_batch(batch.X, batch.y, self.schema, min_rows=rcfg.min_batch_rows)
            base = {
                "event_type": "monitoring", "batch_index": batch.batch_index, "scenario": batch.scenario,
                "intensity": batch.intensity, "n_rows": int(len(batch.X)), "validation": report.to_dict(),
                "model_version": v_before,
            }
            if not report.valid:
                log.warning("batch rejected by validation: %s", report.errors)
                return self.history.append({**base, "status": "invalid", "action": "none",
                                            "batch_fingerprint": None,
                                            "message": "batch failed validation and was not processed"})
            X = align_columns(batch.X, self.schema)
            y = None if batch.y is None else pd.Series(np.asarray(batch.y), dtype="int64")
            fp = Batch(X=X, y=y).fingerprint()
            if self.history.has_batch(fp):
                log.info("batch %s already processed; skipping (no state change, no retraining)", fp)
                return {**base, "status": "duplicate_skipped", "action": "none", "batch_fingerprint": fp,
                        "message": "identical batch was already processed; nothing was recorded or retrained"}
            ids = batch.row_ids if (batch.row_ids is not None and len(batch.row_ids) == len(X)) else content_row_ids(X)

            st = self.state
            st.batch_counter += 1
            seq = st.batch_counter

            # --- drift detection
            results: list[dict] = []
            for det in self._detectors:
                try:
                    results.append(det.detect(X).to_dict())
                except Exception as exc:
                    log.exception("detector %s failed", det.name)
                    results.append({"detector": det.name, "error": f"{type(exc).__name__}: {exc}", "drifted": None})
            ok = [r for r in results if "error" not in r]
            n_flagged = int(sum(bool(r["drifted"]) for r in ok))
            need = min(rcfg.min_detectors_drifted, max(1, len(ok)))
            dataset_drift = bool(ok) and n_flagged >= need

            proba = positive_proba(self._model, X)
            pdrift = prediction_drift(self._pred_ref, proba, s.drift.prediction_alpha)

            # --- performance (only if labels exist)
            perf = baseline_metric = drop = degraded = None
            if y is not None:
                perf = compute_metrics(y, self._model.predict(X), proba)
                baseline_metric = self._record.reference_metrics.get(rcfg.performance_metric)  # type: ignore[union-attr]
                cur = perf.get(rcfg.performance_metric)
                if baseline_metric is not None and cur is not None:
                    drop = float(baseline_metric - cur)
                    degraded = bool(drop >= rcfg.max_metric_drop - 1e-12)
                self.buffer.add(X, y, ids, fp, seq)

            st.drift_streak = st.drift_streak + 1 if dataset_drift else 0
            if degraded is True:
                st.perf_streak += 1
            elif degraded is False:
                st.perf_streak = 0  # unlabeled batches leave the streak unchanged

            since = None if st.last_retrain_batch is None else seq - st.last_retrain_batch
            labeled_rows = len(self.buffer.window(rcfg.window_batches))
            decision = evaluate_rules(
                rcfg, drift_streak=st.drift_streak, perf_streak=st.perf_streak, labeled_rows=labeled_rows,
                batches_since_retrain=since,
                drift_detail=f"{n_flagged}/{len(ok)} detectors flagged the latest batch.",
                perf_detail=(f"{rcfg.performance_metric}={perf.get(rcfg.performance_metric):.3f} vs baseline "
                             f"{baseline_metric:.3f}." if (perf and baseline_metric is not None
                                                          and perf.get(rcfg.performance_metric) is not None) else ""),
            )

            action, retrain_ev = "none", None
            if decision.trigger:
                if allow_retrain:
                    retrain_ev = self._retrain(reasons=decision.reasons, seq=seq, forced=False)
                    action = {"promoted": "retrain_promoted", "rejected": "retrain_rejected"}.get(
                        retrain_ev["status"], "retrain_skipped")
                else:
                    action = "retrain_suppressed"

            event = {
                **base, "status": "ok", "batch_fingerprint": fp, "batch_seq": seq, "labeled": y is not None,
                "drift": {"detectors": results, "n_flagged": n_flagged, "n_active": len(ok),
                          "dataset_drift": dataset_drift, "min_detectors_drifted": need},
                "prediction_drift": pdrift, "performance": perf, "performance_baseline": baseline_metric,
                "performance_drop": drop, "performance_degraded": degraded,
                "trigger": decision.to_dict(), "action": action,
                "retrain_event_id": retrain_ev["event_id"] if retrain_ev else None,
                "model_version_after": self._record.version,  # type: ignore[union-attr]
                "streaks": {"drift": st.drift_streak, "performance": st.perf_streak},
            }
            event["mlflow_run_id"] = self.tracker.log_monitoring(
                event, {d.name: d.config() for d in self._detectors})
            event["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            stored = self.history.append(event)
            self.state_store.save()
            self._remember_batch(X, y, stored)
            return stored

    def _remember_batch(self, X: pd.DataFrame, y, event: dict) -> None:
        meta = {k: event.get(k) for k in ("batch_seq", "scenario", "batch_index", "intensity", "batch_fingerprint")}
        self._last_batch = (X, y, meta)
        out = X.copy()
        if y is not None:
            out["label"] = y.to_numpy()
        out.to_csv(self.paths.last_batch_file, index=False)

    def last_batch(self):
        """(features, labels-or-None, meta) of the most recent processed batch, or ``None``."""
        with self._lock:
            if self._last_batch is not None:
                return self._last_batch
            f = self.paths.last_batch_file
            if f.exists():
                df = pd.read_csv(f)
                y = df.pop("label").astype(int) if "label" in df.columns else None
                last = next((e for e in reversed(self.history.all())
                             if e.get("event_type") == "monitoring" and e.get("status") == "ok"), {})
                meta = {k: last.get(k) for k in ("batch_seq", "scenario", "batch_index", "intensity", "batch_fingerprint")}
                return df[self.features], y, meta
            return None

    # ------------------------------------------------------------------ retraining
    def _retrain(self, *, reasons: list[str], seq: int, forced: bool) -> dict:
        s, rcfg = self.settings, self.settings.retrain
        st = self.state
        cur = self._record  # type: ignore[assignment]
        base = {"event_type": "retrain", "forced": forced, "trigger_reasons": list(reasons), "batch_seq": seq,
                "previous_version": cur.version, "window_batches": rcfg.window_batches}
        window = self.buffer.window(rcfg.window_batches)
        if len(window) < rcfg.min_labeled_rows:
            return self.history.append({**base, "status": "skipped",
                                        "message": f"only {len(window)} labeled rows buffered (< {rcfg.min_labeled_rows})"})
        wfp = hashlib.sha256("|".join(sorted(window["batch_fp"].astype(str).unique())).encode()).hexdigest()[:16]
        if st.last_window_fp == wfp:
            return self.history.append({**base, "status": "skipped",
                                        "message": "identical training window was already used by the previous retraining attempt"})

        hold = holdout_mask(window["row_id"], rcfg.partition_salt, rcfg.holdout_fraction)
        w_train, w_hold = window[~hold], window[hold]
        if rcfg.candidate_data == "window_only":
            X_fit = w_train[self.features].reset_index(drop=True)
            y_fit = w_train["label"].reset_index(drop=True).astype("int64")
        else:
            X_fit = pd.concat([self.splits.X_train.reset_index(drop=True), w_train[self.features]], ignore_index=True)
            y_fit = pd.concat([self.splits.y_train.reset_index(drop=True), w_train["label"].reset_index(drop=True)],
                              ignore_index=True)
        if y_fit.nunique() < 2:
            return self.history.append({**base, "status": "skipped",
                                        "message": "candidate training data contains a single class; not trained"})
        # Leakage guard: no row id may be on both sides.
        assert set(w_train["row_id"]).isdisjoint(set(w_hold["row_id"])), "train/holdout row-id overlap"

        candidate = train_model(X_fit, y_fit, s.model, s.data.seed)
        decision = evaluate_candidate(
            self._model, candidate, (w_hold[self.features], w_hold["label"].to_numpy()),
            (self.splits.X_ref, self.splits.y_ref), s.promotion,
        )
        rec = self.registry.register(
            candidate, params=model_params(s.model, s.data.seed),
            metrics=decision.candidate_metrics, reference_metrics=decision.candidate_metrics.get("holdout", {}),
            train_rows=len(X_fit), data_fingerprint=wfp, trigger_reason=" | ".join(reasons)[:500],
            decision=decision.to_dict(),
        )
        reference_refreshed = False
        if decision.promote:
            self.registry.promote(rec.version)
            self._model, self._record = candidate, self.registry.get(rec.version)
            self._reference = window[self.features].reset_index(drop=True)
            self._reference.to_csv(self.paths.reference_file, index=False)
            self._refit_monitors()
            reference_refreshed = True
            st.promotions += 1
            status = "promoted"
            log.info("candidate v%d PROMOTED (%s)", rec.version, "; ".join(decision.reasons))
        else:
            self.registry.reject(rec.version, "; ".join(decision.reasons))
            status = "rejected"
            log.info("candidate v%d REJECTED, v%d stays deployed (%s)", rec.version, cur.version,
                     "; ".join(decision.reasons))
        st.retrain_attempts += 1
        st.last_retrain_batch = seq
        st.drift_streak = st.perf_streak = 0
        st.last_window_fp = wfp

        ev = {**base, "status": status, "candidate_version": rec.version, "decision": decision.to_dict(),
              "candidate_metrics": decision.candidate_metrics, "current_metrics": decision.current_metrics,
              "train_rows": len(X_fit), "holdout_rows": int(len(w_hold)),
              "reference_refreshed": reference_refreshed, "params": rec.params,
              "candidate_data": rcfg.candidate_data,
              "deployed_version_after": self._record.version}
        run_id = self.tracker.log_retrain(ev, candidate, asdict(rcfg), asdict(s.promotion))
        if run_id:
            ev["mlflow_run_id"] = run_id
            updated_rec = self.registry.update(rec.version, mlflow_run_id=run_id)
            if decision.promote:
                self._record = updated_rec
        stored = self.history.append(ev)
        self.state_store.save()
        return stored

    def retrain(self, *, force: bool = False, reason: str | None = None) -> dict:
        """Manual entry point. Without ``force`` the configured rules decide; with it they are bypassed
        (promotion gates still apply)."""
        with self._lock:
            self._require_model()
            st = self.state
            if not force:
                since = None if st.last_retrain_batch is None else st.batch_counter - st.last_retrain_batch
                d = evaluate_rules(
                    self.settings.retrain, drift_streak=st.drift_streak, perf_streak=st.perf_streak,
                    labeled_rows=len(self.buffer.window(self.settings.retrain.window_batches)),
                    batches_since_retrain=since,
                )
                if not d.trigger:
                    return {"event_type": "retrain", "status": "not_triggered", "trigger": d.to_dict(),
                            "message": "retraining rules are not satisfied (use force=true to override)"}
                reasons = d.reasons
            else:
                reasons = [f"manual force: {reason or 'no reason given'}"]
            return self._retrain(reasons=reasons, seq=st.batch_counter, forced=force)

    # ------------------------------------------------------------------ scenarios
    def run_scenario(self, scenario: ScenarioConfig, n_batches: int, allow_retrain: bool = True) -> list[dict]:
        sim = DriftSimulator(self.splits, scenario, self.settings.data.batch_size)
        return [self.process_batch(sim.batch(i), allow_retrain=allow_retrain) for i in range(n_batches)]

    # ------------------------------------------------------------------ introspection
    def model_info(self) -> dict:
        with self._lock:
            self._require_model()
            return {
                **self._record.to_dict(),  # type: ignore[union-attr]
                "n_features": len(self.features), "feature_names": self.features,
                "positive_class": "malignant (label 1)", "detectors": list(self.settings.detectors),
                "retrain_policy": self.settings.retrain.policy,
            }

    def metrics_summary(self) -> dict:
        with self._lock:
            evs = self.history.all()
            mon = [e for e in evs if e["event_type"] == "monitoring" and e.get("status") == "ok"]
            ret = [e for e in evs if e["event_type"] == "retrain"]
            count = lambda status: sum(1 for e in ret if e.get("status") == status)  # noqa: E731
            latest = mon[-1] if mon else None
            return {
                "model_version": self._record.version if self._record else None,
                "baseline_reference_metrics": self._record.reference_metrics if self._record else {},
                "counts": {
                    "monitoring_runs": len(mon),
                    "invalid_batches": sum(1 for e in evs if e.get("status") == "invalid"),
                    "retrain_attempts": self.state.retrain_attempts,
                    "promoted": count("promoted"), "rejected": count("rejected"), "skipped": count("skipped"),
                },
                "latest": None if latest is None else {
                    "batch_seq": latest["batch_seq"], "scenario": latest["scenario"],
                    "dataset_drift": latest["drift"]["dataset_drift"], "n_flagged": latest["drift"]["n_flagged"],
                    "performance": latest["performance"], "action": latest["action"],
                    "prediction_drift": latest["prediction_drift"],
                },
                "state": asdict(self.state),
            }

    def reset(self) -> ModelRecord:
        """Delete all runtime state (registry, history, buffer, reference) and redeploy a fresh baseline.
        Paths come only from settings; MLflow's database is left untouched."""
        with self._lock:
            self.history.clear()
            self.state_store.reset()
            self.buffer.clear()
            for f in (self.paths.reference_file, self.paths.last_batch_file):
                if f.exists():
                    f.unlink()
            shutil.rmtree(self.paths.registry_dir, ignore_errors=True)
            self._model = self._record = None
            self._last_batch = None
            self._detectors, self._pred_ref = [], None
            self._reference = self.splits.X_ref.reset_index(drop=True)
            return self.ensure_baseline()
