"""FastAPI application.

Run:  ``uvicorn driftguard.app.main:app --port 8000``  (interactive docs at /docs)

The API is synchronous by design (local prototype). Retraining a small scikit-learn model takes about a
second; for production use you would move it to a background worker. State is shared with the dashboard
through the artifacts directory -- run only ONE writer (API *or* dashboard actions) at a time.
Clients can never supply file paths or code; only numeric records and enum/int parameters are accepted.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request

from driftguard import __version__
from driftguard.app.schemas import (
    HealthResponse, MonitorRequest, PredictRequest, PredictResponse, RetrainRequest, SimulateRequest,
)
from driftguard.core.config import Settings
from driftguard.core.logging import configure_logging
from driftguard.core.utils import to_jsonable
from driftguard.data.simulation import Batch, DriftSimulator, ScenarioConfig
from driftguard.pipelines.pipeline import (
    BatchValidationError, DriftGuardPipeline, InsufficientDataError, NoModelError,
)


def get_pipeline(request: Request) -> DriftGuardPipeline:
    pipe = getattr(request.app.state, "pipeline", None)
    if pipe is None:
        raise HTTPException(status_code=503, detail="service is not initialised yet")
    return pipe


def create_app(settings: Settings | None = None, pipeline: DriftGuardPipeline | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "pipeline", None) is None:
            s = settings or Settings.from_env()
            configure_logging(s.log_level)
            pipe = DriftGuardPipeline(s)
            if s.auto_train:
                pipe.ensure_baseline()
            app.state.pipeline = pipe
        yield

    app = FastAPI(
        title="DriftGuard API", version=__version__, lifespan=lifespan,
        description="Data-drift detection, model monitoring and adaptive retraining (Breast Cancer Wisconsin demo).",
    )
    if pipeline is not None:
        app.state.pipeline = pipeline

    def _no_model(exc: NoModelError):
        return HTTPException(status_code=503, detail=str(exc))

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(pipe: DriftGuardPipeline = Depends(get_pipeline)):
        v = pipe.registry.current_version()
        return HealthResponse(status="ok" if v is not None else "no_model", model_loaded=v is not None,
                              model_version=v, version=__version__)

    @app.get("/model/info", tags=["model"])
    def model_info(pipe: DriftGuardPipeline = Depends(get_pipeline)):
        try:
            return to_jsonable(pipe.model_info())
        except NoModelError as exc:
            raise _no_model(exc)

    @app.post("/predict", response_model=PredictResponse, tags=["model"])
    def predict(req: PredictRequest, pipe: DriftGuardPipeline = Depends(get_pipeline)):
        try:
            return pipe.predict(pd.DataFrame(req.records))
        except NoModelError as exc:
            raise _no_model(exc)
        except BatchValidationError as exc:
            raise HTTPException(status_code=422, detail={"message": "invalid records", "errors": exc.report.errors})

    @app.post("/monitor/run", tags=["monitoring"])
    def monitor_run(req: MonitorRequest, pipe: DriftGuardPipeline = Depends(get_pipeline)):
        try:
            if req.records is not None:
                X = pd.DataFrame(req.records)
                y = pd.Series(req.labels, name="malignant") if req.labels is not None else None
                batch = Batch(X=X, y=y, scenario="external")
            else:
                sc = ScenarioConfig(name=req.scenario, drift_start=req.drift_start, seed=req.seed)
                batch = DriftSimulator(pipe.splits, sc, pipe.settings.data.batch_size).batch(req.batch_index)
            event = pipe.process_batch(batch, allow_retrain=req.allow_retrain)
        except NoModelError as exc:
            raise _no_model(exc)
        if event.get("status") == "invalid":
            raise HTTPException(status_code=422, detail={"message": event["message"], "errors": event["validation"]["errors"]})
        return to_jsonable(event)

    @app.post("/simulate/drift", tags=["monitoring"])
    def simulate_drift(req: SimulateRequest, pipe: DriftGuardPipeline = Depends(get_pipeline)):
        seed = req.seed if req.seed is not None else int(time.time()) % 1_000_000
        sc = ScenarioConfig(name=req.scenario, drift_start=req.drift_start, seed=seed)
        try:
            events = pipe.run_scenario(sc, req.n_batches, allow_retrain=req.allow_retrain)
        except NoModelError as exc:
            raise _no_model(exc)
        rows = []
        for e in events:
            ok = e.get("status") == "ok"
            rows.append({
                "batch_index": e["batch_index"], "status": e["status"], "intensity": e["intensity"],
                "dataset_drift": e["drift"]["dataset_drift"] if ok else None,
                "n_flagged": e["drift"]["n_flagged"] if ok else None,
                "f1": (e.get("performance") or {}).get("f1") if ok else None,
                "model_version": e["model_version"], "action": e["action"],
            })
        return to_jsonable({
            "scenario": req.scenario, "seed": seed, "n_batches": req.n_batches, "batches": rows,
            "deployed_version": pipe.current_record().version,
            "retrain_attempts": sum(1 for r in rows if r["action"] in ("retrain_promoted", "retrain_rejected")),
            "duplicates_skipped": sum(1 for r in rows if r["status"] == "duplicate_skipped"),
        })

    @app.post("/retrain", tags=["retraining"])
    def retrain(req: RetrainRequest, pipe: DriftGuardPipeline = Depends(get_pipeline)):
        try:
            result = pipe.retrain(force=req.force, reason=req.reason)
        except NoModelError as exc:
            raise _no_model(exc)
        except InsufficientDataError as exc:  # pragma: no cover - reserved
            raise HTTPException(status_code=409, detail=str(exc))
        if result.get("status") == "skipped":
            raise HTTPException(status_code=409, detail=result.get("message", "retraining skipped"))
        return to_jsonable(result)

    @app.get("/metrics", tags=["monitoring"])
    def metrics(pipe: DriftGuardPipeline = Depends(get_pipeline)):
        return to_jsonable(pipe.metrics_summary())

    @app.get("/history", tags=["monitoring"])
    def history(limit: int = Query(50, ge=1, le=500),
                event_type: str | None = Query(None, pattern="^(monitoring|retrain)$"),
                pipe: DriftGuardPipeline = Depends(get_pipeline)):
        return {"events": pipe.history.recent(limit=limit, event_type=event_type)}

    return app


app = create_app()
