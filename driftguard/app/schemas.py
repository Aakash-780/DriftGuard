"""Pydantic request/response schemas for the REST API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

ScenarioName = Literal["baseline", "covariate", "sudden", "gradual", "concept"]


class PredictRequest(BaseModel):
    records: list[dict[str, float]] = Field(
        ..., min_length=1, max_length=1000,
        description="Rows as {feature_name: value}. Names must match GET /model/info feature_names.",
    )


class PredictResponse(BaseModel):
    model_version: int
    predictions: list[int]
    probabilities: list[float]
    labels: list[str]


class MonitorRequest(BaseModel):
    """Supply EITHER ``records`` (+ optional ``labels``) OR a ``scenario`` to generate a batch."""

    records: Optional[list[dict[str, float]]] = Field(None, min_length=1, max_length=5000)
    labels: Optional[list[int]] = Field(None, description="Optional 0/1 ground truth (1 = malignant).")
    scenario: Optional[ScenarioName] = None
    batch_index: int = Field(0, ge=0, le=10_000)
    drift_start: int = Field(4, ge=0, le=1000)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    allow_retrain: bool = True

    @model_validator(mode="after")
    def _exactly_one_source(self):
        if (self.records is None) == (self.scenario is None):
            raise ValueError("provide exactly one of 'records' or 'scenario'")
        if self.labels is not None and self.records is None:
            raise ValueError("'labels' can only accompany 'records'")
        return self


class SimulateRequest(BaseModel):
    scenario: ScenarioName
    n_batches: int = Field(10, ge=1, le=30)
    drift_start: int = Field(4, ge=0, le=29)
    seed: Optional[int] = Field(None, ge=0, le=2**31 - 1, description="Omit for a time-based seed.")
    allow_retrain: bool = True


class RetrainRequest(BaseModel):
    force: bool = Field(False, description="Bypass the trigger rules (promotion gates still apply).")
    reason: Optional[str] = Field(None, max_length=200)

    @field_validator("reason")
    @classmethod
    def _strip(cls, v):
        return v.strip() if v else v


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[int] = None
    version: str
