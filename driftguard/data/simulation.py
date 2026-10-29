"""Reproducible simulation of incoming data under normal and drifted conditions.

Scenarios (all seeded; batch ``i`` is a pure function of ``(seed, i)``):

``baseline``   no change; batches are bootstrap samples of the stream pool.
``covariate``  *pure covariate shift*: rows are re-sampled with exponential tilting on a subset of
               features. Each row keeps its own label, so P(y|x) is exactly preserved while P(x) moves.
               Because P(y|x) is unchanged, accuracy may or may not degrade -- that is the point.
``sudden``     step change at ``drift_start``: a subset of features is translated by
               ``shift_magnitude`` reference standard deviations (like a sensor re-calibration).
               Labels stay attached to their original rows, so this can also move rows across the
               decision boundary; whether performance falls is measured, never assumed.
``gradual``    same translation as ``sudden`` but ramped linearly over ``ramp_batches`` batches.
``concept``    *concept drift*: P(x) is untouched but labels are flipped (probability ``flip_prob``)
               for rows where ``concept_feature`` is above its reference mean, i.e. P(y|x) changes.
               Feature-based detectors cannot see this by construction.

Batches are bootstrap draws (with replacement) from a finite stream pool; see README limitations.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from driftguard.data.loader import DataSplits, TARGET_NAME

SCENARIOS = ("baseline", "covariate", "sudden", "gradual", "concept")


@dataclass(frozen=True)
class ScenarioConfig:
    name: str = "baseline"
    drift_start: int = 4
    ramp_batches: int = 4
    shift_magnitude: float = 1.5  # in reference standard deviations
    feature_fraction: float = 0.4
    tilt_beta: float = 1.5
    flip_prob: float = 0.8
    concept_feature: str = "worst perimeter"
    # sudden/gradual only: False = random +/- sign per shifted feature (can create physically implausible
    # combinations, e.g. radius up while perimeter goes down); True = every shifted feature moves upward.
    coherent_signs: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.name not in SCENARIOS:
            raise ValueError(f"unknown scenario {self.name!r}; choose from {SCENARIOS}")
        if self.seed < 0 or self.drift_start < 0:
            raise ValueError("seed and drift_start must be non-negative")

    @property
    def is_drift(self) -> bool:
        return self.name != "baseline"

    def intensity(self, batch_index: int) -> float:
        """Drift intensity in [0, 1] for a batch index (0 = no drift)."""
        if self.name == "baseline" or batch_index < self.drift_start:
            return 0.0
        if self.name == "gradual":
            return min(1.0, (batch_index - self.drift_start + 1) / max(1, self.ramp_batches))
        return 1.0


@dataclass
class Batch:
    X: pd.DataFrame
    y: pd.Series | None = None
    row_ids: np.ndarray | None = None
    batch_index: int = 0
    scenario: str = "external"
    intensity: float = 0.0
    meta: dict = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Content hash used to recognise a batch that was already processed."""
        h = hashlib.sha256()
        h.update("|".join(map(str, self.X.columns)).encode("utf-8"))
        h.update(np.ascontiguousarray(self.X.to_numpy(dtype="float64")).tobytes())
        if self.y is None:
            h.update(b"nolabel")
        else:
            h.update(np.ascontiguousarray(self.y.to_numpy(dtype="int64")).tobytes())
        return h.hexdigest()[:20]


class DriftSimulator:
    def __init__(self, splits: DataSplits, scenario: ScenarioConfig, batch_size: int = 100):
        self.scenario = scenario
        self.batch_size = int(batch_size)
        self.features = list(splits.feature_names)
        self._pool_X = splits.X_stream
        self._pool_y = splits.y_stream
        self._test_X = splits.X_test
        self._test_y = splits.y_test
        ref = splits.X_ref
        self.mean = ref.mean().to_numpy(dtype=float)
        std = ref.std(ddof=0).to_numpy(dtype=float)
        self.std = np.where(std == 0, 1.0, std)
        if scenario.concept_feature not in self.features:
            raise ValueError(f"concept_feature {scenario.concept_feature!r} is not a feature")
        self.concept_idx = self.features.index(scenario.concept_feature)
        rng = np.random.default_rng([scenario.seed, 9001])
        n = len(self.features)
        k = max(1, int(round(scenario.feature_fraction * n)))
        self.sel = np.sort(rng.choice(n, size=k, replace=False))
        self.signs = rng.choice([-1.0, 1.0], size=k)
        if scenario.coherent_signs:  # drawn above anyway so the RNG stream is unchanged
            self.signs = np.ones(k)

    # ------------------------------------------------------------------ core sampling
    def _draw(self, pool_X: pd.DataFrame, pool_y: pd.Series, n: int, intensity: float,
              rng_sample: np.random.Generator, rng_flip: np.random.Generator):
        sc = self.scenario
        Xp = pool_X.to_numpy(dtype=float)
        yp = pool_y.to_numpy(dtype=int)
        m = len(Xp)
        p = np.full(m, 1.0 / m)
        if sc.name == "covariate" and intensity > 0:
            z = (Xp - self.mean) / self.std
            score = z[:, self.sel].mean(axis=1)
            logits = sc.tilt_beta * intensity * score
            logits = logits - logits.max()
            p = np.exp(logits)
            p = p / p.sum()
        # Always pass p so RNG consumption is identical across scenarios (clean comparisons).
        idx = rng_sample.choice(m, size=n, replace=True, p=p)
        X = Xp[idx].copy()
        y = yp[idx].copy()
        if sc.name in ("sudden", "gradual") and intensity > 0:
            X[:, self.sel] += self.signs * sc.shift_magnitude * intensity * self.std[self.sel]
            X = np.maximum(X, 0.0)  # all features are physical, non-negative measurements
        if sc.name == "concept" and intensity > 0:
            z = (X[:, self.concept_idx] - self.mean[self.concept_idx]) / self.std[self.concept_idx]
            flip = (z > 0) & (rng_flip.random(n) < sc.flip_prob * intensity)
            y = np.where(flip, 1 - y, y)
        return X, y, pool_X.index.to_numpy()[idx]

    def batch(self, batch_index: int) -> Batch:
        intensity = self.scenario.intensity(batch_index)
        seed = self.scenario.seed
        rs = np.random.default_rng([seed, int(batch_index), 0])
        rf = np.random.default_rng([seed, int(batch_index), 1])
        X, y, ids = self._draw(self._pool_X, self._pool_y, self.batch_size, intensity, rs, rf)
        return Batch(
            X=pd.DataFrame(X, columns=self.features),
            y=pd.Series(y, name=TARGET_NAME),
            row_ids=np.array([f"s{i}" for i in ids], dtype=object),
            batch_index=int(batch_index),
            scenario=self.scenario.name,
            intensity=float(intensity),
        )

    def regime_batch(self, intensity: float = 1.0, n: int = 300) -> Batch:
        """Sample from the *test* pool under the given drift intensity (never seen by the pipeline)."""
        seed = self.scenario.seed
        rs = np.random.default_rng([seed, 777_000, 0])
        rf = np.random.default_rng([seed, 777_000, 1])
        X, y, ids = self._draw(self._test_X, self._test_y, n, intensity, rs, rf)
        return Batch(
            X=pd.DataFrame(X, columns=self.features),
            y=pd.Series(y, name=TARGET_NAME),
            row_ids=np.array([f"t{i}" for i in ids], dtype=object),
            batch_index=-1,
            scenario=self.scenario.name,
            intensity=float(intensity),
            meta={"split": "test"},
        )
