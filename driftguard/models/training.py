"""Model construction and training."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from driftguard.core.config import DataConfig, ModelConfig
from driftguard.data.loader import make_splits


def build_estimator(cfg: ModelConfig, seed: int):
    if cfg.model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=seed,
            n_jobs=1,
        )
    if cfg.model_type == "logistic_regression":
        return make_pipeline(
            StandardScaler(), LogisticRegression(C=cfg.C, max_iter=2000, random_state=seed)
        )
    raise ValueError(f"unknown model_type {cfg.model_type!r}")


def model_params(cfg: ModelConfig, seed: int) -> dict:
    if cfg.model_type == "random_forest":
        return {
            "model_type": cfg.model_type, "n_estimators": cfg.n_estimators,
            "max_depth": cfg.max_depth, "min_samples_leaf": cfg.min_samples_leaf,
            "random_state": seed,
        }
    return {"model_type": cfg.model_type, "C": cfg.C, "random_state": seed}


def train_model(X: pd.DataFrame, y, cfg: ModelConfig, seed: int):
    model = build_estimator(cfg, seed)
    model.fit(X, y)
    return model


@lru_cache(maxsize=32)
def get_baseline_model(data_cfg: DataConfig, model_cfg: ModelConfig):
    """Deterministically trained baseline for a (data, model) config. Cached; never mutated."""
    splits = make_splits(data_cfg)
    return train_model(splits.X_train, splits.y_train, model_cfg, data_cfg.seed)
