"""Dataset loading and reproducible splitting.

Dataset: Breast Cancer Wisconsin (Diagnostic), shipped with scikit-learn (no download needed).
Label convention: ``malignant = 1`` is the positive class (scikit-learn's raw target has 0 = malignant),
so precision/recall/F1 refer to detecting malignant tumours.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from driftguard.core.config import DataConfig

TARGET_NAME = "malignant"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    data = load_breast_cancer(as_frame=True)
    X = data.data.astype("float64")
    y = (data.target == 0).astype(int).rename(TARGET_NAME)
    return X, y


@dataclass
class DataSplits:
    """Disjoint splits. Treat the contained frames as read-only (they are cached)."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_ref: pd.DataFrame
    y_ref: pd.Series
    X_stream: pd.DataFrame
    y_stream: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_names: tuple[str, ...]
    seed: int

    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.X_train),
            "reference": len(self.X_ref),
            "stream": len(self.X_stream),
            "test": len(self.X_test),
        }

    def train_fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.X_train.to_numpy(dtype="float64")).tobytes())
        h.update(np.ascontiguousarray(self.y_train.to_numpy(dtype="int64")).tobytes())
        return h.hexdigest()[:16]


@lru_cache(maxsize=16)
def make_splits(cfg: DataConfig) -> DataSplits:
    X, y = load_dataset()
    s = cfg.seed
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=cfg.train_frac, stratify=y, random_state=s
    )
    rest = 1.0 - cfg.train_frac
    X_ref, X_rest2, y_ref, y_rest2 = train_test_split(
        X_rest, y_rest, train_size=cfg.reference_frac / rest, stratify=y_rest, random_state=s
    )
    rest2 = 1.0 - cfg.train_frac - cfg.reference_frac
    X_stream, X_test, y_stream, y_test = train_test_split(
        X_rest2, y_rest2, train_size=cfg.stream_frac / rest2, stratify=y_rest2, random_state=s
    )
    return DataSplits(
        X_train=X_train, y_train=y_train, X_ref=X_ref, y_ref=y_ref,
        X_stream=X_stream, y_stream=y_stream, X_test=X_test, y_test=y_test,
        feature_names=tuple(X.columns), seed=s,
    )
