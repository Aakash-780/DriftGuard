"""Model evaluation metrics. Positive class = malignant (label 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")


def positive_proba(model, X: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X))
    return model.predict_proba(X)[:, classes.index(1)]


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Standard binary metrics. ``roc_auc`` is ``None`` when undefined (single class / no scores)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return out


def evaluate_model(model, X: pd.DataFrame, y) -> dict:
    proba = positive_proba(model, X)
    pred = model.predict(X)
    return compute_metrics(y, pred, proba)
