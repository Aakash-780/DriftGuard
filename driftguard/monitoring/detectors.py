"""Drift detectors.

All detectors share one interface: ``fit(reference_df)`` once, then ``detect(batch_df)`` per batch.

========================  =================================  ==========================================
Detector                  What it detects                    Key limitation
========================  =================================  ==========================================
``ks``                    per-feature *marginal* shifts      blind to joint/correlation changes; many
                          (two-sample Kolmogorov-Smirnov)    correlated tests; p-values depend on n
``psi``                   per-feature *magnitude* of         threshold conventions (0.1/0.25) were set
                          distribution change over bins      for large samples; PSI has a positive
                                                             small-sample bias ~ (bins-1)(1/n_ref+1/n_cur)
``domain_classifier``     *multivariate* shift: can a        LINEAR model: sees mean shifts (also coordinated
                          linear classifier tell reference   small ones), not covariance-only changes;
                          from the batch? (cross-val. AUC)   needs enough rows; top coefficients are a hint
``evidently`` (optional)  Evidently's data-drift preset      for small numeric samples it defaults to a
                                                             KS-based test, i.e. close to ``ks`` here
========================  =================================  ==========================================

None of these can detect *concept drift* (a change in P(y|x) with P(x) unchanged): they never see labels.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from driftguard.core.config import DETECTOR_NAMES, DriftConfig


@dataclass
class DriftResult:
    detector: str
    drifted: bool
    score: float  # dataset-level statistic the decision rule uses (see each detector)
    threshold: float
    share_drifted: float | None = None
    n_drifted_features: int | None = None
    feature_scores: dict[str, float] | None = None
    feature_drifted: dict[str, bool] | None = None
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "detector": self.detector, "drifted": bool(self.drifted), "score": self.score,
            "threshold": self.threshold, "share_drifted": self.share_drifted,
            "n_drifted_features": self.n_drifted_features,
            "feature_scores": self.feature_scores, "feature_drifted": self.feature_drifted,
            "details": self.details,
        }


class DriftDetector(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, reference: pd.DataFrame) -> "DriftDetector": ...

    @abstractmethod
    def detect(self, current: pd.DataFrame) -> DriftResult: ...

    @abstractmethod
    def config(self) -> dict: ...


# ---------------------------------------------------------------------- KS
def bh_reject(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg step-up procedure; returns a boolean rejection mask."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, m + 1) / m)
    below = p[order] <= thresh
    reject = np.zeros(m, dtype=bool)
    if below.any():
        k = np.max(np.where(below)[0])
        reject[order[: k + 1]] = True
    return reject


class KSDetector(DriftDetector):
    name = "ks"

    def __init__(self, cfg: DriftConfig):
        if cfg.ks_correction not in ("none", "bonferroni", "bh"):
            raise ValueError("ks_correction must be 'none', 'bonferroni' or 'bh'")
        self.cfg = cfg
        self._ref: np.ndarray | None = None
        self._cols: list[str] = []

    def fit(self, reference: pd.DataFrame) -> "KSDetector":
        self._cols = list(reference.columns)
        self._ref = reference.to_numpy(dtype=float)
        return self

    def detect(self, current: pd.DataFrame) -> DriftResult:
        cur = current[self._cols].to_numpy(dtype=float)
        m = len(self._cols)
        stat = np.empty(m)
        pval = np.empty(m)
        for j in range(m):
            r = stats.ks_2samp(self._ref[:, j], cur[:, j])
            stat[j], pval[j] = r.statistic, r.pvalue
        a = self.cfg.ks_alpha
        if self.cfg.ks_correction == "bonferroni":
            flags = pval < a / m
        elif self.cfg.ks_correction == "bh":
            flags = bh_reject(pval, a)
        else:
            flags = pval < a
        share = float(flags.mean())
        return DriftResult(
            detector=self.name, drifted=share >= self.cfg.ks_min_drifted_share, score=share,
            threshold=self.cfg.ks_min_drifted_share, share_drifted=share,
            n_drifted_features=int(flags.sum()),
            feature_scores=dict(zip(self._cols, stat.tolist())),
            feature_drifted=dict(zip(self._cols, map(bool, flags))),
            details={"p_values": dict(zip(self._cols, pval.tolist())), "statistic": "ks_statistic"},
        )

    def config(self) -> dict:
        c = self.cfg
        return {"alpha": c.ks_alpha, "correction": c.ks_correction,
                "min_drifted_share": c.ks_min_drifted_share}


# ---------------------------------------------------------------------- PSI
def psi_from_proportions(p_ref: np.ndarray, p_cur: np.ndarray, eps: float = 1e-4) -> float:
    """PSI = sum((cur - ref) * ln(cur / ref)); empty bins are floored at ``eps``."""
    a = np.clip(np.asarray(p_ref, dtype=float), eps, None)
    b = np.clip(np.asarray(p_cur, dtype=float), eps, None)
    return float(np.sum((b - a) * np.log(b / a)))


def quantile_edges(ref: np.ndarray, n_bins: int) -> np.ndarray:
    """Inner bin edges from reference quantiles (duplicates removed, so ties give fewer bins)."""
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.unique(np.quantile(ref, qs))


def bin_proportions(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, x, side="right")  # open-ended outer bins
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    return counts / max(1, len(x))


class PSIDetector(DriftDetector):
    name = "psi"

    def __init__(self, cfg: DriftConfig):
        self.cfg = cfg
        self._cols: list[str] = []
        self._edges: list[np.ndarray] = []
        self._p_ref: list[np.ndarray] = []

    def fit(self, reference: pd.DataFrame) -> "PSIDetector":
        self._cols = list(reference.columns)
        self._edges, self._p_ref = [], []
        for c in self._cols:
            x = reference[c].to_numpy(dtype=float)
            e = quantile_edges(x, self.cfg.psi_bins)
            self._edges.append(e)
            self._p_ref.append(bin_proportions(x, e))
        return self

    def detect(self, current: pd.DataFrame) -> DriftResult:
        psis = np.empty(len(self._cols))
        for j, c in enumerate(self._cols):
            p_cur = bin_proportions(current[c].to_numpy(dtype=float), self._edges[j])
            psis[j] = psi_from_proportions(self._p_ref[j], p_cur, self.cfg.psi_epsilon)
        flags = psis >= self.cfg.psi_threshold
        share = float(flags.mean())
        return DriftResult(
            detector=self.name, drifted=share >= self.cfg.psi_min_drifted_share, score=share,
            threshold=self.cfg.psi_min_drifted_share, share_drifted=share,
            n_drifted_features=int(flags.sum()),
            feature_scores=dict(zip(self._cols, psis.tolist())),
            feature_drifted=dict(zip(self._cols, map(bool, flags))),
            details={"mean_psi": float(psis.mean()), "max_psi": float(psis.max()),
                     "feature_threshold": self.cfg.psi_threshold},
        )

    def config(self) -> dict:
        c = self.cfg
        return {"bins": c.psi_bins, "feature_threshold": c.psi_threshold, "epsilon": c.psi_epsilon,
                "min_drifted_share": c.psi_min_drifted_share, "binning": "reference quantiles"}


# ---------------------------------------------------------------------- Domain classifier
class DomainClassifierDetector(DriftDetector):
    name = "domain_classifier"

    def __init__(self, cfg: DriftConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self._ref: np.ndarray | None = None
        self._cols: list[str] = []

    def fit(self, reference: pd.DataFrame) -> "DomainClassifierDetector":
        self._cols = list(reference.columns)
        self._ref = reference.to_numpy(dtype=float)
        return self

    def _make(self):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.cfg.domain_C, class_weight="balanced", max_iter=1000),
        )

    def detect(self, current: pd.DataFrame) -> DriftResult:
        cur = current[self._cols].to_numpy(dtype=float)
        X = np.vstack([self._ref, cur])
        y = np.r_[np.zeros(len(self._ref)), np.ones(len(cur))].astype(int)
        folds = max(2, min(self.cfg.domain_cv_folds, len(cur), len(self._ref)))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
        proba = cross_val_predict(self._make(), X, y, cv=cv, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y, proba))
        top: dict[str, float] = {}
        try:  # which features separate the samples (a hint, not a test)
            model = self._make().fit(X, y)
            coef = np.abs(model[-1].coef_[0])
            order = np.argsort(coef)[::-1][:5]
            top = {self._cols[i]: float(coef[i]) for i in order}
        except Exception:  # pragma: no cover - purely informational
            pass
        return DriftResult(
            detector=self.name, drifted=auc >= self.cfg.domain_auc_threshold, score=auc,
            threshold=self.cfg.domain_auc_threshold,
            details={"cv_folds": folds, "auc_chance_level": 0.5, "top_features": top},
        )

    def config(self) -> dict:
        c = self.cfg
        return {"auc_threshold": c.domain_auc_threshold, "cv_folds": c.domain_cv_folds,
                "C": c.domain_C, "model": "StandardScaler + LogisticRegression(balanced)"}


# ---------------------------------------------------------------------- Prediction drift
def prediction_drift(ref_proba: np.ndarray, cur_proba: np.ndarray, alpha: float) -> dict:
    """KS test on predicted P(malignant). Prediction drift != performance degradation."""
    r = stats.ks_2samp(np.asarray(ref_proba), np.asarray(cur_proba))
    return {
        "drifted": bool(r.pvalue < alpha), "p_value": float(r.pvalue), "statistic": float(r.statistic),
        "alpha": alpha,
        "ref_positive_rate": float(np.mean(np.asarray(ref_proba) >= 0.5)),
        "cur_positive_rate": float(np.mean(np.asarray(cur_proba) >= 0.5)),
    }


# ---------------------------------------------------------------------- factory
def build_detectors(names, cfg: DriftConfig, seed: int = 0) -> list[DriftDetector]:
    out: list[DriftDetector] = []
    for n in names:
        if n == "ks":
            out.append(KSDetector(cfg))
        elif n == "psi":
            out.append(PSIDetector(cfg))
        elif n == "domain_classifier":
            out.append(DomainClassifierDetector(cfg, seed=seed))
        elif n == "evidently":
            from driftguard.monitoring.evidently_adapter import EvidentlyDetector

            out.append(EvidentlyDetector(cfg))
        else:
            raise ValueError(f"unknown detector {n!r}; choose from {DETECTOR_NAMES}")
    return out
