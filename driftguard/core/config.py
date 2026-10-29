"""Central, explicit configuration.

Every threshold used by detectors, retraining rules and promotion gates lives here so that
nothing is hard-coded deep inside the pipeline. The defaults are *starting points chosen a priori*,
not universally correct values; the research experiments report how they behave.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

POLICIES = (
    "never",
    "performance_only",
    "drift_only",
    "drift_and_performance",
    "drift_or_performance",
)
DETECTOR_NAMES = ("ks", "psi", "domain_classifier", "evidently")
DEFAULT_DETECTORS = ("ks", "psi", "domain_classifier")


@dataclass(frozen=True)
class DataConfig:
    """Dataset splitting. Fractions must sum to 1.

    train      -> fits the baseline model (and is re-used when fitting candidates)
    reference  -> reference sample for drift detectors AND the fixed "anchor" validation set
    stream     -> pool from which simulated incoming batches are drawn
    test       -> never touched by pipeline decisions; only used for post-hoc evaluation
    """

    seed: int = 42
    train_frac: float = 0.30
    reference_frac: float = 0.20
    stream_frac: float = 0.35
    test_frac: float = 0.15
    batch_size: int = 100


@dataclass(frozen=True)
class ModelConfig:
    model_type: str = "random_forest"  # "random_forest" | "logistic_regression"
    n_estimators: int = 100
    max_depth: int | None = 8
    min_samples_leaf: int = 2
    C: float = 1.0  # logistic regression only


@dataclass(frozen=True)
class DriftConfig:
    # Kolmogorov-Smirnov (per feature)
    ks_alpha: float = 0.05
    ks_correction: str = "bh"  # "none" | "bonferroni" | "bh" (Benjamini-Hochberg)
    ks_min_drifted_share: float = 0.20  # dataset drift if this share of features drift
    # Population Stability Index (per feature, reference-quantile bins)
    psi_bins: int = 5
    psi_threshold: float = 0.25  # per-feature "significant shift" convention
    psi_epsilon: float = 1e-4  # floor for empty bins
    psi_min_drifted_share: float = 0.20
    # Domain classifier two-sample test (multivariate)
    domain_auc_threshold: float = 0.65
    domain_cv_folds: int = 5
    domain_C: float = 0.1
    # Prediction drift (KS on predicted probabilities)
    prediction_alpha: float = 0.01
    # Evidently (optional adapter)
    evidently_drift_share: float = 0.5


@dataclass(frozen=True)
class RetrainConfig:
    policy: str = "drift_or_performance"
    min_detectors_drifted: int = 2  # detectors that must agree for "dataset drift"
    min_consecutive: int = 2  # consecutive batches a signal must persist
    performance_metric: str = "f1"
    max_metric_drop: float = 0.05  # degradation = baseline metric - batch metric >= this
    cooldown_batches: int = 3  # batches to wait after a retrain attempt
    min_labeled_rows: int = 100  # labeled rows needed in the window before retraining
    window_batches: int = 4  # most recent labeled batches used for retraining
    holdout_fraction: float = 0.30  # share of window rows (by row id) kept for validation
    max_buffer_rows: int = 2000
    min_batch_rows: int = 20  # smaller batches are rejected by validation
    partition_salt: int = 17
    # What the candidate is fitted on: "train_plus_window" = original train split + recent window rows;
    # "window_only" = recent window rows only (drops possibly stale data, e.g. after concept drift).
    candidate_data: str = "train_plus_window"


@dataclass(frozen=True)
class PromotionConfig:
    min_improvement: float = 0.01  # candidate F1 must beat current F1 on recent holdout by this
    min_candidate_f1: float = 0.70  # absolute quality floor on recent holdout
    max_recall_drop: float = 0.10  # safety: do not trade away sensitivity
    max_reference_f1_drop: float = 0.10  # safety: limit forgetting on the anchor set
    min_holdout_rows: int = 30
    min_class_rows: int = 5  # each class needed in the holdout


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def registry_dir(self) -> Path:
        return self.root / "registry"

    @property
    def history_file(self) -> Path:
        return self.root / "history.jsonl"

    @property
    def state_file(self) -> Path:
        return self.root / "state.json"

    @property
    def buffer_file(self) -> Path:
        return self.root / "buffer.csv"

    @property
    def reference_file(self) -> Path:
        return self.root / "reference.csv"

    @property
    def last_batch_file(self) -> Path:
        return self.root / "last_batch.csv"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def mlflow_dir(self) -> Path:
        return self.root / "mlflow"

    @property
    def experiments_dir(self) -> Path:
        return self.root / "experiments"


@dataclass(frozen=True)
class Settings:
    artifacts_dir: Path = Path("artifacts")
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    retrain: RetrainConfig = field(default_factory=RetrainConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    detectors: tuple[str, ...] = DEFAULT_DETECTORS
    mlflow_mode: str = "auto"  # "auto" | "true" | "false"
    mlflow_experiment: str = "driftguard"
    mlflow_tracking_uri: str | None = None
    auto_train: bool = True
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.retrain.policy not in POLICIES:
            raise ValueError(f"unknown policy {self.retrain.policy!r}; choose from {POLICIES}")
        unknown = [d for d in self.detectors if d not in DETECTOR_NAMES]
        if unknown:
            raise ValueError(f"unknown detectors {unknown}; choose from {DETECTOR_NAMES}")
        if self.retrain.candidate_data not in ("train_plus_window", "window_only"):
            raise ValueError("candidate_data must be 'train_plus_window' or 'window_only'")
        if self.mlflow_mode not in ("auto", "true", "false"):
            raise ValueError("mlflow_mode must be 'auto', 'true' or 'false'")
        total = (
            self.data.train_frac + self.data.reference_frac + self.data.stream_frac + self.data.test_frac
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"data fractions must sum to 1 (got {total})")

    @property
    def paths(self) -> Paths:
        return Paths(Path(self.artifacts_dir))

    @property
    def mlflow_uri(self) -> str:
        if self.mlflow_tracking_uri:
            return self.mlflow_tracking_uri
        db = (self.paths.mlflow_dir / "mlflow.db").resolve()
        return "sqlite:///" + db.as_posix()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["artifacts_dir"] = str(self.artifacts_dir)
        d["detectors"] = list(self.detectors)
        return d

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        art = Path(env.get("DRIFTGUARD_ARTIFACTS_DIR", "artifacts")).expanduser().resolve()
        seed = int(env.get("DRIFTGUARD_SEED", "42"))
        raw = env.get("DRIFTGUARD_DETECTORS", ",".join(DEFAULT_DETECTORS))
        detectors = tuple(x.strip() for x in raw.split(",") if x.strip())
        policy = env.get("DRIFTGUARD_POLICY", RetrainConfig().policy)
        return cls(
            artifacts_dir=art,
            data=DataConfig(seed=seed),
            retrain=RetrainConfig(policy=policy),
            detectors=detectors,
            mlflow_mode=env.get("DRIFTGUARD_MLFLOW", "auto").lower(),
            mlflow_experiment=env.get("DRIFTGUARD_MLFLOW_EXPERIMENT", "driftguard"),
            mlflow_tracking_uri=env.get("MLFLOW_TRACKING_URI") or None,
            auto_train=env.get("DRIFTGUARD_AUTO_TRAIN", "true").lower() in ("1", "true", "yes"),
            log_level=env.get("DRIFTGUARD_LOG_LEVEL", "INFO"),
        )
