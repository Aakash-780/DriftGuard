"""Shared experiment configuration and statistics helpers."""
from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from driftguard.core.config import DEFAULT_DETECTORS
from driftguard.data.simulation import SCENARIOS

# (arm label, retraining policy, candidate data strategy)
DEFAULT_ARMS = (
    ("never", "never", "train_plus_window"),
    ("performance_only", "performance_only", "train_plus_window"),
    ("drift_only", "drift_only", "train_plus_window"),
    ("drift_and_performance", "drift_and_performance", "train_plus_window"),
    ("drift_or_performance", "drift_or_performance", "train_plus_window"),
    ("drift_or_performance+window_only", "drift_or_performance", "window_only"),
)


@dataclass(frozen=True)
class ExperimentConfig:
    n_seeds_detection: int = 20
    n_seeds_retraining: int = 10
    seed_start: int = 0
    n_batches: int = 12
    batch_size: int = 100
    drift_start: int = 4
    ramp_batches: int = 4
    scenarios: tuple[str, ...] = SCENARIOS
    detectors: tuple[str, ...] = DEFAULT_DETECTORS
    regime_eval_rows: int = 300
    arms: tuple = DEFAULT_ARMS
    sensitivity_magnitudes: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5)

    @property
    def post_start(self) -> int:
        """First batch index at which every scenario has reached full intensity."""
        return self.drift_start + self.ramp_batches - 1


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "evidently", "mlflow"):
        try:
            info[mod] = importlib.import_module(mod).__version__
        except Exception:
            info[mod] = None
    return info


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def t_ci(values) -> dict:
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    n = len(v)
    if n == 0:
        return {"mean": np.nan, "sd": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    mean = float(v.mean())
    if n < 2:
        return {"mean": mean, "sd": np.nan, "lo": np.nan, "hi": np.nan, "n": n}
    sd = float(v.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return {"mean": mean, "sd": sd, "lo": mean - half, "hi": mean + half, "n": n}


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Minimal GitHub-markdown table (avoids the optional ``tabulate`` dependency)."""
    if df.empty:
        return "_(no rows)_"
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("–" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
