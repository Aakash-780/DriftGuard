"""Pure metric computations for the experiments (unit-tested on hand-made examples).

Definitions (``flag`` = a detector's dataset-level drift decision for one batch):

* **onset**            first batch index with drift intensity > 0 (``drift_start``).
* **false-positive rate (FPR)**   flagged batches / batches with intensity == 0 (no drift present).
* **detection rate (post-onset)** flagged batches / batches with index >= onset.
* **detection rate (full intensity)** flagged batches / batches with intensity == 1
  (for ``gradual`` this excludes the ramp, where drift is still partial).
* **run-level detection rate**    share of runs (seeds) with >= 1 flagged batch at/after onset.
* **detection delay**             (index of first flagged batch >= onset) - onset, in batches, over runs
  where drift was detected; undetected runs are counted separately, not imputed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from driftguard.experiments.common import ExperimentConfig, t_ci, wilson


def _flag_col(name: str) -> str:
    return f"{name}_drifted"


def first_detection_delay(batch_index: np.ndarray, flags: np.ndarray, onset: int) -> float:
    mask = (batch_index >= onset) & (flags.astype(bool))
    if not mask.any():
        return float("nan")
    return float(batch_index[mask].min() - onset)


def summarize_detection(df: pd.DataFrame, detectors: list[str], cfg: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for sc in cfg.scenarios:
        sub = df[df["scenario"] == sc]
        if sub.empty:
            continue
        n_runs = sub["seed"].nunique()
        for det in detectors:
            col = _flag_col(det)
            if col not in sub:
                continue
            pre = sub[sub["intensity"] == 0]
            row = {"scenario": sc, "detector": det, "n_runs": n_runs,
                   "fpr_no_drift_batches": float(pre[col].mean()) if len(pre) else np.nan,
                   "n_no_drift_batches": len(pre)}
            if sc != "baseline":
                post = sub[sub["batch_index"] >= cfg.drift_start]
                full = sub[sub["intensity"] == 1.0]
                delays, detected = [], 0
                for _, g in post.groupby("seed"):
                    d = first_detection_delay(g["batch_index"].to_numpy(), g[col].to_numpy(), cfg.drift_start)
                    if not np.isnan(d):
                        detected += 1
                        delays.append(d)
                row.update({
                    "detection_rate_post_onset": float(post[col].mean()),
                    "detection_rate_full_intensity": float(full[col].mean()),
                    "run_level_detection_rate": detected / n_runs,
                    "n_detected_runs": detected,
                    "mean_delay_batches": float(np.mean(delays)) if delays else np.nan,
                    "median_delay_batches": float(np.median(delays)) if delays else np.nan,
                })
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_fpr(df: pd.DataFrame, detectors: list[str]) -> pd.DataFrame:
    """Pooled false-positive rate over every batch that contains no drift (all scenarios)."""
    pre = df[df["intensity"] == 0]
    rows = []
    for det in detectors:
        col = _flag_col(det)
        if col not in pre:
            continue
        k, n = int(pre[col].sum()), len(pre)
        lo, hi = wilson(k, n)
        rows.append({"detector": det, "false_positives": k, "no_drift_batches": n,
                     "fpr": k / n if n else np.nan, "wilson95_lo": lo, "wilson95_hi": hi})
    return pd.DataFrame(rows)


def summarize_static_performance(df: pd.DataFrame, cfg: ExperimentConfig, metric: str = "f1") -> pd.DataFrame:
    """Never-retrained model: metric before drift vs after drift reached full intensity."""
    rows = []
    for sc in cfg.scenarios:
        sub = df[df["scenario"] == sc]
        deltas, pres, posts = [], [], []
        for _, g in sub.groupby("seed"):
            pre = g[g["batch_index"] < cfg.drift_start][metric].mean()
            post = g[g["batch_index"] >= cfg.post_start][metric].mean()
            pres.append(pre); posts.append(post); deltas.append(post - pre)
        ci = t_ci(deltas)
        rows.append({
            "scenario": sc, "n_seeds": ci["n"], f"mean_{metric}_before": float(np.mean(pres)),
            f"mean_{metric}_after": float(np.mean(posts)), "mean_delta": ci["mean"], "sd_delta": ci["sd"],
            "ci95_lo": ci["lo"], "ci95_hi": ci["hi"],
            "share_seeds_drop_ge_0.05": float(np.mean(np.array(deltas) <= -0.05)),
        })
    return pd.DataFrame(rows)


def summarize_retraining(runs: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Per (scenario, arm): trigger counts, acceptance, delay, performance and recovery."""
    never = runs[runs["arm"] == "never"].set_index(["scenario", "seed"])
    rows = []
    for (sc, arm), g in runs.groupby(["scenario", "arm"], sort=False):
        attempts, promoted = g["n_attempts"].sum(), g["n_promoted"].sum()
        gain = []
        for _, r in g.iterrows():
            key = (sc, r["seed"])
            if key in never.index:
                gain.append(r["mean_f1_post"] - never.loc[key, "mean_f1_post"])
        g_ci, t_ci_ = t_ci(gain), t_ci(g["test_delta_f1"])
        rows.append({
            "scenario": sc, "arm": arm, "n_runs": len(g),
            "mean_attempts": g["n_attempts"].mean(), "mean_promoted": g["n_promoted"].mean(),
            "mean_rejected": g["n_rejected"].mean(),
            "share_runs_with_attempt": float((g["n_attempts"] > 0).mean()),
            "acceptance_rate": float(promoted / attempts) if attempts else np.nan,
            "mean_trigger_delay": g["trigger_delay"].mean(),
            "mean_f1_pre": g["mean_f1_pre"].mean(), "mean_f1_post": g["mean_f1_post"].mean(),
            "instream_gain_vs_never": g_ci["mean"], "gain_ci95_lo": g_ci["lo"], "gain_ci95_hi": g_ci["hi"],
            "test_delta_f1": t_ci_["mean"], "test_delta_ci95_lo": t_ci_["lo"], "test_delta_ci95_hi": t_ci_["hi"],
        })
    return pd.DataFrame(rows)


def summarize_sensitivity(df: pd.DataFrame, detectors: list[str], cfg: ExperimentConfig) -> pd.DataFrame:
    """Per shift magnitude: detection rate at full intensity, pooled no-drift FPR, static-model F1 change."""
    rows = []
    for (mode, mag), g in df.groupby(["sign_mode", "shift_magnitude"]):
        full, pre = g[g["intensity"] == 1.0], g[g["intensity"] == 0]
        row = {"sign_mode": mode, "shift_magnitude": mag, "n_seeds": g["seed"].nunique()}
        for det in detectors:
            row[f"{det}_detection_rate"] = float(full[f"{det}_drifted"].mean())
            row[f"{det}_fpr"] = float(pre[f"{det}_drifted"].mean())
        deltas = [
            gg[gg["batch_index"] >= cfg.post_start]["f1"].mean() - gg[gg["batch_index"] < cfg.drift_start]["f1"].mean()
            for _, gg in g.groupby("seed")
        ]
        ci = t_ci(deltas)
        row.update({"static_f1_delta": ci["mean"], "static_f1_delta_ci95_lo": ci["lo"], "static_f1_delta_ci95_hi": ci["hi"]})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["sign_mode", "shift_magnitude"]).reset_index(drop=True)
