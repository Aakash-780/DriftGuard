"""Experiment 1 -- drift-detector comparison on a model that is never retrained.

For each (seed, scenario): rebuild the splits with that seed, train the baseline, fit every detector on
the reference split, stream ``n_batches`` simulated batches and record each detector's decision plus the
static model's metrics on the batch. Detectors never see labels.
"""
from __future__ import annotations

import time

import pandas as pd

from driftguard.core.config import DataConfig, DriftConfig, ModelConfig
from driftguard.data.loader import make_splits
from driftguard.data.simulation import DriftSimulator, ScenarioConfig
from driftguard.experiments.common import ExperimentConfig
from driftguard.models.evaluation import evaluate_model, positive_proba
from driftguard.models.training import get_baseline_model
from driftguard.monitoring.detectors import build_detectors, prediction_drift


def run_detection_one(seed: int, scenario: str, cfg: ExperimentConfig, shift_magnitude: float | None = None,
                      coherent: bool = False) -> list[dict]:
    dcfg = DataConfig(seed=seed, batch_size=cfg.batch_size)
    drift_cfg = DriftConfig()
    splits = make_splits(dcfg)
    model = get_baseline_model(dcfg, ModelConfig())
    detectors = build_detectors(cfg.detectors, drift_cfg, seed=seed)
    for d in detectors:
        d.fit(splits.X_ref)
    ref_proba = positive_proba(model, splits.X_ref)
    extra = {} if shift_magnitude is None else {"shift_magnitude": shift_magnitude, "coherent_signs": coherent}
    sc = ScenarioConfig(name=scenario, drift_start=cfg.drift_start, ramp_batches=cfg.ramp_batches, seed=seed, **extra)
    sim = DriftSimulator(splits, sc, cfg.batch_size)
    rows = []
    for i in range(cfg.n_batches):
        b = sim.batch(i)
        row = {"seed": seed, "scenario": scenario, "batch_index": i, "intensity": b.intensity}
        if shift_magnitude is not None:
            row["shift_magnitude"] = shift_magnitude
            row["sign_mode"] = "coherent" if coherent else "random"
        m = evaluate_model(model, b.X, b.y)
        row.update({k: m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")})
        for d in detectors:
            r = d.detect(b.X)
            row[f"{d.name}_drifted"] = int(r.drifted)
            row[f"{d.name}_score"] = r.score
        pdr = prediction_drift(ref_proba, positive_proba(model, b.X), drift_cfg.prediction_alpha)
        row["prediction_drifted"] = int(pdr["drifted"])
        row["prediction_pvalue"] = pdr["p_value"]
        rows.append(row)
    return rows


def run_detection_experiment(cfg: ExperimentConfig, verbose: bool = True) -> pd.DataFrame:
    seeds = range(cfg.seed_start, cfg.seed_start + cfg.n_seeds_detection)
    rows, t0 = [], time.time()
    for seed in seeds:
        for sc in cfg.scenarios:
            rows.extend(run_detection_one(seed, sc, cfg))
        if verbose:
            print(f"[detection] seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def run_sensitivity_experiment(cfg: ExperimentConfig, verbose: bool = True) -> pd.DataFrame:
    """Detector power vs shift magnitude for ``sudden``, with random and coherent (all-upward) shift signs."""
    rows, t0 = [], time.time()
    for coherent in (False, True):
        for mag in cfg.sensitivity_magnitudes:
            for seed in range(cfg.seed_start, cfg.seed_start + cfg.n_seeds_detection):
                rows.extend(run_detection_one(seed, "sudden", cfg, shift_magnitude=mag, coherent=coherent))
            if verbose:
                print(f"[sensitivity] signs={'coherent' if coherent else 'random'} magnitude {mag} "
                      f"done ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)
