"""Experiment 2 -- retraining policies, promotion gate and recovery (full pipeline, one fresh
artifact directory per run).

All arms see *identical* simulated batches for a given (seed, scenario) because batch generation does not
depend on the deployed model, so arms can be compared pairwise. ``never`` is the no-retraining control.

Recovery is measured on a post-drift sample drawn from the TEST split (which no pipeline decision ever
touches): F1(final deployed model) - F1(initial baseline model), both on the same fully-drifted test sample.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from driftguard.core.config import DataConfig, RetrainConfig, Settings
from driftguard.core.tracking import Tracker
from driftguard.data.simulation import DriftSimulator, ScenarioConfig
from driftguard.experiments.common import ExperimentConfig
from driftguard.models.evaluation import evaluate_model
from driftguard.pipelines.pipeline import DriftGuardPipeline
from driftguard.pipelines.reporting import monitoring_frame


def run_retraining_one(seed: int, scenario: str, arm: tuple, cfg: ExperimentConfig):
    label, policy, candidate_data = arm
    with tempfile.TemporaryDirectory(prefix="driftguard-exp-") as tmp:
        settings = Settings(
            artifacts_dir=Path(tmp), data=DataConfig(seed=seed, batch_size=cfg.batch_size),
            retrain=RetrainConfig(policy=policy, candidate_data=candidate_data),
            detectors=cfg.detectors, mlflow_mode="false",
        )
        pipe = DriftGuardPipeline(settings, tracker=Tracker.disabled())
        pipe.ensure_baseline()
        baseline_model = pipe.current_model()
        sc = ScenarioConfig(name=scenario, drift_start=cfg.drift_start, ramp_batches=cfg.ramp_batches, seed=seed)
        sim = DriftSimulator(pipe.splits, sc, cfg.batch_size)
        for i in range(cfg.n_batches):
            pipe.process_batch(sim.batch(i))
        events = pipe.history.all()
        mf = monitoring_frame(events)
        regime = sim.regime_batch(1.0, cfg.regime_eval_rows)
        base_m = evaluate_model(baseline_model, regime.X, regime.y)
        fin_m = evaluate_model(pipe.current_model(), regime.X, regime.y)
        final_version = pipe.current_record().version

    retr = [e for e in events if e["event_type"] == "retrain"]
    count = lambda s: sum(1 for e in retr if e["status"] == s)  # noqa: E731
    fired = mf[mf["action"].str.startswith("retrain_") & (mf["action"] != "retrain_suppressed")]
    first_idx = float(fired["batch_index"].min()) if len(fired) else np.nan
    onset = cfg.drift_start if scenario != "baseline" else np.nan
    pre = mf[mf["batch_index"] < cfg.drift_start]["f1"].mean()
    post = mf[mf["batch_index"] >= cfg.post_start]["f1"].mean()
    run_row = {
        "seed": seed, "scenario": scenario, "arm": label, "policy": policy, "candidate_data": candidate_data,
        "n_attempts": count("promoted") + count("rejected"), "n_promoted": count("promoted"),
        "n_rejected": count("rejected"), "n_skipped": count("skipped"),
        "first_trigger_index": first_idx,
        "trigger_delay": (first_idx - onset) if not np.isnan(first_idx) and not np.isnan(onset) else np.nan,
        "final_version": final_version, "mean_f1_pre": pre, "mean_f1_post": post,
        "test_f1_baseline_model": base_m["f1"], "test_f1_final_model": fin_m["f1"],
        "test_delta_f1": fin_m["f1"] - base_m["f1"],
        "test_recall_baseline_model": base_m["recall"], "test_recall_final_model": fin_m["recall"],
    }
    batch_rows = mf[["batch_index", "intensity", "model_version", "n_flagged", "dataset_drift", "action",
                     "accuracy", "precision", "recall", "f1", "roc_auc"]].copy()
    batch_rows.insert(0, "arm", label)
    batch_rows.insert(0, "scenario", scenario)
    batch_rows.insert(0, "seed", seed)
    return run_row, batch_rows


def run_retraining_experiment(cfg: ExperimentConfig, verbose: bool = True):
    seeds = range(cfg.seed_start, cfg.seed_start + cfg.n_seeds_retraining)
    runs, batches, t0 = [], [], time.time()
    for seed in seeds:
        for sc in cfg.scenarios:
            for arm in cfg.arms:
                r, b = run_retraining_one(seed, sc, arm, cfg)
                runs.append(r)
                batches.append(b)
        if verbose:
            print(f"[retraining] seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(runs), pd.concat(batches, ignore_index=True)
