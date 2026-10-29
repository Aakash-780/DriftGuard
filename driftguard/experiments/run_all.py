"""Reproduce every experiment: ``python -m driftguard.experiments.run_all``.

Outputs (default ``artifacts/experiments/``): config.json, detection_batches.csv, retraining_runs.csv,
retraining_batches.csv, derived summary CSVs, figures, RESULTS.md.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from driftguard.core.config import DEFAULT_DETECTORS
from driftguard.core.logging import configure_logging
from driftguard.core.utils import to_jsonable
from driftguard.experiments.common import ExperimentConfig, env_info
from driftguard.experiments.make_report import build_report
from driftguard.experiments.run_detection import run_detection_experiment, run_sensitivity_experiment
from driftguard.experiments.run_retraining import run_retraining_experiment


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/experiments")
    ap.add_argument("--seeds-detection", type=int, default=ExperimentConfig.n_seeds_detection)
    ap.add_argument("--seeds-retraining", type=int, default=ExperimentConfig.n_seeds_retraining)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--batches", type=int, default=ExperimentConfig.n_batches)
    ap.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    ap.add_argument("--with-evidently", action="store_true", help="add the optional Evidently detector")
    ap.add_argument("--quick", action="store_true", help="tiny run for smoke-testing (2 seeds, 8 batches)")
    ap.add_argument("--skip-retraining", action="store_true")
    ap.add_argument("--only-sensitivity", action="store_true", help="(re)run only experiment 3 and rebuild the report")
    a = ap.parse_args(argv)
    configure_logging("WARNING")

    detectors = DEFAULT_DETECTORS + (("evidently",) if a.with_evidently else ())
    cfg = ExperimentConfig(
        n_seeds_detection=2 if a.quick else a.seeds_detection,
        n_seeds_retraining=2 if a.quick else a.seeds_retraining,
        seed_start=a.seed_start, n_batches=8 if a.quick else a.batches, batch_size=a.batch_size,
        detectors=detectors,
    )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(to_jsonable({"config": asdict(cfg), "environment": env_info()}), indent=2), encoding="utf-8")

    if a.only_sensitivity:  # keep the existing config.json's other fields; only add the sweep
        run_sensitivity_experiment(cfg).to_csv(out / "sensitivity_batches.csv", index=False)
        print("wrote", build_report(out))
        return
    det = run_detection_experiment(cfg)
    det.to_csv(out / "detection_batches.csv", index=False)
    run_sensitivity_experiment(cfg).to_csv(out / "sensitivity_batches.csv", index=False)
    if not a.skip_retraining:
        runs, batches = run_retraining_experiment(cfg)
        runs.to_csv(out / "retraining_runs.csv", index=False)
        batches.to_csv(out / "retraining_batches.csv", index=False)
    print("wrote", build_report(out))


if __name__ == "__main__":
    main()
