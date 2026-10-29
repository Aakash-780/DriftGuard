"""Command-line interface: ``python -m driftguard.cli {train,simulate,status,reset}``."""
from __future__ import annotations

import argparse
import json

from driftguard.core.config import Settings
from driftguard.core.logging import configure_logging
from driftguard.core.utils import to_jsonable
from driftguard.data.simulation import SCENARIOS, ScenarioConfig
from driftguard.pipelines.pipeline import DriftGuardPipeline


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="driftguard", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train", help="train and deploy the baseline model if none exists")
    sim = sub.add_parser("simulate", help="run a drift scenario through the pipeline")
    sim.add_argument("--scenario", choices=SCENARIOS, default="sudden")
    sim.add_argument("--batches", type=int, default=12)
    sim.add_argument("--drift-start", type=int, default=4)
    sim.add_argument("--seed", type=int, default=0)
    sim.add_argument("--no-retrain", action="store_true")
    sub.add_parser("status", help="print model and monitoring summary")
    rs = sub.add_parser("reset", help="delete runtime state and retrain the baseline")
    rs.add_argument("--yes", action="store_true", help="confirm deletion")
    a = ap.parse_args(argv)

    settings = Settings.from_env()
    configure_logging(settings.log_level)
    pipe = DriftGuardPipeline(settings)
    if a.cmd == "train":
        rec = pipe.ensure_baseline()
        print(json.dumps(rec.to_dict(), indent=2))
    elif a.cmd == "simulate":
        pipe.ensure_baseline()
        sc = ScenarioConfig(name=a.scenario, drift_start=a.drift_start, seed=a.seed)
        for ev in pipe.run_scenario(sc, a.batches, allow_retrain=not a.no_retrain):
            if ev["status"] != "ok":
                print(f"batch {ev['batch_index']:>2}: {ev['status']} - {ev.get('message', '')}")
                continue
            f1 = (ev["performance"] or {}).get("f1")
            print(f"batch {ev['batch_index']:>2} drift={ev['drift']['n_flagged']}/{ev['drift']['n_active']} "
                  f"f1={f1:.3f} model=v{ev['model_version']} action={ev['action']}")
    elif a.cmd == "status":
        pipe.ensure_baseline()
        print(json.dumps(to_jsonable(pipe.metrics_summary()), indent=2))
    elif a.cmd == "reset":
        if not a.yes:
            raise SystemExit("refusing to delete state without --yes")
        print(json.dumps(pipe.reset().to_dict(), indent=2))


if __name__ == "__main__":
    main()
