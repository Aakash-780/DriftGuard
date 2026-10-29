"""Build plots and RESULTS.md strictly from the saved experiment CSV files (no hand-entered numbers)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from driftguard.experiments.common import ExperimentConfig, md_table  # noqa: E402
from driftguard.experiments.metrics import (  # noqa: E402
    summarize_detection, summarize_fpr, summarize_retraining, summarize_sensitivity, summarize_static_performance,
)

DRIFT_SCENARIOS = ["covariate", "sudden", "gradual", "concept"]


def _cfg_from_json(path: Path) -> ExperimentConfig:
    raw = json.loads(path.read_text())["config"]
    return ExperimentConfig(
        n_seeds_detection=raw["n_seeds_detection"], n_seeds_retraining=raw["n_seeds_retraining"],
        seed_start=raw["seed_start"], n_batches=raw["n_batches"], batch_size=raw["batch_size"],
        drift_start=raw["drift_start"], ramp_batches=raw["ramp_batches"],
        scenarios=tuple(raw["scenarios"]), detectors=tuple(raw["detectors"]),
        regime_eval_rows=raw["regime_eval_rows"], arms=tuple(tuple(a) for a in raw["arms"]),
        sensitivity_magnitudes=tuple(raw.get("sensitivity_magnitudes", ExperimentConfig.sensitivity_magnitudes)),
    )


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_detection(det_sum: pd.DataFrame, fpr: pd.DataFrame, names: list[str], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    sub = det_sum[det_sum["scenario"].isin(DRIFT_SCENARIOS)]
    width = 0.8 / max(1, len(names))
    for i, det in enumerate(names):
        vals = [sub[(sub.scenario == s) & (sub.detector == det)]["detection_rate_full_intensity"].mean()
                for s in DRIFT_SCENARIOS]
        ax.bar(np.arange(len(DRIFT_SCENARIOS)) + i * width, vals, width, label=det)
    ax.set_xticks(np.arange(len(DRIFT_SCENARIOS)) + width * (len(names) - 1) / 2)
    ax.set_xticklabels(DRIFT_SCENARIOS)
    ax.set_ylim(0, 1.05); ax.set_ylabel("detection rate (full-intensity batches)")
    ax.set_title("Detection rate by scenario"); ax.legend(fontsize=8)
    ax = axes[1]
    ax.bar(fpr["detector"], fpr["fpr"],
           yerr=[fpr["fpr"] - fpr["wilson95_lo"], fpr["wilson95_hi"] - fpr["fpr"]], capsize=4, color="#c0504d")
    ax.set_ylim(0, max(0.2, float(fpr["wilson95_hi"].max()) * 1.1)); ax.set_ylabel("false-positive rate")
    ax.set_title("FPR on no-drift batches\n(Wilson 95% CI)"); ax.tick_params(axis="x", rotation=30)
    _save(fig, out / "fig_detection_rates.png")


def plot_delay(det_sum: pd.DataFrame, names: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sub = det_sum[det_sum["scenario"].isin(DRIFT_SCENARIOS)]
    width = 0.8 / max(1, len(names))
    for i, det in enumerate(names):
        vals, labels = [], []
        for s in DRIFT_SCENARIOS:
            r = sub[(sub.scenario == s) & (sub.detector == det)]
            vals.append(float(r["mean_delay_batches"].iloc[0]) if len(r) else np.nan)
            labels.append(f"{int(r['n_detected_runs'].iloc[0])}/{int(r['n_runs'].iloc[0])}" if len(r) else "")
        xs = np.arange(len(DRIFT_SCENARIOS)) + i * width
        bars = ax.bar(xs, np.nan_to_num(vals), width, label=det)
        for x, v, t in zip(xs, vals, labels):
            ax.text(x, (0 if np.isnan(v) else v) + 0.03, t, ha="center", fontsize=7, rotation=90)
    ax.set_xticks(np.arange(len(DRIFT_SCENARIOS)) + width * (len(names) - 1) / 2)
    ax.set_xticklabels(DRIFT_SCENARIOS)
    ax.set_ylabel("mean delay (batches) among runs that detected")
    ax.set_title("Detection delay (labels: runs detected / runs)"); ax.legend(fontsize=8)
    _save(fig, out / "fig_detection_delay.png")


def plot_timelines(df: pd.DataFrame, group_col: str | None, title: str, fname: str, cfg, out: Path,
                   metric: str = "f1") -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey=True)
    for ax, sc in zip(axes.ravel(), cfg.scenarios):
        sub = df[df["scenario"] == sc]
        groups = [(None, sub)] if group_col is None else list(sub.groupby(group_col, sort=False))
        for name, g in groups:
            m = g.groupby("batch_index")[metric].agg(["mean", "std"])
            ax.plot(m.index, m["mean"], marker="o", ms=3, label=name or "static baseline model")
            if group_col is None:
                ax.fill_between(m.index, m["mean"] - m["std"], m["mean"] + m["std"], alpha=0.2)
        if sc != "baseline":
            ax.axvline(cfg.drift_start - 0.5, color="k", ls="--", lw=0.8)
        ax.set_title(sc); ax.set_xlabel("batch index"); ax.set_ylim(0.2, 1.02)
    for ax in axes.ravel()[len(cfg.scenarios):]:
        ax.axis("off")
    axes[0, 0].set_ylabel(metric.upper())
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower right", fontsize=8)
    fig.suptitle(title)
    _save(fig, out / fname)


def plot_sensitivity(sens: pd.DataFrame, names: list[str], out: Path) -> None:
    modes = list(dict.fromkeys(sens["sign_mode"]))
    fig, axes = plt.subplots(1, len(modes), figsize=(6.2 * len(modes), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, mode in zip(axes, modes):
        sub = sens[sens["sign_mode"] == mode]
        for det in names:
            ax.plot(sub["shift_magnitude"], sub[f"{det}_detection_rate"], marker="o", label=det)
        ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("shift magnitude (reference σ, 40% of features)")
        ax2 = ax.twinx()
        ax2.plot(sub["shift_magnitude"], sub["static_f1_delta"], "k--", marker="s", label="static ΔF1 (right axis)")
        ax2.fill_between(sub["shift_magnitude"], sub["static_f1_delta_ci95_lo"], sub["static_f1_delta_ci95_hi"],
                         color="k", alpha=0.1)
        ax2.set_ylim(-0.28, 0.05)
        ax2.set_ylabel("static model ΔF1 after − before (dashed)")
        ax.set_title(f"shift signs: {mode}")
    axes[0].set_ylabel("detection rate (full-intensity batches)")
    h1, l1 = axes[0].get_legend_handles_labels()
    axes[0].legend(h1, l1, fontsize=8, loc="center right")
    _save(fig, out / "fig_sensitivity.png")


def plot_retraining(summary: pd.DataFrame, cfg, out: Path) -> None:
    arms = [a[0] for a in cfg.arms if a[0] != "never"]
    scen = [s for s in cfg.scenarios]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    width = 0.8 / max(1, len(arms))
    for i, arm in enumerate(arms):
        xs = np.arange(len(scen)) + i * width
        att = [summary[(summary.scenario == s) & (summary.arm == arm)]["mean_attempts"].mean() for s in scen]
        axes[0].bar(xs, att, width, label=arm)
        d = [summary[(summary.scenario == s) & (summary.arm == arm)] for s in scen]
        m = [x["test_delta_f1"].mean() for x in d]
        lo = [x["test_delta_ci95_lo"].mean() for x in d]
        hi = [x["test_delta_ci95_hi"].mean() for x in d]
        err = np.array([np.nan_to_num(np.array(m) - np.array(lo)), np.nan_to_num(np.array(hi) - np.array(m))])
        axes[1].bar(xs, m, width, yerr=err, capsize=2, label=arm)
    for ax in axes:
        ax.set_xticks(np.arange(len(scen)) + width * (len(arms) - 1) / 2); ax.set_xticklabels(scen)
    axes[0].set_ylabel("mean retraining attempts per run"); axes[0].set_title("Retraining triggers")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel("ΔF1 on post-drift TEST sample (final − baseline)"); axes[1].set_title("Recovery (95% CI over seeds)")
    axes[0].legend(fontsize=7)
    _save(fig, out / "fig_retraining_outcomes.png")


def build_report(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    cfg = _cfg_from_json(out_dir / "config.json")
    env = json.loads((out_dir / "config.json").read_text())["environment"]
    names = list(cfg.detectors) + ["prediction"]
    parts = ["# DriftGuard experiment results", "",
             "_Generated by `python -m driftguard.experiments.make_report` from the CSV files in this folder. "
             "Every number below is computed from those files; nothing is hand-entered._", ""]

    det_path = out_dir / "detection_batches.csv"
    if det_path.exists():
        det = pd.read_csv(det_path)
        det_sum = summarize_detection(det, names, cfg)
        fpr = summarize_fpr(det, names)
        perf = summarize_static_performance(det, cfg)
        det_sum.to_csv(out_dir / "detection_summary.csv", index=False)
        fpr.to_csv(out_dir / "detection_fpr_pooled.csv", index=False)
        perf.to_csv(out_dir / "static_performance_shift.csv", index=False)
        plot_detection(det_sum, fpr, names, out_dir)
        plot_delay(det_sum, names, out_dir)
        plot_timelines(det, None, "Static (never retrained) model: batch F1, mean ± SD over seeds",
                       "fig_static_f1_timeline.png", cfg, out_dir)
        parts += [
            "## Experiment 1 — drift detectors (model never retrained)", "",
            f"Sample: {cfg.n_seeds_detection} seeds (data split and simulation both vary with the seed) × "
            f"{len(cfg.scenarios)} scenarios × {cfg.n_batches} batches of {cfg.batch_size} rows; "
            f"drift onset at batch {cfg.drift_start}; ramp length {cfg.ramp_batches} (gradual). "
            f"Batches per scenario-detector cell: {cfg.n_seeds_detection * cfg.n_batches}.", "",
            "### False-positive rate (pooled over all batches with no drift present)", "", md_table(fpr), "",
            "### Detection rate, delay (per scenario × detector)", "",
            md_table(det_sum[det_sum["scenario"] != "baseline"][
                ["scenario", "detector", "n_runs", "detection_rate_post_onset", "detection_rate_full_intensity",
                 "run_level_detection_rate", "n_detected_runs", "mean_delay_batches"]]), "",
            "### Static-model performance before vs after drift (F1)", "", md_table(perf), "",
            "![detection](fig_detection_rates.png)", "", "![delay](fig_detection_delay.png)", "",
            "![static](fig_static_f1_timeline.png)", "",
        ]

    sens_path = out_dir / "sensitivity_batches.csv"
    if sens_path.exists():
        sens = summarize_sensitivity(pd.read_csv(sens_path), names, cfg)
        sens.to_csv(out_dir / "sensitivity_summary.csv", index=False)
        plot_sensitivity(sens, names, out_dir)
        parts += [
            "## Experiment 3 — sensitivity to drift magnitude (static model, `sudden` scenario)", "",
            f"{cfg.n_seeds_detection} seeds per magnitude; detection rate over full-intensity batches; FPR over no-drift batches "
            "at that magnitude; `static_f1_delta` = mean batch F1 after − before onset with 95% CI over seeds.", "",
            md_table(sens), "", "![sensitivity](fig_sensitivity.png)", "",
        ]

    runs_path, b_path = out_dir / "retraining_runs.csv", out_dir / "retraining_batches.csv"
    if runs_path.exists() and b_path.exists():
        runs, batches = pd.read_csv(runs_path), pd.read_csv(b_path)
        rs = summarize_retraining(runs, cfg)
        rs.to_csv(out_dir / "retraining_summary.csv", index=False)
        plot_timelines(batches, "arm", "Deployed-model batch F1 by retraining arm (mean over seeds)",
                       "fig_retraining_f1_timeline.png", cfg, out_dir)
        plot_retraining(rs, cfg, out_dir)
        parts += [
            "## Experiment 2 — retraining policies and promotion gate", "",
            f"Sample: {cfg.n_seeds_retraining} seeds × {len(cfg.scenarios)} scenarios × {len(cfg.arms)} arms "
            f"(each a full pipeline run of {cfg.n_batches} batches). Arms see identical data per (seed, scenario).", "",
            "Arms (policy, candidate-data strategy): " + "; ".join(f"`{a[0]}`" for a in cfg.arms), "",
            md_table(rs[["scenario", "arm", "n_runs", "mean_attempts", "mean_promoted", "mean_rejected",
                         "acceptance_rate", "mean_trigger_delay", "mean_f1_post", "instream_gain_vs_never",
                         "gain_ci95_lo", "gain_ci95_hi", "test_delta_f1", "test_delta_ci95_lo",
                         "test_delta_ci95_hi"]]), "",
            "**`test_delta_f1` is the leak-free recovery metric**: F1(final deployed) − F1(baseline) on a post-drift "
            "sample from the untouched TEST split. `instream_gain_vs_never` (paired difference in mean batch F1 "
            "after full drift intensity, arm minus control) is **optimistic**: later batches re-sample the same "
            "199-row stream pool whose rows the candidate was trained on, so treat it as indicative only.", "",
            "![timeline](fig_retraining_f1_timeline.png)", "", "![outcomes](fig_retraining_outcomes.png)", "",
        ]
    parts += ["## Environment and configuration", "", "```json",
              json.dumps({"environment": env, "config": json.loads((out_dir / "config.json").read_text())["config"]},
                         indent=2), "```", ""]
    path = out_dir / "RESULTS.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/experiments")
    print(build_report(Path(ap.parse_args().out)))
