# DriftGuard

**An Automated MLOps Framework for Data Drift Detection, Model Monitoring, and Adaptive Retraining**

A runnable, research-oriented MLOps project: it trains a baseline classifier, simulates incoming data under normal and drifted
conditions, detects drift with several methods, monitors model performance, decides whether to retrain under configurable rules,
validates a candidate against the deployed model, and only then promotes it. Everything is tracked (local model registry,
event history, optional MLflow), exposed through a REST API, and visualised in a Streamlit dashboard. A reproducible experiment
suite evaluates the detectors, the retraining policies and the promotion gate.

> **Read this first — verification status.** The project was authored in a sandbox without network access, so some
> dependencies could not be installed there. The table is deliberately explicit about what was and was not executed.

| Component | Status when this project was produced |
|---|---|
| Data, simulation, detectors (KS / PSI / domain classifier), models, registry, promotion gate, rules, pipeline, CLI, experiments, plots | **Executed**; experiment results in `artifacts/experiments/` come from real runs |
| Test suite (`tests/`) | 89 tests **passed** in the sandbox using a small stand-in runner (real `pytest` was not installable offline); 3 skipped there (2 MLflow tests, the API test module). **Real `pytest` has not been run yet — run it first.** |
| FastAPI app (`driftguard/app/`) | Written and byte-compiled; **never started** (FastAPI not installable offline). Tests in `tests/test_api.py` are written but **were not executed** |
| Streamlit dashboard | Written and byte-compiled; its own logic was smoke-tested against a *stub* `streamlit` module on populated and empty state. **Never rendered in real Streamlit** |
| MLflow integration (`core/tracking.py`) | Written against the MLflow 2.16 fluent API; **never executed** (tests exist and will run in CI). Failure never breaks the pipeline: it degrades to a no-op |
| Evidently adapter | Written against the 0.4.x legacy API; **never executed**. Opt-in only |
| GitHub Actions | Written; never run |

Pinned versions in `requirements.txt` were chosen as a mutually compatible 2024 set but **were not resolved by pip here**;
the numerical core was executed on newer versions (see `artifacts/experiments/config.json`). If a pin conflicts on your machine,
loosen it and note the change. The first thing to do after cloning is `pip install -r requirements.txt && python -m pytest`.

## 1. Problem statement and research objectives

Deployed models degrade silently when the world changes. Practitioners watch for *data drift* because labels arrive late, but drift
is only a proxy: a model can be unaffected by a large input shift, and can collapse under a change (concept drift) that no
input-only detector can see. DriftGuard operationalises that distinction end to end and asks:

1. **RQ1** How do statistical (KS), magnitude-based (PSI) and multivariate (domain-classifier) drift detectors compare in power, false-positive rate and delay?
2. **RQ2** How often does detected drift coincide with real performance loss (feature drift ≠ degradation; degradation ≠ feature drift)?
3. **RQ3** Which retraining trigger policy (performance / drift / both / either) best balances recovery against wasted or harmful retraining?
4. **RQ4** Does a validation-gated promotion protocol prevent harmful model updates, and at what cost?

Results and honest interpretation: [`docs/RESULTS_DISCUSSION.md`](docs/RESULTS_DISCUSSION.md); raw tables and figures:
[`artifacts/experiments/RESULTS.md`](artifacts/experiments/RESULTS.md); precise metric definitions:
[`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md).

## 2. Architecture

```
            simulated / supplied batch
                        │
              ┌─────────▼─────────┐  invalid → recorded, stopped
              │ schema + value    │
              │ validation        │
              └─────────┬─────────┘
                        │ duplicate fingerprint → skipped (no state change, no retrain)
              ┌─────────▼─────────┐      ┌────────────────────────┐
              │ drift detectors   │      │ prediction-drift check │
              │ KS · PSI · domain │      │ (KS on P(malignant))   │
              └─────────┬─────────┘      └───────────┬────────────┘
                        │      labels? → evaluate deployed model (acc/prec/rec/F1/AUC)
              ┌─────────▼────────────────────────────▼───┐
              │ retraining rules (policy, consecutive,    │
              │ cooldown, min labeled rows)               │
              └─────────┬────────────────────────────────┘
                 triggered│
              ┌─────────▼─────────┐   train candidate on  train split + window(train-ids)
              │ candidate training│   validate on         window(holdout-ids)  +  anchor split
              └─────────┬─────────┘
              ┌─────────▼─────────┐  all gates pass → promote (registry pointer swap,
              │ promotion gate    │                    reference + baseline refreshed)
              └─────────┬─────────┘  otherwise      → reject, deployed model untouched
                        │
        history.jsonl · registry/ · buffer.csv · MLflow (optional)
                        │
              ┌─────────┴──────────┐
              │ FastAPI    Streamlit│  both call the same DriftGuardPipeline service
              └────────────────────┘
```

Layout (a `driftguard/` Python package inside the repository root, so tests, CI files and requirements sit beside it and
`python -m driftguard...` works without installation):

```
driftguard-project/
├── driftguard/
│   ├── app/           main.py (FastAPI factory + routes), schemas.py (Pydantic)
│   ├── core/          config.py (all thresholds), logging.py, tracking.py (MLflow wrapper), utils.py
│   ├── data/          loader.py (splits), simulation.py (scenarios), validation.py, partition.py (leak-free id split)
│   ├── models/        training.py, evaluation.py, registry.py (versioned, atomic pointer), promotion.py (gate)
│   ├── monitoring/    detectors.py (KS, PSI, domain classifier, prediction drift), evidently_adapter.py (optional)
│   ├── pipelines/     pipeline.py (orchestration), rules.py (triggers), store.py (history/state/buffer), reporting.py
│   ├── dashboard/     app.py (Streamlit)
│   ├── experiments/   run_detection.py, run_retraining.py, metrics.py, make_report.py, run_all.py
│   └── cli.py         train | simulate | status | reset
├── tests/             pytest suite (+ conftest fixtures)
├── docs/              EXPERIMENT_PROTOCOL.md, RESULTS_DISCUSSION.md
├── artifacts/         runtime state (git-ignored) + experiments/ (committed results)
├── .github/workflows/ci.yml · requirements.txt · requirements-evidently.txt · Makefile · .env.example · .streamlit/config.toml
```

### Training / validation / test separation

| Split | Used for | Never used for |
|---|---|---|
| `train` (30 %) | baseline fit; included in every candidate fit | evaluation |
| `reference` (20 %) | drift reference; fixed *anchor* set that guards against forgetting in promotion | training |
| `stream` (35 %) | source of simulated incoming batches | – |
| `test` (15 %) | post-hoc evaluation in the experiments only | any pipeline decision |
| buffered labeled window | rows split **by hash of row id** into candidate-train (70 %) and recent-holdout (30 %); a row id is always on the same side, even when re-sampled into many batches (asserted at runtime) | – |

## 3. Technology stack

Python · scikit-learn (Random Forest baseline, logistic-regression option) · SciPy/NumPy · Pandas · FastAPI + Pydantic v2 + Uvicorn ·
MLflow (SQLite backend, no server required) · Streamlit · Matplotlib · Joblib · pytest · GitHub Actions · Evidently AI (optional).
Rationale for the pins and the Evidently decision: see *Dependency decisions* below.

## 4. Installation

Requires Python 3.10–3.12.

```bash
# macOS / Linux
git clone <your-repo-url> driftguard-project && cd driftguard-project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                       # run this first; see "verification status"
```
```powershell
# Windows (PowerShell)
py -3 -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest
```

Windows differences: use `py -3`, activate with `.venv\Scripts\Activate.ps1` (cmd: `.venv\Scripts\activate.bat`), set variables with
`$env:NAME="value"` (cmd: `set NAME=value`) instead of `export`/`source .env`, and use the commands below instead of `make`
(or install GNU Make). Nothing else is platform-specific.

Configuration is by environment variables (see `.env.example`); defaults work out of the box. Load a file on macOS/Linux with
`set -a; source .env; set +a`. **Run all commands from the repository root** so that `artifacts/` resolves consistently for the API,
dashboard and CLI.

## 5. Dataset and drift simulation

*Breast Cancer Wisconsin (Diagnostic)* ships inside scikit-learn (569 rows × 30 numeric features), so no download is needed and
runs are reproducible offline. Positive class = **malignant** (label 1). Splits are stratified and seeded.

| Scenario | What changes | P(x) | P(y\|x) |
|---|---|---|---|
| `baseline` | nothing | same | same |
| `covariate` | rows re-sampled with exponential tilt on a subset of features; each row keeps its own label (also raises the malignant share, i.e. a prior shift) | **changes** | exactly preserved |
| `sudden` | at `drift_start`, a subset of features shifts by 1.5 reference-σ | **changes** | rows may cross the boundary |
| `gradual` | the same shift ramped over 4 batches | **changes** | as above |
| `concept` | features untouched; labels flipped (p = 0.8) where `worst perimeter` is above its reference mean | same | **changes** |

Feature drift and concept drift are different failure modes: no feature detector can see `concept`, and `covariate` drift is
flagged without necessarily hurting accuracy. Batch `i` is a pure function of `(seed, i)`. Batches are bootstrap draws from a
finite pool (limitation below).

## 6. Quick start

```bash
python -m driftguard.cli train                       # train + register + deploy the baseline (idempotent)
python -m driftguard.cli simulate --scenario sudden --batches 12 --seed 0
python -m driftguard.cli status                      # JSON summary
python -m driftguard.cli reset --yes                 # wipe runtime state and retrain
```

Actual output of `simulate --scenario sudden --batches 8 --seed 0` (MLflow disabled, fresh state):

```
batch  0 drift=0/3 f1=0.909 model=v1 action=none
batch  1 drift=0/3 f1=0.921 model=v1 action=none
batch  2 drift=0/3 f1=0.925 model=v1 action=none
batch  3 drift=0/3 f1=0.963 model=v1 action=none
batch  4 drift=3/3 f1=0.880 model=v1 action=none
batch  5 drift=3/3 f1=0.897 model=v1 action=retrain_promoted
batch  6 drift=3/3 f1=0.974 model=v2 action=none
batch  7 drift=3/3 f1=0.988 model=v2 action=none
```
Drift starts at batch 4 and is flagged at once, but retraining waits for `min_consecutive = 2` batches of persistence. Running the
same command again prints `duplicate_skipped` for every batch: identical batches are never reprocessed. (The in-stream F1 after
promotion is optimistic; see the discussion document.)

### Baseline model
`python -m driftguard.cli train` writes `artifacts/registry/v0001/{model.joblib,meta.json}` and `artifacts/registry/current.json`.
`meta.json` records hyper-parameters, validation and test metrics and the training-data fingerprint.

### MLflow
MLflow uses a local SQLite database (`artifacts/mlflow/mlflow.db`) and a local artifact folder; **no server is required to record
runs**. The server/UI is only for browsing:

```bash
python -m driftguard.cli train                                   # creates the DB on first run
mlflow ui --backend-store-uri sqlite:///$(pwd)/artifacts/mlflow/mlflow.db --port 5000     # macOS/Linux
# Windows PowerShell:  mlflow ui --backend-store-uri "sqlite:///$((Get-Location).Path -replace '\\','/')/artifacts/mlflow/mlflow.db" --port 5000
```
Open <http://127.0.0.1:5000>, experiment `driftguard`. Runs: `baseline-vN`, `monitor-batch-N` (drift scores/flags per detector,
performance metrics, trigger decision, tags for scenario/action/model version, detector configuration as parameters) and `retrain-vN`
(hyper-parameters, candidate vs deployed metrics, rule and gate configuration, trigger reasons, `decision.json`, the model artifact;
promoted models are also registered as `driftguard-classifier` in the MLflow model registry). Set `DRIFTGUARD_MLFLOW=false` to turn it off.
The authoritative serving registry is the local `artifacts/registry/`; MLflow is the audit trail.

### FastAPI
```bash
uvicorn driftguard.app.main:app --port 8000        # interactive docs: http://127.0.0.1:8000/docs
```
On start-up the baseline is trained if absent (`DRIFTGUARD_AUTO_TRAIN=true`). The API is synchronous (a retrain takes about a second
on a laptop); for real deployments move training to a worker. It accepts only numeric records and enum/integer parameters — never file
paths or code. **Run only one writer at a time** (API *or* dashboard actions) because both mutate the same `artifacts/` state.

| Method | Path | Purpose | Notes |
|---|---|---|---|
| GET | `/health` | liveness + deployed version | |
| GET | `/model/info` | deployed model metadata, feature names, policy | 503 if no model |
| POST | `/predict` | predict for ≤ 1000 records | 422 on missing/extra/NaN/non-numeric fields |
| POST | `/monitor/run` | monitor a supplied batch (`records` + optional `labels`) **or** a generated one (`scenario`) | 422 invalid batch or both/neither source; duplicates return `status: duplicate_skipped` |
| POST | `/simulate/drift` | run a scenario of 1–30 batches through the pipeline | omit `seed` for a time-based seed |
| POST | `/retrain` | rule-checked retrain, or `force: true` to bypass rules (gates still apply) | 409 if nothing/too little new labeled data or window already used |
| GET | `/metrics` | counts, latest drift/performance, baseline metrics, state | |
| GET | `/history` | recent events (`limit` ≤ 500, `event_type` = monitoring \| retrain) | newest first |

Examples (response bodies abbreviated with `…`; field names are exact, values vary):

```bash
curl -s localhost:8000/health
# {"status":"ok","model_loaded":true,"model_version":1,"version":"0.1.0"}

curl -s -X POST localhost:8000/simulate/drift -H 'content-type: application/json' \
     -d '{"scenario":"sudden","n_batches":10,"drift_start":4,"seed":0}'
# {"scenario":"sudden","seed":0,"n_batches":10,"batches":[{"batch_index":0,"status":"ok","intensity":0.0,
#   "dataset_drift":false,"n_flagged":0,"f1":…,"model_version":1,"action":"none"}, …],
#  "deployed_version":…,"retrain_attempts":…,"duplicates_skipped":0}

curl -s -X POST localhost:8000/monitor/run -H 'content-type: application/json' \
     -d '{"scenario":"concept","batch_index":6,"drift_start":4,"seed":1,"allow_retrain":false}'

curl -s -X POST localhost:8000/retrain -H 'content-type: application/json' -d '{"force":true,"reason":"demo"}'
# {"event_type":"retrain","status":"promoted"|"rejected","candidate_version":…,"decision":{"promote":…,
#   "reasons":[…],"checks":{"f1_improvement":{"passed":…,"value":…,"threshold":0.01}, …}}, …}

# /predict needs all 30 feature names (GET /model/info lists them). Build a valid payload from the dataset:
python - <<'PY' > payload.json
import json; from driftguard.data.loader import make_splits; from driftguard.core.config import DataConfig
print(json.dumps({"records": make_splits(DataConfig()).X_test.head(2).to_dict("records")}))
PY
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d @payload.json
# {"model_version":1,"predictions":[…],"probabilities":[…],"labels":["malignant"|"benign", …]}
```

### Streamlit dashboard
```bash
streamlit run driftguard/dashboard/app.py          # http://localhost:8501
```
Sidebar: scenario, batches, drift start, seed, automatic-retraining switch, **Run scenario**, and a confirmed **Reset**. Tabs:
*Overview* (model status, latest drift/F1, counts), *Drift monitoring* (detector table, score history, per-feature scores, reference vs
incoming histogram, thresholds), *Model performance* (metric history, per-scenario comparison next to drift-flag rate),
*Retraining* (policy, history, per-check promotion results, manual force-retrain), *Experiment tracking* (registry, MLflow runs),
*Research results* (saved experiment tables/figures; empty-state hint if none). Every control calls the same service as the API and
every chart reads recorded state; there are no hard-coded series. The dashboard talks to the service **in-process** rather than over HTTP.

### Tests
```bash
python -m pytest              # all tests; API tests run when fastapi+httpx are installed, MLflow tests when mlflow is
```
Coverage: validation, simulation determinism and drift semantics (concept drift leaves X untouched; covariate shift preserves row
labels), detector behaviour on controlled data (including a documented limitation), BH/PSI/metric arithmetic, registry and promotion
gate (promote / reject / small holdout / regression), trigger-rule truth table, full pipeline behaviour (duplicate and invalid
batches, persistence, rejection keeps the deployed model, promotion updates it, leakage guard), experiment metric arithmetic on
hand-built examples, API endpoints, MLflow round-trip. Tests are deterministic and need no network or paid services.

### Reproduce the research experiments
```bash
python -m driftguard.experiments.run_all            # full: 20 seeds (detection), 10 seeds (retraining); ~8 min on one CPU core
python -m driftguard.experiments.run_all --quick    # 2 seeds smoke run
python -m driftguard.experiments.make_report        # rebuild tables/figures from the saved CSVs only
python -m driftguard.experiments.run_all --with-evidently   # add the optional detector (needs requirements-evidently.txt)
```
Outputs in `artifacts/experiments/`: `config.json` (configuration + library versions), `detection_batches.csv`,
`sensitivity_batches.csv`, `retraining_runs.csv`, `retraining_batches.csv`, derived `*_summary.csv`, five-plus `fig_*.png`, and `RESULTS.md`.

## 7. Detectors, assumptions and limitations

| Detector | Detects | Assumptions / limitations | Configuration (defaults, all in `DriftConfig`) |
|---|---|---|---|
| KS | per-feature marginal shift | independent-ish features; blind to joint changes; many correlated tests → BH correction; p-values depend on n | α 0.05, BH, dataset drift if ≥ 20 % features |
| PSI | per-feature shift magnitude | thresholds are conventions; small-sample PSI is biased upward ≈ (bins−1)(1/n_ref+1/n_cur), so bins are kept few | 5 quantile bins, feature threshold 0.25, ε 1e-4, ≥ 20 % features |
| Domain classifier | multivariate shift separable by a **linear** model | cannot see covariance-only changes (tested); no per-feature p-value | 5-fold CV AUC ≥ 0.65, C 0.1 |
| Prediction drift | change in the model's output distribution | not a cause; also fires on benign class-mix changes | KS α 0.01 |
| Evidently (opt.) | Evidently's `DataDriftPreset` | KS-based for small numeric data → close to `ks`; adapter unverified | share 0.5 |

The retraining vote requires `min_detectors_drifted = 2` of the active feature detectors. None of the detectors can detect concept drift.
Thresholds are *a-priori defaults, not universal truths*; they are configurable and were not tuned on the experiment results.

## 8. Dependency decisions
* `numpy 1.26 / pandas 2.2 / scipy 1.13 / scikit-learn 1.5` are pinned together because MLflow 2.16 requires `numpy<2` and `pandas<3`. The code avoids newer-only APIs and was also run on numpy 2.4 / pandas 3.0 / scikit-learn 1.8 (the sandbox's versions).
* Evidently 0.4.40 (legacy API) is **optional** (`requirements-evidently.txt`) because its large dependency tree could not be resolved offline against the other pins; the three built-in detectors satisfy the "at least three methods" goal without it. CI has a separate, non-blocking job for it.
* Streamlit 1.38: the dashboard uses `st.image(use_column_width=True)` for that version.
* Pydantic 2.9 is required by the API schemas (`model_validator`, `field_validator`).

## 9. Generating genuine screenshots for the report
Use real runs only; do not edit images.

1. `python -m driftguard.cli reset --yes` (clean state), then `streamlit run driftguard/dashboard/app.py`.
2. **Overview, no data**: capture the empty state. Sidebar → scenario `sudden`, 10 batches, drift start 4, seed 1, **Run scenario** → capture *Overview*.
3. *Drift monitoring*: select a post-onset batch; capture the detector table, score history and the `worst perimeter` histogram. Capture a pre-onset batch for contrast.
4. Run `concept` (new seed) and capture *Model performance* showing F1 falling while drift flags stay off; run `covariate` and capture the opposite.
5. *Retraining*: capture a promoted and a rejected event with the **Promotion checks** table. Use *Force retrain now* twice to show the duplicate-window skip.
6. `mlflow ui …` → capture the experiment run list and one `retrain-vN` run (params, metrics, artifacts).
7. `uvicorn …` → capture `/docs` and a `/simulate/drift` response.
8. *Research results* tab / `artifacts/experiments/fig_*.png` for the experiment chapter. Record the commit hash and `config.json` in the caption.

## 10. Known limitations and future work
* **Synthetic drift.** Scenarios are known, clean and begin at a known batch; real drift is gradual, mixed, unlabeled and entangled with seasonality. Feature detectors are not validated on real production data here.
* **Small pool, bootstrap batches.** Batches resample 199 stream rows; repeated rows shrink effective sample size and make in-stream post-retraining metrics optimistic (use the test-split metric).
* **One small dataset**; absolute numbers do not transfer.
* **Concept drift is not recovered** by the current recipe (see the discussion document), and nothing in the project claims otherwise.
* **Futile repeated retraining**: after a *rejected* candidate the drift reference is not refreshed, so persistent drift keeps re-triggering (cooldown-limited). Exponential back-off or "accept the new regime without retraining" logic is future work.
* **Promotion on small holdouts** (≈100–130 rows) has sampling noise larger than the 0.01 F1 margin; bootstrap/paired tests would be more defensible.
* Labels are assumed to arrive with each batch (no label delay modelling); the API/dashboard share state through files (single writer); the pipeline is synchronous; joblib files must only be loaded from trusted directories.
* The linear domain classifier misses covariance-only change; a gradient-boosted classifier or MMD would extend coverage.
# DriftGuard
