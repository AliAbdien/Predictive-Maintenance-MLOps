# Predictive-Maintenance-MLOps

A deliberately simple, well-understood classifier (XGBoost on tabular sensor
data) wrapped in the full production-engineering layer around it: experiment
tracking and a model registry (MLflow), a served REST API (FastAPI), data-drift
monitoring against the training distribution, containerization, and a CI/CD
pipeline. The model is intentionally the least interesting part of this repo -
the point is everything *around* it.

Local-first, same as my other projects: no cloud ML platform, no managed
model-serving product. Everything here runs on your own machine with `docker
compose up`.

## Why this project

[AI-Visual-Inspector](../AI-Visual-Inspector) and my MAHLE internship are both
about *building* a model. This project is about what happens *after* a model
works: how do you know it's still trustworthy six months later, when the
sensors it depends on start drifting? A model that scores 0.97 ROC-AUC in a
notebook and is never checked again is not a production system.

## What it does

```
Sensor reading (temperature, rpm, torque, tool wear)
        │
        ▼
FastAPI /predict  ──► loads the current @champion model version
        │               from the MLflow registry (an alias, not a
        │               hardcoded path - promoting a new model is
        │               a registry alias move, not a redeploy)
        ▼
Prediction returned + logged to a JSONL prediction log
        │
        ▼
Streamlit dashboard reads that log and runs a Kolmogorov-Smirnov
test per sensor against the training distribution → flags exactly
which sensor has drifted, not just "something changed"
```

## Real results

Dataset: [AI4I 2020 Predictive Maintenance Dataset](https://github.com/jonathanwvd/awesome-industrial-datasets/blob/master/markdown/ai4i_2020_predictive_maintenance_dataset.md)
(Matzka, 2020) - 10,000 industrial sensor readings, 339 real machine-failure
events (3.39% - a realistic factory failure rate, not an artificially balanced
toy set). An 80/20 stratified split keeps that rate identical in train and
test.

**The failure-mode breakdown columns the dataset ships (`TWF`, `HDF`, `PWF`,
`OSF`, `RNF`) are dropped from the feature set on purpose.** They're a
diagnosis of *which* failure happened - using them as inputs would be target
leakage, since in production you don't have tomorrow's failure diagnosis
today. This is the single easiest mistake to make with this dataset and it's
called out explicitly in `src/data.py`.

Two models, trained on the exact same pipeline, compared honestly (not
cherry-picked - both are logged to MLflow either way):

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Random Forest (class-weighted) | 0.557 | 0.721 | 0.628 | 0.961 | 0.634 |
| **XGBoost** (`scale_pos_weight`) | **0.635** | **0.794** | **0.706** | **0.970** | **0.815** |

XGBoost wins and is registered as `@champion`. The model is selected on
**PR-AUC**, not accuracy or even ROC-AUC - at a 3.4% positive rate, a model
that never predicts a failure scores 96.6% accuracy while being worthless,
and ROC-AUC is optimistic under heavy class imbalance. PR-AUC is what
actually reflects "when this model raises an alarm, how often is it right,
and how many real failures does it catch."

### Drift detection, actually caught something

`scripts/simulate_stream.py --mode drift` replays 200 real held-out sensor
readings against the live API, and after row 100 applies a believable
shop-floor shift: hotter ambient air (+6K), hotter process temperature (+4K),
and a batch of already-worn tooling mistakenly put back into rotation (+140
min tool wear). The dashboard's KS-test correctly separates the three
sensors that actually shifted from the two that didn't:

| Feature | KS statistic | p-value | Drift flagged? |
|---|---|---|---|
| Air temperature [K] | 0.432 | <0.001 | ✅ yes |
| Process temperature [K] | 0.393 | <0.001 | ✅ yes |
| Tool wear [min] | 0.339 | <0.001 | ✅ yes |
| Rotational speed [rpm] | 0.087 | 0.097 | ❌ no (correctly) |
| Torque [Nm] | 0.081 | 0.146 | ❌ no (correctly) |

That's the whole point of the monitoring layer: it didn't just say "traffic
looks different", it named the three sensors that actually moved and left
the two untouched ones alone.

## Tech stack

- **Model**: XGBoost + scikit-learn (`ColumnTransformer` + `Pipeline`)
- **Experiment tracking & registry**: MLflow (SQLite backend locally, a real
  `mlflow server` container in `docker-compose.yml`)
- **Serving**: FastAPI, model loaded from the registry by alias at startup
  (and reloadable via `POST /reload-model` without a restart)
- **Monitoring**: Streamlit dashboard, drift via two-sample Kolmogorov-Smirnov
  test (`scipy.stats.ks_2samp`) per numeric feature
- **Packaging**: Docker (separate images for API / dashboard / MLflow server),
  orchestrated with `docker-compose.yml`
- **CI/CD**: GitHub Actions - lint (`ruff`), tests (`pytest`), a sanity
  training run, then a Docker image build, on every push

## Project structure

```
Predictive-Maintenance-MLOps/
├── src/
│   ├── data.py           # loading, the leakage-column guard, train/test split
│   ├── train.py          # trains + compares both models, registers @champion in MLflow
│   ├── schemas.py         # pydantic request/response models
│   ├── drift.py            # KS-test drift detection
│   └── prediction_log.py    # JSONL log every /predict call writes to
├── api/
│   └── main.py                # FastAPI app: /predict, /health, /reload-model
├── monitoring/
│   └── dashboard.py             # Streamlit: volume, failure rate, drift table
├── scripts/
│   └── simulate_stream.py         # replays real data at the API, optionally with injected drift
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.dashboard
│   └── Dockerfile.mlflow
├── docker-compose.yml               # mlflow + api + dashboard, one command
├── .github/workflows/ci.yml           # lint, test, sanity-train, build images
├── tests/                               # pytest - data integrity, drift math, API contract
└── data/ai4i2020.csv                      # AI4I 2020 dataset (committed - 500KB, no PII)
```

## Setup

### Docker (recommended - the whole stack, one command)

```bash
docker compose up --build -d mlflow
# wait for mlflow to report healthy, then train + register a model into it:
docker compose run --rm api python -m src.train --tracking-uri http://mlflow:5000
docker compose up --build -d api dashboard
```

- API: http://localhost:8000/docs (interactive Swagger UI)
- MLflow UI: http://localhost:5000
- Monitoring dashboard: http://localhost:8501

Generate some traffic to see the dashboard do something:

```bash
pip install requests
python -m scripts.simulate_stream --mode drift --n 200
```

### Local (no Docker)

```bash
pip install -r requirements-dev.txt
python -m src.train                       # trains, logs to MLflow, registers @champion
uvicorn api.main:app --reload             # in one terminal
python -m scripts.simulate_stream --mode drift --n 200   # in another
streamlit run monitoring/dashboard.py     # in a third
```

### Tests

```bash
pytest tests/ -v
ruff check .
```

## Design notes

- **The model is loaded by registry alias, not a file path.** `@champion` is
  a pointer MLflow lets you move between model versions; promoting a
  retrained model to production is `client.set_registered_model_alias(...)`,
  not editing a deploy config or rebuilding the API image.
- **Every prediction is logged, not just the model's decision.** The drift
  dashboard is only as good as the data it has to compare - if you don't log
  what the model actually saw in production, you can never tell whether the
  world moved.
- **Drift is measured per-feature, not globally.** "Something changed" isn't
  actionable on a factory floor; "the ambient air temperature sensor drifted"
  is - it tells you where to go look.
- **Class imbalance is handled explicitly**, not left to sklearn's defaults:
  `class_weight="balanced"` for the Random Forest, an explicit
  `scale_pos_weight` (computed from the real ~28.5:1 negative:positive ratio
  in the training split) for XGBoost. And the model is picked on PR-AUC, the
  metric that's actually informative at a 3.4% positive rate.

## Limitations

- `docker-compose.yml` is validated (`docker compose config` parses cleanly,
  and every service ran and was tested individually in a plain Python
  environment) but the full multi-container stack itself wasn't build-tested
  in the environment this was developed in - it had no outbound access to
  Docker Hub. Runs on a machine with normal internet access.
- The simulated stream (`scripts/simulate_stream.py`) replays real held-out
  data with a synthetic shift applied for the drift demo - it's a stand-in
  for a real live sensor feed, not one.
- Drift detection alerts (flags a feature in the dashboard); it doesn't yet
  trigger an automatic retrain. That's a reasonable next step, not something
  claimed here.

## Companion projects

Part of the same portfolio as [AI-Portfolio-Agent](../AI-Portfolio-Agent)
(RAG + tool-calling agent), [AI-Dev-Crew](../AI-Dev-Crew) (multi-agent
orchestration), and [AI-Visual-Inspector](../AI-Visual-Inspector) (computer
vision + VLM explanation) - this one is deliberately about the engineering
discipline around a model rather than the model itself.
