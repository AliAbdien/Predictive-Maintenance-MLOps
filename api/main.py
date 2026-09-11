"""FastAPI serving layer for the predictive-maintenance model.

Loads the current @champion model version straight from the MLflow model
registry at startup (not a hardcoded pickle path) - promoting a new model is
an MLflow alias move, not a redeploy. Every prediction is logged to
monitoring/predictions.jsonl for the drift dashboard.
"""
from __future__ import annotations

import os
import sys

from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction_log import log_prediction  # noqa: E402
from src.schemas import HealthResponse, PredictionResponse, SensorReading  # noqa: E402

MODEL_NAME = "predictive-maintenance-classifier"
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
DECISION_THRESHOLD = float(os.environ.get("DECISION_THRESHOLD", "0.5"))

_state: dict = {"model": None, "model_version": None, "error": None}


def _load_model() -> bool:
    """Load the @champion model into _state. Returns whether it succeeded.

    On failure, an *already-loaded* model is deliberately left in place: a
    bad `/reload-model` call (registry down, alias unset, whatever) should
    not take down a server that was serving predictions just fine a moment
    ago. Only the very first startup load can leave `_state["model"]` at its
    initial `None` - there's nothing prior to fall back to yet.
    """
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        _state["model"] = model
        _state["model_version"] = mv.version
        _state["error"] = None
        return True
    except Exception as exc:  # noqa: BLE001 - surfaced via /health and /reload-model, not swallowed
        _state["error"] = str(exc)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title="Predictive Maintenance API",
    description="Serves failure predictions for industrial equipment from sensor readings.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if _state["model"] is None:
        return HealthResponse(status="degraded", model_loaded=False)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=MODEL_NAME,
        model_version=str(_state["model_version"]),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading) -> PredictionResponse:
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {_state['error'] or 'unknown error'}",
        )

    row = pd.DataFrame(
        [
            {
                "Type": reading.type,
                "Air temperature [K]": reading.air_temperature_k,
                "Process temperature [K]": reading.process_temperature_k,
                "Rotational speed [rpm]": reading.rotational_speed_rpm,
                "Torque [Nm]": reading.torque_nm,
                "Tool wear [min]": reading.tool_wear_min,
            }
        ]
    )

    probability = float(_state["model"].predict_proba(row)[0, 1])
    decision = probability >= DECISION_THRESHOLD

    log_prediction(features=row.iloc[0].to_dict(), probability=probability, decision=decision)

    return PredictionResponse(
        failure_probability=round(probability, 6),
        is_predicted_failure=decision,
        decision_threshold=DECISION_THRESHOLD,
        model_version=str(_state["model_version"]),
        model_alias=MODEL_ALIAS,
    )


@app.post("/reload-model")
def reload_model() -> dict:
    """Pick up a newly-promoted @champion version without restarting the process.

    A failed reload keeps serving whatever model was already loaded - it
    reports the failure rather than silently taking the API down.
    """
    before = _state["model_version"]
    succeeded = _load_model()
    return {
        "previous_version": before,
        "current_version": _state["model_version"],
        "reload_succeeded": succeeded,
        "serving": _state["model"] is not None,
        "error": _state["error"] if not succeeded else None,
    }
