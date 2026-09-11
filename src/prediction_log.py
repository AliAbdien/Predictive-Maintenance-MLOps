"""Append-only JSONL log of every prediction the API serves.

This is the raw material the monitoring dashboard's drift detection reads -
without logging what the model actually saw in production, you can't ever
tell whether the world has drifted away from the training data.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_PATH = os.environ.get("PREDICTION_LOG_PATH", "monitoring/predictions.jsonl")
_lock = threading.Lock()


def log_prediction(features: dict, probability: float, decision: bool, log_path: str = DEFAULT_LOG_PATH) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "failure_probability": probability,
        "is_predicted_failure": decision,
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def read_predictions(log_path: str = DEFAULT_LOG_PATH) -> list[dict]:
    path = Path(log_path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
