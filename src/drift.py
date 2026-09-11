"""Data-drift detection: are the sensor readings the API is seeing in
production still the same distribution the model was trained on?

Uses a two-sample Kolmogorov-Smirnov test per numeric feature, comparing the
training set's distribution against a window of recent live predictions. KS
is the standard choice here because it's non-parametric (no assumption about
the shape of a sensor's distribution) and gives an interpretable p-value per
feature, which is what actually matters in a factory: not "drift happened"
but "which sensor drifted".
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import ks_2samp

from src.data import NUMERIC_FEATURES, load_raw, prepare_features

ALERT_P_VALUE = 0.05


def reference_distribution(path: str = "data/ai4i2020.csv") -> pd.DataFrame:
    return prepare_features(load_raw(path))


def compute_drift(reference: pd.DataFrame, live: pd.DataFrame) -> list[dict]:
    """Per numeric feature, run a KS test of live vs reference. Returns one
    row per feature with the statistic, p-value, and whether it crosses the
    alert threshold."""
    results = []
    for col in NUMERIC_FEATURES:
        if col not in live.columns or live[col].dropna().empty:
            continue
        stat, p_value = ks_2samp(reference[col].dropna(), live[col].dropna())
        results.append(
            {
                "feature": col,
                "ks_statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 6),
                "drift_detected": bool(p_value < ALERT_P_VALUE),
                "reference_mean": round(float(reference[col].mean()), 3),
                "live_mean": round(float(live[col].mean()), 3),
                "n_live": int(live[col].count()),
            }
        )
    return results


def live_features_from_predictions(predictions: list[dict]) -> pd.DataFrame:
    """predictions: list of records as written by src.prediction_log."""
    if not predictions:
        return pd.DataFrame(columns=NUMERIC_FEATURES)
    return pd.DataFrame([p["features"] for p in predictions])
