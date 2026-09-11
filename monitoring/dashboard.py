"""Streamlit monitoring dashboard: prediction volume, failure-rate over time,
and per-feature data-drift status against the training distribution.

Run:
    streamlit run monitoring/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.drift import compute_drift, live_features_from_predictions, reference_distribution
from src.prediction_log import read_predictions

st.set_page_config(page_title="Predictive Maintenance - Monitoring", layout="wide")
st.title("🏭 Predictive Maintenance — Production Monitoring")

predictions = read_predictions()

if not predictions:
    st.info(
        "No predictions logged yet. Send some traffic to the API first, e.g.:\n\n"
        "`python -m scripts.simulate_stream --mode drift --n 200`"
    )
    st.stop()

df = pd.DataFrame(
    [
        {
            "timestamp": p["timestamp"],
            "failure_probability": p["failure_probability"],
            "is_predicted_failure": p["is_predicted_failure"],
        }
        for p in predictions
    ]
)
df["timestamp"] = pd.to_datetime(df["timestamp"])

col1, col2, col3 = st.columns(3)
col1.metric("Predictions logged", len(df))
col2.metric("Predicted-failure rate", f"{df['is_predicted_failure'].mean() * 100:.2f}%")
col3.metric("Mean failure probability", f"{df['failure_probability'].mean():.4f}")

st.subheader("Failure probability over time")
st.line_chart(df.set_index("timestamp")["failure_probability"])

st.subheader("Data drift — live sensor readings vs. training distribution")
st.caption(
    "Two-sample Kolmogorov-Smirnov test per feature. A feature is flagged when "
    "p < 0.05, i.e. the live readings are unlikely to come from the same "
    "distribution the model was trained on. This is what actually matters in "
    "production: an alert on *which sensor* drifted, not just a global "
    "'something changed'."
)

reference = reference_distribution()
live = live_features_from_predictions(predictions)
drift_results = compute_drift(reference, live)
drift_df = pd.DataFrame(drift_results)

if drift_df.empty:
    st.warning("Not enough live data yet to compute drift.")
else:
    def highlight_drift(row):
        color = "background-color: #7f1d1d" if row["drift_detected"] else ""
        return [color] * len(row)

    st.dataframe(
        drift_df.style.apply(highlight_drift, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    drifted = drift_df[drift_df["drift_detected"]]
    if len(drifted) > 0:
        st.error(
            f"⚠️ Drift detected in {len(drifted)} feature(s): "
            f"{', '.join(drifted['feature'].tolist())}. "
            "Consider retraining or investigating the sensor(s) before trusting new predictions."
        )
    else:
        st.success("No drift detected in any monitored feature.")

st.subheader("Recent predictions")
st.dataframe(df.tail(20).sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
