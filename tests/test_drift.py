import pandas as pd

from src.data import NUMERIC_FEATURES
from src.drift import compute_drift


def _flat_frame(**overrides) -> pd.DataFrame:
    base = {
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
    }
    base.update(overrides)
    import numpy as np

    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {col: base[col] + rng.normal(0, 1, n) for col in NUMERIC_FEATURES}
    )


def test_no_drift_when_distributions_match():
    reference = _flat_frame()
    live = _flat_frame()
    results = compute_drift(reference, live)
    assert all(not r["drift_detected"] for r in results)


def test_drift_detected_on_shifted_feature():
    reference = _flat_frame()
    # Shift only air temperature by a lot - a real sensor/process change.
    live = _flat_frame(**{"Air temperature [K]": 320.0})
    results = {r["feature"]: r for r in compute_drift(reference, live)}
    assert results["Air temperature [K]"]["drift_detected"] is True
    # Untouched features should not be falsely flagged.
    assert results["Torque [Nm]"]["drift_detected"] is False
