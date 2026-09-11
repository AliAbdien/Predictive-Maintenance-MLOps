"""Data loading and preprocessing for the AI4I 2020 predictive maintenance dataset.

Source: AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository,
Matzka 2020), 10,000 synthetic-but-realistic industrial sensor readings with real
machine failure labels across 5 failure modes.

Important: the dataset ships 5 failure-mode flag columns (TWF, HDF, PWF, OSF, RNF)
that are a *breakdown* of the target itself (their OR is essentially the target).
Using them as input features would be target leakage - a model "predicting"
failure from columns that already encode which failure happened would report
near-perfect accuracy while being useless in production, where you don't have
tomorrow's failure diagnosis available today. They are dropped from the feature
set on purpose; this is called out explicitly because it is the single easiest
mistake to make with this dataset.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/ai4i2020.csv"

# Columns that leak the target (breakdown of *why* the machine failed, only
# knowable after the fact) - excluded from the feature set.
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

ID_COLUMNS = ["UDI", "Product ID"]

TARGET_COLUMN = "Machine failure"

CATEGORICAL_FEATURES = ["Type"]

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return just the columns a live prediction request would actually supply."""
    return df[FEATURE_COLUMNS].copy()


def load_dataset(path: str = RAW_PATH, test_size: float = 0.2, random_state: int = 42):
    """Load the raw CSV and return a stratified train/test split of (X, y)."""
    df = load_raw(path)
    X = prepare_features(df)
    y = df[TARGET_COLUMN].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_dataset()
    print(f"train: {X_train.shape}, test: {X_test.shape}")
    print(f"train failure rate: {y_train.mean():.4f}, test failure rate: {y_test.mean():.4f}")
