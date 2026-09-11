"""Train, evaluate, and register a predictive-maintenance classifier with MLflow.

Two models are trained honestly and compared (not cherry-picked): a
RandomForestClassifier and an XGBoost classifier, both wrapped in the same
preprocessing pipeline (one-hot encode `Type`, pass the sensor readings
through). Class imbalance (3.4% failure rate) is handled with class weighting
rather than ignored - training on raw accuracy here would let a model that
never predicts a failure score ~96.6% while being useless.

Usage:
    python -m src.train                 # trains both, logs to MLflow, registers the better one
    mlflow ui --backend-store-uri mlruns # inspect runs at http://localhost:5000
"""
from __future__ import annotations

import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_dataset

MODEL_NAME = "predictive-maintenance-classifier"
EXPERIMENT_NAME = "predictive-maintenance"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("type_ohe", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )


def build_candidates(random_state: int = 42) -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()
    return {
        "random_forest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=5,
                        learning_rate=0.05,
                        # ~28.5x more negatives than positives in the training split
                        scale_pos_weight=28.5,
                        eval_metric="logloss",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }


def main(tracking_uri: str = "sqlite:///mlflow.db", register: bool = True) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_dataset()
    candidates = build_candidates()

    results: dict[str, dict[str, float]] = {}
    run_ids: dict[str, str] = {}

    for name, pipeline in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            pipeline.fit(X_train, y_train)
            metrics = evaluate(pipeline, X_test, y_test)

            mlflow.log_param("model_type", name)
            mlflow.log_params(
                {"n_train": len(X_train), "n_test": len(X_test), "failure_rate": round(y_train.mean(), 4)}
            )
            mlflow.log_metrics(metrics)

            signature = mlflow.models.infer_signature(X_test, pipeline.predict_proba(X_test))
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                signature=signature,
                input_example=X_test.head(3),
                # Pipelines containing an XGBoost step trip MLflow's skops
                # trusted-type check (skops flags xgboost's Booster/XGBClassifier
                # as untrusted by default). We pickle instead of skops here -
                # this is our own model artifact, not a third-party one, so the
                # skops sandboxing skops exists for isn't the relevant threat model.
                serialization_format="pickle",
            )

            results[name] = metrics
            run_ids[name] = run.info.run_id

            print(f"[{name}] " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    # Pick the winner on PR-AUC (the right metric for a 3.4%-positive-rate
    # problem - ROC-AUC is optimistic under heavy imbalance, plain accuracy
    # is actively misleading).
    best_name = max(results, key=lambda n: results[n]["pr_auc"])
    print(f"\nBest model by PR-AUC: {best_name} (pr_auc={results[best_name]['pr_auc']:.4f})")

    if register:
        model_uri = f"runs:/{run_ids[best_name]}/model"
        result = mlflow.register_model(model_uri, MODEL_NAME)
        # MLflow deprecated numbered "stages" (Staging/Production) in favor of
        # aliases - a mutable pointer you can move between versions. "champion"
        # is the alias the serving API resolves at load time (models:/<name>@champion),
        # so promoting a new version later is a one-line alias move, no redeploy.
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(name=MODEL_NAME, alias="champion", version=result.version)
        print(f"Registered {MODEL_NAME} v{result.version} -> @champion")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    main(tracking_uri=args.tracking_uri, register=not args.no_register)
