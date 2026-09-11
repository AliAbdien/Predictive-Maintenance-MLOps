"""Simulate a live stream of sensor readings hitting the API.

Two modes:
  --mode normal  sends real held-out readings, unmodified (no drift expected)
  --mode drift   sends real readings with a shop-floor-realistic shift applied
                 partway through (a hotter shift, a batch of more worn tools) -
                 so the monitoring dashboard's KS-test drift detector has
                 something real to actually catch, not a synthetic toy.

Usage:
    python -m scripts.simulate_stream --mode drift --n 300 --sleep 0.02
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests

from src.data import load_raw, prepare_features

API_URL_DEFAULT = "http://localhost:8000/predict"


def to_payload(row: pd.Series) -> dict:
    return {
        "type": row["Type"],
        "air_temperature_k": float(row["Air temperature [K]"]),
        "process_temperature_k": float(row["Process temperature [K]"]),
        "rotational_speed_rpm": float(row["Rotational speed [rpm]"]),
        "torque_nm": float(row["Torque [Nm]"]),
        "tool_wear_min": float(row["Tool wear [min]"]),
    }


def apply_drift(row: pd.Series) -> pd.Series:
    """A believable factory-floor shift: hotter shop-floor air (poor summer
    ventilation) pushes both temperatures up, and a batch of already-worn
    tooling was mistakenly put back into rotation."""
    row = row.copy()
    row["Air temperature [K]"] += 6.0
    row["Process temperature [K]"] += 4.0
    row["Tool wear [min]"] += 140
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["normal", "drift"], default="normal")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--url", default=API_URL_DEFAULT)
    parser.add_argument("--drift-after", type=int, default=None, help="Row index drift starts at (default: halfway)")
    args = parser.parse_args()

    df = prepare_features(load_raw()).sample(n=args.n, random_state=7).reset_index(drop=True)
    drift_after = args.drift_after if args.drift_after is not None else args.n // 2

    sent, failed = 0, 0
    for i, row in df.iterrows():
        if args.mode == "drift" and i >= drift_after:
            row = apply_drift(row)
        try:
            resp = requests.post(args.url, json=to_payload(row), timeout=5)
            resp.raise_for_status()
            sent += 1
        except requests.RequestException as exc:
            failed += 1
            print(f"  request {i} failed: {exc}", file=sys.stderr)
        if args.sleep:
            time.sleep(args.sleep)
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{args.n} sent")

    print(f"Done. sent={sent} failed={failed} mode={args.mode} drift_after={drift_after if args.mode == 'drift' else 'n/a'}")


if __name__ == "__main__":
    main()
