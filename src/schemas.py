"""Pydantic request/response models for the serving API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """One machine's current sensor readings - exactly what a live prediction needs."""

    type: Literal["L", "M", "H"] = Field(
        ..., description="Product quality variant: L(ow), M(edium), or H(igh)."
    )
    air_temperature_k: float = Field(..., ge=250, le=350, description="Air temperature in Kelvin.")
    process_temperature_k: float = Field(
        ..., ge=250, le=350, description="Process temperature in Kelvin."
    )
    rotational_speed_rpm: float = Field(..., ge=0, description="Tool rotational speed in rpm.")
    torque_nm: float = Field(..., ge=0, description="Torque in Newton-metres.")
    tool_wear_min: float = Field(..., ge=0, description="Cumulative tool wear in minutes.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "M",
                    "air_temperature_k": 298.1,
                    "process_temperature_k": 308.6,
                    "rotational_speed_rpm": 1551,
                    "torque_nm": 42.8,
                    "tool_wear_min": 0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    failure_probability: float = Field(..., description="Model's estimated probability of machine failure.")
    is_predicted_failure: bool = Field(..., description="failure_probability >= decision threshold.")
    decision_threshold: float
    model_version: str
    model_alias: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
