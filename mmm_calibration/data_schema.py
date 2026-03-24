"""
Data Schema + Validation
========================

Defines minimal required columns and basic validation checks for repo datasets.

Why this exists:
- Prevents silent failures (wrong column names, negative spend, missing fields)
- Makes the pipeline robust and "professionally-reviewable"
- Documents the contract between /data/raw and the modeling code

Datasets covered:
- marketing_spend.csv: long format (date, channel, spend)
- outcome_sales.csv: time series outcome (date, y)
- controls.csv: time series controls (date, macro/price/promo/seasonality)
- geo_experiment_results.csv: incrementality evidence for calibration
    (experiment_id, channel, start/end, lift_mean, lift_se, ...)

Validation Philosophy:
- Lightweight checks only (columns + obvious constraints)
- Heavy QA belongs in production systems; this repo stays canonical and readable
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import pandas as pd


@dataclass(frozen=True)
class RequiredColumns:
    marketing_spend: List[str] = ("date", "channel", "spend")
    outcome_sales: List[str] = ("date", "y")
    geo_experiment_results: List[str] = (
        "experiment_id", "channel", "start_date", "end_date",
        "incremental_lift_mean", "incremental_lift_se"
    )


def validate_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def validate_nonnegative(df: pd.DataFrame, col: str, name: str) -> None:
    if (df[col] < 0).any():
        raise ValueError(f"{name}: column '{col}' contains negative values.")
