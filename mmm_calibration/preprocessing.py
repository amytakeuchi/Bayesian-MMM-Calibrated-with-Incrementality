"""
Preprocessing + Feature Engineering
===================================

Loads raw MMM inputs and converts them into model-ready matrices.

What this file does:
- Reads /data/raw CSVs (spend, outcome, controls)
- Pivots long-format spend (date, channel, spend) into wide channel matrix
- Joins outcome + controls + spend into a single aligned time index
- Standardizes spend and controls (optional, recommended for stable sampling)
- Returns a ModelMatrix object with:
    - y (outcome)
    - X_spend (channel spend features)
    - X_controls (control regressors)
    - metadata (channels, scalers, merged dataframe)

Why this matters:
- MMMs fail silently when dates don’t align or missing values shift the design matrix.
- Standardization dramatically improves MCMC geometry (especially with weak priors).

Design choice:
- We keep feature engineering minimal and explicit.
- More sophisticated seasonality or holiday handling can be added later without changing the API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from .data_schema import RequiredColumns, validate_columns, validate_nonnegative


@dataclass(frozen=True)
class ModelMatrix:
    dates: np.ndarray                 # shape (T,)
    y: np.ndarray                     # shape (T,)
    X_spend: np.ndarray               # shape (T, C) raw spend (maybe standardized)
    X_controls: np.ndarray            # shape (T, K)
    channels: List[str]
    control_cols: List[str]
    spend_scaler: Optional[Dict[str, Tuple[float, float]]]  # mean,std per channel
    controls_scaler: Optional[Dict[str, Tuple[float, float]]]
    df_merged: pd.DataFrame           # for reference/debug


def load_raw_data(paths, cfg):
    spend = pd.read_csv(paths.data_raw / "marketing_spend.csv")
    outcome = pd.read_csv(paths.data_raw / "outcome_sales.csv")
    controls = pd.read_csv(paths.data_raw / "controls.csv")

    req = RequiredColumns()
    validate_columns(spend, list(req.marketing_spend), "marketing_spend.csv")
    validate_columns(outcome, list(req.outcome_sales), "outcome_sales.csv")
    validate_nonnegative(spend, "spend", "marketing_spend.csv")

    # parse dates
    for df in (spend, outcome, controls):
        df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])

    return spend, outcome, controls


def _standardize_series(x: pd.Series) -> Tuple[pd.Series, Tuple[float, float]]:
    mu = float(x.mean())
    sd = float(x.std(ddof=0)) if float(x.std(ddof=0)) > 1e-12 else 1.0
    return (x - mu) / sd, (mu, sd)


def build_model_matrix(
    spend: pd.DataFrame,
    outcome: pd.DataFrame,
    controls: pd.DataFrame,
    cfg
) -> ModelMatrix:

    # Pivot spend to wide: (date x channel)
    spend_wide = spend.pivot_table(index=cfg.date_col, columns=cfg.channel_col, values=cfg.spend_col, aggfunc="sum").fillna(0.0)

    channels = cfg.channels or list(spend_wide.columns)
    channels = list(channels)

    # Ensure consistent columns
    spend_wide = spend_wide.reindex(columns=channels).fillna(0.0)

    # Merge outcome + controls
    df = (
        outcome.set_index(cfg.date_col)
        .join(controls.set_index(cfg.date_col), how="left")
        .join(spend_wide, how="left")
        .reset_index()
        .sort_values(cfg.date_col)
    )

    # Fill missing controls if any
    for c in cfg.control_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(df[c].median())

    y = df[cfg.y_col].astype(float).to_numpy()
    dates = df[cfg.date_col].to_numpy()

    spend_scaler = None
    controls_scaler = None

    # Standardize spend columns channel-wise
    X_spend_df = df[channels].copy()
    if cfg.standardize_spend:
        spend_scaler = {}
        for ch in channels:
            X_spend_df[ch], stats = _standardize_series(X_spend_df[ch])
            spend_scaler[ch] = stats

    # Controls matrix
    X_ctrl_df = df[cfg.control_cols].copy()
    if cfg.standardize_controls:
        controls_scaler = {}
        for c in cfg.control_cols:
            X_ctrl_df[c], stats = _standardize_series(X_ctrl_df[c])
            controls_scaler[c] = stats

    X_spend = X_spend_df.to_numpy(dtype=float)
    X_controls = X_ctrl_df.to_numpy(dtype=float)

    return ModelMatrix(
        dates=dates,
        y=y,
        X_spend=X_spend,
        X_controls=X_controls,
        channels=channels,
        control_cols=cfg.control_cols,
        spend_scaler=spend_scaler,
        controls_scaler=controls_scaler,
        df_merged=df
    )
