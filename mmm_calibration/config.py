"""
Configuration for Bayesian MMM Calibration
==========================================

Defines strongly-typed configuration objects used across the repo.

Why this exists:
- Keeps model assumptions explicit and reviewable
- Enables reproducibility across notebooks/scripts
- Makes it easy to compare scenarios (uncalibrated vs calibrated, different adstock/saturation)

Key Concepts:
- PathsConfig: standardizes repo directories (data/raw, data/processed, models, reports)
- MMMConfig: defines modeling choices
    - feature standardization (spend/controls)
    - adstock parameters (default + per-channel override)
    - Hill saturation parameters (fixed for identifiability in this canonical repo)
    - priors (weakly informative baseline; experiment-informed priors applied later)
    - MCMC sampling controls (draws/tune/chains/target_accept)
    - budget optimization settings

This file is intentionally boring: it acts as the single source of truth for assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict


@dataclass(frozen=True)
class PathsConfig:
    repo_root: Path = field(default_factory=lambda: Path("."))
    data_raw: Path = field(init=False)
    data_processed: Path = field(init=False)
    models: Path = field(init=False)
    reports: Path = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "data_raw", self.repo_root / "data" / "raw")
        object.__setattr__(self, "data_processed", self.repo_root / "data" / "processed")
        object.__setattr__(self, "models", self.repo_root / "models")
        object.__setattr__(self, "reports", self.repo_root / "reports")
        self.data_processed.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)
        (self.reports / "figures").mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class MMMConfig:
    # Data
    date_col: str = "date"
    y_col: str = "y"
    channel_col: str = "channel"
    spend_col: str = "spend"

    channels: Optional[List[str]] = None  # if None, inferred from spend data

    # Feature engineering
    standardize_spend: bool = True
    standardize_controls: bool = True

    # Adstock (geometric)
    adstock_default_lambda: float = 0.5
    adstock_lambdas: Optional[Dict[str, float]] = None  # per channel; if None uses default

    # Saturation (Hill)
    hill_alpha: float = 1.5
    hill_theta: float = 1.0  # in standardized spend units; interpretable after scaling
    learn_hill_params: bool = False  # keep false for this canonical repo (identifiability)

    # Priors
    intercept_sigma: float = 2.0
    beta_sigma: float = 1.0          # default sigma for channel betas (uncalibrated)
    beta_mu: float = 0.0
    sigma_y_exponential: float = 1.0

    # Controls
    control_cols: List[str] = field(default_factory=lambda: ["price_index", "promo", "macro_index", "seasonality"])

    # Sampling
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.9
    random_seed: int = 123

    # Budget optimization
    n_posterior_samples_for_opt: int = 400
    risk_aversion: float = 0.0  # 0 = expected value; >0 penalizes variance in objective
