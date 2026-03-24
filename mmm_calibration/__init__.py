"""
MMM Calibration Package
=======================

A lightweight, production-minded reference implementation of a Bayesian Media Mix Model (MMM)
calibrated with incrementality evidence (geo experiments).

Design Principles:
- Separation of concerns: preprocessing, transforms, modeling, calibration, diagnostics, decisions
- Reproducibility: scripts + deterministic config
- Decision focus: posterior uncertainty is first-class (not just point estimates)

Typical Workflow:
1) Build features (wide matrices for channels + controls)
2) Fit uncalibrated Bayesian MMM (observational baseline)
3) Convert geo experiment lift estimates into priors
4) Fit calibrated MMM (belief updating)
5) Compare models (posterior shift + PPC)
6) Optimize budgets under uncertainty

This module exposes the "public API" of the package for scripts and notebooks.
"""


from .config import MMMConfig, PathsConfig
from .preprocessing import load_raw_data, build_model_matrix
from .model import fit_mmm
from .calibration import build_geo_priors
from .budget_optimization import optimize_budget
