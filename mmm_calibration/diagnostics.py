"""
Diagnostics + Reporting
=======================

Provides lightweight diagnostics to evaluate:
- Posterior stability (summary of beta)
- Predictive fit (posterior predictive RMSE)
- Quick model report payloads for saving into /reports

Why this exists:
- Senior DS work is judged by diagnostics and model criticism, not just fitting code.
- Calibrated vs uncalibrated should be compared with:
    - posterior shifts
    - uncertainty changes
    - predictive behavior

Design choice:
- Keep diagnostics minimal and interpretable.
- Advanced checks (LOO/WAIC, residual autocorrelation, coverage) can be added later.
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np


def summarize_beta(idata, channels: list[str]) -> Dict[str, Dict[str, float]]:
    """
    Return posterior mean and 90% interval for each channel beta.
    """
    beta = idata.posterior["beta"].values  # (chain, draw, C)
    beta_flat = beta.reshape(-1, beta.shape[-1])

    out = {}
    for i, ch in enumerate(channels):
        vals = beta_flat[:, i]
        out[ch] = {
            "mean": float(np.mean(vals)),
            "p05": float(np.quantile(vals, 0.05)),
            "p95": float(np.quantile(vals, 0.95)),
        }
    return out


def posterior_predictive_rmse(idata, y_true: np.ndarray) -> float:
    """
    RMSE of posterior predictive mean vs observed.
    """
    y_pp = idata.posterior_predictive["y_obs"].values  # (chain, draw, T)
    y_mean = y_pp.mean(axis=(0, 1))
    return float(np.sqrt(np.mean((y_true - y_mean) ** 2)))


def basic_model_report(fit_result) -> Dict[str, Any]:
    mm = fit_result.model_matrix
    rmse = posterior_predictive_rmse(fit_result.idata, mm.y)
    beta_summary = summarize_beta(fit_result.idata, mm.channels)

    return {
        "rmse_pp_mean": rmse,
        "beta_summary": beta_summary,
        "priors_used": fit_result.priors_used,
        "n_obs": int(len(mm.y)),
        "n_channels": int(len(mm.channels)),
        "n_controls": int(mm.X_controls.shape[1]),
    }
