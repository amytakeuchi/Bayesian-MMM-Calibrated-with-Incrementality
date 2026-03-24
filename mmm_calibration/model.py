"""
Bayesian MMM Model (PyMC)
========================

Fits a baseline Bayesian Media Mix Model using observational time-series data.

Model structure:
- Transform spend via:
    1) Adstock (carryover)
    2) Hill saturation (diminishing returns)
- Predict outcome y with:
    y_t ~ Normal(mu_t, sigma)
    mu_t = intercept + X_controls @ gamma + X_sat @ beta

Key outputs:
- Posterior draws for:
    - beta (channel effects)
    - gamma (control effects)
    - sigma (noise)
- Posterior predictive distribution for y

Calibration hook:
- This function accepts optional `priors_by_channel` which modifies the prior
  for each beta_c. *That is the bridge for geo-experiment calibration.

Design choices:
- Keeps functional form stable (same model before/after calibration)
- Calibration changes beliefs (priors) rather than rewriting the model
- Sampling settings are controlled via MMMConfig for reproducibility

This file is the statistical "engine" of the repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np

from .adstock import geometric_adstock_2d
from .saturation import hill_saturation


@dataclass(frozen=True)
class FitResult:
    idata: Any
    model: Any
    model_matrix: Any
    priors_used: Dict[str, Dict[str, float]]


def fit_mmm(model_matrix, cfg, priors_by_channel: Optional[Dict[str, Dict[str, float]]] = None) -> FitResult:
    """
    Fits a Bayesian MMM:
      y ~ Normal(mu, sigma)
      mu = intercept + X_controls @ gamma + sum_c beta_c * Hill(Adstock(spend_c))
    Notes:
      - We keep Hill params fixed by default for identifiability in this canonical repo.
      - Calibration is implemented as informative priors on beta_c.
    """
    import pymc as pm

    X_spend = model_matrix.X_spend
    X_ctrl = model_matrix.X_controls
    y = model_matrix.y
    channels = model_matrix.channels

    # lambdas per channel
    lam = np.array([ (cfg.adstock_lambdas or {}).get(ch, cfg.adstock_default_lambda) for ch in channels ], dtype=float)

    X_ad = geometric_adstock_2d(X_spend, lam)
    X_sat = hill_saturation(X_ad, alpha=cfg.hill_alpha, theta=cfg.hill_theta)

    priors_used = {}
    for ch in channels:
        if priors_by_channel and ch in priors_by_channel:
            priors_used[ch] = {
                "beta_mu": float(priors_by_channel[ch]["beta_mu"]),
                "beta_sigma": float(priors_by_channel[ch]["beta_sigma"]),
            }
        else:
            priors_used[ch] = {"beta_mu": float(cfg.beta_mu), "beta_sigma": float(cfg.beta_sigma)}

    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0.0, sigma=cfg.intercept_sigma)

        # controls
        gamma = pm.Normal("gamma", mu=0.0, sigma=1.0, shape=X_ctrl.shape[1])

        # channel coefficients
        beta_mu_vec = np.array([priors_used[ch]["beta_mu"] for ch in channels], dtype=float)
        beta_sd_vec = np.array([priors_used[ch]["beta_sigma"] for ch in channels], dtype=float)
        beta = pm.Normal("beta", mu=beta_mu_vec, sigma=beta_sd_vec, shape=len(channels))

        mu = intercept + pm.math.dot(X_ctrl, gamma) + pm.math.dot(X_sat, beta)

        sigma = pm.Exponential("sigma", lam=cfg.sigma_y_exponential)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=cfg.draws,
            model=model,
            tune=cfg.tune,
            chains=cfg.chains,
            target_accept=cfg.target_accept,
            random_seed=cfg.random_seed,
            # var_names=["y_obs"],
            progressbar=True,
            return_inferencedata=True,
            # var_names=["intercept", "gamma", "beta", "sigma", "y_obs"],
            # extend_inferencedata=True,
            )
        # ADD THIS LINE HERE:
        # This populates the 'posterior_predictive' group in idata
        idata.extend(pm.sample_posterior_predictive(idata, model=model, progressbar=False))

    return FitResult(idata=idata, model=model, model_matrix=model_matrix, priors_used=priors_used)
