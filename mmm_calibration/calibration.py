"""
Experiment-to-Prior Calibration
===============================

Converts incrementality evidence (geo experiments) into informative priors
for the Bayesian MMM.

What this file does:
- Reads geo experiment summary stats:
    - incremental_lift_mean
    - incremental_lift_se
- Aggregates multiple experiments per channel using precision weighting
- Produces per-channel priors for beta:
    beta_c ~ Normal(mu_c, sigma_c)

Why this matters:
- Observational MMM is weakly identified and often over-attributes to media.
- Geo experiments provide causal anchors but are sparse and noisy.
- Bayesian priors allow integration without pretending experiments are perfect.

Key design decisions:
- Keep mapping intentionally simple in the canonical repo:
    - lift_mean -> prior mean
    - lift_se -> prior std (with a floor to avoid overconfidence)
- Provide a `strength` knob to tighten/loosen priors for sensitivity analysis

This is the core "calibration" contribution of the project.
"""

from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd

from .data_schema import RequiredColumns, validate_columns


def build_geo_priors(
    geo_exp: pd.DataFrame,
    channels: list[str],
    default_mu: float,
    default_sigma: float,
    sigma_floor: float = 0.10,
    strength: float = 1.0
) -> Dict[str, Dict[str, float]]:
    """
    Convert geo-experiment results into priors on beta (channel effectiveness).

    Strategy (deliberately simple and defensible):
      - Use incremental_lift_mean as prior mean for beta (on the saturated scale).
      - Use incremental_lift_se as prior sigma, with a floor to avoid overconfidence.
      - 'strength' scales tightness: strength > 1 tightens priors; <1 loosens.

    This keeps the repo focused on *belief updating*, not unit conversions.
    """
    # 1. DATA VALIDATION
    # Checks if the input CSV/DataFrame has the required columns: channel, lift_mean, and lift_se.
    # If columns are missing, it stops early to prevent calculation errors.
    req = RequiredColumns()
    validate_columns(geo_exp, list(req.geo_experiment_results), "geo_experiment_results.csv")

    # 2. DATA CLEANING
    # We create a copy to avoid modifying the original data and ensure channel names are strings
    # so they match our 'channels' list exactly.
    geo_exp = geo_exp.copy()
    geo_exp["channel"] = geo_exp["channel"].astype(str)

    # 3. BASELINE INITIALIZATION
    # We start by giving every channel a "weak" default prior. 
    # If a channel has no experiment data, it will keep these defaults.
    priors = {ch: {"beta_mu": float(default_mu), "beta_sigma": float(default_sigma)} for ch in channels}

    # 4. ITERATING THROUGH CHANNELS
    # We look at each marketing channel one by one to see if we have experimental evidence for it.
    # If multiple experiments per channel: precision-weighted average
    for ch in channels:
        df = geo_exp[geo_exp["channel"] == ch]
        # 5. CHECK FOR DATA EXISTENCE
        # If no experiment exists for this specific channel, we skip to the next one.
        if df.empty:
            continue

        # 6. EXTRACTING EXPERIMENT STATS
        # We grab the Lift (the 'what') and the Standard Error (the 'certainty').
        mu = df["incremental_lift_mean"].astype(float).to_numpy()
        se = df["incremental_lift_se"].astype(float).to_numpy()

        # 7. APPLYING THE UNCERTAINTY FLOOR
        # We ensure the Standard Error isn't too small. If an experiment claims to be 100% 
        # perfect (se=0), we force it to at least 0.10 to keep the Bayesian model flexible.
        se = np.maximum(se, sigma_floor)

        # 8. CALCULATING PRECISION (WEIGHTS)
        # Precision is 1 / Variance (se^2). High-confidence tests (low SE) get high weights.
        # This is the "Meta-Analysis" math that handles multiple tests for one channel.
        #
        w = 1.0 / (se**2)
        # 9. COMPUTING PRECISION-WEIGHTED MEAN (mu_hat)
        # Instead of a simple average, we multiply each result by its weight. 
        # A very "sure" test will pull the average toward itself more than a "noisy" test.
        mu_hat = float(np.sum(w * mu) / np.sum(w))
        
        # 10. COMPUTING AGGREGATED UNCERTAINTY (sigma_hat)
        # The new uncertainty of combined tests is the inverse square root of the sum of precisions.
        # Mathematically, the more tests we have, the smaller this sigma becomes.
        # effective sigma: inverse sqrt(sum precision)
        sigma_hat = float(1.0 / np.sqrt(np.sum(w)))

        # 11. APPLYING THE STRENGTH TUNER
        # We divide the sigma by the 'strength' parameter. 
        # If strength > 1, the prior becomes "tighter" (narrower bell curve), forcing the MMM to obey.
        # If strength < 1, the prior becomes "looser," letting the model ignore the experiment more easily.
        # adjusted_sigma = sigma_hat / max(strength, 1e-6)

        # 12. FINAL PRIOR ASSIGNMENT
        # We save the calculated mean and the final sigma (ensuring it doesn't drop below the floor).
        priors[ch] = {
            "beta_mu": mu_hat,
            "beta_sigma": float(max(sigma_floor, sigma_hat / max(strength, 1e-6))),
        }

    return priors
