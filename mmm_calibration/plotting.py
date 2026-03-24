"""
Plotting Utilities (Repo Figures)
=================================

Generates key figures used in reports and README narratives.

Primary figures supported:
- Posterior shift plot: uncalibrated vs calibrated beta intervals
- Saturation curve illustration: before/after conceptual comparison

Why this exists:
- This repo is meant to be read by humans.
- Clear visual storytelling is often more persuasive than raw metrics.

Design:
- Keep plotting functions small, deterministic, and easy to call from scripts.
- Avoid heavy dependencies; matplotlib only.
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_saturation_before_after(
    channel: str,
    hill_alpha: float,
    theta_before: float,
    theta_after: float,
    out_path: str,
    x_max: float = 3.0,
    n: int = 200
) -> None:
    """
    Simple figure: Hill curve before/after calibration.
    Here theta_after is a conceptual change (can wire this later if you decide to
    calibrate theta too). For now, it's a clean illustrative plot for the repo.
    """
    x = np.linspace(0, x_max, n)
    def hill(x, a, th):
        x = np.maximum(x, 0.0)
        return (x**a) / (x**a + th**a + 1e-12)

    y1 = hill(x, hill_alpha, theta_before)
    y2 = hill(x, hill_alpha, theta_after)

    plt.figure()
    plt.plot(x, y1, label="Uncalibrated")
    plt.plot(x, y2, label="Calibrated")
    plt.title(f"Saturation Curve: {channel}")
    plt.xlabel("Standardized Spend (proxy)")
    plt.ylabel("Response (Hill saturation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_beta_posterior_shift(
    beta_uncal: Dict[str, Dict[str, float]],
    beta_cal: Dict[str, Dict[str, float]],
    # out_path: str
) -> None:
    """
    Bar/interval summary: mean and 90% CI for uncalibrated vs calibrated betas.
    Input format from diagnostics.summarize_beta().
    """
    channels = list(beta_uncal.keys())

    def arr(d, key):
        return np.array([d[ch][key] for ch in channels], dtype=float)

    mu_u = arr(beta_uncal, "mean")
    lo_u = arr(beta_uncal, "p05")
    hi_u = arr(beta_uncal, "p95")

    mu_c = arr(beta_cal, "mean")
    lo_c = arr(beta_cal, "p05")
    hi_c = arr(beta_cal, "p95")

    x = np.arange(len(channels))
    width = 0.35

    plt.figure(figsize=(10, 4))
    plt.errorbar(x - width/2, mu_u, yerr=[mu_u - lo_u, hi_u - mu_u], fmt="o", label="Uncalibrated")
    plt.errorbar(x + width/2, mu_c, yerr=[mu_c - lo_c, hi_c - mu_c], fmt="o", label="Calibrated")
    plt.xticks(x, channels, rotation=0)
    plt.title("Channel Effect Posterior: Uncalibrated vs Calibrated")
    plt.ylabel("beta (effect on saturated scale)")
    plt.legend()
    plt.tight_layout()
    # plt.savefig(out_path, dpi=200)
    plt.close()

def _normal_pdf(x, mu, sigma):
    return (1.0 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-0.5 * ((x - mu)/sigma)**2)

def _plot_prior_vs_posterior_1d(mu_prior, sd_prior, posterior_samples, title, xpad=4.0):
    mu_post = float(np.mean(posterior_samples))
    sd_post = float(np.std(posterior_samples))

    xmin = min(mu_prior - xpad*sd_prior, mu_post - xpad*sd_post)
    xmax = max(mu_prior + xpad*sd_prior, mu_post + xpad*sd_post)
    x = np.linspace(xmin, xmax, 400)

    plt.figure(figsize=(4, 2))
    plt.plot(x, _normal_pdf(x, mu_prior, sd_prior), label="Prior (Normal)")
    plt.hist(posterior_samples, bins=40, density=True, alpha=0.35, label="Posterior samples")
    plt.axvline(mu_prior, linestyle="--")
    plt.axvline(mu_post, linestyle="--")
    plt.title(title)
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.show()