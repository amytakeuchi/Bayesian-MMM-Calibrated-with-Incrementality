"""
Calibration Visualization Module
=================================

Visualization utilities for comparing uncalibrated vs calibrated MMM models.

This module provides:
- Helper functions for extracting beta samples from InferenceData
- 3-way prior→posterior shift visualization (uncal prior, geo prior, posterior)
- Forest plot comparison (uncal vs cal with HDI intervals)

Usage:
------
```python
from calibration_viz import (
    plot_prior_vs_posterior_shift,
    plot_beta_forest_pre_post
)

# 3-way comparison
plot_prior_vs_posterior_shift(
    idata_cal=idata_cal,
    channels=['search', 'social', 'tv', 'youtube'],
    geo_priors=geo_priors,
    weak_mu=0.0,
    weak_sigma=0.5
)

# Forest plot
plot_beta_forest_pre_post(
    idata_uncal=idata_uncal,
    idata_cal=idata_cal,
    channels=['search', 'social', 'tv', 'youtube']
)
```

Author: Portfolio Project - Bayesian MMM Calibration
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from typing import Dict, List, Tuple, Optional


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _safe_get_beta_samples(idata, channels: List[str]) -> Optional[np.ndarray]:
    """
    Extract beta samples from InferenceData, handling different naming conventions.
    
    Tries common naming patterns: beta, betas, beta_channel, beta_media.
    Adjusts for models that use different variable names.
    
    Parameters
    ----------
    idata : arviz.InferenceData
        InferenceData object containing posterior samples
    channels : list of str
        Channel names (used for dimension checking)
        
    Returns
    -------
    np.ndarray or None
        Beta samples with shape (chain, draw, channel) or None if not found
        
    Raises
    ------
    KeyError
        If beta variable cannot be found in idata.posterior
    """
    # Try common naming conventions
    candidates = ["beta", "betas", "beta_channel", "beta_media"]
    
    for name in candidates:
        if name in idata.posterior.data_vars:
            arr = idata.posterior[name]
            # expected dims: chain, draw, channel
            # If dims include channel-like dim, map it.
            dims = arr.dims
            ch_dim = None
            for d in dims:
                if arr.sizes[d] == len(channels):
                    ch_dim = d
                    break
            if ch_dim is None:
                continue
                
            return np.asarray(arr.values)
    
    raise KeyError(f"Could not find beta variable in idata.posterior. "
                   f"Check model var name.")


def _extract_beta_samples(idata, channels: List[str]) -> Dict[str, np.ndarray]:
    """
    Extract and flatten beta posterior samples for each channel.
    
    Returns dict: channel -> 1D array of posterior samples (flattened across chains/draws)
    
    Parameters
    ----------
    idata : arviz.InferenceData
        InferenceData with beta posteriors
    channels : list of str
        Channel names
        
    Returns
    -------
    dict
        {channel_name: flattened_samples} for each channel
        
    Examples
    --------
    >>> beta_samples = _extract_beta_samples(idata, ['search', 'social'])
    >>> beta_samples['search'].shape
    (4000,)  # e.g., 2 chains × 2000 draws
    """
    if "beta" not in idata.posterior.data_vars:
        raise KeyError("Could not find 'beta' in idata.posterior. "
                      "Check your model variable name.")
    
    beta = idata.posterior["beta"].values  # (chain, draw, C)
    beta_flat = beta.reshape(-1, beta.shape[-1])  # (chain*draw, C)
    
    return {ch: beta_flat[:, i] for i, ch in enumerate(channels)}


def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Normal distribution probability density function.
    
    Parameters
    ----------
    x : array-like
        Points at which to evaluate PDF
    mu : float
        Mean of the distribution
    sigma : float
        Standard deviation (must be > 0)
        
    Returns
    -------
    np.ndarray
        PDF values at each x
        
    Examples
    --------
    >>> x = np.linspace(-3, 3, 100)
    >>> pdf = _normal_pdf(x, mu=0, sigma=1)
    >>> pdf.max()  # Should be ~0.399 at x=0
    0.3989...
    """
    sigma = max(float(sigma), 1e-12)  # Avoid division by zero
    return (1.0 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-0.5 * ((x - mu)/sigma)**2)


def _kde_like_hist(samples: np.ndarray, bins: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple density proxy without seaborn/scipy: normalized histogram (step curve).
    
    Returns bin centers and normalized histogram values suitable for plotting.
    
    Parameters
    ----------
    samples : array-like
        Posterior samples to estimate density
    bins : int, default=60
        Number of histogram bins
        
    Returns
    -------
    centers : np.ndarray
        Bin centers (x-coordinates)
    hist : np.ndarray
        Normalized histogram values (y-coordinates, sums to 1)
        
    Examples
    --------
    >>> samples = np.random.normal(0, 1, 1000)
    >>> centers, hist = _kde_like_hist(samples, bins=30)
    >>> plt.plot(centers, hist)
    """
    hist, edges = np.histogram(samples, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def _mean_and_hdi(samples: np.ndarray, hdi_prob: float = 0.94) -> Tuple[float, float, float]:
    """
    Compute mean and highest density interval (HDI) for posterior samples.
    
    Parameters
    ----------
    samples : array-like
        Posterior samples (1D)
    hdi_prob : float, default=0.94
        Probability mass for HDI (e.g., 0.94 for 94% HDI)
        
    Returns
    -------
    mean : float
        Posterior mean
    lower : float
        Lower bound of HDI
    upper : float
        Upper bound of HDI
        
    Examples
    --------
    >>> samples = np.random.normal(0, 1, 1000)
    >>> m, lo, hi = _mean_and_hdi(samples, hdi_prob=0.94)
    >>> print(f"Mean: {m:.3f}, 94% HDI: [{lo:.3f}, {hi:.3f}]")
    Mean: -0.012, 94% HDI: [-1.876, 1.902]
    """
    m = float(np.mean(samples))
    hdi = az.hdi(samples, hdi_prob=hdi_prob)  # returns array([low, high])
    return m, float(hdi[0]), float(hdi[1])


# ============================================================================
# MAIN VISUALIZATION FUNCTIONS
# ============================================================================

def plot_prior_vs_posterior_shift(
    idata_cal,
    channels: List[str],
    geo_priors: Dict[str, Dict[str, float]],
    weak_mu: float,
    weak_sigma: float,
    bins: int = 60,
    cols: int = 2,
    xpad: float = 5.0,
    title: str = "Prior vs Posterior Shift (Uncal Prior vs Geo Prior vs Posterior)"
) -> plt.Figure:
    """
    Visualize 3-way Bayesian updating: Weak prior → Geo prior → Posterior.
    
    Creates a grid of subplots (one per channel) showing:
    - Blue curve: Weak uncalibrated prior N(weak_mu, weak_sigma)
    - Orange curve: Geo-experiment informed prior N(geo_mu, geo_sigma)
    - Green histogram: Calibrated posterior (after updating with obs data)
    
    This visualization shows how the posterior "chooses" between the two priors
    based on their relative precision and the observational data.
    
    Parameters
    ----------
    idata_cal : arviz.InferenceData
        Calibrated model posterior (with geo-experiment priors)
    channels : list of str
        Channel names (e.g., ['search', 'social', 'tv', 'youtube'])
    geo_priors : dict
        Geo-experiment priors: {channel: {'beta_mu': float, 'beta_sigma': float}}
    weak_mu : float
        Mean of weak uncalibrated prior (typically 0.0)
    weak_sigma : float
        Std dev of weak prior (typically 0.5)
    bins : int, default=60
        Number of bins for posterior histogram
    cols : int, default=2
        Number of columns in subplot grid
    xpad : float, default=5.0
        Extra padding for x-axis range (in multiples of std dev)
    title : str
        Overall figure title
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure (can be saved with fig.savefig(...))
        
    Examples
    --------
    >>> fig = plot_prior_vs_posterior_shift(
    ...     idata_cal=idata,
    ...     channels=['search', 'social', 'tv', 'youtube'],
    ...     geo_priors=geo_priors,
    ...     weak_mu=0.0,
    ...     weak_sigma=0.5
    ... )
    >>> fig.savefig('prior_posterior_shift.png', dpi=100, bbox_inches='tight')
    
    Notes
    -----
    - If posterior is close to weak prior (blue), observational data dominated
    - If posterior is close to geo prior (orange), experiments were influential
    - Typically with 156 weeks >> 2 experiments, posterior stays near weak prior
    """
    beta_samples = _extract_beta_samples(idata_cal, channels)
    
    n = len(channels)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
    axes = np.atleast_1d(axes).flatten()  # Handle single subplot case
    
    for i, ch in enumerate(channels):
        ax = axes[i]
        
        # Get geo-experiment prior parameters
        mu_weak = float(weak_mu)
        sd_weak = float(weak_sigma)
        mu_geo = float(geo_priors.get(ch, {}).get("beta_mu", mu_weak))
        sd_geo = float(geo_priors.get(ch, {}).get("beta_sigma", sd_weak))
        
        # Get posterior samples
        s = beta_samples[ch]
        mu_post = float(np.mean(s))
        sd_post = float(np.std(s))
        
        # Define x-range wide enough for all 3 distributions
        xmin = min(mu_weak - xpad*sd_weak, mu_geo - xpad*sd_geo, mu_post - xpad*sd_post)
        xmax = max(mu_weak + xpad*sd_weak, mu_geo + xpad*sd_geo, mu_post + xpad*sd_post)
        x = np.linspace(xmin, xmax, 500)
        
        # Plot weak prior (analytical - blue line)
        ax.plot(x, _normal_pdf(x, mu_weak, sd_weak), 
                label="Uncal prior (weak)", linewidth=2, color='blue')
        
        # Plot geo prior (analytical - orange line)
        ax.plot(x, _normal_pdf(x, mu_geo, sd_geo), 
                label="Geo prior", linewidth=2, color='orange')
        
        # Plot posterior density (hist-density curve - green)
        xc, yc = _kde_like_hist(s, bins=bins)
        ax.plot(xc, yc, label="Posterior (calibrated)", linewidth=2, color='green')
        
        # Add vertical lines for means
        ax.axvline(mu_weak, linestyle="--", color='blue', linewidth=1, alpha=0.5)
        ax.axvline(mu_geo, linestyle="--", color='orange', linewidth=1, alpha=0.5)
        ax.axvline(mu_post, linestyle="--", color='green', linewidth=1, alpha=0.5)
        
        # Formatting
        ax.set_title(f"Channel: {ch}", fontsize=11, fontweight='bold')
        ax.set_xlabel("β")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_yticks([])  # Density scale not important, hide ticks
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(title, y=1.02, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_beta_forest_pre_post(
    idata_uncal,
    idata_cal,
    channels: List[str],
    hdi_prob: float = 0.94,
    title: str = "β Forest Plot: Uncalibrated vs Calibrated (Mean + 94% HDI)"
) -> plt.Figure:
    """
    Forest plot comparing beta credible intervals before/after calibration.
    
    Shows horizontal error bars for HDI (highest density interval) with point
    estimates (means) for both uncalibrated and calibrated models.
    
    Parameters
    ----------
    idata_uncal : arviz.InferenceData
        Uncalibrated model posterior
    idata_cal : arviz.InferenceData
        Calibrated model posterior
    channels : list of str
        Channel names
    hdi_prob : float, default=0.94
        HDI probability (0.94 = 94% credible interval)
    title : str
        Plot title
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
        
    Examples
    --------
    >>> fig = plot_beta_forest_pre_post(
    ...     idata_uncal=idata_uncal,
    ...     idata_cal=idata_cal,
    ...     channels=['search', 'social', 'tv', 'youtube'],
    ...     hdi_prob=0.94
    ... )
    >>> plt.show()
    
    Notes
    -----
    - Blue bars: Uncalibrated (94% HDI)
    - Orange bars: Calibrated (94% HDI)
    - Dots: Posterior means
    - Overlapping intervals suggest minimal calibration impact
    """
    s_uncal = _extract_beta_samples(idata_uncal, channels)
    s_cal = _extract_beta_samples(idata_cal, channels)
    
    # Compute stats for each channel
    mu_u, lo_u, hi_u = [], [], []
    mu_c, lo_c, hi_c = [], [], []
    
    for ch in channels:
        m, lo, hi = _mean_and_hdi(s_uncal[ch], hdi_prob=hdi_prob)
        mu_u.append(m)
        lo_u.append(lo)
        hi_u.append(hi)
        
        m, lo, hi = _mean_and_hdi(s_cal[ch], hdi_prob=hdi_prob)
        mu_c.append(m)
        lo_c.append(lo)
        hi_c.append(hi)
    
    # Convert to numpy arrays for plotting
    mu_u, lo_u, hi_u = map(np.array, (mu_u, lo_u, hi_u))
    mu_c, lo_c, hi_c = map(np.array, (mu_c, lo_c, hi_c))
    
    # Create figure
    y = np.arange(len(channels))
    offset = 0.15  # Offset for visual separation
    
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(channels) + 2))
    
    # Plot horizontal error bars (HDI intervals)
    ax.errorbar(
        mu_u, y - offset,
        xerr=[mu_u - lo_u, hi_u - mu_u],
        fmt='o', capsize=3, label=f"Uncalibrated ({int(hdi_prob*100)}% HDI)",
        color='blue', markersize=6, linewidth=1.5
    )
    
    ax.errorbar(
        mu_c, y + offset,
        xerr=[mu_c - lo_c, hi_c - mu_c],
        fmt='o', capsize=3, label=f"Calibrated ({int(hdi_prob*100)}% HDI)",
        color='orange', markersize=6, linewidth=1.5
    )
    
    # Add zero reference line
    ax.axvline(0.0, linestyle="--", color='gray', linewidth=1, alpha=0.5)
    
    # Formatting
    ax.set_yticks(y)
    ax.set_yticklabels(channels)
    ax.set_xlabel("β (channel effect on saturated/adstocked scale)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    return fig


# ============================================================================
# CONVENIENCE FUNCTION: Generate Comparison Summary Table
# ============================================================================

def generate_comparison_table(
    idata_uncal,
    idata_cal,
    channels: List[str],
    geo_priors: Dict[str, Dict[str, float]],
    hdi_prob: float = 0.94
) -> 'pd.DataFrame':
    """
    Generate a comparison table showing all key statistics.
    
    Returns DataFrame with columns:
    - Channel
    - Geo_Prior_Mean
    - Uncal_Mean, Uncal_HDI_Lower, Uncal_HDI_Upper
    - Cal_Mean, Cal_HDI_Lower, Cal_HDI_Upper
    - Change_Abs, Change_Pct
    
    Parameters
    ----------
    idata_uncal : arviz.InferenceData
        Uncalibrated posterior
    idata_cal : arviz.InferenceData
        Calibrated posterior
    channels : list of str
        Channel names
    geo_priors : dict
        Geo-experiment priors
    hdi_prob : float, default=0.94
        HDI probability
        
    Returns
    -------
    pd.DataFrame
        Comparison table (requires pandas)
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = generate_comparison_table(idata_uncal, idata_cal, channels, geo_priors)
    >>> print(df)
    """
    import pandas as pd
    
    s_uncal = _extract_beta_samples(idata_uncal, channels)
    s_cal = _extract_beta_samples(idata_cal, channels)
    
    rows = []
    for ch in channels:
        # Geo prior
        geo_mean = geo_priors.get(ch, {}).get('beta_mu', np.nan)
        
        # Uncalibrated
        m_u, lo_u, hi_u = _mean_and_hdi(s_uncal[ch], hdi_prob=hdi_prob)
        
        # Calibrated
        m_c, lo_c, hi_c = _mean_and_hdi(s_cal[ch], hdi_prob=hdi_prob)
        
        # Changes
        change_abs = m_c - m_u
        change_pct = 100 * change_abs / abs(m_u) if m_u != 0 else 0
        
        rows.append({
            'Channel': ch,
            'Geo_Prior_Mean': geo_mean,
            'Uncal_Mean': m_u,
            'Uncal_HDI_Lower': lo_u,
            'Uncal_HDI_Upper': hi_u,
            'Cal_Mean': m_c,
            'Cal_HDI_Lower': lo_c,
            'Cal_HDI_Upper': hi_c,
            'Change_Abs': change_abs,
            'Change_Pct': change_pct
        })
    
    return pd.DataFrame(rows)


# ============================================================================
# MODULE METADATA
# ============================================================================

__all__ = [
    'plot_prior_vs_posterior_shift',
    'plot_beta_forest_pre_post',
    'generate_comparison_table',
    '_extract_beta_samples',
    '_mean_and_hdi'
]

__version__ = '1.0.0'
__author__ = 'Bayesian MMM Portfolio Project'


if __name__ == '__main__':
    print(f"""
Calibration Visualization Module v{__version__}
{'='*80}

Available functions:
- plot_prior_vs_posterior_shift()  # 3-way comparison
- plot_beta_forest_pre_post()       # Forest plot (uncal vs cal)
- generate_comparison_table()       # Summary statistics

Helper functions:
- _extract_beta_samples()
- _mean_and_hdi()
- _normal_pdf()
- _kde_like_hist()

Import with:
    from calibration_viz import plot_prior_vs_posterior_shift, plot_beta_forest_pre_post

Documentation available via help(function_name)
    """)
