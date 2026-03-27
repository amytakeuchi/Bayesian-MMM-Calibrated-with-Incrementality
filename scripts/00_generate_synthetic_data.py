"""
Synthetic MMM Dataset Generator
===============================

Generates synthetic-but-realistic datasets for demonstrating Bayesian MMM calibration
with incrementality evidence.

What it simulates:
- Multiple marketing channels with spend dynamics
- Adstock (carryover) + Hill saturation (diminishing returns)
- Controls (price, promo, macro, seasonality)
- Outcome time series y built from known ground truth
- Geo experiment summaries providing noisy causal lift signals

Outputs (written to /data/raw):
- marketing_spend.csv: (date, channel, spend)
- outcome_sales.csv: (date, y)
- controls.csv: (date, controls...)
- geo_experiment_results.csv: (experiment_id, channel, start/end, lift_mean, lift_se, ...)

Why synthetic:
- Allows validation against known truth
- Enables clean demonstration of calibration effects
- Makes failure modes explainable and reproducible

This script is the starting point for a fully self-contained repo demo.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------
@dataclass
class SimConfig:
    # Time
    start_date: str = "2023-01-02"   # Monday
    n_weeks: int = 156               # 3 years weekly
    freq: str = "W-MON"

    # Channels
    channels: tuple = ("search", "social", "youtube", "tv")

    # Spend generation
    base_spend: dict = None          # filled in __post_init__
    spend_cv: float = 0.35           # coefficient of variation
    spend_trend_strength: float = 0.10

    # Demand baseline + controls
    base_level: float = 50_000.0
    base_growth_per_week: float = 0.0015
    season_amp: float = 0.12

    promo_prob: float = 0.12
    promo_lift: float = 0.10

    price_sd: float = 0.015
    price_sensitivity: float = -0.9

    macro_sd: float = 0.02
    macro_sensitivity: float = 0.7

    # True media response (adstock + Hill saturation)
    # Each channel has different half-saturation and effectiveness.
    hill_alpha: dict = None
    hill_theta: dict = None
    beta: dict = None                 # channel effectiveness scaling

    # Adstock persistence
    adstock_lambda: dict = None       # 0..1

    # Observation noise
    obs_noise_sd: float = 0.06        # multiplicative-ish

    # Geo experiment simulation
    n_experiments: int = 8
    experiment_len_weeks: int = 6
    experiment_geo_fraction: float = 0.25
    experiment_noise_scale: float = 1.2
    experiment_bias_sd: float = 0.01  # small bias to mimic imperfect experiments

    # Random
    seed: int = 11

    def __post_init__(self):
        if self.base_spend is None:
            self.base_spend = {
                "search": 18000,
                "social": 14000,
                "youtube": 22000,
                "tv": 30000,
            }
        if self.hill_alpha is None:
            self.hill_alpha = {
                "search": 1.2,
                "social": 1.3,
                "youtube": 1.6,
                "tv": 1.4,
            }
        if self.hill_theta is None:
            self.hill_theta = {
                "search": 16000,
                "social": 14000,
                "youtube": 28000,
                "tv": 42000,
            }
        if self.beta is None:
            # Interpretable: marginal effectiveness *after* saturation/adstock
            self.beta = {
                "search": 0.08,
                "social": 0.06,
                "youtube": 0.10,
                "tv": 0.12,
            }
        if self.adstock_lambda is None:
            self.adstock_lambda = {
                "search": 0.20,
                "social": 0.35,
                "youtube": 0.55,
                "tv": 0.70,
            }


# -----------------------------
# Helpers
# -----------------------------
def hill(x: np.ndarray, alpha: float, theta: float) -> np.ndarray:
    """Hill saturation in [0,1)."""
    x = np.maximum(x, 0.0)
    return (x ** alpha) / (x ** alpha + theta ** alpha + 1e-12)


def geometric_adstock(x: np.ndarray, lam: float) -> np.ndarray:
    """Simple geometric adstock."""
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for t in range(len(x)):
        carry = x[t] + lam * carry
        out[t] = carry
    return out


# -----------------------------
# Data generation
# -----------------------------
def generate_controls(cfg: SimConfig, dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    n = len(dates)
    t = np.arange(n)

    seasonality = cfg.season_amp * np.sin(2 * np.pi * t / 52) + 0.04 * np.cos(2 * np.pi * t / 26)
    macro_index = rng.normal(0.0, cfg.macro_sd, size=n)

    promo = (rng.uniform(size=n) < cfg.promo_prob).astype(int)
    # Add light promo seasonality: more promos in Q4-ish
    promo = np.maximum(promo, (rng.uniform(size=n) < (0.03 + 0.04 * (seasonality > 0.08))).astype(int))

    price_index = 1.0 + rng.normal(0.0, cfg.price_sd, size=n) + 0.01 * (seasonality < -0.05)

    controls = pd.DataFrame({
        "date": dates.date.astype(str),
        "price_index": price_index,
        "promo": promo,
        "macro_index": macro_index,
        "seasonality": seasonality,
    })
    return controls


def generate_spend(cfg: SimConfig, dates: pd.DatetimeIndex, controls: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(dates)
    t = np.arange(n)

    # A shared spend trend component (e.g., growth over time)
    spend_trend = np.exp(cfg.spend_trend_strength * (t / n - 0.5))

    rows = []
    for ch in cfg.channels:
        base = cfg.base_spend[ch]

        # Correlate spend with promo and macro slightly to create realistic endogeneity
        promo = controls["promo"].to_numpy()
        macro = controls["macro_index"].to_numpy()
        season = controls["seasonality"].to_numpy()

        # Deterministic mean spend before random variation
        mean_spend_t = base * spend_trend * (1.0 + 0.10 * promo + 0.05 * macro + 0.06 * np.maximum(season, 0))

        # Lognormal noise around mean spend
        sigma = np.sqrt(np.log(1 + cfg.spend_cv**2))
        spend = rng.lognormal(mean=np.log(np.maximum(mean_spend_t, 1e-6)), sigma=sigma)

        tmp = pd.DataFrame({
            "date": dates.date.astype(str),
            "channel": ch,
            "spend": spend.astype(float),
        })
        rows.append(tmp)

    spend_df = pd.concat(rows, ignore_index=True)
    return spend_df


def generate_outcome(cfg: SimConfig, dates: pd.DatetimeIndex, spend_df: pd.DataFrame, controls: pd.DataFrame, rng: np.random.Generator):
    n = len(dates)
    t = np.arange(n)

    # baseline demand (multiplicative dynamics)
    baseline = cfg.base_level * np.exp(cfg.base_growth_per_week * t)

    # controls multiplicative effect
    price = controls["price_index"].to_numpy()
    promo = controls["promo"].to_numpy()
    macro = controls["macro_index"].to_numpy()
    season = controls["seasonality"].to_numpy()

    controls_mult = np.exp(
        cfg.price_sensitivity * (price - 1.0)
        + cfg.macro_sensitivity * macro
        + season
    ) * (1.0 + cfg.promo_lift * promo)

    # media contributions
    media_mult = np.ones(n, dtype=float)

    # build spend matrix per channel
    spend_wide = spend_df.pivot(index="date", columns="channel", values="spend").reindex(controls["date"]).fillna(0.0)

    true_channel_contrib = {}
    for ch in cfg.channels:
        x = spend_wide[ch].to_numpy()
        ad = geometric_adstock(x, cfg.adstock_lambda[ch])
        sat = hill(ad, cfg.hill_alpha[ch], cfg.hill_theta[ch])
        # multiplicative lift contribution
        contrib = cfg.beta[ch] * sat
        true_channel_contrib[ch] = contrib
        media_mult *= (1.0 + contrib)

    mu = baseline * controls_mult * media_mult

    # observation noise (lognormal)
    eps = rng.normal(0.0, cfg.obs_noise_sd, size=n)
    y = np.exp(np.log(np.maximum(mu, 1e-6)) + eps)

    outcome = pd.DataFrame({
        "date": dates.date.astype(str),
        "y": y.astype(float),
    })

    # also return ground truth contributions (helpful for debugging/plots, you can drop later)
    truth = pd.DataFrame({"date": dates.date.astype(str)})
    for ch in cfg.channels:
        truth[f"true_contrib_{ch}"] = true_channel_contrib[ch]
    truth["true_mu"] = mu.astype(float)

    return outcome, truth


def simulate_geo_experiments(cfg: SimConfig, spend_df: pd.DataFrame, outcome_df: pd.DataFrame, truth_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Create synthetic geo-experiment results per channel.
    We approximate "incremental lift" over a short window by using the ground-truth media
    contribution for that channel over that window, then add noise + mild bias.
    """
    dates = pd.to_datetime(outcome_df["date"])
    n = len(dates)

    results = []
    channels = list(cfg.channels)

    # 1. Randomly select time windows for the experiments
    # We avoid the very beginning and end of the dataset to ensure enough buffer
    # choose experiment windows
    possible_start = np.arange(20, n - cfg.experiment_len_weeks - 1)
    starts = rng.choice(possible_start, size=cfg.n_experiments, replace=False)
    starts.sort()

    for i, s in enumerate(starts, start=1):
        # Rotate through channels so each one gets tested
        ch = channels[(i - 1) % len(channels)]
        start_dt = dates.iloc[s]
        end_dt = dates.iloc[s + cfg.experiment_len_weeks - 1]

        # 2. Look at the "Hidden Truth"
        # We find what the channel *actually* contributed during this specific window
        window_mask = (dates >= start_dt) & (dates <= end_dt)
        y_window = outcome_df.loc[window_mask, "y"].to_numpy()

        # "True" incremental lift proxy: average multiplicative contribution of channel
        true_c = truth_df.loc[window_mask, f"true_contrib_{ch}"].to_numpy()
        true_lift_mean = float(np.mean(true_c))  # approx percent lift

        # 3. Calculate Spend
        # Sum up how much was spent on this channel during the experiment period    
        # spend during test
        spend_window = spend_df[(spend_df["channel"] == ch) & (pd.to_datetime(spend_df["date"]).between(start_dt, end_dt))]["spend"].to_numpy()
        spend_during_test = float(np.sum(spend_window))

        # 4. Model 'Real-World' Imperfection
        # Real experiments aren't perfect. We calculate a Standard Error (SE) that depends on volatility and sample size, scaled
        # based on how volatile the sales data (y) was during the window.
        # Use outcome volatility in window as a proxy
        se_base = np.std(np.log(np.maximum(y_window, 1e-6))) / np.sqrt(len(y_window))
        lift_se = float(cfg.experiment_noise_scale * se_base)

        # 5. Add Noise and Bias
        # In reality, tests can be slightly biased (bias) and always have random noise.
        # This makes the calibration task realistic for the MMM.
        # Add mild bias + noise to mimic imperfect geo experiments
        bias = rng.normal(0.0, cfg.experiment_bias_sd)
        noisy_lift = true_lift_mean + bias + rng.normal(0.0, lift_se)

        # 6. Record the Experiment result
        results.append({
            "experiment_id": f"exp_{i:02d}",
            "channel": ch,
            "start_date": start_dt.date().isoformat(),
            "end_date": end_dt.date().isoformat(),
            "incremental_lift_mean": float(noisy_lift),
            "incremental_lift_se": float(max(lift_se, 1e-4)),
            "spend_during_test": spend_during_test,
            "notes": "Synthetic geo result: centered near ground-truth channel lift with noise and mild bias."
        })

    return pd.DataFrame(results)


def main():
    cfg = SimConfig()
    rng = np.random.default_rng(cfg.seed)

    repo_root = Path(".")
    raw_dir = repo_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(cfg.start_date, periods=cfg.n_weeks, freq=cfg.freq)

    # STEP 1: Generate 'external' factors like holidays, price changes, or the economy
    # 1) Controls
    controls = generate_controls(cfg, dates, rng)

    # STEP 2: Generate marketing spend patterns (often correlated with controls)
    # 2) Spend
    spend_df = generate_spend(cfg, dates, controls, rng)

    # STEP 3: Combine controls + spend into the final 'Sales' outcome (y)
    # This creates the 'truth_df' (the hidden blueprint of what caused what)
    # 3) Outcome
    outcome_df, truth_df = generate_outcome(cfg, dates, spend_df, controls, rng)

    # STEP 4: Run the simulated experiments based on that hidden blueprint
    # 4) Geo experiments (calibration bridge)
    geo_exp = simulate_geo_experiments(cfg, spend_df, outcome_df, truth_df, rng)

    # SAVE EVERYTHING: These CSVs are what you'd typically receive from a client
    # Write CSVs
    spend_df.to_csv(raw_dir / "marketing_spend.csv", index=False)
    outcome_df.to_csv(raw_dir / "outcome_sales.csv", index=False)
    controls.to_csv(raw_dir / "controls.csv", index=False)
    geo_exp.to_csv(raw_dir / "geo_experiment_results.csv", index=False)

    # Optional: write a minimal data dictionary
    data_dict = {
        "marketing_spend.csv": ["date", "channel", "spend"],
        "outcome_sales.csv": ["date", "y"],
        "controls.csv": ["date", "price_index", "promo", "macro_index", "seasonality"],
        "geo_experiment_results.csv": [
            "experiment_id", "channel", "start_date", "end_date",
            "incremental_lift_mean", "incremental_lift_se", "spend_during_test", "notes"
        ],
        "notes": {
            "frequency": cfg.freq,
            "channels": list(cfg.channels),
            "dgp": "y = baseline * controls_mult * Π(1 + beta_c * Hill(Adstock(spend_c))) * noise",
            "geo_experiments": "incremental_lift_mean is noisy proxy of true channel contribution over experiment window",
        }
    }
    (raw_dir / "data_dictionary.json").write_text(json.dumps(data_dict, indent=2))

    print("Wrote synthetic MMM datasets to:")
    for f in ["marketing_spend.csv", "outcome_sales.csv", "controls.csv", "geo_experiment_results.csv", "data_dictionary.json"]:
        print(" -", raw_dir / f)


if __name__ == "__main__":
    main()
