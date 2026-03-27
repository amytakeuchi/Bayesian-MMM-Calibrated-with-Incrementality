"""
Build Experiment-Informed Priors (Geo → Beta Priors)
===================================================

Converts geo-experiment lift estimates into Bayesian priors for the MMM.

What it does:
- Loads geo_experiment_results.csv
- Aggregates experiments per channel (precision weighting)
- Produces per-channel Normal priors:
    beta_c ~ Normal(mu_c, sigma_c)

Design:
- Preserves uncertainty (SE) instead of forcing point-match calibration
- Applies sigma floors to avoid overconfidence
- Supports a `strength` knob for sensitivity analysis

Outputs (written to /data/processed):
- geo_priors.json

This is the calibration “bridge” between causal experiments and MMM inference.
"""

from pathlib import Path
import json
import joblib
import pandas as pd

from mmm_calibration.config import MMMConfig, PathsConfig
from mmm_calibration.calibration import build_geo_priors


def main():
    paths = PathsConfig(repo_root=Path("."))
    cfg = MMMConfig()

    mm = joblib.load(paths.data_processed / "model_matrix.joblib")
    geo_exp = pd.read_csv(paths.data_raw / "geo_experiment_results.csv")

    priors = build_geo_priors(
        geo_exp=geo_exp,
        channels=mm.channels,
        default_mu=cfg.beta_mu,
        default_sigma=cfg.beta_sigma,
        sigma_floor=0.10,
        strength=1.0
    )

    out_json = paths.data_processed / "geo_priors.json"
    out_json.write_text(json.dumps(priors, indent=2))
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
