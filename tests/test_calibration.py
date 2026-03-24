import numpy as np
import pandas as pd

from mmm_calibration.calibration import build_geo_priors


def test_build_geo_priors_precision_weighted_and_floor():
    geo_exp = pd.DataFrame({
        "experiment_id": ["exp_01", "exp_02", "exp_03"],
        "channel": ["youtube", "youtube", "search"],
        "start_date": ["2024-01-01"] * 3,
        "end_date": ["2024-02-01"] * 3,
        "incremental_lift_mean": [0.10, 0.20, 0.05],
        "incremental_lift_se": [0.10, 0.20, 0.30],
    })

    priors = build_geo_priors(
        geo_exp=geo_exp,
        channels=["youtube", "search", "social"],
        default_mu=0.0,
        default_sigma=1.0,
        sigma_floor=0.10,
        strength=1.0
    )

    # social has no experiments -> defaults
    assert priors["social"]["beta_mu"] == 0.0
    assert priors["social"]["beta_sigma"] == 1.0

    # youtube: precision weighted average
    mu = np.array([0.10, 0.20])
    se = np.array([0.10, 0.20])
    w = 1.0 / (se**2)
    expected_mu = float((w * mu).sum() / w.sum())
    expected_sigma = float(1.0 / np.sqrt(w.sum()))
    expected_sigma = max(0.10, expected_sigma)

    assert abs(priors["youtube"]["beta_mu"] - expected_mu) < 1e-12
    assert abs(priors["youtube"]["beta_sigma"] - expected_sigma) < 1e-12

    # sigma floor respected
    assert priors["youtube"]["beta_sigma"] >= 0.10
