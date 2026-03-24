import numpy as np
import pandas as pd

from mmm_calibration.config import MMMConfig
from mmm_calibration.preprocessing import build_model_matrix


def test_build_model_matrix_shapes_and_no_nans():
    cfg = MMMConfig(
        channels=["search", "social"],
        control_cols=["price_index", "promo"],
        standardize_spend=True,
        standardize_controls=True,
    )

    # minimal synthetic inputs
    dates = pd.date_range("2024-01-01", periods=10, freq="W-MON")

    spend = pd.DataFrame({
        "date": np.repeat(dates, 2),
        "channel": ["search", "social"] * len(dates),
        "spend": np.random.RandomState(0).rand(len(dates) * 2) * 1000,
    })

    outcome = pd.DataFrame({
        "date": dates,
        "y": np.random.RandomState(1).rand(len(dates)) * 500 + 1000
    })

    controls = pd.DataFrame({
        "date": dates,
        "price_index": 1.0 + np.random.RandomState(2).randn(len(dates)) * 0.01,
        "promo": (np.random.RandomState(3).rand(len(dates)) > 0.8).astype(int),
    })

    mm = build_model_matrix(spend, outcome, controls, cfg)

    assert mm.X_spend.shape == (len(dates), 2)
    assert mm.X_controls.shape == (len(dates), 2)
    assert mm.y.shape == (len(dates),)

    assert np.isfinite(mm.X_spend).all()
    assert np.isfinite(mm.X_controls).all()
    assert np.isfinite(mm.y).all()

    # standardization sanity: mean approx 0 for each channel
    assert abs(mm.X_spend[:, 0].mean()) < 1e-6
    assert abs(mm.X_spend[:, 1].mean()) < 1e-6
