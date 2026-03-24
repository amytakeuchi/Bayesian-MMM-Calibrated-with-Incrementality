import numpy as np
import xarray as xr

from mmm_calibration.budget_optimization import optimize_budget


def _fake_idata(beta_draws: np.ndarray):
    """
    Create a minimal ArviZ-like idata structure containing posterior['beta'].
    beta_draws shape: (N, C)
    """
    # convert to (chain, draw, C) with 1 chain
    beta_3d = beta_draws[None, :, :]
    posterior = xr.Dataset({"beta": (("chain", "draw", "beta_dim_0"), beta_3d)})
    class Fake:
        def __init__(self, posterior):
            self.posterior = posterior
    return Fake(posterior)


def test_optimize_budget_weights_sum_to_one_and_nonnegative():
    rng = np.random.default_rng(0)

    # 3 channels, make channel 0 strongest
    beta_draws = rng.normal(loc=[0.20, 0.10, 0.05], scale=[0.02, 0.02, 0.02], size=(200, 3))
    idata = _fake_idata(beta_draws)

    res = optimize_budget(
        idata=idata,
        channels=["a", "b", "c"],
        total_budget=10.0,
        hill_alpha=1.5,
        hill_theta=5.0,
        n_samples=150,
        risk_aversion=0.0,
        step=0.20,
        seed=0
    )

    w = np.array([res["best_weights"][k] for k in ["a", "b", "c"]], dtype=float)

    assert (w >= -1e-12).all()
    assert abs(w.sum() - 1.0) < 1e-9

    spend = np.array([res["best_spend"][k] for k in ["a", "b", "c"]], dtype=float)
    assert abs(spend.sum() - 10.0) < 1e-9
