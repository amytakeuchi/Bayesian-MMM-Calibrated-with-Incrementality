from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from .adstock import geometric_adstock_2d
from .saturation import hill_saturation


def _sample_posterior_beta(idata, n: int, rng: np.random.Generator) -> np.ndarray:
    beta = idata.posterior["beta"].values  # (chain, draw, C)
    beta_flat = beta.reshape(-1, beta.shape[-1])
    idx = rng.choice(beta_flat.shape[0], size=min(n, beta_flat.shape[0]), replace=False)
    return beta_flat[idx]


def expected_incremental_from_spend(
    spend_vec: np.ndarray,
    beta_draws: np.ndarray,
    hill_alpha: float,
    hill_theta: float
) -> np.ndarray:
    """
    Compute expected incremental (relative) lift proxy:
      sum_c beta_c * Hill(spend_c)
    Here spend_vec is assumed already in the same space MMM was trained on
    (typically standardized and already adstocked-ish; for simplicity we treat
     it as a static single-period decision).
    """
    sat = hill_saturation(spend_vec[None, :], alpha=hill_alpha, theta=hill_theta)[0]  # (C,)
    # For each draw: dot(sat, beta)
    return beta_draws @ sat


def optimize_budget(
    idata,
    channels: list[str],
    total_budget: float,
    hill_alpha: float,
    hill_theta: float,
    n_samples: int = 400,
    risk_aversion: float = 0.0,
    step: float = 0.05,
    seed: int = 123
) -> Dict[str, object]:
    """
    Simple robust optimizer:
      - discretize allocation weights in increments of `step`
      - evaluate expected objective over posterior beta draws
      - return best allocation and uncertainty distribution

    Objective: mean(lift) - risk_aversion * var(lift)
    where lift = beta_draw dot Hill(spend)
    """
    rng = np.random.default_rng(seed)
    C = len(channels)
    beta_draws = _sample_posterior_beta(idata, n=n_samples, rng=rng)

    # Generate weight grids for C channels (simple recursive for small C)
    weights_list = []

    def rec_build(prefix, remaining, k):
        if k == C - 1:
            w_last = remaining
            weights_list.append(prefix + [w_last])
            return
        w = 0.0
        while w <= remaining + 1e-9:
            rec_build(prefix + [w], remaining - w, k + 1)
            w += step

    rec_build([], 1.0, 0)
    W = np.array(weights_list, dtype=float)  # (G, C)

    # Convert to spend vectors
    spend_mat = W * total_budget

    best_idx = None
    best_obj = -np.inf
    obj_vals = []

    for i in range(spend_mat.shape[0]):
        lift_draws = expected_incremental_from_spend(
            spend_vec=spend_mat[i],
            beta_draws=beta_draws,
            hill_alpha=hill_alpha,
            hill_theta=hill_theta
        )
        m = float(np.mean(lift_draws))
        v = float(np.var(lift_draws))
        obj = m - risk_aversion * v
        obj_vals.append(obj)
        if obj > best_obj:
            best_obj = obj
            best_idx = i

    best_w = W[best_idx]
    best_spend = spend_mat[best_idx]

    # posterior distribution of optimality is hard; we provide lift distribution for chosen allocation
    best_lift_draws = expected_incremental_from_spend(best_spend, beta_draws, hill_alpha, hill_theta)

    return {
        "channels": channels,
        "total_budget": float(total_budget),
        "step": float(step),
        "risk_aversion": float(risk_aversion),
        "best_weights": {ch: float(best_w[j]) for j, ch in enumerate(channels)},
        "best_spend": {ch: float(best_spend[j]) for j, ch in enumerate(channels)},
        "best_lift_mean": float(np.mean(best_lift_draws)),
        "best_lift_p10": float(np.quantile(best_lift_draws, 0.10)),
        "best_lift_p90": float(np.quantile(best_lift_draws, 0.90)),
        "search_space_size": int(W.shape[0]),
    }
