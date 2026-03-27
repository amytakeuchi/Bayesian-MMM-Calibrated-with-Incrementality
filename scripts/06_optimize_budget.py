"""
Budget Optimization Using Calibrated Posterior
==============================================

Uses posterior draws from the calibrated MMM to recommend budget allocation
across channels under uncertainty.

What it does:
- Loads calibrated posterior draws
- Searches over candidate allocations (grid)
- Evaluates expected lift and uncertainty bands (P10/P90)
- Outputs a decision-ready recommendation

Outputs (written to /reports):
- budget_optimization.json

Why this is “FAANG-grade”:
- It treats uncertainty as part of the product decision.
- It produces ranges and risk-aware guidance instead of single-point ROAS.
"""

from pathlib import Path
import json
import joblib
import arviz as az

from mmm_calibration.config import MMMConfig, PathsConfig
from mmm_calibration.budget_optimization import optimize_budget


def main():
    paths = PathsConfig(repo_root=Path("."))
    cfg = MMMConfig()

    mm = joblib.load(paths.data_processed / "model_matrix.joblib")
    idata = az.from_netcdf(paths.models / "calibrated_idata.nc")

    # Use standardized space budget for demonstration
    # In practice you'd map real dollars -> standardized spend units.
    total_budget = 4.0  # "budget units" in standardized scale

    result = optimize_budget(
        idata=idata,
        channels=mm.channels,
        total_budget=total_budget,
        hill_alpha=cfg.hill_alpha,
        hill_theta=cfg.hill_theta,
        n_samples=cfg.n_posterior_samples_for_opt,
        risk_aversion=cfg.risk_aversion,
        step=0.10,
        seed=cfg.random_seed
    )

    out_json = paths.reports / "budget_optimization.json"
    out_json.write_text(json.dumps(result, indent=2))
    print("Saved:", out_json)
    print("Best spend:", result["best_spend"])
    print("Lift mean [p10, p90]:", result["best_lift_mean"], result["best_lift_p10"], result["best_lift_p90"])


if __name__ == "__main__":
    main()
