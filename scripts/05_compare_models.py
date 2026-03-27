"""
Compare Uncalibrated vs Calibrated MMM
======================================

Produces the core evaluation story for the repo.

What it compares:
- Posterior shift in channel effects (beta): mean + credible intervals
- Predictive behavior (posterior predictive RMSE)
- Optional: qualitative checks like whether calibration reduces implausible ROI

Outputs (written to /reports):
- model_comparison.json
- figures/fig1_prior_to_posterior_shift.png

Why this matters:
- Demonstrates senior-level modeling discipline:
  calibration is justified via measurable posterior movement and uncertainty changes.
"""

from pathlib import Path
import json
import arviz as az
import joblib

from mmm_calibration.config import PathsConfig, MMMConfig
from mmm_calibration.diagnostics import basic_model_report, summarize_beta
from mmm_calibration.plotting import plot_beta_posterior_shift


def main():
    paths = PathsConfig(repo_root=Path("."))
    cfg = MMMConfig()

    mm = joblib.load(paths.data_processed / "model_matrix.joblib")

    idata_u = az.from_netcdf(paths.models / "uncalibrated_idata.nc")
    idata_c = az.from_netcdf(paths.models / "calibrated_idata.nc")

    # Lightweight reports
    # (We create FitResult-like wrappers for report convenience.)
    class _FR:
        def __init__(self, idata):
            self.idata = idata
            self.model_matrix = mm
            self.priors_used = joblib.load(paths.models / ("calibrated_priors.joblib" if "calibrated" in str(idata) else "uncalibrated_priors.joblib"))

    report = {
        "uncalibrated": {
            "rmse_pp_mean": float(((idata_u.posterior_predictive["y_obs"].mean(("chain","draw")).values - mm.y) ** 2).mean() ** 0.5),
            "beta_summary": summarize_beta(idata_u, mm.channels),
        },
        "calibrated": {
            "rmse_pp_mean": float(((idata_c.posterior_predictive["y_obs"].mean(("chain","draw")).values - mm.y) ** 2).mean() ** 0.5),
            "beta_summary": summarize_beta(idata_c, mm.channels),
        }
    }

    out_report = paths.reports / "model_comparison.json"
    out_report.write_text(json.dumps(report, indent=2))
    print("Saved:", out_report)

    # Plot posterior shift
    fig_path = paths.reports / "figures" / "fig1_beta_posterior_shift.png"
    plot_beta_posterior_shift(
        beta_uncal=report["uncalibrated"]["beta_summary"],
        beta_cal=report["calibrated"]["beta_summary"],
        out_path=str(fig_path)
    )
    print("Saved:", fig_path)


if __name__ == "__main__":
    main()
