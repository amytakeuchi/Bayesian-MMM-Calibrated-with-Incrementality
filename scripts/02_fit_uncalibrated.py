"""
Fit Uncalibrated Bayesian MMM
=============================

Fits the baseline Bayesian MMM using observational data only.

Purpose:
- Establish what the MMM believes without any experimental anchoring
- Quantify typical failure modes:
    - over-attribution
    - wide uncertainty
    - unstable marginal ROI curves

Pipeline:
1) Load processed model matrix
2) Fit MMM with weakly informative priors on channel effects (beta)
3) Save posterior draws and posterior predictive samples

Outputs (written to /models):
- uncalibrated_idata.nc

This artifact becomes the baseline for calibration comparison.
"""

from pathlib import Path
import joblib
import arviz as az

from mmm_calibration.config import MMMConfig, PathsConfig
from mmm_calibration.model import fit_mmm


def main():
    # Initialize path configurations to locate data and model folders
    paths = PathsConfig(repo_root=Path("."))
    # Load model settings (hyperparameters, distributions, etc.)
    cfg = MMMConfig()

    # Load the pre-processed 'model_matrix' which contains your spend and target variables  
    mm = joblib.load(paths.data_processed / "model_matrix.joblib")
    # Run the Bayesian inference (MCMC) to estimate channel impacts (betas)
    # Note: 'priors_by_channel=None' means we are using default, uninformative priors
    fit = fit_mmm(mm, cfg, priors_by_channel=None)

    # Define the file path for the output 'InferenceData' (idata) object
    out_idata = paths.models / "uncalibrated_idata.nc"
    # Save the full posterior distribution to a NetCDF file for later analysis
    az.to_netcdf(fit.idata, out_idata)
    # Store a record of the priors used to compare against future calibrated versions
    joblib.dump(fit.priors_used, paths.models / "uncalibrated_priors.joblib")

# Log completion for the pipeline   
    print("Saved:", out_idata)


if __name__ == "__main__":
    main()
