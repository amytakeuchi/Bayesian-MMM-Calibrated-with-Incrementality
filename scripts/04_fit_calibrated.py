"""
Fit Calibrated Bayesian MMM
===========================

Fits the Bayesian MMM with priors informed by incrementality experiments.

Purpose:
- Update MMM beliefs using causal anchors
- Reduce over-attribution and stabilize ROI / contribution estimates

Pipeline:
1) Load processed model matrix
2) Load geo_priors.json
3) Fit MMM using channel-specific priors on beta
4) Save posterior draws and posterior predictive samples

Outputs (written to /models):
- calibrated_idata.nc

This calibrated model is used for comparison and budget optimization.
"""

from pathlib import Path
import json
import joblib
import arviz as az

from mmm_calibration.config import MMMConfig, PathsConfig
from mmm_calibration.model import fit_mmm


def main():
    paths = PathsConfig(repo_root=Path("."))
    cfg = MMMConfig()

    mm = joblib.load(paths.data_processed / "model_matrix.joblib")
    priors = json.loads((paths.data_processed / "geo_priors.json").read_text())

    fit = fit_mmm(mm, cfg, priors_by_channel=priors)

    out_idata = paths.models / "calibrated_idata.nc"
    az.to_netcdf(fit.idata, out_idata)
    joblib.dump(fit.priors_used, paths.models / "calibrated_priors.joblib")

    print("Saved:", out_idata)


if __name__ == "__main__":
    main()
