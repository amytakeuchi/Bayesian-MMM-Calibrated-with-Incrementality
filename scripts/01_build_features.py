"""
Feature Builder (Raw → Model Matrix)
===================================

Loads raw CSVs from /data/raw and produces a model-ready feature object.

What it does:
- Validates required columns
- Aligns dates across outcome, controls, and channel spend
- Pivots spend into a (T, C) channel matrix
- Standardizes spend and controls for stable Bayesian sampling
- Saves output to /data/processed for reuse in multiple scripts/notebooks

Outputs:
- model_matrix.parquet or model_matrix.joblib (depending on your implementation)
- (Optional) scalers/metadata for mapping back to original units

Why this matters:
- Prevents “notebook-only” pipelines.
- Creates a reproducible artifact that guarantees consistent inputs across models.
"""

from pathlib import Path
import joblib

from mmm_calibration.config import MMMConfig, PathsConfig
from mmm_calibration.preprocessing import load_raw_data, build_model_matrix


def main():
    paths = PathsConfig(repo_root=Path("."))
    cfg = MMMConfig()

    spend, outcome, controls = load_raw_data(paths, cfg)
    mm = build_model_matrix(spend, outcome, controls, cfg)

    out_path = paths.data_processed / "model_matrix.joblib"
    joblib.dump(mm, out_path)
    print("Saved model matrix:", out_path)


if __name__ == "__main__":
    main()
