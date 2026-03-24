# ============================================================================
# ADD THIS TO THE END OF NOTEBOOK 03
# Section: Save Artifacts for Notebook 04
# ============================================================================

"""
This cell saves all necessary artifacts from Notebook 03 so that
Notebook 04 can load them for ROI analysis and budget optimization.

Files saved:
1. idata_calibrated.nc - Calibrated model posterior
2. model_artifacts_cal.pkl - Model matrix, config, geo_priors
3. comparison_results.pkl - Pre/post calibration comparison (optional)
"""

import pickle
import arviz as az
from pathlib import Path

# ============================================================================
# 1. Save Calibrated InferenceData
# ============================================================================

print("="*80)
print("SAVING ARTIFACTS FOR NOTEBOOK 04")
print("="*80)

outputs_dir = repo_root / 'outputs'
outputs_dir.mkdir(parents=True, exist_ok=True)

# Save calibrated model
idata_cal_path = outputs_dir / 'idata_calibrated.nc'
az.to_netcdf(idata_cal, idata_cal_path)
print(f"✓ Saved: {idata_cal_path}")

# ============================================================================
# 2. Save Model Artifacts
# ============================================================================

# Package all necessary artifacts
artifacts = {
    'model_matrix': mm,  # ModelMatrix object with X_spend, X_controls, y, channels, etc.
    'config': cfg,  # MMMConfig object
    'geo_priors': geo_priors,  # Dict of geo-experiment priors
    'channels': mm.channels,  # List of channel names
    'y_max': outcome[cfg.y_col].max(),  # Max scaling factor for unscaling
    'dates': mm.dates,  # Time index (optional but useful)
}

artifacts_path = outputs_dir / 'model_artifacts_cal.pkl'
with open(artifacts_path, 'wb') as f:
    pickle.dump(artifacts, f)

print(f"✓ Saved: {artifacts_path}")
print(f"  Contains: model_matrix, config, geo_priors, channels, y_max")

# ============================================================================
# 3. Optional: Save Comparison Results
# ============================================================================

# If you have comparison results, save them too
if 'idata_uncal' in locals():
    comparison_artifacts = {
        'idata_uncal': idata_uncal,
        'idata_cal': idata_cal,
        'beta_uncal': beta_uncal if 'beta_uncal' in locals() else None,
        'beta_cal': beta_cal if 'beta_cal' in locals() else None,
        'comparison_df': comparison_df if 'comparison_df' in locals() else None
    }
    
    comparison_path = outputs_dir / 'comparison_results.pkl'
    with open(comparison_path, 'wb') as f:
        pickle.dump(comparison_artifacts, f)
    
    print(f"✓ Saved: {comparison_path}")
    print(f"  Contains: uncal/cal models and comparison results")

# ============================================================================
# 4. Verify Files for Notebook 04
# ============================================================================

print("\n" + "="*80)
print("VERIFICATION - Files Required for Notebook 04")
print("="*80)

required_files = [
    outputs_dir / 'idata_calibrated.nc',
    outputs_dir / 'model_artifacts_cal.pkl'
]

all_present = True
for filepath in required_files:
    if filepath.exists():
        size_kb = filepath.stat().st_size / 1024
        print(f"✓ {filepath.name:30s} ({size_kb:>8.1f} KB)")
    else:
        print(f"✗ {filepath.name:30s} MISSING!")
        all_present = False

if all_present:
    print("\n✅ All required files saved successfully!")
    print("   Notebook 04 is ready to run.")
else:
    print("\n⚠️  Some required files are missing.")
    print("   Please check the code above for errors.")

print("="*80)
