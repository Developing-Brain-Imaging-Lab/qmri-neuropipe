# Free-Water Elimination DTI (FWE-DTI)

For every accepted field and default, see {ref}`Diffusion model fitting <diffusion-model-fitting>`.

The Free-Water Elimination DTI model fits a bi-tensor model to separate the signal into two compartments:
1.  **Tissue Compartment:** Modeled as a diffusion tensor (like standard DTI).
2.  **Free Water Compartment:** Modeled as an isotropic sphere with the diffusivity of free water.

This corrects DTI metrics (like FA and MD) for Partial Volume Effects (PVE) from CSF or edema, which is crucial in atrophy or peri-ventricular regions.

## Backend

This model uses **DIPY** (`dipy.reconst.fwdti`).

## Configuration

```yaml
dmri:
  modeling:
    fwe_dti:
      enabled: true
      fit_method: "NLLS" # Options: NLLS
      # parameters can be nested here if preferred
```

### Options

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `fit_method` | Optimization algorithm. `"NLLS"` (Non-Linear Least Squares) is standard for this non-linear model. | `"NLLS"` |
| `sigma` | Noise standard deviation (if known) to weight the fit. | `None` |
| `metrics` | Metrics to save. | `["fa", "md", "ad", "rd", "f"]` |

## Outputs

- **Corrected FA:** `sub-<id>_..._model-FWDTI_FA.nii.gz`
- **Corrected MD:** `sub-<id>_..._model-FWDTI_MD.nii.gz`
- **Free Water Fraction (f):** `sub-<id>_..._model-FWDTI_FW.nii.gz` (Value between 0 and 1 representing the volume fraction of free water in the voxel).
