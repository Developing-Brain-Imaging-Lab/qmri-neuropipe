# Diffusion Tensor Imaging (DTI)

DTI models the diffusion signal as a 3D Gaussian distribution (ellipsoid), characterized by a symmetric 3x3 tensor. It is the most fundamental dMRI model, providing metrics about white matter integrity.

## Backends

`qmri-neuropipe` supports multiple backends for DTI fitting. The default and recommended backend is **DIPY**.

- **dipy**: Pure Python implementation. Supports:
    - **WLLS** (Weighted Linear Least Squares) - Recommended default.
    - **OLS** (Ordinary Least Squares)
    - **NLLS** (Non-Linear Least Squares)
    - **RESTORE** (Robust Estimation of Tensors by Outlier Rejection)
    - **IRLS** (Iterative Reweighted Least Squares) - Robust fitting with custom weighting.
- **fsl**: Uses FSL's `dtifit`.
- **mrtrix**: Uses MRtrix3's `dwi2tensor` and `tensor2metric`.

## Configuration

To enable DTI, add the following to your `preproc.yaml`:

```yaml
dmri:
  modeling:
    dti:
      enabled: true
      method: "dipy" # Options: "dipy", "fsl", "mrtrix"
      parameters:
        sub_method: "WLLS" # For dipy: WLLS (default), OLS, NLLS, RESTORE
        metrics: ["fa", "md", "ad", "rd", "l1", "l2", "l3", "v1", "v2", "v3", "tensor_mrtrix"]
```

### Options

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `method` | Backend software to use (`dipy`, `fsl`, `mrtrix`). | `"dipy"` |
| `fit_method` | (DIPY only) Solver algorithm: `WLLS`, `OLS`, `NLLS`, `RESTORE`, `IRLS`. | `"WLLS"` |
| `weights_method` | (DIPY IRLS only) Robust method: `gm`, `cauchy`, `wls_m_est`. | `"wls_m_est"` |
| `weights_cutoff` | (DIPY IRLS only) Z-score threshold for outlier rejection. | `None` |
| `metrics` | List of metrics to compute: `fa`, `md`, `ad`, `rd`, `color_fa`, `l1`, `l2`, `l3`, `v1`, `v2`, `v3`, `tensor`, `tensor_fsl`, `tensor_mrtrix`, `evals`, `evecs`. | `["fa", "md", "ad", "rd"]` |
| `save_tensor` | (FSL only) Whether to save the full tensor file. | `false` |

## Outputs

All outputs are saved in the `dwi` directory, following BIDS naming conventions.

- **FA (Fractional Anisotropy):** `sub-<id>_ses-<id>_model-DTI_FA.nii.gz`
- **MD (Mean Diffusivity):** `sub-<id>_ses-<id>_model-DTI_MD.nii.gz`
- **AD (Axial Diffusivity):** `sub-<id>_ses-<id>_model-DTI_AD.nii.gz`
- **RD (Radial Diffusivity):** `sub-<id>_ses-<id>_model-DTI_RD.nii.gz`
- **Color FA:** `sub-<id>_ses-<id>_model-DTI_DECFA.nii.gz` (Direction Encoded Color Map)
- **Eigenvalues:** `sub-<id>_ses-<id>_model-DTI_L1.nii.gz`, `L2`, `L3`
- **Eigenvectors:** `sub-<id>_ses-<id>_model-DTI_V1.nii.gz`, `V2`, `V3`
- **Tensor (FSL/image basis):** `sub-<id>_ses-<id>_model-DTI_tensor.nii.gz`
- **Tensor (MRtrix/world basis):** `sub-<id>_ses-<id>_model-DTI_tensorMRTRIX.nii.gz`

## Notes

- `tensorMRTRIX` is the preferred tensor output for `mrview`.
- The DIPY backend now reorients `tensorMRTRIX` and `V1/V2/V3` into the image world basis instead of only reordering tensor components. This avoids left-right or axis-swap artifacts in `mrview` on non-canonical images.
- The FSL backend now exposes `L1/L2/L3`, `V1/V2/V3`, and derives `RD` and `DECFA` when requested.
