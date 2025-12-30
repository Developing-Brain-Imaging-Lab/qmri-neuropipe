# Diffusion Tensor Imaging (DTI)

DTI models the diffusion signal as a 3D Gaussian distribution (ellipsoid), characterized by a symmetric 3x3 tensor. It is the most fundamental dMRI model, providing metrics about white matter integrity.

## Backends

`qmri-neuropipe` supports multiple backends for DTI fitting. The default and recommended backend is **DIPY**.

- **dipy**: Pure Python implementation. Supports Weighted Linear Least Squares (WLLS), Ordinary Least Squares (OLS), Non-Linear Least Squares (NLLS), and RESTORE (robust to outliers).
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
        metrics: ["fa", "md", "ad", "rd", "color_fa"]
```

### Options

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `method` | Backend software to use. | `"dipy"` |
| `sub_method` | (DIPY only) Solver algorithm. | `"WLLS"` |
| `metrics` | List of metrics to compute. | `["fa", "md", "ad", "rd"]` |
| `save_tensor` | (FSL only) Whether to save the full tensor file. | `false` |

## Outputs

All outputs are saved in the `dwi` directory, following BIDS naming conventions.

- **FA (Fractional Anisotropy):** `sub-<id>_ses-<id>_model-DTI_FA.nii.gz`
- **MD (Mean Diffusivity):** `sub-<id>_ses-<id>_model-DTI_MD.nii.gz`
- **AD (Axial Diffusivity):** `sub-<id>_ses-<id>_model-DTI_AD.nii.gz`
- **RD (Radial Diffusivity):** `sub-<id>_ses-<id>_model-DTI_RD.nii.gz`
- **Color FA:** `sub-<id>_ses-<id>_model-DTI_DECFA.nii.gz` (Direction Encoded Color Map)
