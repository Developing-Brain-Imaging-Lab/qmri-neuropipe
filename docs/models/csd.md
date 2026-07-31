# Constrained Spherical Deconvolution (CSD)

For every accepted field and default, see {ref}`Diffusion model fitting <diffusion-model-fitting>`.

CSD estimates the Fiber Orientation Distribution (FOD) by deconvolving the diffusion signal with a tissue response function. It is capable of resolving crossing fibers, which DTI cannot do.

## Backend

**MRtrix3** (`dwi2response` and `dwi2fod`).

> **Requirements:** MRtrix3 must be installed and available in the system path.

## Configuration

To enable CSD, add a `csd` section to the `modeling` block in `preproc.yaml`.

```yaml
dmri:
  modeling:
    csd:
      enabled: true
      method: "msmt_csd" # or "csd" for single-shell
      parameters:
         response_algorithm: "dhollander" # Algorithm for response function estimation
         fod_algorithm: "msmt_csd" # Algorithm for FOD estimation
         lmax: [4, 8, 8] # Max spherical harmonic order for each tissue type
```

### Options

| Parameter | Desc | Default |
| :--- | :--- | :--- |
| `method` | High-level method selector (`"msmt_csd"` or `"csd"`). | `"msmt_csd"` |
| `response_algorithm` | Algorithm for `dwi2response` (e.g., `"dhollander"`, `"tournier"`). | `"dhollander"` |
| `fod_algorithm` | Algorithm for `dwi2fod`. | `"msmt_csd"` |
| `lmax` | Maximum spherical harmonic order(s). Pass a list for multi-tissue (e.g. `[4, 8, 8]`). | `None` (MRtrix default) |

## Outputs

Outputs are Fiber Orientation Distributions (FODs) saved as spherical harmonic coefficients.

- **WM FOD:** `*_model-CSD_wmFOD.nii.gz` (White Matter)
- **GM FOD:** `*_model-CSD_gmFOD.nii.gz` (Gray Matter, if MSMT)
- **CSF FOD:** `*_model-CSD_csfFOD.nii.gz` (CSF, if MSMT)

If using single-shell CSD, the output might simply be:
- **FOD:** `*_model-CSD_FOD.nii.gz`
