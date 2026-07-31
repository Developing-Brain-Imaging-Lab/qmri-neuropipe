# Diffusion Kurtosis Imaging (DKI)

For every accepted field and default, see {ref}`Diffusion model fitting <diffusion-model-fitting>`.

DKI extends DTI by quantifying the non-Gaussianity of water diffusion. This provides metrics that may better reflect tissue complexity and heterogeneity than standard DTI indices.

## Backend

**DIPY** (`dipy.reconst.dki`).

## Configuration

```yaml
dmri:
  modeling:
    dki:
      enabled: true
      method: "dipy"
      parameters:
        metrics: ["mk", "ak", "rk", "fa", "md"]
```

### Options

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `metrics` | List of metrics to compute. | `["mk", "ak", "rk", "fa", "md"]` |

## Outputs

- **MK (Mean Kurtosis):** `*_model-DKI_MK.nii.gz`
- **AK (Axial Kurtosis):** `*_model-DKI_AK.nii.gz`
- **RK (Radial Kurtosis):** `*_model-DKI_RK.nii.gz`
- **FA/MD:** Standard tensor metrics are also derived from the kurtosis tensor.
