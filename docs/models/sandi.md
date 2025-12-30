# SANDI (Soma and Neurite Density Imaging)

SANDI is a multi-compartment model designed to estimate the density of cell bodies (somas) in addition to neurites. This is particularly useful for gray matter microstructural imaging.

## Backend

**AMICO**: Accelerated Microstructure Imaging via Convex Optimization.

> **Note:** Requires `amico` python package to be installed and configured.

## Configuration

```yaml
dmri:
  modeling:
    sandi:
      enabled: true
      method: "amico"
```

## Outputs

- **fsoma (Soma Volume Fraction):** `*_model-SANDI_fsoma.nii.gz`
- **fneurite (Neurite Volume Fraction):** `*_model-SANDI_fneurite.nii.gz`
- **fextra (Extra-cellular Fraction):** `*_model-SANDI_fextra.nii.gz`
