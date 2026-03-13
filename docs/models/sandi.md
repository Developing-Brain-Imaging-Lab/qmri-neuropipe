# SANDI (Soma and Neurite Density Imaging)

SANDI is a multi-compartment model designed to estimate the density of cell bodies (somas) in addition to neurites. This is particularly useful for gray matter microstructural imaging.

## Backends

- **dmipy**: Default native backend. Supports voxel-wise GNL-aware fitting.
- **amico**: Legacy accelerated backend, kept for compatibility.

## Configuration

```yaml
dmri:
  modeling:
    sandi:
      enabled: true
      method: "dmipy"
      parallel_diffusivity: 1.7e-9
      iso_diffusivity: 3.0e-9
```

## Outputs

- **fsoma (Soma Volume Fraction):** `*_model-SANDI_fsoma.nii.gz`
- **fneurite (Neurite Volume Fraction):** `*_model-SANDI_fneurite.nii.gz`
- **fextra (Extra-cellular Fraction):** `*_model-SANDI_fextra.nii.gz`
- **Rsoma (Soma Radius):** `*_model-SANDI_Rsoma.nii.gz`
