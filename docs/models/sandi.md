# SANDI (Soma and Neurite Density Imaging)

For every accepted field and default, see {ref}`Diffusion model fitting <diffusion-model-fitting>`.

SANDI is a multi-compartment model designed to estimate the density of cell bodies (somas) in addition to neurites. This is particularly useful for gray matter microstructural imaging.

## Backends

- **dmipy**: Default native backend. Supports CPU `brute2fine`/`mix` and
  vectorized JAX CPU/GPU fitting. The JAX path supports voxelwise GNL without
  assigning corrected measurements back to nominal shells.
- **amico**: Legacy accelerated backend, kept for compatibility.

## Configuration

```yaml
dmri:
  modeling:
    sandi:
      enabled: true
      method: "dmipy"
      parameters:
        solver: "jax"
        device: "gpu"
        gradient_nonlinearity: true
        soma_diffusivity: 3.0e-9
        solver_kwargs:
          batch_size: 4000
          maxiter: 300
```

## Outputs

- **fsoma (Soma Volume Fraction):** `*_model-SANDI_fsoma.nii.gz`
- **fneurite (Neurite Volume Fraction):** `*_model-SANDI_fneurite.nii.gz`
- **fextra (Extra-cellular Fraction):** `*_model-SANDI_fextra.nii.gz`
- **Rsoma (Soma Radius):** `*_model-SANDI_Rsoma.nii.gz`
