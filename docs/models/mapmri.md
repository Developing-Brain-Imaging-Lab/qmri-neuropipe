# MAP-MRI (Mean Apparent Propagator MRI)

MAP-MRI provides a comprehensive description of the diffusion signal using a basis of Hermite functions (in 1D, 2D, or 3D). It can estimate the full Ensemble Average Propagator (EAP) and derive advanced metrics describing restriction and pore geometry.

## Backend

**DIPY** (`dipy.reconst.mapmri`).

## Configuration

```yaml
dmri:
  modeling:
    mapmri:
      enabled: true
      method: "dipy"
      parameters:
        radial_order: 6
        laplacian_regularization: true
        laplacian_weighting: 0.2
        positivity_constraint: true
        cvxpy_solver: "SCS" # Options: SCS (default), ECOS, CVXOPT
        metrics: ["rtop", "ng", "msd"]
```

### Options

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `radial_order` | Maximum radial order of the MAP basis. Higher values = more detail but more noise sensitivity. | `6` |
| `laplacian_regularization` | Enable regularization to enforce smoothness. | `true` |
| `positivity_constraint` | Enforce positive probability density. | `false` |
| `global_constraints` | Use global optimization (slower). | `false` |
| `cvxpy_solver` | Optimization solver used by cvxpy (e.g., `"SCS"`, `"ECOS"`, `"CVXOPT"`). | `None` |
| `metrics` | List of metrics: `rtop`, `rtap`, `rtpp`, `msd`, `qiv`, `ng`, `ng_par`, `ng_perp`. | `["rtop", "rtap", "rtpp", "qiv", "msd"]` |

## Outputs

- **RTOP (Return to Origin Probability):** `*_model-MAPMRI_RTOP.nii.gz`
- **RTAP (Return to Axis Probability):** `*_model-MAPMRI_RTAP.nii.gz`
- **RTPP (Return to Plane Probability):** `*_model-MAPMRI_RTPP.nii.gz`
- **MSD (Mean Squared Displacement):** `*_model-MAPMRI_MSD.nii.gz`
- **QIV (Q-Space Inverse Variance):** `*_model-MAPMRI_QIV.nii.gz`
- **NG (Non-Gaussivity):** `*_model-MAPMRI_NG.nii.gz`
- **NG Parallel:** `*_model-MAPMRI_NG_PAR.nii.gz`
- **NG Perpendicular:** `*_model-MAPMRI_NG_PERP.nii.gz`
