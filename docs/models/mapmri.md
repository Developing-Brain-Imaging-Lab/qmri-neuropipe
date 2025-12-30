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
```

### Options

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `radial_order` | Maximum radial order of the MAP basis. Higher values = more detail but more noise sensitivity. | `6` |
| `laplacian_regularization` | Enable regularization to enforce smoothness. | `true` |
| `positivity_constraint` | Enforce positive probability density. | `false` |
| `global_constraints` | Use global optimization (slower). | `false` |

## Outputs

- **RTOP (Return to Origin Probability):** `*_model-MAPMRI_RTOP.nii.gz`
- **RTAP (Return to Axis Probability):** `*_model-MAPMRI_RTAP.nii.gz`
- **RTPP (Return to Plane Probability):** `*_model-MAPMRI_RTPP.nii.gz`
- **MSD (Mean Squared Displacement):** `*_model-MAPMRI_MSD.nii.gz`
- **QIV (Q-Space Inverse Variance):** `*_model-MAPMRI_QIV.nii.gz`
