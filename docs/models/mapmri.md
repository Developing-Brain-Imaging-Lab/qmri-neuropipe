# MAP-MRI (Mean Apparent Propagator MRI)

For every accepted field and default, see {ref}`Diffusion model fitting <diffusion-model-fitting>`.

MAP-MRI provides a comprehensive description of the diffusion signal using a basis of Hermite functions (in 1D, 2D, or 3D). It can estimate the full Ensemble Average Propagator (EAP) and derive advanced metrics describing restriction and pore geometry.

## Backend

**DIPY** (`dipy.reconst.mapmri`).

Constrained MAP-MRI fitting also requires CVXPY. Install it with the MAP-MRI
extra for local environments:

```bash
pip install -e ".[mapmri]"
```

The Docker and Apptainer definitions include this extra and verify that CVXPY
can be imported while the image is built.

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
        metrics: ["rtop", "ng", "parng", "perng", "peaks"]
        peak_npeaks: 3
        peak_relative_threshold: 0.5
        peak_min_separation_angle: 25
```

### Options

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `radial_order` | Maximum radial order of the MAP basis. Higher values = more detail but more noise sensitivity. | `6` |
| `laplacian_regularization` | Enable regularization to enforce smoothness. | `true` |
| `positivity_constraint` | Enforce positive probability density. | `false` |
| `global_constraints` | Use global optimization (slower). | `false` |
| `cvxpy_solver` | Optimization solver used by cvxpy (e.g., `"SCS"`, `"ECOS"`, `"CVXOPT"`). | `None` |
| `metrics` | List of metrics: `rtop`, `rtap`, `rtpp`, `msd`, `qiv`, `ng`, `ng_par`, `ng_perp`, `peaks`. Aliases `parng` / `ng_parallel` and `perng` / `ng_perpendicular` are also accepted. | `["rtop", "rtap", "rtpp", "qiv", "msd"]` |
| `peak_npeaks` | Number of peak directions to save in the `PEAKS` volume. | `3` |
| `peak_relative_threshold` | Relative ODF threshold used for peak finding. | `0.5` |
| `peak_min_separation_angle` | Minimum angular separation between accepted peaks, in degrees. | `25` |

## Outputs

- **RTOP (Return to Origin Probability):** `*_model-MAPMRI_RTOP.nii.gz`
- **RTAP (Return to Axis Probability):** `*_model-MAPMRI_RTAP.nii.gz`
- **RTPP (Return to Plane Probability):** `*_model-MAPMRI_RTPP.nii.gz`
- **MSD (Mean Squared Displacement):** `*_model-MAPMRI_MSD.nii.gz`
- **QIV (Q-Space Inverse Variance):** `*_model-MAPMRI_QIV.nii.gz`
- **NG (Non-Gaussivity):** `*_model-MAPMRI_NG.nii.gz`
- **NG Parallel:** `*_model-MAPMRI_NG_PAR.nii.gz`
- **NG Perpendicular:** `*_model-MAPMRI_NG_PERP.nii.gz`
- **Peaks:** `*_model-MAPMRI_PEAKS.nii.gz` (4D, default 9 components: `peak1_xyz`, `peak2_xyz`, `peak3_xyz`)

Accepted aliases are normalized to the canonical output names above:
- `parng`, `ng_parallel` -> `NG_PAR`
- `perng`, `ng_perpendicular` -> `NG_PERP`
