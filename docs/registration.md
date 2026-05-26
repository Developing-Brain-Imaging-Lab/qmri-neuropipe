# Registration & Coregistration

The `qmri-neuropipe` provides flexible registration support across both anatomical and diffusion pipelines. The main coregistration logic is handled by the `CoregistrationStep`.

## Supported Methods

### 1. ANTs (Recommended)
The ANTs backend uses **ANTsPy** for high-performance registration. It is highly configurable and supports complex non-linear transforms.

**Configuration:**
```yaml
coregistration:
  enabled: true
  method: "ants"
  options:
    transform_type: "Rigid"  # Options: Rigid, Affine, SyN, etc.
    interpolation: "linear"   # Options: linear, nearest, cubic (bspline)
    # Advanced: any ANTsPy registration keyword argument
    grad_step: 0.1
    reg_iterations: [100, 50, 25]
```

**Common ANTs Options:**
| Parameter | Description |
| :--- | :--- |
| `transform_type` | Type of transform to use (e.g., `"Rigid"`, `"Affine"`, `"SyN"`, `"QuickSyN"`). |
| `interpolation` | Interpolation method for the result image. |
| `aff_niters` | Number of iterations for affine stage. |
| `syn_niters` | Number of iterations for SyN stage. |
| `mask` | Optional brain mask to guide registration. |

### 2. FSL
FSL coregistration uses `flirt` for linear registration and `bbregister` (if FreeSurfer outputs are present).

**Configuration:**
```yaml
coregistration:
  enabled: true
  method: "fsl"
  options:
    dof: 6                   # Degrees of freedom (6=Rigid, 12=Affine)
    cost: "bbr"             # Cost function (bbr, corratio, mutualinfo)
    searchrx: "-180 180"    # X-axis search range
    interpolation: "sinc"   # Output interpolation
```

## Global Options

The following options apply regardless of the backend:

| Option | Description | Default |
| :--- | :--- | :--- |
| `apply_method` | Strategy for applying the transform to DWI data. `"native"` (direct) or `"mrtrix"` (handles gradient rotation). | `"native"` |
| `output_resolution` | Resolution of the transformed image. `"anatomical"` (match structural reference) or `"dwi"` (original DWI resolution). | `"anatomical"` |
| `reference_image` | The target image for coregistration (`"T1w"`, `"T2w"`, `"supersynth"`, or `"supersynth_multivariate"`). SuperSynth modes generate synthetic contrasts for transform estimation. | `"T1w"` |
| `supersynth_input` | Anatomical contrast used to generate the SuperSynth target when `reference_image: "supersynth"` (`"auto"`, `"T1w"`, `"T2w"`). | `"auto"` |
| `supersynth_registration` | Set to `"multivariate"` with `reference_image: "supersynth"` to estimate an ANTs transform using synthetic T1w and T2w pairs, then apply it to the original moving image. | unset |
| `multivariate_metric` | ANTs metric for additional SuperSynth contrast channels. | `"Mattes"` |
| `multivariate_weight` | ANTs weight for each additional contrast channel. | `0.5` |
| `multivariate_sampling` | ANTs sampling parameter for additional contrast channels. | `32` |
