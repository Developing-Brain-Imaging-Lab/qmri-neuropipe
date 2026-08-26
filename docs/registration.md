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

### 3. SynthMorph

SynthMorph uses FreeSurfer's `mri_synthmorph` executable. Generic dMRI
coregistration supports the linear `rigid` and `affine` models. With a
SuperSynth reference, the transform is estimated from the matched synthetic
contrasts, converted for the original image geometries, and applied to the full
4D DWI with b-vector rotation.

```yaml
coregistration:
  enabled: true
  method: synthmorph
  reference_image: supersynth
  options:
    synthmorph_model: rigid
    output_resolution: anatomical
    interpolation: linear
    # synthmorph_register_args: ""
```

Deformable SynthMorph models are intentionally rejected for dMRI
coregistration because one spatially varying transform cannot be represented by
a single global b-vector rotation.

## Global Options

The following options apply regardless of the backend:

| Option | Description | Default |
| :--- | :--- | :--- |
| `apply_method` | Strategy for applying the transform to DWI data. `"native"` (direct) or `"mrtrix"` (handles gradient rotation). | `"native"` |
| `output_resolution` | Resolution of the transformed image. `"anatomical"` (match structural reference) or `"dwi"` (original DWI resolution). | `"anatomical"` |
| `reference_image` | The target image for coregistration (`"T1w"`, `"T2w"`, `"supersynth"`, or `"supersynth_multivariate"`). For dMRI, SuperSynth modes synthesize matching contrasts from a mean b0 and the anatomical input, estimate their transform with the configured backend, then apply it to the original 4D DWI. | `"T1w"` |
| `supersynth_input` | Anatomical source used for dMRI SuperSynth registration (`"auto"`, `"T1w"`, `"T2w"`). The DWI source is an automatically extracted mean b0. | `"auto"` |
| `supersynth_registration` | Set to `"multivariate"` with `reference_image: "supersynth"` to use synthetic T1w and T2w pairs. `reference_image: "supersynth_multivariate"` is equivalent. | unset |
| `supersynth_b0_threshold` | Maximum b-value included in the mean b0 supplied to SuperSynth. | `50` |
| `multivariate_metric` | ANTs metric for additional SuperSynth contrast channels. | `"Mattes"` |
| `multivariate_weight` | ANTs weight for each additional contrast channel. | `0.5` |
| `multivariate_sampling` | ANTs sampling parameter for additional contrast channels. | `32` |

Single-contrast SuperSynth registration supports the ANTs, FSL, FreeSurfer, and
SynthMorph coregistration backends. FreeSurfer uses `mri_coreg` for the
arbitrary synthetic image pair. SynthMorph uses a linear rigid or affine model.
Multivariate synthetic T1w+T2w estimation uses ANTs; other backends log a
warning and use the synthetic T1w pair.
