# Diffusion Workflow

The **Diffusion Pipeline** (`--pipeline dmri`) processes Diffusion Weighted Imaging (DWI) data, performing corrections for noise, artifacts, and distortion, followed by coregistration.

## Overview

The pipeline handles single-shell and multi-shell data and supports advanced distortion correction strategies using reverse phase-encoding (PE) data.

## Step Method Summary

| Step | Config Key | Methods / Backends |
| --- | --- | --- |
| Reorient | `dmri.preprocessing.reorient` | `mrtrix` (`mrconvert` stride standardization with b-vector rotation) |
| Resample | `dmri.preprocessing.resample` | `freesurfer` (`mri_convert`) |
| Gradient check | `dmri.preprocessing.grad_check` | `mrtrix` (`dwigradcheck`) |
| Distortion correction | `dmri.preprocessing.distcorr` | `topup`, `synb0`, `drbuddi`, `topup+drbuddi`, `none` |
| Denoising | `dmri.preprocessing.denoising` | `mrtrix`, `ants`, `mppca`, `patch2self`, `nlmeans`, `wavelets`, `gaussian` |
| Gibbs unringing | `dmri.preprocessing.degibbs` | `mrtrix`, `dipy` |
| Eddy correction | `dmri.preprocessing.eddy` | `eddy`, `eddy-correct`, `two-pass` |
| Outlier removal | `dmri.preprocessing.outliers` | `manual`, `eddy_qc`, `threshold` |
| Gradient nonlinearity | `dmri.preprocessing.grad_nonlin` | `native_ge`, `tortoise` |
| Bias correction | `dmri.preprocessing.bias_correction` | `ants`, `mrtrix` |
| Coregistration | `dmri.preprocessing.coregistration` | `ants`, `fsl`, `freesurfer` |
| Brain masking | `dmri.preprocessing.brain_masking` | `fsl`, `mrtrix`, `ants`, `freesurfer`, `synthstrip`, `hd-bet` |
| Normalization | `dmri.normalization` | `ants`, `synthmorph`, `robust_iterative` |

**Main Class**: `qmri_neuropipe.workflows.pipelines.dmri.DMRIPipeline`

See [Tool Reference](../tool_reference.md) for the full list of tools and config keys.

## Anatomical Integration
If an `anat` directory exists for the subject, the pipeline can optionally run the full **Anatomical Workflow** first (see [Anatomical Workflow](anatomical.md)) and use the resulting T1w/T2w images as references for coregistration.

## Steps

### 1. QC & Audit
Audits inputs (bvals, bvecs, phase encoding directions). Optionally runs `MRIQC` on raw data.

**Available tools**
*   `mriqc` (optional)

**Config**
```yaml
qc:
  mriqc:
    enabled: true
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `qc.mriqc.enabled` | bool | false | Run MRIQC on raw inputs |

### 2. Distortion Correction
Corrects for susceptibility-induced distortions.
*   **Topup**: Uses reverse-PE (AP/PA) data (FSL `topup`).
*   **Synb0**: Synthesizes a distortion-free b0 from T1w (useful if no reverse-PE data exists).
*   **Native DRBUDDI**: ANTs-based reverse-PE refinement that runs after Eddy on the merged DWI.
*   **Combined Topup + DRBUDDI**: Runs Topup before merge, then Eddy, then the native DRBUDDI refinement.
*   **Method Config**: `dmri.preprocessing.distcorr.method` (`synb0`, `topup`, `drbuddi`, `topup+drbuddi`, `none`).

**Available tools**
*   `topup` (FSL)
*   `synb0` (DIPY Synb0)
*   `drbuddi` (native ANTs-based reverse-PE refinement)
*   `none` (skip)

**Config**
```yaml
dmri:
  preprocessing:
    distcorr:
      method: topup+drbuddi   # topup | synb0 | drbuddi | topup+drbuddi | none
      fallback: true  # allow Synb0 fallback when Topup inputs missing
      config: /path/to/topup.cnf
      drbuddi:
        transform_type: SyNOnly
        interpolator: linear
        symmetric_pairwise: true
        pe_axis_constraint: 1.0
        registration_options: {}
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `dmri.preprocessing.distcorr.method` | str | `none` | `topup`, `synb0`, `drbuddi`, `topup+drbuddi`, `none` |
| `dmri.preprocessing.distcorr.fallback` | bool | false | Allow Synb0 fallback |
| `dmri.preprocessing.distcorr.config` | path | none | Topup config file |
| `dmri.preprocessing.distcorr.drbuddi.transform_type` | str | `SyNOnly` | Native DRBUDDI ANTs transform |
| `dmri.preprocessing.distcorr.drbuddi.interpolator` | str | `linear` | Output resampling interpolator |
| `dmri.preprocessing.distcorr.drbuddi.symmetric_pairwise` | bool | `true` | Use symmetric blip-up/blip-down half-warps |
| `dmri.preprocessing.distcorr.drbuddi.pe_axis_constraint` | float | `1.0` | Constrain warp mostly to the PE axis |
| `dmri.preprocessing.distcorr.drbuddi.registration_options` | dict | `{}` | Extra ANTs registration kwargs |

### Distortion-Correction QC

When report generation is enabled, the native `drbuddi` and `topup+drbuddi` modes now add:

*   mean b0 before and after the post-Eddy refinement
*   residual blip-up vs blip-down mean b0 maps before and after refinement
*   a residual summary table per phase-encoding axis

### 3. Denoising
MP-PCA denoising to improve SNR.
*   **Method**: `mrtrix` (`dwidenoise`) or `dipy`.
*   **Config**: `dmri.preprocessing.denoising`

**Available tools**
*   `mrtrix` (dwidenoise)
*   `ants` (DenoiseImage)
*   `mppca` (DIPY MP-PCA)
*   `patch2self` (DIPY)
*   `nlmeans` (DIPY)
*   `wavelets` (PyWavelets)
*   `gaussian` (SciPy)

**Config**
```yaml
dmri:
  preprocessing:
    denoising:
      enabled: true
      method: mppca
      patch_radius: 2
      block_radius: 5
      mask_dilation: 2
      pca_method: eig
      model: ridge     # for patch2self
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `dmri.preprocessing.denoising.enabled` | bool | false | Enable step |
| `dmri.preprocessing.denoising.method` | str | `mrtrix` | `mrtrix`, `ants`, `mppca`, `patch2self`, `nlmeans`, `wavelets`, `gaussian` |
| `dmri.preprocessing.denoising.patch_radius` | int | 2 | MP-PCA |
| `dmri.preprocessing.denoising.block_radius` | int | 5 | MP-PCA |
| `dmri.preprocessing.denoising.mask_dilation` | int | 2 | Temporary mask dilation |
| `dmri.preprocessing.denoising.pca_method` | str | `eig` | MP-PCA |
| `dmri.preprocessing.denoising.model` | str | `ridge` | Patch2Self |

### 4. Gibbs Unringing
Removes Gibbs ringing artifacts.
*   **Method**: `mrtrix` (`mrdegibbs`).
*   **Config**: `dmri.preprocessing.degibbs`

**Available tools**
*   `mrtrix` (mrdegibbs)
*   `dipy` (gibbs_removal)

**Config**
```yaml
dmri:
  preprocessing:
    degibbs:
      enabled: true
      method: mrtrix
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `dmri.preprocessing.degibbs.enabled` | bool | false | Enable step |
| `dmri.preprocessing.degibbs.method` | str | `mrtrix` | `mrtrix`, `dipy` |

### 5. Eddy Current Correction
Corrects for eddy currents and subject motion.
*   **Method**: `fsl` (`eddy_openmp` / `eddy_cuda`).
*   **QC**: Automatically runs `eddy_quad` for quality metrics.
*   **Config**: `dmri.preprocessing.eddy`

**Available tools**
*   `eddy` (FSL eddy)
*   `eddy-correct` (FSL legacy)
*   `two-pass` (two-pass eddy correction)

**Config**
```yaml
dmri:
  preprocessing:
    eddy:
      enabled: true
      method: eddy
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `dmri.preprocessing.eddy.enabled` | bool | false | Enable step |
| `dmri.preprocessing.eddy.method` | str | `eddy` | `eddy`, `eddy-correct`, `two-pass` |

### 6. Outlier Removal
Detects and removes outlier volumes/slices based on Eddy QC or manual thresholds.
*   **Config**: `dmri.preprocessing.outliers`

**Available tools**
*   `manual` (explicit indices)
*   `eddy_qc` (from eddy outlier map)
*   `threshold` (threshold-based)

**Config**
```yaml
dmri:
  preprocessing:
    outliers:
      enabled: true
      method: manual      # manual | eddy_qc | threshold
      manual_indices: [0, 1, 2]
      threshold: 0.05
      volumes_file: /path/to/eddy_outlier_map
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `dmri.preprocessing.outliers.enabled` | bool | false | Enable step |
| `dmri.preprocessing.outliers.method` | str | `manual` | `manual`, `eddy_qc`, `threshold` |
| `dmri.preprocessing.outliers.manual_indices` | list[int] | none | Manual volume indices |
| `dmri.preprocessing.outliers.threshold` | float | 0.05 | Threshold mode |
| `dmri.preprocessing.outliers.volumes_file` | path | none | Eddy outlier map |

### 7. Bias Correction
Corrects B1 field inhomogeneity in the DWI series (often on the b0 or mean b0).
*   **Method**: `ants` (N4).
*   **Config**: `dmri.preprocessing.bias_correction`

**Available tools**
*   `ants` (N4BiasFieldCorrection)
*   `mrtrix` (dwibiascorrect)

**Config**
```yaml
dmri:
  preprocessing:
    bias_correction:
      enabled: true
      method: ants
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `dmri.preprocessing.bias_correction.enabled` | bool | false | Enable step |
| `dmri.preprocessing.bias_correction.method` | str | `ants` | `ants`, `mrtrix` |

### 8. Coregistration
Registers the DWI series (average b0) to the structural T1w reference.
*   **Method**: `ants` (Rigid/Affine) or `fsl` (`flirt`/`bbregister`).
*   **Config**: `dmri.preprocessing.coregistration`
*   **Detailed Guide**: See [Registration & Coregistration](../registration.md) for advanced options.

**Available tools**
*   `ants`
*   `fsl`
*   `freesurfer`

**Config**
```yaml
dmri:
  preprocessing:
    coregistration:
      enabled: true
      method: ants
      options:
        apply_method: native     # native | mrtrix
        output_resolution: anatomical  # anatomical | dwi | native
        interpolation: linear    # linear | nearest | sinc | cubic
        dof: 6
        cost: normmi
        transform_type: Rigid
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `dmri.preprocessing.coregistration.enabled` | bool | false | Enable step |
| `dmri.preprocessing.coregistration.method` | str | `ants` | `ants`, `fsl`, `freesurfer` |
| `dmri.preprocessing.coregistration.options.apply_method` | str | `native` | `native`, `mrtrix` |
| `dmri.preprocessing.coregistration.options.output_resolution` | str | `anatomical` | `anatomical`, `dwi`, `native` |
| `dmri.preprocessing.coregistration.options.interpolation` | str | `linear` | `linear`, `nearest`, `sinc`, `cubic` |
| `dmri.preprocessing.coregistration.options.dof` | int | 6 | FSL |
| `dmri.preprocessing.coregistration.options.cost` | str | `normmi` | FSL |
| `dmri.preprocessing.coregistration.options.transform_type` | str | `Rigid` | ANTs |

### 9. Gradient Nonlinearity Correction (Optional)
Corrects for scanner gradient nonlinearities using either the native GE implementation or TORTOISE (requires gradient coefficients).

**Available tools**
*   `native_ge` (native GE gradient nonlinearity tensor generation / alignment)
*   `tortoise` (TORTOISE gradient nonlinearity correction)

**Config**
```yaml
dmri:
  preprocessing:
    grad_nonlin:
      enabled: true
      coeff_file: /path/to/coeffs.dat
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `dmri.preprocessing.grad_nonlin.enabled` | bool | false | Enable step |
| `dmri.preprocessing.grad_nonlin.method` | str | `tortoise` | `native_ge`, `tortoise` |
| `dmri.preprocessing.grad_nonlin.coeff_file` | path | required | Gradient coefficients `.dat` |

### 10. Brain Masking
Generates a final brain mask for the DWI data.
*   **Method**: `mrtrix` (`dwi2mask`) or based on T1w mask projection.

**Available tools**
*   `mrtrix` (dwi2mask)
*   `fsl` (bet)
*   `ants` (antsBrainExtraction)
*   `freesurfer` (mri_watershed)
*   `synthstrip` (mri_synthstrip)
*   `hd-bet` (HD-BET)

**Config**
```yaml
dmri:
  preprocessing:
    brain_masking:
      enabled: true
      method: mrtrix
      mask_input: b0      # b0 | average
      apply_mask: true
      use_gpu: false
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `dmri.preprocessing.brain_masking.enabled` | bool | false | Enable step |
| `dmri.preprocessing.brain_masking.method` | str | `mrtrix` | `mrtrix`, `fsl`, `ants`, `freesurfer`, `synthstrip`, `hd-bet` |
| `dmri.preprocessing.brain_masking.mask_input` | str | `b0` | `b0`, `average` |
| `dmri.preprocessing.brain_masking.apply_mask` | bool | true | Apply mask to data |
| `dmri.preprocessing.brain_masking.use_gpu` | bool | false | Some tools support GPU |

## Outputs

*   `*desc-preproc_dwi.nii.gz`: Fully preprocessed DWI 4D series.
*   `*desc-preproc_dwi.bval/.bvec`: Rotated/Corrected gradient table.
*   `*desc-brain_mask.nii.gz`: Binary brain mask.
*   `qc/`: Quality control metrics and reports (Eddy QC).
*   `report.html`: Visual pipeline report.

## Configuration Example

```yaml
dmri:
  preprocessing:
    distcorr:
      method: "topup"
      fallback: true # Use Synb0 if Topup fails/missing data
    denoising:
      method: "mrtrix"
      patch_radius: 2
    eddy:
      enabled: true
      method: "eddy"
```
