# Anatomical Workflow

The **Anatomical Pipeline** (`--pipeline anat`) standardizes T1w (and optionally T2w) structural images for downstream analysis or as references for diffusion coregistration.

## Overview

The pipeline processes T1w and T2w images independently and then coregisters them.

**Main Class**: `qmri_neuropipe.workflows.pipelines.anat.AnatPreprocessingWorkflow`

See [Tool Reference](../tool_reference.md) for the full list of tools and config keys.

## Steps

### 0. QC (Optional)
Runs `MRIQC` on available anatomical inputs when enabled.

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
| `qc.mriqc.enabled` | bool | false | Run MRIQC on inputs |

### 1. Resample & Reorient
Images are resampled to a consistent resolution (configurable) and reoriented to standard orientation (e.g., MNI) using `fslreorient2std` logic.

**Available tools**
*   Resample: `freesurfer` (`mri_convert`)
*   Reorient: `fsl` (`fslreorient2std`)

**Config**
```yaml
anat:
  preprocessing:
    resample:
      enabled: true
      resolution: 1.0
    reorient:
      enabled: true
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `anat.preprocessing.resample.enabled` | bool | false | Enable resampling |
| `anat.preprocessing.resample.resolution` | float | none | Target resolution (mm) |
| `anat.preprocessing.reorient.enabled` | bool | false | Enable reorientation |

### 2. Denoising
Noise reduction is applied to improve segmentation quality.
*   **Method**: `ants` (DenoiseImage) or others.
*   **Config**: `anat.preprocessing.denoising`

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
anat:
  preprocessing:
    denoising:
      enabled: true
      method: ants
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `anat.preprocessing.denoising.enabled` | bool | false | Enable step |
| `anat.preprocessing.denoising.method` | str | `mrtrix` | `mrtrix`, `ants`, `mppca`, `patch2self`, `nlmeans`, `wavelets`, `gaussian` |
| `anat.preprocessing.denoising.patch_radius` | int | 2 | MP-PCA |
| `anat.preprocessing.denoising.block_radius` | int | 5 | MP-PCA |
| `anat.preprocessing.denoising.mask_dilation` | int | 2 | Temporary mask dilation |
| `anat.preprocessing.denoising.pca_method` | str | `eig` | MP-PCA |
| `anat.preprocessing.denoising.model` | str | `ridge` | Patch2Self |

### 3. Gibbs Unringing
Removes Gibbs ringing artifacts (optional).
*   **Method**: `mrtrix` (`mrdegibbs`) or `dipy`.
*   **Config**: `anat.preprocessing.degibbs`

**Available tools**
*   `mrtrix` (mrdegibbs)
*   `dipy` (gibbs_removal)

**Config**
```yaml
anat:
  preprocessing:
    degibbs:
      enabled: true
      method: mrtrix
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `anat.preprocessing.degibbs.enabled` | bool | false | Enable step |
| `anat.preprocessing.degibbs.method` | str | `mrtrix` | `mrtrix`, `dipy` |

### 4. Bias Correction
Corrects for B1 field inhomogeneity (N4 Bias Correction).
*   **Method**: `ants` (N4).
*   **Config**: `anat.preprocessing.bias_correction`

**Available tools**
*   `ants` (N4BiasFieldCorrection)
*   `mrtrix` (dwibiascorrect)

**Config**
```yaml
anat:
  preprocessing:
    bias_correction:
      enabled: true
      method: ants
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `anat.preprocessing.bias_correction.enabled` | bool | false | Enable step |
| `anat.preprocessing.bias_correction.method` | str | `ants` | `ants`, `mrtrix` |

### 5. Sharpening (Optional)
Sharpens the image to enhance edges.
*   **Config**: `anat.preprocessing.sharpen`

**Available tools**
*   `ants` (iMath Sharpen)

**Config**
```yaml
anat:
  preprocessing:
    sharpen:
      enabled: true
      method: ants
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `anat.preprocessing.sharpen.enabled` | bool | false | Enable step |
| `anat.preprocessing.sharpen.method` | str | `ants` | `ants` |

### 6. Coregistration
Aligns T1w and T2w images if both are present.
*   **Direction**: Configurable (T1w -> T2w or T2w -> T1w).
*   **Method**: `fsl` (`flirt`) or `ants`.
*   **Config**: `anat.preprocessing.coregistration`
*   **Detailed Guide**: See [Registration & Coregistration](../registration.md) for advanced options.

**Available tools**
*   `fsl`
*   `ants`
*   `freesurfer`

**Config**
```yaml
anat:
  preprocessing:
    coregistration:
      enabled: true
      method: fsl
      reference_image: t1w  # t1w | t2w
      options:
        dof: 6
        cost: normmi
        interpolation: trilinear
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `anat.preprocessing.coregistration.enabled` | bool | false | Enable step |
| `anat.preprocessing.coregistration.method` | str | `fsl` | `fsl`, `ants`, `freesurfer` |
| `anat.preprocessing.coregistration.reference_image` | str | `t1w` | `t1w`, `t2w` |
| `anat.preprocessing.coregistration.options.dof` | int | 6 | FSL |
| `anat.preprocessing.coregistration.options.cost` | str | `normmi` | FSL |
| `anat.preprocessing.coregistration.options.interpolation` | str | `trilinear` | FSL |

### 7. Brain Masking
Generates a binary brain mask from the reference structural image.
*   **Method**: `ants` (ANTs brain extraction) or others.
*   **Config**: `anat.preprocessing.brain_masking`

**Available tools**
*   `fsl` (bet)
*   `mrtrix` (dwi2mask)
*   `ants` (antsBrainExtraction)
*   `freesurfer` (mri_watershed)
*   `synthstrip` (mri_synthstrip)
*   `hd-bet` (HD-BET)

**Config**
```yaml
anat:
  preprocessing:
    brain_masking:
      enabled: true
      method: ants
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `anat.preprocessing.brain_masking.enabled` | bool | false | Enable step |
| `anat.preprocessing.brain_masking.method` | str | `ants` | `mrtrix`, `fsl`, `ants`, `freesurfer`, `synthstrip`, `hd-bet` |

### 8. Recon-all (Optional)
Runs FreeSurfer `recon-all` surface reconstruction.
*   **Config**: `anat.preprocessing.recon_all.enabled: true`

**Available tools**
*   `freesurfer` (recon-all)

**Config**
```yaml
anat:
  preprocessing:
    recon_all:
      enabled: true
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `anat.preprocessing.recon_all.enabled` | bool | false | Enable step |

### 9. SuperSynth (Optional)

Runs FreeSurfer `mri_super_synth`, a modality-agnostic U-Net that produces brain
region segmentation, MNI atlas registration, and synthetic 1 mm isotropic T1w,
T2w, and FLAIR images from any 3D brain volume — regardless of resolution or
contrast.  It supports in vivo, ex vivo, cerebrum-only, and single-hemisphere
acquisitions.

> **Requires** a FreeSurfer development build newer than October 2025.

**Available tools**
*   `freesurfer` (`mri_super_synth`)

**Config**
```yaml
anat:
  super_synth:
    enabled: true
    mode: invivo         # invivo | exvivo | cerebrum | left-hemi | right-hemi
    sharpen_synths: false
    device: null         # null = tool default (cuda when available)
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `anat.super_synth.enabled` | bool | false | Enable step |
| `anat.super_synth.mode` | str | `invivo` | Input type — `invivo`, `exvivo`, `cerebrum`, `left-hemi`, `right-hemi` |
| `anat.super_synth.sharpen_synths` | bool | false | Sharpen synthetic T1w/T2w/FLAIR predictions |
| `anat.super_synth.device` | str\|null | null | `cpu` or `cuda`; omit to use tool default |

**Outputs** (written to `<output_dir>/super_synth/sub-<id>/[ses-<id>/]`)
| File | Description |
| --- | --- |
| `seg.nii.gz` | Brain region segmentation |
| `T1w.nii.gz` | Synthetic 1 mm isotropic T1w |
| `T2w.nii.gz` | Synthetic 1 mm isotropic T2w |
| `FLAIR.nii.gz` | Synthetic 1 mm isotropic FLAIR |

**Notes**
- The step sets `super_synth_dir` and `super_synth_outputs` in the pipeline
  context.  If no `preprocessed_t1w` is present, the synthetic T1w is injected
  automatically so downstream steps receive a structural reference.
- The tool also performs MNI registration internally and writes Dice scores for
  QC; both are preserved in the output directory.
- See [Tool Reference](../tool_reference.md#supersynth-anatomical) for the full
  parameter list and output-file notes.

### 10. Nonlinear Registration (Optional)
Registers the structural image to a template (e.g., MNI).
*   **Config**: `anat.preprocessing.normalization`

**Available tools**
*   `ants` (SyN)

**Config**
```yaml
anat:
  preprocessing:
    normalization:
      enabled: true
      template: /path/to/template.nii.gz
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `anat.preprocessing.normalization.enabled` | bool | false | Enable step |
| `anat.preprocessing.normalization.template` | path | required | Template image |

## Outputs

*   `*desc-preproc_T1w.nii.gz`: Preprocessed T1w image.
*   `*desc-preproc_T2w.nii.gz`: Preprocessed T2w image (aligned to T1w).
*   `*desc-preproc_mask.nii.gz`: Binary brain mask.
*   `report.html`: Visual quality report (if enabled).

## Configuration Example

```yaml
anat:
  preprocessing:
    denoising:
      enabled: true
    bias_correction:
      method: "ants"
    coregistration:
      reference_image: "t1w" # Align T2w to T1w
```
