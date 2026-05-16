# Anatomical Workflow

The **Anatomical Pipeline** (`--pipeline anat`) standardizes T1w (and optionally T2w) structural images for downstream analysis or as references for diffusion coregistration.

## Overview

The pipeline processes T1w and T2w images independently and then coregisters them.

## Step Method Summary

| Step | Config Key | Methods / Backends |
| --- | --- | --- |
| Resample | `anat.preprocessing.resample` | `freesurfer` (`mri_convert`) |
| Reorient | `anat.preprocessing.reorient` | `fsl` (`fslreorient2std`) |
| Denoising | `anat.preprocessing.denoising` | `mrtrix`, `ants`, `mppca`, `patch2self`, `nlmeans`, `wavelets`, `gaussian` |
| Gibbs unringing | `anat.preprocessing.degibbs` | `mrtrix`, `dipy` |
| Bias correction | `anat.preprocessing.bias_correction` | `ants`, `mrtrix` |
| Sharpening | `anat.preprocessing.sharpen` | `ants` |
| Coregistration | `anat.preprocessing.coregistration` | `ants`, `fsl`, `freesurfer` |
| Brain masking | `anat.preprocessing.brain_masking` | `fsl`, `mrtrix`, `ants`, `freesurfer`, `synthstrip`, `hd-bet` |
| Recon-all | `anat.preprocessing.recon_all` | `standard`, `clinical` |
| Normalization | `anat.preprocessing.normalization` | `ants` |
| SuperSynth | `anat.super_synth` | `freesurfer` (`mri_super_synth`) |

## Input Selection

If multiple anatomical acquisitions exist in the BIDS `anat/` folder, the pipeline
can select a specific `T1w` or `T2w` using `anat.input.t1w_match` and
`anat.input.t2w_match`.

This is useful for datasets with multiple `acq-*`, `run-*`, `rec-*`, or similar
entity variants.

**Config**
```yaml
anat:
  input:
    t1w_match:
      entities:
        acq: memprage
        run: "2"
    t2w_match:
      entities:
        acq: space
```

**Alternative**
```yaml
anat:
  input:
    t1w_match:
      json_fields:
        ProtocolName: MPRAGE_0p8mm
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `anat.input.t1w_match.entities` | dict | none | Match BIDS entities such as `acq`, `run`, `rec`, `ce`, `dir` |
| `anat.input.t1w_match.json_fields` | dict | none | Match JSON sidecar fields |
| `anat.input.t1w_match.bids_name` | str | none | Match full BIDS file stem |
| `anat.input.t2w_match.entities` | dict | none | Same behavior for T2w |
| `anat.input.t2w_match.json_fields` | dict | none | Same behavior for T2w |
| `anat.input.t2w_match.bids_name` | str | none | Same behavior for T2w |

Matching is strict:
*   exactly one match is required when a selector is provided
*   zero matches raise an error
*   multiple matches raise an error

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
| `anat.preprocessing.coregistration.options.cost` | str | `normmi` | FSL cost function; use `bbr` for boundary-based registration |
| `anat.preprocessing.coregistration.options.interpolation` | str | `trilinear` | FSL |
| `anat.preprocessing.coregistration.options.wm_seg_method` | str | `fast` | WM mask method for BBR (see below) |

#### White Matter Segmentation for BBR Coregistration

When `cost: bbr` is set, the pipeline needs a white matter mask to drive
boundary-based registration. The `wm_seg_method` option controls how that mask
is generated:

| Method | Backend | Notes |
| --- | --- | --- |
| `fast` | FSL FAST | Default; fast but less accurate |
| `synthseg` | FreeSurfer `mri_synthseg` | Better accuracy; requires FreeSurfer |
| `supersynth` | FreeSurfer `mri_super_synth` | Best accuracy; requires FreeSurfer dev build newer than Oct 2025; **automatically falls back to `synthseg`** if `mri_super_synth` is not available |

**Example — BBR with SuperSynth WM segmentation:**
```yaml
anat:
  preprocessing:
    coregistration:
      enabled: true
      method: fsl
      options:
        cost: bbr
        wm_seg_method: supersynth   # falls back to synthseg if unavailable
```

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
Runs FreeSurfer surface reconstruction on the preprocessed T1w image.
*   **Config**: `anat.preprocessing.recon_all.enabled: true`

Two methods are supported:

*   `standard` — runs the standard `recon-all` pipeline (full cortical surface reconstruction).
*   `clinical` — runs `recon-all-clinical.sh`, a faster streamlined variant designed for clinical-quality (lower-resolution or non-isotropic) data that may not be suitable for the full pipeline.

**Available tools**
*   `freesurfer` (`recon-all` / `recon-all-clinical.sh`)

**Config**
```yaml
anat:
  preprocessing:
    recon_all:
      enabled: true
      method: standard        # standard | clinical
      subjects_dir: /path/to/freesurfer_subjects
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `anat.preprocessing.recon_all.enabled` | bool | false | Enable step |
| `anat.preprocessing.recon_all.method` | str | `standard` | `standard` (recon-all) or `clinical` (recon-all-clinical.sh) |
| `anat.preprocessing.recon_all.subjects_dir` | path | `<bids_dir>/derivatives/freesurfer` | Optional external FreeSurfer `SUBJECTS_DIR` to reuse or write recon-all outputs |

### 9. SuperSynth (Optional)

Runs FreeSurfer `mri_super_synth`, a modality-agnostic U-Net that produces brain
region segmentation, MNI atlas registration, and synthetic 1 mm isotropic T1w,
T2w, and FLAIR images from any 3D brain volume — regardless of resolution or
contrast. It supports in vivo, ex vivo, cerebrum-only, and single-hemisphere
acquisitions.

> **Requires** a FreeSurfer development build newer than October 2025.

**Available tools**
*   `freesurfer` (`mri_super_synth`)

**Config**
```yaml
anat:
  super_synth:
    enabled: true
    mode: invivo             # invivo | exvivo | cerebrum | left-hemi | right-hemi
    sharpen_synths: false
    device: null             # null = auto (cuda when available), cpu, or cuda
    compute_volumes: false
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `anat.super_synth.enabled` | bool | false | Enable step |
| `anat.super_synth.mode` | str | `invivo` | Input type — `invivo`, `exvivo`, `cerebrum`, `left-hemi`, `right-hemi` |
| `anat.super_synth.sharpen_synths` | bool | false | Sharpen synthetic T1w/T2w/FLAIR predictions |
| `anat.super_synth.device` | str\|null | null | `cpu` or `cuda`; null = tool default (cuda when available) |
| `anat.super_synth.compute_volumes` | bool | false | Extract per-region anatomical volumes from `seg.nii.gz` using the FreeSurfer LUT |

#### Volume Extraction (`compute_volumes`)

When `compute_volumes: true`, the pipeline reads the SuperSynth `seg.nii.gz`
segmentation using the FreeSurfer Look-Up Table (LUT) and computes the volume
(mm³) of every labeled region. Results are saved as:

```
<output_dir>/super_synth/sub-<id>/[ses-<id>/]
    sub-XX_ses-YY_desc-supersynth_volumes.csv
```

The CSV contains one row per region with columns `label`, `name`, and
`volume_mm3`. When the study tracker is enabled (`tracker.enabled: true`),
volumes are also logged to the `Volume_Statistics` sheet of the tracker
workbook.

**Outputs** (written to `<output_dir>/super_synth/sub-<id>/[ses-<id>/]`)
| File | Description |
| --- | --- |
| `seg.nii.gz` | Brain region segmentation (FreeSurfer LUT labels) |
| `T1w.nii.gz` | Synthetic 1 mm isotropic T1w |
| `T2w.nii.gz` | Synthetic 1 mm isotropic T2w |
| `FLAIR.nii.gz` | Synthetic 1 mm isotropic FLAIR |
| `sub-XX_ses-YY_desc-supersynth_volumes.csv` | Per-region volumes (when `compute_volumes: true`) |

**Notes**
- The step sets `super_synth_dir` and `super_synth_outputs` in the pipeline
  context. If no `preprocessed_t1w` is present, the synthetic T1w is injected
  automatically so downstream steps receive a structural reference.
- The tool also performs MNI registration internally and writes Dice scores for
  QC; both are preserved in the output directory.
- See [Tool Reference](../tool_reference.md#supersynth-anatomical) for the full
  parameter list and output-file notes.

### 10. Nonlinear Normalization (Optional)
Registers the structural image to a template (e.g., MNI) using ANTs SyN.
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
      space_entity: MNI152NLin2009cAsym   # BIDS space entity for output filenames
      space_name: MNI                     # human-readable label used in reports
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `anat.preprocessing.normalization.enabled` | bool | false | Enable step |
| `anat.preprocessing.normalization.template` | path | required | Template image to register to |
| `anat.preprocessing.normalization.space_entity` | str | none | BIDS `space-` entity written into output filenames (e.g., `MNI152NLin2009cAsym`) |
| `anat.preprocessing.normalization.space_name` | str | none | Human-readable space label used in pipeline reports and the study tracker |

## Outputs

*   `*desc-preproc_T1w.nii.gz`: Preprocessed T1w image.
*   `*desc-preproc_T2w.nii.gz`: Preprocessed T2w image (aligned to T1w).
*   `*desc-preproc_mask.nii.gz`: Binary brain mask.
*   `super_synth/sub-<id>/[ses-<id>/]`: SuperSynth outputs including segmentation, synthetics, and (optionally) volumes CSV.
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
      reference_image: "t1w"   # Align T2w to T1w
      options:
        cost: bbr
        wm_seg_method: supersynth
    recon_all:
      enabled: true
      method: clinical
    normalization:
      enabled: true
      template: /path/to/MNI152_T1_1mm.nii.gz
      space_entity: MNI152NLin2009cAsym
      space_name: MNI
  super_synth:
    enabled: true
    mode: invivo
    compute_volumes: true
```

## Study Tracker

When `tracker.enabled: true`, the anatomical pipeline records step statuses,
segmentation method, and (if SuperSynth `compute_volumes` is enabled)
per-region volumes to the study tracker. See [Study Tracker](../study_tracker.md).
