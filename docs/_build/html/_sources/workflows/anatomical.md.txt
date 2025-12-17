# Anatomical Workflow

The **Anatomical Pipeline** (`--pipeline anat`) standardizes T1w (and optionally T2w) structural images for downstream analysis or as references for diffusion coregistration.

## Overview

The pipeline processes T1w and T2w images independently and then coregisters them.

**Main Class**: `qmri_neuropipe.workflows.pipelines.anat.AnatPreprocessingWorkflow`

## Steps

### 1. Resample & Reorient
Images are resampled to a consistent resolution (configurable) and reoriented to standard orientation (e.g., MNI) using `fslreorient2std` logic.

### 2. Denoising
Noise reduction is applied to improve segmentation quality.
*   **Method**: `ants` (DenoiseImage) or others.
*   **Config**: `anat.preprocessing.denoising`

### 3. Gibbs Unringing
Removes Gibbs ringing artifacts (optional).
*   **Method**: `mrtrix` (`mrdegibbs`) or `dipy`.
*   **Config**: `anat.preprocessing.degibbs`

### 4. Bias Correction
Corrects for B1 field inhomogeneity (N4 Bias Correction).
*   **Method**: `ants` (N4).
*   **Config**: `anat.preprocessing.bias_correction`

### 5. Sharpening (Optional)
Sharpens the image to enhance edges.
*   **Config**: `anat.preprocessing.sharpen`

### 6. Coregistration
Aligns T1w and T2w images if both are present.
*   **Direction**: Configurable (T1w -> T2w or T2w -> T1w).
*   **Method**: `fsl` (`flirt`) or `ants`.
*   **Config**: `anat.preprocessing.coregistration`

### 7. Brain Masking
Generates a binary brain mask from the reference structural image.
*   **Method**: `ants` (ANTs brain extraction) or others.
*   **Config**: `anat.preprocessing.brain_masking`

### 8. Recon-all (Optional)
Runs FreeSurfer `recon-all` surface reconstruction.
*   **Config**: `anat.preprocessing.recon_all.enabled: true`

### 9. Nonlinear Registration (Optional)
Registers the structural image to a template (e.g., MNI).
*   **Config**: `anat.preprocessing.normalization`

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
