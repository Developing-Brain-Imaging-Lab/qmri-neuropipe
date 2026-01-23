# Diffusion Workflow

The **Diffusion Pipeline** (`--pipeline dmri`) processes Diffusion Weighted Imaging (DWI) data, performing corrections for noise, artifacts, and distortion, followed by coregistration.

## Overview

The pipeline handles single-shell and multi-shell data and supports advanced distortion correction strategies using reverse phase-encoding (PE) data.

**Main Class**: `qmri_neuropipe.workflows.pipelines.dmri.DMRIPipeline`

## Anatomical Integration
If an `anat` directory exists for the subject, the pipeline can optionally run the full **Anatomical Workflow** first (see [Anatomical Workflow](anatomical.md)) and use the resulting T1w/T2w images as references for coregistration.

## Steps

### 1. QC & Audit
Audits inputs (bvals, bvecs, phase encoding directions). Optionally runs `MRIQC` on raw data.

### 2. Distortion Correction
Corrects for susceptibility-induced distortions.
*   **Topup**: Uses reverse-PE (AP/PA) data (FSL `topup`).
*   **Synb0**: Synthesizes a distortion-free b0 from T1w (useful if no reverse-PE data exists).
*   **Method Config**: `dmri.preprocessing.distcorr.method` (`synb0`, `topup`, `none`).

### 3. Denoising
MP-PCA denoising to improve SNR.
*   **Method**: `mrtrix` (`dwidenoise`) or `dipy`.
*   **Config**: `dmri.preprocessing.denoising`

### 4. Gibbs Unringing
Removes Gibbs ringing artifacts.
*   **Method**: `mrtrix` (`mrdegibbs`).
*   **Config**: `dmri.preprocessing.degibbs`

### 5. Eddy Current Correction
Corrects for eddy currents and subject motion.
*   **Method**: `fsl` (`eddy_openmp` / `eddy_cuda`).
*   **QC**: Automatically runs `eddy_quad` for quality metrics.
*   **Config**: `dmri.preprocessing.eddy`

### 6. Outlier Removal
Detects and removes outlier volumes/slices based on Eddy QC or manual thresholds.
*   **Config**: `dmri.preprocessing.outliers`

### 7. Bias Correction
Corrects B1 field inhomogeneity in the DWI series (often on the b0 or mean b0).
*   **Method**: `ants` (N4).
*   **Config**: `dmri.preprocessing.bias_correction`

### 8. Coregistration
Registers the DWI series (average b0) to the structural T1w reference.
*   **Method**: `ants` (Rigid/Affine) or `fsl` (`flirt`/`bbregister`).
*   **Config**: `dmri.preprocessing.coregistration`
*   **Detailed Guide**: See [Registration & Coregistration](../registration.md) for advanced options.

### 9. Gradient Nonlinearity Correction (Optional)
Corrects for scanner gradient nonlinearities using `Tortoise` or similar tools (requires specific gradient coefficients).

### 10. Brain Masking
Generates a final brain mask for the DWI data.
*   **Method**: `mrtrix` (`dwi2mask`) or based on T1w mask projection.

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
