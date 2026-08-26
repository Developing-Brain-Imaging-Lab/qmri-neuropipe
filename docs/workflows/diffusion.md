# Diffusion Workflow

The **Diffusion Pipeline** (`--pipeline dmri`) processes Diffusion Weighted Imaging (DWI) data, performing corrections for noise, artifacts, and distortion, followed by coregistration, model fitting, and atlas-based analysis.

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
| Tractography | `dmri.modeling.tractography` | MRtrix, ACT, SIFT/SIFT2, TractSeg, pyAFQ |
| Tract-specific analysis | `dmri.modeling.tractography.tract_specific` | Bundle extraction, tractometry, track-density maps, connectomes |
| Analysis | `dmri.analysis` | Atlas-based ROI statistics |

**Main Class**: `qmri_neuropipe.workflows.pipelines.dmri.DMRIPipeline`

See {ref}`Diffusion preprocessing <diffusion-preprocessing>`
for every preprocessing field and
{ref}`Diffusion model fitting <diffusion-model-fitting>` for
all modeling, tractography, and tract-analysis controls.

For streamline workflows, see [MRtrix Tractography](tractography.md),
[TractSeg](../models/tractseg.md), [Extracting Specific Tracts](tract_extraction.md),
and [Tractometry and Connectomes](tractometry.md).

## Anatomical Integration
If an `anat` directory exists for the subject, the pipeline can optionally run the full **Anatomical Workflow** first (see [Anatomical Workflow](anatomical.md)) and use the resulting T1w/T2w images as references for coregistration.

## Resume From a Step

With `skip_existing: true`, completed dMRI derivatives are reused by default. Use `rerun_from_step` when you want earlier steps to remain cached but the selected step and every later step to be regenerated.

```yaml
dmri:
  preprocessing:
    rerun_from_step: eddy
```

For modeling-only reruns:

```yaml
dmri:
  modeling:
    rerun_from_step: dti
```

Accepted aliases include `force_from_step`, `start_at_step`, and `resume_from_step`. Common dMRI step names include `merge`, `reorient`, `topup`, `distcorr`, `denoise`, `degibbs`, `gibbs`, `eddy`, `bias`, `coregistration`, `brain_masking`, `modeling`, and model names such as `dti`, `dki`, `noddi`, `sandi`, `mapmri`, `csd`, `fwe_dti`, `tractseg`, and `pyafq`.

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

`synb0` uses DIPY's TensorFlow-backed Synb0 model. Install the `synb0` extra
for local runs, or rebuild the container from a source tree where `.[all]`
includes TensorFlow.

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
      synb0:
        device: cpu  # cpu | gpu
        t1w_source: raw  # raw | supersynth | prefer_supersynth | dwi_supersynth
        supersynth_input: auto  # auto | T1w | T2w | dwi
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
| `dmri.preprocessing.distcorr.fallback` | bool | false | Allow Synb0 fallback when reverse-PE data is missing |
| `dmri.preprocessing.distcorr.synb0.device` | str | `cpu` | `cpu` hides GPUs from TensorFlow; use `gpu` only when the container TensorFlow/CUDA stack is compatible with the host driver |
| `dmri.preprocessing.distcorr.synb0.gpu_ids` | list\|int\|str | global `gpu_ids` | GPU IDs exposed to TensorFlow when `device: gpu` |
| `dmri.preprocessing.distcorr.synb0.t1w_source` | str | `raw` | `raw`, `supersynth`, `prefer_supersynth`, `dwi_supersynth`; controls whether Synb0 uses the anatomical T1w directly or a SuperSynth-generated T1w |
| `dmri.preprocessing.distcorr.synb0.supersynth_input` | str | `auto` | `auto`, `T1w`, `T2w`, `dwi`; selects which image SuperSynth uses when generating the T1w for Synb0. `dwi` uses a 3D mean b0 extracted from the diffusion image |
| `dmri.preprocessing.distcorr.config` | path | none | Topup config file |
| `dmri.preprocessing.distcorr.drbuddi.transform_type` | str | `SyNOnly` | ANTs transform type (`SyNOnly`, `SyN`, `Rigid`, `Affine`) |
| `dmri.preprocessing.distcorr.drbuddi.interpolator` | str | `linear` | Output resampling interpolator (`linear`, `nearest`, `cubic`) |
| `dmri.preprocessing.distcorr.drbuddi.symmetric_pairwise` | bool | true | Use symmetric blip-up/blip-down half-warps |
| `dmri.preprocessing.distcorr.drbuddi.pe_axis_constraint` | float | 1.0 | Constrain warp to the PE axis (0.0–1.0) |
| `dmri.preprocessing.distcorr.drbuddi.registration_options` | dict | {} | Extra ANTs registration kwargs passed through to the registration call |

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

Three methods are available:

*   `manual` — explicitly specify volume indices to remove via `manual_indices`.
*   `eddy_qc` — parse the Eddy outlier map (produced by `eddy` with `--repol`) to identify and exclude slice/volume outliers automatically. Requires the eddy outlier map file.
*   `threshold` — remove volumes whose mean signal falls below a specified threshold fraction.

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
| `dmri.preprocessing.outliers.manual_indices` | list[int] | none | Volume indices to remove (manual mode) |
| `dmri.preprocessing.outliers.threshold` | float | 0.05 | Signal threshold fraction (threshold mode) |
| `dmri.preprocessing.outliers.volumes_file` | path | none | Eddy outlier map file (eddy_qc mode) |

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
      reference_image: T1w  # T1w | T2w | supersynth
      supersynth_input: auto  # auto | T1w | T2w, used when reference_image is supersynth
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
| `dmri.preprocessing.coregistration.method` | str | `ants` | `ants`, `fsl`, `freesurfer`, `synthmorph` |
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

### 11. Normalization (Optional)
Registers preprocessed DWI-derived maps to a standard template.
*   **Config**: `dmri.normalization`

Three methods are available:

*   `ants` — standard ANTs SyN nonlinear registration.
*   `synthmorph` — learning-based diffeomorphic registration (FreeSurfer SynthMorph); does not require skull-stripping.
*   `robust_iterative` — iterative registration scheme that alternates between updating the template and refining individual subject warps; improves registration quality for heterogeneous cohorts.

**Config**
```yaml
dmri:
  normalization:
    enabled: true
    method: ants                  # ants | synthmorph | robust_iterative
    template: /path/to/FA_template.nii.gz
    registration_target: FA       # metric map used as moving image (FA, MD, etc.)
    include_all_metrics: false    # warp all metric maps after primary registration
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `dmri.normalization.enabled` | bool | false | Enable normalization |
| `dmri.normalization.method` | str | `ants` | `ants`, `synthmorph`, `robust_iterative` |
| `dmri.normalization.template` | path | required | Template image |
| `dmri.normalization.registration_target` | str | `FA` | Which metric map is used as the moving image for registration |
| `dmri.normalization.include_all_metrics` | bool | false | After computing the warp from `registration_target`, apply it to all available metric maps |

### 12. Atlas-Based Analysis

Registers atlas label sets to the preprocessed (or normalized) DWI-derived maps and extracts per-ROI statistics.

For the full atlas configuration format, see [Analysis](../analysis.md).

**Config**
```yaml
dmri:
  analysis:
    enabled: true
    atlases:
      JHU:
        labels: /path/to/JHU-ICBM-labels-1mm.nii.gz
        template: /path/to/JHU-ICBM-FA-1mm.nii.gz
      XTRACT:
        labels: /path/to/xtract_labels.nii.gz
        template: /path/to/MNI152_T1_1mm.nii.gz
    registration_target: FA          # default for diffusion; which metric to register atlases to
    registration_template: /path/to/FA_template.nii.gz  # override per-atlas templates for all atlases
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `dmri.analysis.enabled` | bool | false | Enable atlas analysis |
| `dmri.analysis.atlases` | dict | none | Dictionary of named atlases; each entry requires `labels` and optionally `template` |
| `dmri.analysis.atlases.<name>.labels` | path | required | NIfTI label image |
| `dmri.analysis.atlases.<name>.template` | path | none | Atlas-space template used for registration; falls back to `registration_template` if not set |
| `dmri.analysis.registration_target` | str | `FA` | Metric map used as the fixed image when registering atlases |
| `dmri.analysis.registration_template` | path | none | Global template override applied to all atlases that do not specify their own |

**Outputs**

Atlas outputs are written alongside the model output folders:

```
<output_dir>/dwi/atlases/<AtlasName>/
    sub-XX_ses-YY_atlas-<AtlasName>_desc-<metric>_roiStats.csv

<output_dir>/dwi/statistics/
    sub-XX_ses-YY_desc-roiStats_stats.csv   # combined across all atlases and metrics
```

The combined `desc-roiStats_stats.csv` aggregates every atlas and every metric into a single wide-format CSV for cohort-level analysis.

## Outputs

*   `*desc-preproc_dwi.nii.gz`: Fully preprocessed DWI 4D series.
*   `*desc-preproc_dwi.bval/.bvec`: Rotated/Corrected gradient table.
*   `*desc-brain_mask.nii.gz`: Binary brain mask.
*   `dwi/atlases/`: Per-atlas ROI statistics CSVs.
*   `dwi/statistics/desc-roiStats_stats.csv`: Combined ROI statistics across atlases.
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
    outliers:
      enabled: true
      method: eddy_qc
  normalization:
    enabled: true
    method: robust_iterative
    registration_target: FA
    include_all_metrics: true
    template: /path/to/FA_template.nii.gz
  analysis:
    enabled: true
    atlases:
      JHU:
        labels: /path/to/JHU-ICBM-labels-1mm.nii.gz
        template: /path/to/JHU-ICBM-FA-1mm.nii.gz
    registration_target: FA
```

## Study Tracker

When `tracker.enabled: true`, the diffusion pipeline records step statuses
(Denoising, Eddy_Correction, etc.), model fits, atlas names, DWI motion
(absolute and relative), SNR, and outlier counts to the study tracker. Cohort
CSVs are exported after each subject completes to `tracker_reports/`. See
[Study Tracker](../study_tracker.md).
