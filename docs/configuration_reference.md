# Configuration Reference

This page summarizes the pipeline steps, supported methods, and configuration keys available in YAML or JSON configuration files.

The examples use YAML. The same structure can be represented directly as JSON by replacing indentation with nested objects and using `true`/`false` booleans.

## General Pattern

Most processing steps follow this structure:

```yaml
<workflow>:
  preprocessing:
    <step_name>:
      enabled: true
      method: <method_name>
      options:
        key: value
```

Many steps also accept step-specific keys directly under the step block. When both direct keys and `options` or `parameters` are supported, nested values are flattened by the workflow builder before the step is created.

## Resume and Step-Level Reruns

When `skip_existing: true`, the pipeline normally reuses existing derivative files and skips completed steps. To rerun from a specific step while still allowing earlier steps to use cached outputs, set `rerun_from_step` at the relevant workflow scope. The selected step and every subsequent step in that workflow are forced to run even if their intermediate outputs already exist.

Accepted key aliases are `rerun_from_step`, `force_from_step`, `start_at_step`, and `resume_from_step`.

```yaml
dmri:
  preprocessing:
    rerun_from_step: eddy

dmri:
  modeling:
    rerun_from_step: dti

anat:
  preprocessing:
    rerun_from_step: brain_masking

relaxometry:
  preprocessing:
    rerun_from_step: motion_correction
```

Common step aliases include `merge`, `reorient`, `denoise`, `gibbs`/`degibbs`, `topup`/`distcorr`, `eddy`, `bias`, `coregistration`, `brain_masking`, `normalization`, `freesurfer`/`recon_all`, `supersynth`, `segmentation`, `motion_correction`, `b1`, `modeling`, `atlas`, `stats`, and `acqparams`.

## Command and Provenance Reporting

Every command executed through `qmri_neuropipe.core.run.run_cmd()` is recorded with:

- timestamp
- command label
- exact command string
- working directory, when provided
- explicit environment overrides, when provided
- return code

Command records are written to the normal log at DEBUG level, attached to per-step provenance records in `provenance.json`, and included in the HTML/PDF report command appendix. dMRI steps reported through the execution engine also show their step-specific commands in the step section.

## Anatomical Workflow

Root key: `anat`

```yaml
anat:
  preprocessing:
    rerun_from_step: null
    reorient:
      enabled: true
    denoising:
      enabled: true
      method: ants
    degibbs:
      enabled: true
      method: mrtrix
    bias_correction:
      enabled: true
      method: ants
    brain_masking:
      enabled: true
      method: synthstrip
    coregistration:
      enabled: true
      method: ants
    normalization:
      enabled: true
      method: ants
      template: /path/to/template.nii.gz
    recon_all:
      enabled: false
  super_synth:
    enabled: false
```

| Step | Key | Methods | Notes |
| --- | --- | --- | --- |
| Resample | `anat.preprocessing.resample` | FreeSurfer `mri_convert` | Configure with `enabled: true` and `resolution` |
| Reorient | `anat.preprocessing.reorient` | FSL `fslreorient2std` | No `method` switch currently |
| Denoising | `anat.preprocessing.denoising` | `mrtrix`, `ants`, `mppca`, `patch2self`, `nlmeans`, `wavelets`, `gaussian` | Anatomical default is commonly `ants` |
| Gibbs unringing | `anat.preprocessing.degibbs` or `anat.preprocessing.gibbs` | `mrtrix`, `dipy` | `degibbs` takes precedence over `gibbs` |
| Bias correction | `anat.preprocessing.bias_correction` | `ants`, `mrtrix` | `mrtrix` is primarily DWI-oriented and requires gradients for DWI inputs |
| Sharpening | `anat.preprocessing.sharpen` | `ants` | Optional ANTs sharpening |
| T1w/T2w coregistration | `anat.preprocessing.coregistration` | `ants`, `fsl`, `freesurfer` | Can use SuperSynth references |
| Brain masking | `anat.preprocessing.brain_masking` | `fsl`, `mrtrix`, `ants`, `freesurfer`, `synthstrip`, `hd-bet` | `use_gpu` is available for GPU-capable tools |
| FreeSurfer reconstruction | `anat.preprocessing.recon_all` | `standard`, `clinical` | `clinical` uses `recon-all-clinical` when configured |
| Normalization | `anat.preprocessing.normalization` | `ants`, `fsl` | `ants` is the primary nonlinear mode; `fsl` is affine-oriented |
| Segmentation/statistics | `anat.segmentation` | atlas-driven extraction | Configure `atlas_file`, `atlas_labels`, `metrics`, `atlas_threshold` |
| SuperSynth | `anat.super_synth` | FreeSurfer `mri_super_synth` | Produces segmentation and synthetic T1w/T2w/FLAIR when available |

### Anatomical Coregistration Options

```yaml
anat:
  preprocessing:
    coregistration:
      enabled: true
      method: ants
      reference_image: supersynth_multivariate
      supersynth_input: auto
      supersynth_mode: invivo
      supersynth_device: cpu
      supersynth_registration: multivariate
      transform_type: Rigid
      interpolation: linear
```

| Key | Values | Notes |
| --- | --- | --- |
| `method` | `ants`, `fsl`, `freesurfer` | Transform estimation backend |
| `reference_image` | `T1w`, `T2w`, `supersynth`, `supersynth_multivariate` | SuperSynth modes synthesize registration contrasts |
| `supersynth_input` | `auto`, `T1w`, `T2w` | Source contrast for synthetic target |
| `supersynth_registration` | `single`, `multivariate` | Multivariate uses synthetic T1w and T2w channels |
| `transform_type` | `Rigid`, `Affine`, `SyN`, `SyNOnly` | Used by ANTs paths |
| `cost` | FSL cost names such as `normmi`, `mutualinfo`, `corratio` | Used by FSL paths |
| `dof` | integer | Used by FSL paths |

For dMRI coregistration, `reference_image: supersynth` extracts a mean b0,
runs SuperSynth separately on that image and the selected anatomical image,
registers the synthetic T1w pair with the configured ANTs, FSL, or FreeSurfer
backend, and applies the transform to the original 4D DWI. FreeSurfer uses
`mri_coreg` for this arbitrary-volume registration. Use
`reference_image: supersynth_multivariate` or
`supersynth_registration: multivariate` to add the synthetic T2w pair when
using ANTs. Other backends fall back to the synthetic T1w pair.

### SuperSynth Options

```yaml
anat:
  super_synth:
    enabled: true
    mode: invivo
    sharpen_synths: false
    device: cpu
    compute_volumes: true
```

| Key | Values | Notes |
| --- | --- | --- |
| `mode` | `invivo`, `exvivo`, `cerebrum`, `left-hemi`, `right-hemi` | Passed to `mri_super_synth` |
| `device` | `cpu`, `cuda`, null | Omit or null to use FreeSurfer default |
| `sharpen_synths` | bool | Sharpen synthetic contrasts |
| `compute_volumes` | bool | Compute per-label volumes from SuperSynth segmentation |

## Diffusion Preprocessing

Root key: `dmri.preprocessing`

```yaml
dmri:
  preprocessing:
    rerun_from_step: null
    reorient:
      enabled: true
    distcorr:
      method: synb0
      fallback: false
      synb0:
        device: cpu
        t1w_source: dwi_supersynth
    grad_check:
      enabled: true
    denoising:
      enabled: true
      method: mrtrix
    degibbs:
      enabled: true
      method: mrtrix
    motion_correction:
      method: eddy
    eddy:
      method: eddy
    bias_correction:
      enabled: true
      method: mrtrix
    coregistration:
      enabled: true
      method: fsl
    brain_masking:
      enabled: true
      method: synthstrip
```

| Step | Key | Methods | Notes |
| --- | --- | --- | --- |
| Merge | `dmri.preprocessing.merging` | internal merge | Used when multiple DWI series need shared downstream correction |
| Reorient | `dmri.preprocessing.reorient` | MRtrix `mrconvert` | Standardizes stride/orientation and gradient sidecars |
| Resample | `dmri.preprocessing.resample` | FreeSurfer `mri_convert` | Configure `resolution` |
| Distortion correction | `dmri.preprocessing.distcorr.method` | `none`, `topup`, `synb0`, `drbuddi`, `topup+drbuddi` | `topup+drbuddi` runs native TOPUP then native DRBUDDI refinement |
| Gradient nonlinearity | `dmri.preprocessing.grad_nonlin` | `tortoise`, `native_ge`, `native`, `latest_native` | Aliases normalize to TORTOISE or native GE paths |
| Gradient check | `dmri.preprocessing.grad_check` | MRtrix `dwigradcheck` | Exports corrected FSL gradient tables |
| Denoising | `dmri.preprocessing.denoising` | `mrtrix`, `ants`, `mppca`, `patch2self`, `nlmeans`, `wavelets`, `gaussian` | MP-PCA uses `patch_radius`, `block_radius`, `pca_method` |
| Gibbs unringing | `dmri.preprocessing.degibbs` | `mrtrix`, `dipy` | Removes Gibbs ringing |
| Motion/eddy correction | `dmri.preprocessing.motion_correction.method` | `none`, `eddy`, `niifreeze` | `eddy` delegates to `dmri.preprocessing.eddy.method` |
| Eddy backend | `dmri.preprocessing.eddy.method` | `eddy`, `eddy-correct`, `two-pass` | `eddy` also enables EddyQuad QC |
| Native DRBUDDI refinement | `dmri.preprocessing.distcorr.drbuddi` | ANTs registration | Requires eddy-corrected data for current workflow |
| Outlier removal | `dmri.preprocessing.outliers.method` | `manual`, `threshold`, `eddy_qc`, `deep_learning` | `deep_learning` is a placeholder/experimental method path |
| Bias correction | `dmri.preprocessing.bias_correction` | `ants`, `mrtrix` | `mrtrix` uses `dwibiascorrect` |
| Coregistration | `dmri.preprocessing.coregistration` | `ants`, `fsl`, `freesurfer` | Can target acquired or SuperSynth anatomical references |
| Brain masking | `dmri.preprocessing.brain_masking` | `fsl`, `mrtrix`, `ants`, `freesurfer`, `synthstrip`, `hd-bet` | `mask_input` can use `b0` or `average` |
| Final GNL tensor alignment | `dmri.preprocessing.grad_nonlin.enabled` | internal | Aligns GNL tensor to final DWI space when GNL correction is enabled |

### Synb0 Options

```yaml
dmri:
  preprocessing:
    distcorr:
      method: synb0
      synb0:
        device: cpu
        gpu_ids: 0
        t1w_source: dwi_supersynth
        supersynth_input: dwi
        supersynth_mode: invivo
        supersynth_device: cpu
        supersynth_sharpen_synths: false
```

| Key | Values | Notes |
| --- | --- | --- |
| `device` | `cpu`, `gpu` | Controls TensorFlow execution for DIPY Synb0. `cpu` hides GPUs with `CUDA_VISIBLE_DEVICES=-1` |
| `gpu_ids` | int, list, string | Exposed to TensorFlow when `device: gpu` |
| `t1w_source` | `raw`, `supersynth`, `prefer_supersynth`, `dwi_supersynth`, `supersynth_dwi`, `diffusion_supersynth`, `b0_supersynth` | Selects acquired or synthetic T1w source |
| `supersynth_input` | `auto`, `T1w`, `T2w`, `dwi`, `b0`, `diffusion`, `mean_b0` | `dwi`/aliases synthesize T1w from a 3D mean b0 extracted from the DWI |
| `supersynth_mode` | SuperSynth modes | Passed to `mri_super_synth` |
| `supersynth_device` | `cpu`, `cuda`, null | Device for SuperSynth, separate from Synb0 TensorFlow device |
| `supersynth_sharpen_synths` | bool | Sharpen SuperSynth outputs |

### Coregistration Options

```yaml
dmri:
  preprocessing:
    coregistration:
      enabled: true
      method: ants
      reference_image: supersynth
      supersynth_input: auto
      transform_type: Rigid
      interpolation: linear
      apply_method: native
      output_resolution: anatomical
```

| Key | Values | Notes |
| --- | --- | --- |
| `method` | `ants`, `fsl`, `freesurfer` | Registration backend |
| `reference_image` | `T1w`, `T2w`, `supersynth`, `synthT1w`, `synthetic_t1w` | Target reference |
| `apply_method` | `native`, `mrtrix` | `mrtrix` applies transforms with gradient handling when supported |
| `output_resolution` | `anatomical`, `dwi`, `native` | Output grid |
| `transform_type` | `Rigid`, `Affine`, `SyN`, `SyNOnly` | ANTs transform |
| `interpolation` | `linear`, `nearestNeighbor`, `bSpline`, `genericLabel` | ANTs interpolation |
| `cost` | FSL cost names | FSL registration cost |
| `dof` | integer | FSL degrees of freedom |

## Diffusion Modeling

Root key: `dmri.modeling`

```yaml
dmri:
  modeling:
    rerun_from_step: null
    dti:
      enabled: true
      method: dipy
      fit_method: WLLS
      metrics: [fa, md, ad, rd]
    dki:
      enabled: false
      method: dipy
    csd:
      enabled: false
      method: msmt_csd
    noddi:
      enabled: false
      method: dmipy
    tractography:
      mrtrix:
        enabled: false
        algorithm: iFOD2
        select: 10000000
        act:
          enabled: false
          algorithm: fsl
          validate: true
          seed_gmwmi: true
        filtering:
          method: none
      tractseg:
        enabled: false
      pyafq:
        enabled: false
      tract_specific:
        enabled: false
        bundles: []
        metrics: []
```

| Model | Key | Methods | Common options |
| --- | --- | --- | --- |
| DTI | `dmri.modeling.dti` or `dmri.modeling.tensor` | `dipy`, `fsl`, `mrtrix` | `fit_method`, `metrics` |
| DKI | `dmri.modeling.dki` | `dipy` | `fit_method`, `metrics` |
| CSD/FOD | `dmri.modeling.csd` | `msmt_csd`, `csd` | `lmax`, response options |
| NODDI | `dmri.modeling.noddi` | `dmipy`, `amico` | Model-specific parameters |
| SANDI | `dmri.modeling.sandi` | `dmipy`, `amico` | Model-specific parameters |
| Microglia | `dmri.modeling.microglia` | `dmipy` | Model-specific parameters |
| NEXI | `dmri.modeling.nexi` | `nexi` | Model-specific parameters |
| MAPMRI | `dmri.modeling.mapmri` | `dipy` | MAPMRI model options |
| Free-water DTI | `dmri.modeling.fwe_dti` or `dmri.modeling.fwdti` | `dipy` | `fit_method` |
| MRtrix tractography | `dmri.modeling.tractography.mrtrix` | `iFOD2`, `SD_STREAM`, `Tensor_Det`, `Tensor_Prob` | `select`, `cutoff`, `minlength`, `maxlength`, `act`, `filtering` |
| TractSeg | `dmri.modeling.tractography.tractseg` | `tractseg` | `options` passed to TractSeg wrapper |
| pyAFQ | `dmri.modeling.tractography.pyafq` | `pyafq` | `options` passed to pyAFQ wrapper |
| Tract-specific analysis | `dmri.modeling.tractography.tract_specific` | MRtrix | `bundles`, `metrics`, `track_density`, `connectome` |

### MRtrix tractography options

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `mrtrix.enabled` | bool | false | Generate a whole-brain tractogram |
| `mrtrix.algorithm` | str | `iFOD2` | FOD or tensor tracking algorithm |
| `mrtrix.select` | int | 10000000 | Requested output streamline count |
| `mrtrix.cutoff` | float | MRtrix default | FOD amplitude cutoff |
| `mrtrix.minlength` / `maxlength` | float | MRtrix default | Streamline length constraints in mm |
| `mrtrix.act.enabled` | bool | false | Enable anatomically constrained tractography |
| `mrtrix.act.five_tt` | path | auto | Existing 5TT image; otherwise generate from T1w |
| `mrtrix.act.algorithm` | str | `fsl` | `5ttgen` backend |
| `mrtrix.act.validate` | bool | true | Run `5ttcheck` |
| `mrtrix.act.seed_gmwmi` | bool | true | Seed from the GMWMI image |
| `mrtrix.act.backtrack` | bool | true | Enable ACT backtracking |
| `mrtrix.act.crop_at_gmwmi` | bool | true | Crop endpoints at the GMWMI |
| `mrtrix.filtering.method` | str | `none` | `none`, `sift`, or `sift2` |
| `tract_specific.bundles` | list | [] | TractSeg names or ROI definitions |
| `tract_specific.metrics` | list | [] | `MODEL.METRIC` names or image mappings |
| `tract_specific.streamline_statistic` | str | `mean` | Statistic produced per streamline by `tcksample` |
| `tract_specific.profiles.enabled` | bool | false | Generate fixed-node along-tract profiles |
| `tract_specific.profiles.nodes` | int | 100 | Number of points per resampled streamline |
| `tract_specific.track_density.enabled` | bool | false | Create a TDI per extracted bundle |
| `tract_specific.connectome.enabled` | bool | false | Create a parcellation-based connectome |
| `tract_specific.connectome.nodes` | path | none | Required DWI-space integer node image |

The old flat `tractography.enabled`, `algorithm`, and `n_streamlines` keys are
translated for compatibility, but new configurations should use `mrtrix`.

### Modeling Fit Method Notes

| Model | Method | Fit options |
| --- | --- | --- |
| DTI DIPY | `dipy` | `WLLS`, `OLS`, `NLLS`, `RESTORE` |
| DTI FSL | `fsl` | FSL `dtifit` options |
| DTI MRtrix | `mrtrix` | MRtrix tensor fitting and `tensor2metric` outputs |
| DKI DIPY | `dipy` | `WLLS`, `OLS`, `IRLS`/weighted options depending on DIPY support |
| Free-water DTI DIPY | `dipy` | Usually `NLLS` or `WLS` |

## Relaxometry Workflow

Root key: `relaxometry`

```yaml
relaxometry:
  preprocessing:
    rerun_from_step: null
    reorient:
      enabled: true
    denoising:
      enabled: true
      method: mrtrix
    degibbs:
      enabled: true
      method: dipy
    motion_correction:
      enabled: true
      method: ants
    b1:
      enabled: false
      method: afi
  masking:
    enabled: true
    method: fsl
```

| Step | Key | Methods | Notes |
| --- | --- | --- | --- |
| Reorient | `relaxometry.preprocessing.reorient` | FSL `fslreorient2std` | No method switch currently |
| Denoising | `relaxometry.preprocessing.denoising` | `mrtrix`, `ants`, `mppca`, `patch2self`, `nlmeans`, `wavelets`, `gaussian` | Same denoising step as anatomical/dMRI |
| Gibbs unringing | `relaxometry.preprocessing.degibbs` | `mrtrix`, `dipy` | Same Gibbs step |
| Motion correction | `relaxometry.preprocessing.motion_correction` | `ants`, `fsl` | Series registration |
| B1 mapping | `relaxometry.preprocessing.b1` | `afi`, `external`, `hifi` | `registration.method` can be `ants` or `fsl` |
| Brain masking | `relaxometry.masking` or `relaxometry.preprocessing.brain_masking` | `fsl`, `mrtrix`, `ants`, `freesurfer`, `synthstrip`, `hd-bet` | Uses common brain masking step |

## Import Workflow

Root key: `import`

```yaml
import:
  enabled: true
  method: dcm2bids
```

| Key | Methods | Notes |
| --- | --- | --- |
| `import.method` | `dcm2bids`, `dcm2niix` | DICOM import backend |

## fMRI Workflow

The fMRI workflow currently wraps external BIDS-app style tools.

| Step | Key | Methods / Tools |
| --- | --- | --- |
| fMRIPrep | fMRI workflow config | `fmriprep` |
| HCP-style workflow | fMRI workflow config | HCP wrapper where configured |

## Common Method Reference

### Denoising

```yaml
denoising:
  enabled: true
  method: mppca
  patch_radius: 2
  block_radius: 5
  pca_method: eig
  model: ridge
```

| Method | Backend | Key options |
| --- | --- | --- |
| `mrtrix` | MRtrix `dwidenoise` | `mask_dilation`, `noise_map` outputs |
| `ants` | ANTs `DenoiseImage` | ANTs defaults |
| `mppca` | DIPY MP-PCA | `patch_radius`, `block_radius`, `pca_method` |
| `patch2self` | DIPY Patch2Self | `model` |
| `nlmeans` | DIPY NLMeans | DIPY denoise options |
| `wavelets` | PyWavelets | `threshold_method` |
| `gaussian` | SciPy Gaussian filter | smoothing sigma options where exposed |

### Brain Masking

```yaml
brain_masking:
  enabled: true
  method: synthstrip
  mask_input: b0
  apply_mask: true
  use_gpu: false
```

| Method | Backend |
| --- | --- |
| `fsl` | FSL BET |
| `mrtrix` | MRtrix `dwi2mask` |
| `ants` | ANTs-style masking path |
| `freesurfer` | FreeSurfer tools |
| `synthstrip` | FreeSurfer SynthStrip |
| `hd-bet` | HD-BET |

### Registration and Normalization

| Step | Methods | Notes |
| --- | --- | --- |
| Coregistration | `ants`, `fsl`, `freesurfer` | Linear/cross-modal alignment |
| Nonlinear normalization | `ants`, `fsl` | ANTs is the primary nonlinear option |
| dMRI normalization | `ants`, `synthmorph`, `robust_iterative` | Used by dMRI normalization workflow where enabled |
