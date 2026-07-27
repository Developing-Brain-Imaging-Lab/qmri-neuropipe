# Tool Reference

This page lists the implemented tools for each workflow step and the matching configuration keys.

## Workflow Method Summary

### Anatomical

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

### Diffusion

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

### Relaxometry

| Step | Config Key | Methods / Backends |
| --- | --- | --- |
| Reorient | `relaxometry.preprocessing.reorient` | `fsl` (`fslreorient2std`) |
| Denoising | `relaxometry.preprocessing.denoising` | `mrtrix`, `ants`, `mppca`, `patch2self`, `nlmeans`, `wavelets`, `gaussian` |
| Gibbs unringing | `relaxometry.preprocessing.degibbs` | `mrtrix`, `dipy` |
| Motion correction | `relaxometry.preprocessing.motion_correction` | `ants`, `fsl` |
| B1 mapping | `relaxometry.preprocessing.b1` | `afi`, `external`, `hifi` |
| Brain masking | `relaxometry.masking` or `relaxometry.preprocessing.brain_masking` | `fsl`, `mrtrix`, `ants`, `freesurfer`, `synthstrip`, `hd-bet` |

## QC

**Tools**
- `mriqc` (optional)

**Config**
```yaml
qc:
  mriqc:
    enabled: true
```

## Denoising

**Tools**
- `mrtrix` (dwidenoise)
- `ants` (DenoiseImage)
- `mppca` (DIPY MP-PCA)
- `patch2self` (DIPY)
- `nlmeans` (DIPY)
- `wavelets` (PyWavelets)
- `gaussian` (SciPy)

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
      model: ridge

anat:
  preprocessing:
    denoising:
      enabled: true
      method: ants
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | bool | false | Enable step |
| `method` | str | `mrtrix` | `mrtrix`, `ants`, `mppca`, `patch2self`, `nlmeans`, `wavelets`, `gaussian` |
| `patch_radius` | int | 2 | MP-PCA |
| `block_radius` | int | 5 | MP-PCA |
| `mask_dilation` | int | 2 | Temporary mask dilation |
| `pca_method` | str | `eig` | MP-PCA |
| `model` | str | `ridge` | Patch2Self |

## Gibbs Unringing

**Tools**
- `mrtrix` (mrdegibbs)
- `dipy` (gibbs_removal)

**Config**
```yaml
dmri:
  preprocessing:
    degibbs:
      enabled: true
      method: mrtrix

anat:
  preprocessing:
    degibbs:
      enabled: true
      method: mrtrix
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | bool | false | Enable step |
| `method` | str | `mrtrix` | `mrtrix`, `dipy` |

## Bias Correction

**Tools**
- `ants` (N4BiasFieldCorrection)
- `mrtrix` (dwibiascorrect)

**Config**
```yaml
dmri:
  preprocessing:
    bias_correction:
      enabled: true
      method: ants

anat:
  preprocessing:
    bias_correction:
      enabled: true
      method: ants
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | bool | false | Enable step |
| `method` | str | `ants` | `ants`, `mrtrix` |

## Brain Masking

**Tools**
- `mrtrix` (dwi2mask)
- `fsl` (bet)
- `ants` (antsBrainExtraction)
- `freesurfer` (mri_watershed)
- `synthstrip` (mri_synthstrip)
- `hd-bet` (HD-BET)

**Config**
```yaml
dmri:
  preprocessing:
    brain_masking:
      enabled: true
      method: mrtrix
      mask_input: b0
      apply_mask: true
      use_gpu: false

anat:
  preprocessing:
    brain_masking:
      enabled: true
      method: ants
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | bool | false | Enable step |
| `method` | str | `mrtrix` (dmri) / `ants` (anat) | `mrtrix`, `fsl`, `ants`, `freesurfer`, `synthstrip`, `hd-bet` |
| `mask_input` | str | `b0` | `b0`, `average` (dmri) |
| `apply_mask` | bool | true | Apply mask to data |
| `use_gpu` | bool | false | Some tools support GPU |

> **Note:** The pipeline always saves brain masks as 3D NIfTI files. If a pre-existing mask on disk has more than 3 dimensions (e.g. from an older run), it is automatically detected and squeezed to 3D on the next run, using the first volume.

## Coregistration

**Tools**
- `ants`
- `fsl`
- `freesurfer`

**Config**
```yaml
dmri:
  preprocessing:
    coregistration:
      enabled: true
      method: ants
      options:
        apply_method: native
        output_resolution: anatomical
        interpolation: linear
        dof: 6
        cost: normmi
        transform_type: Rigid

anat:
  preprocessing:
    coregistration:
      enabled: true
      method: fsl
      reference_image: t1w
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | bool | false | Enable step |
| `method` | str | `ants` (dmri) / `fsl` (anat) | `ants`, `fsl`, `freesurfer` |
| `reference_image` | str | `t1w` | `t1w`, `t2w`, `supersynth`, or `supersynth_multivariate`; dMRI SuperSynth modes synthesize matching contrasts from a mean b0 and the anatomical input |
| `supersynth_input` | str | `auto` | `auto`, `T1w`, `T2w`; selects the anatomical source. The dMRI source is an automatically extracted mean b0 |
| `supersynth_registration` | str | none | Set to `multivariate` with `reference_image: supersynth` to use synthetic T1w and T2w pairs with ANTs. FSL and FreeSurfer use the synthetic T1w pair |
| `supersynth_b0_threshold` | float | `50` | Maximum b-value included in the mean b0 used by SuperSynth |
| `multivariate_metric` | str | `Mattes` | ANTs metric for additional SuperSynth contrast channels |
| `multivariate_weight` | float | `0.5` | ANTs weight for each additional contrast channel |
| `multivariate_sampling` | int | `32` | ANTs sampling parameter for additional contrast channels |
| `options.apply_method` | str | `native` | `native`, `mrtrix` |
| `options.output_resolution` | str | `anatomical` | `anatomical`, `dwi`, `native` |
| `options.interpolation` | str | `linear` | `linear`, `nearest`, `sinc`, `cubic` |
| `options.dof` | int | 6 | FSL |
| `options.cost` | str | `normmi` | FSL |
| `options.transform_type` | str | `Rigid` | ANTs |

## WM Segmentation (BBR Coregistration)

The `wm_seg_method` option controls which tool is used to generate the white-matter
segmentation mask used during boundary-based registration (BBR).

| Method | Description |
|--------|-------------|
| `fast` | FSL FAST (3-class tissue segmentation). Fast but less accurate. |
| `synthseg` | FreeSurfer `mri_synthseg`. Better accuracy; requires FreeSurfer. |
| `supersynth` | FreeSurfer `mri_super_synth`. Best accuracy; requires FreeSurfer dev build (>Oct 2025). **Automatically falls back to `synthseg`** if `mri_super_synth` is not found on PATH or in `$FREESURFER_HOME/bin/`. |

**Config**
```yaml
anat:
  preprocessing:
    coregistration:
      wm_seg_method: supersynth  # fast | synthseg | supersynth
```

## Anatomical Input Selection

Use these selectors when multiple anatomical acquisitions exist and you want to
choose a specific `T1w` or `T2w` scan without hard-coding a full path.

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
| `anat.input.t1w_match.json_fields` | dict | none | Match sidecar JSON fields |
| `anat.input.t1w_match.bids_name` | str | none | Match full BIDS file stem |
| `anat.input.t2w_match.entities` | dict | none | Same behavior for T2w |
| `anat.input.t2w_match.json_fields` | dict | none | Same behavior for T2w |
| `anat.input.t2w_match.bids_name` | str | none | Same behavior for T2w |

**Notes**
*   If no selector is provided, the pipeline keeps the existing behavior and uses the discovered anatomical inputs as-is.
*   If a selector is provided, exactly one match is required.

## Normalization (dMRI)

**Tools**
- `ants`
- `synthmorph` (FreeSurfer mri_synthmorph)
- `robust_iterative` (iterative SynthMorph + ANTs)

**Config**
```yaml
dmri:
  normalization:
    enabled: true
    template: /path/to/template.nii.gz
    driving_metric: FA
    space_name: MNI
    tool: ants
    transform_type: SyN
    save_transforms: true
    include_all_metrics: true
    # For synthmorph only:
    synthmorph_model: deform
    synthmorph_args: ""
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | bool | false | Enable step |
| `template` | path | required | Template image |
| `driving_metric` | str | `FA` | Metric used for registration |
| `space_name` | str | `Standard` | BIDS `space-` label |
| `tool` | str | `ants` | `ants`, `synthmorph`, `robust_iterative` |
| `transform_type` | str | `SyN` | ANTs only |
| `save_transforms` | bool | true | ANTs only |
| `include_all_metrics` | bool | true | Normalize all model outputs found |
| `robust_iterative.iterations` | int | 2 | Number of iterative SynthMorph + ANTs rounds |
| `robust_iterative.synthmorph.enabled` | bool | true | Enable SynthMorph stage inside each round |
| `robust_iterative.synthmorph.model` | str | `joint` | `joint`, `deform`, `affine`, `rigid` |
| `robust_iterative.ants.enabled` | bool | true | Enable ANTs refinement inside each round |
| `robust_iterative.ants.transform_type` | str | `SyN` | ANTs transform for refinement |
| `robust_iterative.apply.default_scalar_interpolator` | str | `linear` | Interpolator used when applying saved robust manifests |
| `synthmorph_transform_ext` | str | model-dependent | Output extension for `-t`: `.lta` for `rigid`/`affine`, `.nii.gz` for `deform`/`joint` |
| `synthmorph_model` | str | `joint` | Model passed to `mri_synthmorph register -m` (`joint`, `deform`, `affine`, or `rigid`) |
| `synthmorph_register_args` | str | none | Extra args for `mri_synthmorph register` |
| `synthmorph_apply_args` | str | none | Extra args for `mri_synthmorph apply` |
| `synthmorph_args` | str | none | Legacy passthrough (register/apply) |
| `synthmorph_moving_flag` | str | `--mov` | Deprecated (not used in register/apply) |
| `synthmorph_target_flag` | str | `--targ` | Deprecated (not used in register/apply) |
| `synthmorph_output_flag` | str | `--o` | Deprecated (not used in register/apply) |

**Notes**
- If gradient nonlinearity correction is enabled in preprocessing or modeling, the pipeline
  will reuse the cached GNL tensor from the preprocessed outputs and only recompute if missing
  or forced.

## Distortion Correction

**Tools**
- `topup` (FSL)
- `synb0` (DIPY Synb0)
- `drbuddi` (native ANTs-based reverse-PE refinement)
- `none` (skip)

**Config**
```yaml
dmri:
  preprocessing:
    distcorr:
      method: topup+drbuddi
      fallback: true
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
| `method` | str | `none` | `topup`, `synb0`, `drbuddi`, `topup+drbuddi`, `none` |
| `fallback` | bool | false | Allow Synb0 fallback |
| `synb0.device` | str | `cpu` | `cpu` hides GPUs from TensorFlow; use `gpu` only when TensorFlow/CUDA is compatible with the host driver |
| `synb0.gpu_ids` | list\|int\|str | global `gpu_ids` | GPU IDs exposed to TensorFlow when `synb0.device: gpu` |
| `synb0.t1w_source` | str | `raw` | `raw`, `supersynth`, `prefer_supersynth`, `dwi_supersynth`; controls whether Synb0 uses the anatomical T1w directly or a SuperSynth-generated T1w |
| `synb0.supersynth_input` | str | `auto` | `auto`, `T1w`, `T2w`, `dwi`; selects which image SuperSynth uses when generating the T1w for Synb0. `dwi` uses a 3D mean b0 extracted from the diffusion image |
| `config` | path | none | Topup config file |
| `drbuddi.transform_type` | str | `SyNOnly` | Native DRBUDDI ANTs transform |
| `drbuddi.interpolator` | str | `linear` | Output resampling interpolator |
| `drbuddi.symmetric_pairwise` | bool | `true` | Use symmetric blip-up/blip-down half-warps |
| `drbuddi.pe_axis_constraint` | float | `1.0` | Constrain warp mostly to the PE axis |
| `drbuddi.registration_options` | dict | `{}` | Extra ANTs registration kwargs |

## Eddy Current Correction

**Tools**
- `eddy` (FSL eddy)
- `eddy-correct` (FSL legacy)
- `two-pass` (two-pass eddy correction)

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
| `enabled` | bool | false | Enable step |
| `method` | str | `eddy` | `eddy`, `eddy-correct`, `two-pass` |

## Outlier Removal

**Tools**
- `manual` (explicit indices)
- `eddy_qc` (from eddy outlier map)
- `threshold` (threshold-based)

**Config**
```yaml
dmri:
  preprocessing:
    outliers:
      enabled: true
      method: manual
      manual_indices: [0, 1, 2]
      threshold: 0.05
      volumes_file: /path/to/eddy_outlier_map
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | bool | false | Enable step |
| `method` | str | `manual` | `manual`, `eddy_qc`, `threshold` |
| `manual_indices` | list[int] | none | Manual volume indices |
| `threshold` | float | 0.05 | Threshold mode |
| `volumes_file` | path | none | Eddy outlier map |

## Gradient Nonlinearity Correction

**Tools**
- `native_ge`
- `tortoise`

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
| `enabled` | bool | false | Enable step |
| `method` | str | `tortoise` | `native_ge`, `tortoise` |
| `coeff_file` | path | required | Gradient coefficients `.dat` |

## Motion Correction (Relaxometry)

**Tools**
- `ants`
- `fsl`

**Config**
```yaml
relaxometry:
  preprocessing:
    motion_correction:
      enabled: true
      method: ants
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | bool | false | Enable step |
| `method` | str | `ants` | `ants`, `fsl` |

## SuperSynth (Anatomical)

A modality-agnostic U-Net that produces brain segmentation, MNI atlas registration,
and synthetic 1 mm isotropic T1w, T2w, and FLAIR images from any 3D brain volume.
Accepts inputs of any resolution or contrast, including low-field scans, ex vivo
tissue, cerebrum-only acquisitions, and single hemispheres.

Requires a FreeSurfer development build newer than **October 2025**.

**Tools**
- `freesurfer` (`mri_super_synth`)

**Config**
```yaml
anat:
  super_synth:
    enabled: true
    mode: invivo        # invivo | exvivo | cerebrum | left-hemi | right-hemi
    sharpen_synths: false
    device: null        # null = tool default (cuda when available), or "cpu" / "cuda"
    compute_volumes: false
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `anat.super_synth.enabled` | bool | false | Enable step |
| `anat.super_synth.mode` | str | `invivo` | `invivo`, `exvivo`, `cerebrum`, `left-hemi`, `right-hemi` |
| `anat.super_synth.sharpen_synths` | bool | false | Sharpen synthetic T1w/T2w/FLAIR predictions |
| `anat.super_synth.device` | str\|null | null | Compute device: `cpu` or `cuda` |
| `anat.super_synth.compute_volumes` | bool | false | Load `seg.nii.gz`, map label IDs to names via `$FREESURFER_HOME/FreeSurferColorLUT.txt` (falls back to `ROI_{id}`), and save a CSV (`sub-XX_ses-YY_desc-supersynth_volumes.csv`) with columns `label_id`, `label_name`, `n_voxels`, `volume_mm3`. Volumes are also logged to the study tracker `Volume_Statistics` sheet. |

**Outputs** (written to `super_synth/sub-<id>/[ses-<id>/]`)
| File | Description |
| --- | --- |
| `seg.nii.gz` | Brain region segmentation |
| `T1w.nii.gz` | Synthetic 1 mm isotropic T1w |
| `T2w.nii.gz` | Synthetic 1 mm isotropic T2w |
| `FLAIR.nii.gz` | Synthetic 1 mm isotropic FLAIR |

**Context keys set**
| Key | Value |
| --- | --- |
| `super_synth_dir` | Path to the per-subject output directory |
| `super_synth_outputs` | Dict mapping `seg`, `synth_t1w`, `synth_t2w`, `synth_flair` → `Path` |
| `preprocessed_t1w` | Set to the synthetic T1w `ImageFile` if not already present in context |

**Notes**
- Output filenames above reflect common FreeSurfer Synth tool conventions; verify
  against your specific build and update `_OUTPUT_STEMS` in
  `lib/anat/super_synth.py` if they differ.
- The tool also registers the input to MNI space and writes Dice scores for QC.
  These files are preserved in `super_synth_dir` but not currently parsed into
  pipeline context.
- Threads default to `-1` (all available cores); override via `n_cpus` in the
  top-level config.

## Step-Level Reruns

When `skip_existing: true`, the workflow normally skips steps whose expected outputs already exist. Set `rerun_from_step` at the relevant scope to force that step and all later steps while still allowing earlier steps to use cached derivatives.

```yaml
dmri:
  preprocessing:
    rerun_from_step: eddy

dmri:
  modeling:
    rerun_from_step: dti

anat:
  preprocessing:
    rerun_from_step: normalization

relaxometry:
  preprocessing:
    rerun_from_step: motion_correction
```

Accepted key aliases: `force_from_step`, `start_at_step`, `resume_from_step`.

Common step aliases: `merge`, `reorient`, `denoise`, `gibbs`, `degibbs`, `topup`, `distcorr`, `eddy`, `bias`, `coregistration`, `brain_masking`, `normalization`, `freesurfer`, `recon_all`, `supersynth`, `segmentation`, `motion_correction`, `b1`, `modeling`, `atlas`, `stats`, and `acqparams`.

## B1 Mapping (Relaxometry)

**Tools**
- `afi`
- `external`
- `hifi`

**Config**
```yaml
relaxometry:
  preprocessing:
    b1:
      method: afi
      smoothing_fwhm: 0.0
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `method` | str | `afi` | `afi`, `external`, `hifi` |
| `smoothing_fwhm` | float | 0.0 | Optional smoothing |

## Analysis & Statistics (Atlas Registration / ROI Extraction)

The shared analysis module (`lib/common/analysis.py`) handles atlas registration and
ROI statistics extraction across all modalities (diffusion, relaxometry, anatomical).

**Config**
```yaml
dmri:
  analysis:
    enabled: true
    atlases:
      - name: JHU
        template: /path/to/JHU-ICBM-FA-1mm.nii.gz
        labels: /path/to/JHU-labels.nii.gz
        label_names: /path/to/JHU-label-names.txt
        registration_template: /path/to/JHU-ICBM-FA-1mm.nii.gz
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | bool | false | Enable atlas registration and ROI extraction |
| `atlases` | list | none | List of atlas configurations |
| `atlases[].name` | str | required | Atlas label used in output filenames |
| `atlases[].template` | path | required | Atlas template image |
| `atlases[].labels` | path | required | Atlas label image |
| `atlases[].label_names` | path | none | Text file mapping label IDs to region names |
| `atlases[].registration_template` | path | none | Override the registration target when the atlas template has different contrast from the primary reference (e.g. FA-contrast templates used with a T1w SPGR reference). If omitted, the atlas `template` is used directly. |

**Output**

ROI statistics are written as **CSV** files (one per atlas, per subject/session) with
columns for region name, mean, standard deviation, and voxel count for each
fitted metric.

See [Atlas Registration & ROI Statistics](analysis.md) for full configuration options.
# MRtrix tractography tools

The tractography workflow uses `5ttgen`, `5ttcheck`, and `5tt2gmwmi` for ACT;
`tckgen` for tracking; `tcksift` or `tcksift2` for filtering; `tckedit` for
bundle extraction; `tckresample`, `tcksample`, and `tckmap` for tractometry; and
`tck2connectome` for connectivity matrices. See
[MRtrix Tractography](workflows/tractography.md) for configuration.
