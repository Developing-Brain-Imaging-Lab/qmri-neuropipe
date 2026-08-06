# Pipeline Option Reference

This is the authoritative configuration-file reference for processing,
modeling, normalization, tractography, and quantitative analysis. The tables
are organized by the exact YAML path consumed by the workflow builders.
Command-line flags are documented separately in [CLI usage](CLI_USAGE_GUIDE.md).

## Reading the tables

- A dotted name is the full path from the YAML root.
- `null` means that the pipeline discovers the input or lets the backend choose.
- Step blocks are disabled unless their `enabled` value says otherwise.
- For supported steps, keys in `options` or `parameters` are flattened into the
  step block. A direct key wins when both spellings are present.
- Backend option dictionaries are passed through only where a table explicitly
  says so. Unknown keys elsewhere should be treated as configuration errors,
  even if an older release silently ignored them.
- Paths may be absolute or relative to the directory from which the command is
  launched. Environment variables in configuration values are expanded.

## Global execution options

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `bids_dir` | path | required | Input BIDS dataset. |
| `output_dir` | path | required | Derivatives root. |
| `work_dir` | path | required | Scratch and resumable intermediate files. |
| `participant_label` | string/list | all | Subject IDs, with or without `sub-`. |
| `session_label` | string/list | all | Session IDs, with or without `ses-`. |
| `subjects_file` | path | null | Text/CSV subject selection file. |
| `n_cpus` | integer | `1` | Process/thread budget supplied to supporting tools. |
| `memory_gb` | number | `8.0` | Requested memory budget in GiB. |
| `use_gpu` | boolean | `false` | Global GPU preference; a step-specific value wins. |
| `gpu_ids` | integer/list/string | null | Visible GPU IDs. JAX fits use one selected GPU per fit. |
| `skip_existing` | boolean | `true` | Reuse complete, valid derivatives. |
| `save_intermediates` | boolean | `false` | Publish intermediate derivatives. Legacy alias: `save_intermediate`. |
| `stop_on_error` | boolean | `false` | Abort the subject batch after the first error. |
| `log_level` | string | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR`. |
| `debug` / `verbose` | boolean | `false` | Increase diagnostic output. |
| `study_name` | string | null | Name recorded by the study tracker and reports. |
| `force` | boolean | `false` | Force execution despite reusable outputs. |
| `qc.mriqc.enabled` | boolean | `false` | Run MRIQC on raw inputs when supported. |
| `qc.mriqc.modalities` | list | workflow defaults | MRIQC modalities; dMRI defaults to `dwi`, `T1w`, and `T2w` when available. |

Workflow scopes accept `rerun_from_step`; aliases are `force_from_step`,
`start_at_step`, and `resume_from_step`. The selected step and all later steps
run again, while earlier valid outputs remain reusable.

## Shared image-processing steps

The following blocks are used by anatomical, diffusion, and relaxometry
workflows. Prefix the key with the workflow path shown in its workflow section.

### Resampling and reorientation

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `resample.enabled` | boolean | `false` | Enable voxel-size resampling with FreeSurfer `mri_convert`. |
| `resample.resolution` | number/list | null | Isotropic millimetres as one number, or three target voxel sizes. |
| `reorient.enabled` | boolean | `false` | Reorient to a standard orientation. dMRI also rotates/preserves gradient sidecars. |

### Denoising

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `denoising.enabled` | boolean | `false` | Enable denoising. |
| `denoising.method` | string | workflow-specific | `mrtrix`, `ants`, `mppca`, `patch2self`, `nlmeans`, `wavelets`, or `gaussian`. |
| `denoising.patch_radius` | integer | `2` | Spatial patch radius for MP-PCA/NLMeans. May also appear under `parameters`. |
| `denoising.block_radius` | integer | `5` | NLMeans search-block radius. May also appear under `parameters`. |
| `denoising.mask_dilation` | integer | `2` | Number of temporary-mask dilation passes. |
| `denoising.pca_method` | string | `eig` | MP-PCA decomposition: `eig` or `svd`. May also appear under `parameters`. |
| `denoising.model` | string | `ridge` | Patch2Self regression model. |
| `denoising.sigma` | number/null | estimated | NLMeans noise standard deviation. |
| `denoising.wavelet` | string | `db4` | PyWavelets wavelet family. |
| `denoising.threshold_method` | string | `BayesShrink` | Wavelet threshold rule. |

MP-PCA and Patch2Self require a 4D image. The anatomical workflow defaults
`method` to `ants`; dMRI and relaxometry default it to `mrtrix`.

### Gibbs unringing, bias correction, and sharpening

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `degibbs.enabled` | boolean | `false` | Enable Gibbs-ringing removal. `gibbs` is an anatomical compatibility alias. |
| `degibbs.method` | string | `mrtrix` | `mrtrix` or `dipy`. |
| `bias_correction.enabled` | boolean | `false` | Enable bias-field correction. |
| `bias_correction.method` | string | `ants` | `ants` (N4) or `mrtrix` (`dwibiascorrect`; DWI needs gradients). |
| `sharpen.enabled` | boolean | `false` | Enable anatomical sharpening. |
| `sharpen.method` | string | `ants` | Currently only ANTs sharpening is implemented. |

### Brain masking

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `brain_masking.enabled` | boolean | `false` | Enable mask generation. |
| `brain_masking.method` | string | workflow-specific | `fsl`, `mrtrix`, `ants`, `freesurfer`, `synthstrip`, or `hd-bet`. |
| `brain_masking.mask_input` | string | `b0` | For 4D/DWI input, build the mask from `b0` or `average`. |
| `brain_masking.apply_mask` | boolean | `true` | Also write a skull-stripped image, not only the binary mask. |
| `brain_masking.use_gpu` | boolean/null | global setting | GPU selection for a GPU-capable backend. |
| `brain_masking.output_skull_stripped` | boolean | `false` | Anatomical alias that enables final skull-stripped derivative export. |

### Coregistration

Direct keys and `coregistration.options` are merged; direct keys win.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `coregistration.enabled` | boolean | `false` | Enable alignment to an anatomical reference. |
| `coregistration.method` | string | `ants` | `ants`, `fsl`, or `freesurfer`. |
| `coregistration.reference_image` | string | `T1w` | `T1w`, `T2w`, `supersynth`, or `supersynth_multivariate`; dMRI also accepts `synthT1w` and `synthetic_t1w`. |
| `coregistration.transform_type` | string | `Rigid` | ANTs transform (`Rigid`, `Affine`, `SyN`, `SyNOnly`, or another ANTsPy transform name). |
| `coregistration.interpolation` | string | `linear` | ANTs interpolation, such as `linear`, `nearestNeighbor`, `bSpline`, or `genericLabel`. |
| `coregistration.dof` | integer | `6` | FSL FLIRT degrees of freedom. |
| `coregistration.cost` | string | `normmi` | FSL cost function, for example `normmi`, `mutualinfo`, or `corratio`. |
| `coregistration.apply_method` | string | `native` | `native` or `mrtrix`; the latter applies a DWI transform with MRtrix gradient handling. |
| `coregistration.application_mode` | string | `resample` | `resample` changes voxel data; `header` updates only geometry and requires native/DWI output resolution. |
| `coregistration.output_resolution` | string | `anatomical` | `anatomical`, `dwi`, or `native` output grid. |
| `coregistration.registration_moving` | path | normal moving image | Alternate contrast used only to estimate the transform. |
| `coregistration.registration_fixed` | path | selected target | Alternate fixed contrast used only to estimate the transform. |
| `coregistration.application_fixed` | path | selected target | Reference grid used when applying the estimated transform. |
| `coregistration.registration_moving_extras` | list | `[]` | Extra moving channels for multivariate ANTs registration. |
| `coregistration.registration_fixed_extras` | list | `[]` | Extra fixed channels paired with the moving extras. |
| `coregistration.multivariate_metric` | string | `Mattes` | ANTs metric for extra channels. |
| `coregistration.multivariate_weight` | number | `0.5` | Weight assigned to each extra-channel metric. |
| `coregistration.multivariate_sampling` | integer | `32` | Sampling parameter for extra-channel metrics. |
| `coregistration.supersynth_input` | string | `auto` | `auto`, `T1w`, or `T2w`; dMRI/Synb0 also accepts `dwi`, `b0`, `diffusion`, and `mean_b0`. |
| `coregistration.supersynth_registration` | string | `single` | `single` or ANTs `multivariate` synthetic-contrast registration. |
| `coregistration.supersynth_mode` | string | anatomical setting | `invivo`, `exvivo`, `cerebrum`, `left-hemi`, or `right-hemi`. |
| `coregistration.supersynth_device` | string/null | null | FreeSurfer SuperSynth device, for example `cpu` or `cuda`. |
| `coregistration.supersynth_sharpen_synths` | boolean | `false` | Sharpen synthetic registration contrasts. |
| `coregistration.subjects_dir` | path | discovered | FreeSurfer `SUBJECTS_DIR`. Aliases: `freesurfer_dir`, `freesurfer_subjects_dir`. |
| `coregistration.subject_id` | string | discovered | FreeSurfer subject. Aliases: `fs_subject_id`, `freesurfer_subject_id`. |
| `coregistration.skull_strip` | boolean/object | `false` | Skull-strip registration inputs. Alias: `skull_strip_registration`. |
| `coregistration.skull_strip_method` | string | workflow default | Mask backend. Aliases: `skullstrip_method`, `brain_extraction_method`. |
| `coregistration.skull_strip_moving` / `skull_strip_fixed` | boolean | `true` | Select which side(s) to strip. |
| `coregistration.skull_strip_use_gpu` | boolean | `false` | GPU preference for registration-input masking. |
| `coregistration.skull_strip_mask_input` | string | `b0` | `b0` or `average` for a 4D registration input. |

(anatomical-processing)=
## Anatomical processing

### Input selection

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `anat.input.primary_modality` | string | `auto` | `auto`, `T1w`, or `T2w`; controls the primary structural image. |
| `anat.input.t1w_match` / `t2w_match` | object | null | Strict selector; must resolve to exactly one image. |
| `...match.entities` | object | null | Match BIDS entities such as `acq`, `run`, `rec`, `ce`, or `dir`. |
| `...match.json_fields` | object | null | Match exact JSON sidecar values. |
| `...match.bids_name` | string | null | Match an exact BIDS stem. |

Shared steps live under `anat.preprocessing`. Additional anatomical controls:

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `anat.preprocessing.force_run` | boolean | `false` | Force anatomical preprocessing. |
| `anat.preprocessing.use_freesurfer` | boolean | `false` | Compatibility switch that enables reconstruction. Also accepted at `anat.use_freesurfer`. |
| `anat.preprocessing.skull_stripped_outputs` | boolean | `false` | Export masked T1w/T2w derivatives. |
| `anat.preprocessing.recon_all.enabled` | boolean | `false` | Run FreeSurfer reconstruction. |
| `...recon_all.method` | string | `standard` | `standard` or `clinical`. |
| `...recon_all.args` | string | `-all` | Main recon-all argument string. |
| `...recon_all.extra_args` | string | empty | Additional backend arguments. |
| `...recon_all.subjects_dir` | path | output-derived | FreeSurfer subject directory root. |
| `anat.super_synth.enabled` | boolean | `false` | Run `mri_super_synth`. |
| `anat.super_synth.mode` | string | `invivo` | `invivo`, `exvivo`, `cerebrum`, `left-hemi`, or `right-hemi`. |
| `anat.super_synth.sharpen_synths` | boolean | `false` | Sharpen generated contrasts. |
| `anat.super_synth.device` | string/null | null | FreeSurfer-selected device if omitted; otherwise `cpu` or `cuda`. |
| `anat.super_synth.compute_volumes` | boolean | `false` | Export per-label volumes from the synthetic segmentation. |

### Anatomical normalization

The block is `anat.preprocessing.normalization`.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | boolean | `false` | Enable normalization. |
| `template` | path | required | Fixed standard-space template. |
| `method` | string | `ants` | `ants` or affine-oriented `fsl`. |
| `options` | object | `{}` | Backend registration options; ANTs accepts `transform_type`, `interpolation`, and ANTsPy registration kwargs. |
| `save_transforms` | boolean | `true` | Publish forward/inverse transforms. Alias: `save_transform`. |
| `space_entity` | string | `Standard` | BIDS `space-` entity. Aliases: `space_name`, `space`. |
| `skull_stripped_outputs` | boolean | preprocessing setting | Export normalized skull-stripped derivatives. |

(diffusion-preprocessing)=
## Diffusion preprocessing

All keys below are under `dmri.preprocessing`.

| Block/key | Type | Default | Description |
| --- | --- | --- | --- |
| `merging.enabled` | boolean | `true` for multiple series | Merge compatible DWI runs when downstream correction needs one series. |
| `grad_check.enabled` | boolean | `false` | Run MRtrix `dwigradcheck` and use/export corrected gradients. |
| `distcorr.method` | string | `none` | `none`, `topup`, `synb0`, `drbuddi`, or `topup+drbuddi`. |
| `distcorr.fallback` | boolean | `false` | Use Synb0 when TOPUP was requested but reverse-PE data are unavailable. |
| `distcorr.config` | path | FSL default | TOPUP configuration file. |
| `distcorr.coregister_inputs` | boolean | `false` | Coregister reverse-PE inputs before TOPUP. |
| `motion_correction.method` | string | `none` | `none`, `eddy`, or `niifreeze`. Enabling the legacy `eddy` block implies `eddy`. |
| `eddy.enabled` | boolean | `false` | Legacy switch for Eddy motion correction. |
| `eddy.method` | string | `eddy` | `eddy`, `eddy-correct`, or `two-pass`. |
| `eddy.mask_dilation` | integer | `3` | Dilation passes for the temporary Eddy mask. |
| `eddy.options` | object | `{}` | FSL Eddy options. Keys become `--key=value`; booleans become flags. Pipeline-generated `acqp`, `index`, `mask`, `bvecs`, and `bvals` take precedence. |
| `eddy.optimize_memory` | boolean | `false` | Enable the wrapper's memory-optimized Eddy mode. |
| `outliers.enabled` | boolean | `false` | Enable volume/slice outlier removal. |
| `outliers.method` | string | `threshold` | `manual`, `threshold`, or `eddy_qc`; `deep_learning` is reserved/experimental. |
| `outliers.manual_indices` | integer list | null | Zero-based volumes removed by `manual`. |
| `outliers.volumes_file` | path | null | File containing indices or an Eddy outlier map, depending on method. |
| `outliers.threshold` | number | `0.05` | Low-signal threshold fraction used by `threshold`. |
| `bias_correction.*` | shared block | disabled | See shared steps. |
| `brain_masking.*` | shared block | disabled | dMRI method default is `mrtrix`. |
| `coregistration.*` | shared block | disabled | See the complete coregistration table. |

### NiiFreeze

When `motion_correction.method: niifreeze`, the implementation reads these
keys from the top-level configuration for compatibility:

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `b0_thresh` | number | `5` | b-value threshold used to identify b0 volumes. |
| `model` | string | `b0` | NiiFreeze registration model. |
| `strategy` | string | `random` | Volume-selection strategy. |
| `seed` | integer | `2021` | Reproducibility seed. |

### Synb0 and DRBUDDI

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `distcorr.synb0.device` | string | `cpu` | `cpu` or `gpu`; CPU hides CUDA devices from TensorFlow. |
| `distcorr.synb0.gpu_ids` | integer/list/string | global `gpu_ids` | GPUs exposed to Synb0. |
| `distcorr.synb0.t1w_source` | string | `raw` | `raw`, `supersynth`, `prefer_supersynth`, `dwi_supersynth`, or documented aliases `supersynth_dwi`, `diffusion_supersynth`, `b0_supersynth`. |
| `distcorr.synb0.supersynth_input` | string | `auto` | `auto`, `T1w`, `T2w`, `dwi`, `b0`, `diffusion`, or `mean_b0`. |
| `distcorr.synb0.anatomical_input` | string | `auto` | Preferred acquired anatomical contrast. |
| `distcorr.synb0.anatomical_series_mode` | string | `mean` | How multiple volumes in the anatomical input are reduced. |
| `distcorr.synb0.anatomical_series_index` | integer | `0` | Volume used by index-based selection. |
| `distcorr.synb0.registration_method` | string | `direct` | Synthetic/anatomical input registration strategy. |
| `distcorr.synb0.registration_tool` | string | `fsl` | Registration backend. Alias: `registration_backend`. |
| `distcorr.synb0.skull_strip_registration_inputs` | boolean | `false` | Mask inputs before Synb0 registration. |
| `distcorr.synb0.registration_skull_strip_method` | string | `synthstrip` | Masking backend for those inputs. |
| `distcorr.synb0.supersynth_mode` | string | anatomical setting | SuperSynth mode. |
| `distcorr.synb0.supersynth_device` | string/null | anatomical setting | SuperSynth device, independent of TensorFlow device. |
| `distcorr.synb0.supersynth_sharpen_synths` | boolean | anatomical setting | Sharpen synthetic T1w. |
| `distcorr.drbuddi.transform_type` | string | `SyNOnly` | ANTs transform for reverse-PE refinement. |
| `distcorr.drbuddi.interpolator` | string | `linear` | Final resampling interpolation. |
| `distcorr.drbuddi.symmetric_pairwise` | boolean | `true` | Estimate symmetric half-warps. |
| `distcorr.drbuddi.pe_axis_constraint` | number | `1.0` | Warp constraint along the PE axis, from `0` to `1`. |
| `distcorr.drbuddi.registration_options` | object | `{}` | Extra ANTs registration keyword arguments. |
| `tortoise_v4.synthetic_reverse_pe.enabled` | boolean | `false` | Experimental Synb0/TOPUP field forward-warp into b0-only synthetic reverse-PE data for TORTOISE DRBUDDI. |
| `tortoise_v4.synthetic_reverse_pe.forward_warp_backend` | string | `fugue` | Forward-distortion backend; currently only FSL FUGUE is supported. |
| `tortoise_v4.synthetic_reverse_pe.fugue_unwarpdir` | string | `auto` | FUGUE voxel-axis direction (`x`, `x-`, `y`, `y-`, `z`, `z-`); protocol validation is recommended, especially for the x axis. |
| `tortoise_v4.synthetic_reverse_pe.intensity_correction` | boolean | `true` | Request FUGUE pixel-shift intensity correction. |
| `tortoise_v4.synthetic_reverse_pe.duplicate_volumes` | integer | `2` | Number of duplicate synthetic b0 volumes supplied as the down series. |
| `tortoise_v4.synthetic_reverse_pe.repol_policy` | string | `disable` | `disable`, `error`, or `allow`; protects b0-only synthetic down processing. |
| `tortoise_v4.synthetic_reverse_pe.effective_echo_spacing` | number/null | BIDS metadata | Explicit FUGUE dwell-time override in seconds. |
| `tortoise_v4.synthetic_reverse_pe.total_readout_time` | number/null | BIDS metadata | Readout-time override used to derive echo spacing and write synthetic metadata. |

### Gradient nonlinearity (GNL)

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `grad_nonlin.enabled` | boolean | `false` | Build a spatial gradient tensor and propagate it to final DWI/model space. |
| `grad_nonlin.method` | string | `tortoise` | `tortoise`, `native_ge`, `native`, or `latest_native`; aliases normalize to TORTOISE or native GE. |
| `grad_nonlin.coeff_file` | path | required to compute | Scanner gradient-coefficient file. |
| `grad_nonlin.map_path` / `map_file` | path | null | Use an existing GNL tensor rather than compute one. |
| `grad_nonlin.force` | boolean | `false` | Recompute/re-align despite reusable output. |

The same keys may be placed at `dmri.modeling.grad_nonlin` to supply GNL when
preprocessing is reused. A preprocessing map is preferred when both exist.

(diffusion-model-fitting)=
## Diffusion model fitting

Every model block accepts `enabled` (default `false`), `method` (listed below),
and `nthreads` (default `n_cpus`). `parameters` and `options` are accepted where
noted and flattened into the model call.

### Tensor, kurtosis, MAP-MRI, and CSD

| Block/key | Type | Default | Description |
| --- | --- | --- | --- |
| `dti` / `tensor` | block | disabled | DTI. Methods: `dipy`, `fsl`, `mrtrix`. |
| `dti.fit_method` | string | `WLLS` | DIPY `WLLS`, `OLS`, `NLLS`, or `RESTORE`; passed to the chosen backend where supported. |
| `dti.metrics` | list | `fa, md, ad, rd, color_fa, evals, evecs` | Requested DTI derivatives. |
| `dti.b0_threshold` | number | `50` | DIPY b0 threshold. |
| `dti.smoothing_fwhm` | number/null | null | Optional spatial smoothing before fitting. |
| `dti.Delta_file` / `delta_file` | path | DWI sidecars | Big-/small-delta timing overrides. |
| `dki.fit_method` | string | `WLLS` | DIPY fit method; aliases are normalized to DIPY-supported names. |
| `dki.metrics` | list | `mk, ak, rk, fa, md` | Requested DKI derivatives. |
| `dki.mean_signal` | boolean | `false` | Use the mean-signal DKI variant. |
| `dki.b0_threshold`, `smoothing_fwhm`, `Delta_file`, `delta_file` | mixed | as DTI | Same meanings as DTI. |
| `fwe_dti` / `fwdti` | block | disabled | DIPY free-water DTI. Only `method: dipy`. |
| `fwe_dti.fit_method` | string | `NLLS` | DIPY free-water tensor fit method. |
| `fwe_dti.metrics` | list | `fa, md, ad, rd, f` | Requested tensor/free-water derivatives. |
| `mapmri.laplacian` | boolean | `true` | Enable Laplacian regularization. |
| `mapmri.positivity` | boolean | `true` | Enforce propagator positivity. |
| `mapmri.global_constraints` | boolean | `false` | Use global rather than local constraints. |
| `mapmri.metrics` | list | `rtop, rtap, rtpp, qiv, msd, ng` | Requested scalar maps; add `peaks` for peak directions. |
| `mapmri.peak_npeaks` | integer | `3` | Maximum peaks per voxel. |
| `mapmri.peak_relative_threshold` | number | `0.5` | Relative peak-amplitude threshold. |
| `mapmri.peak_min_separation_angle` | number | `25` | Minimum peak separation in degrees. |
| `mapmri.b0_threshold`, `smoothing_fwhm`, `Delta_file`, `delta_file` | mixed | as DTI | Same meanings as DTI. Other keys are passed to DIPY `MapmriModel`. |
| `csd.method` | string | `msmt_csd` | `msmt_csd` or single-tissue `csd`. |
| `csd.response_algorithm` | string | `dhollander` | MRtrix `dwi2response` algorithm. |
| `csd.lmax` | integer/list/null | MRtrix default | Spherical-harmonic order(s). |

### NODDI, SANDI, Microglia, and NEXI

| Block/key | Type | Default | Description |
| --- | --- | --- | --- |
| `noddi.method` | string | `dmipy` | `dmipy` or `amico`. |
| `noddi.metrics` | list | `odi, ficvf, fiso` | Requested maps. AMICO exports ICVF, ODI, and FISO. |
| `noddi.model_type` | string | `standard` | `standard` or spherical-mean `smt`. |
| `noddi.distribution` | string | `Watson` | Orientation distribution for standard dmipy NODDI. |
| `noddi.parallel_diffusivity` | number | `1.7e-9` | Fixed intra-neurite axial diffusivity in m2/s. |
| `noddi.iso_diffusivity` | number | `3.0e-9` | Fixed isotropic diffusivity in m2/s. |
| `noddi.fiso_file` | path | null | External isotropic-fraction constraint. |
| `sandi.method` | string | `amico` | `amico` or `dmipy`. |
| `sandi.soma_diffusivity` | number | backend-specific | AMICO uses `3.0` um2/ms; dmipy uses SI units and accepts legacy `iso_diffusivity`. |
| `sandi.soma_radii` | number list | `[1, 3.8, 6.6, 9.4, 12.2, 15]` | AMICO radius grid in um. |
| `sandi.neurite_diffusivities` | number list | `[0.25, 1.1667, 2.0833, 3]` | AMICO grid in um2/ms. |
| `sandi.extra_diffusivities` | number list | `[0.25, 0.9375, 1.625, 2.3125, 3]` | AMICO extracellular grid in um2/ms. |
| `sandi.l1_regularization` | number | `0` | AMICO L1 coefficient. |
| `sandi.l2_regularization` | number | `0.00075` | AMICO L2 coefficient. |
| `sandi.Delta_file` / `delta_file` | path | DWI sidecars | Required PGSE timing for dmipy SANDI. |
| `microglia.method` | string | `dmipy` | Four-compartment dmipy model. |
| `microglia.parallel_diffusivity` | number | `1.0e-9` | Cylinder axial diffusivity in m2/s. |
| `microglia.iso_diffusivity` | number | `3.0e-9` | Isotropic diffusivity in m2/s. |
| `microglia.small_diameter` / `large_diameter` | number | `8e-6` / `16e-6` | Initial diameters in metres. |
| `microglia.small_diameter_bounds` | pair | `(5e-6, 11e-6)` | Small-cylinder fit bounds. |
| `microglia.large_diameter_bounds` | pair | `(12e-6, 18e-6)` | Large-cylinder fit bounds. |
| `microglia.Ns` | integer | `5` | Brute-force grid samples per parameter. |
| `microglia.maxiter` | integer | `300` | Optimizer iteration limit. |
| `microglia.N_sphere_samples` | integer | `30` | Sphere-model sampling resolution. |
| `nexi.method` | string | `nexi` | NEXI Rice-mean fitter. |
| `nexi.td_file` / `td_path` | path | required | Per-volume diffusion-time file. |
| `nexi.lowb_noisemap` / `lowb_noisemap_file` | path | required | Noise map estimated from low-b data. |
| `nexi.metrics` | list | `t_ex, di, de, f, sigma` | Requested maps; `tex` and `t-ex` alias `t_ex`. |
| `nexi.debug` | boolean | `false` | Enable NEXI diagnostics. |

dmipy legacy blocks (`noddi`, dmipy `sandi`, and `microglia`) share:

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `solver` | string | `brute2fine` | `brute2fine`, `mix`, or `jax`, subject to model capabilities. |
| `device` | string | `auto` | `auto`, `cpu`, or `gpu`; GPU requires JAX. |
| `gpu_device` | integer | first configured GPU | GPU index visible to this process. |
| `solver_kwargs` | object | `{}` | Solver controls such as grid sizes, batch size, and iteration limits. |
| `jax_cache_dir` | path | null | Persistent JAX compilation cache. |
| `jax_log_compiles` | boolean | `false` | Log JAX compilation. |
| `heartbeat_interval` | seconds | `30` | Liveness interval during long fits. |

### Registry-driven dmipy models

`dmri.modeling.dmipy.models` is a list of objects or a mapping keyed by model
name. Available names and capability details come from `qmri-tools dmipy-models`.

| Entry key | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | string | required in list form | Allow-listed registry name. |
| `enabled` | boolean | `true` | Disable an entry without removing it. |
| `solver` | string | `brute2fine` | Capability-validated `brute2fine`, `mix`, or `jax`. |
| `device` | string | `auto` | `auto`, `cpu`, or `gpu`. |
| `gpu_device` | integer | selected global GPU | One GPU per fit. |
| `nthreads` | integer | `n_cpus` | Fit thread budget. |
| `factory_kwargs` | object | `{}` | Registered model-factory keyword arguments. |
| `solver_options` | object | `{}` | dmipy solver keyword arguments. |
| `gradient_nonlinearity` | boolean | `true` | Use a context GNL tensor when the model supports it. |
| `jax_cache_dir` | path | null | Persistent compilation cache. |
| `jax_log_compiles` | boolean | `false` | Emit JAX compilation logging. |
| `heartbeat_interval` | seconds | `30` | Fit heartbeat interval. |
| `delta_file` / `Delta_file` / `TE_file` | path | DWI metadata/null | Acquisition timing overrides. |

Duplicate model names, unsupported model/solver/GNL combinations, and enabling
a registry model alongside its legacy block are rejected before fitting.

(tractography-and-tract-analysis)=
## Tractography and tract analysis

All keys are under `dmri.modeling.tractography`.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `mrtrix.enabled` | boolean | `false` | Generate a whole-brain MRtrix tractogram. |
| `mrtrix.algorithm` | string | `iFOD2` | `iFOD2`, `SD_STREAM`, `Tensor_Det`, `Tensor_Prob`, or another supported `tckgen` algorithm. |
| `mrtrix.select` | integer | `10000000` | Output streamline count. Legacy name: flat `n_streamlines`. |
| `mrtrix.cutoff` | number | MRtrix default | FOD amplitude cutoff. |
| `mrtrix.minlength` / `maxlength` | number | MRtrix default | Streamline length limits in mm. |
| `mrtrix.seed_image` | path | auto | Explicit seed image. |
| `mrtrix.seed_dynamic` | path | auto FOD | Dynamic-seeding FOD image. |
| `mrtrix.options` | object | `{}` | Additional `tckgen` options; booleans become flags. |
| `mrtrix.act.enabled` | boolean | `false` | Anatomically constrained tractography. |
| `mrtrix.act.five_tt` / `5tt` | path | generated | Existing 5TT image. |
| `mrtrix.act.algorithm` | string | `fsl` | `5ttgen` algorithm. |
| `mrtrix.act.validate` | boolean | `true` | Run `5ttcheck`. |
| `mrtrix.act.seed_gmwmi` | boolean | `true` | Generate/use a GMWMI seed. |
| `mrtrix.act.backtrack` | boolean | `true` | Allow ACT backtracking. |
| `mrtrix.act.crop_at_gmwmi` | boolean | `true` | Crop streamline endpoints at GMWMI. |
| `mrtrix.act.input` | string | `auto` | Anatomical input selection. |
| `mrtrix.act.mode`, `device`, `sharpen_synths` | mixed | anatomical settings | SuperSynth controls when a synthetic anatomical input is required. |
| `mrtrix.act.b0_threshold` | number | `50` | b0 threshold for DWI-derived anatomical handling. |
| `mrtrix.filtering.method` | string | `none` | `none`, `sift`, or `sift2`. |
| `mrtrix.filtering.term_number` | integer | backend default | Target streamline count for SIFT. |
| `mrtrix.filtering.options` | object | `{}` | Extra SIFT/SIFT2 backend arguments. |

### TractSeg and pyAFQ

The following keys live inside `tractseg.options` or `pyafq.options`.

| TractSeg key | Type | Default | Description |
| --- | --- | --- | --- |
| `output_type` | string | `tract_segmentation` | TractSeg output directory/type, such as `tract_segmentation`, `endings_segmentation`, `TOM`, or `dm_regression`, depending on the installed TractSeg release. |
| `preprocess` | boolean | `true` | Use pipeline MNI transforms when present; otherwise request TractSeg's internal preprocessing. |
| `bundles` / `bundle_names` | list | null | Downstream bundle selection; TractSeg still segments its standard bundle set. |
| `mni_template` | path | context MNI reference | Template for manual forward/inverse warping. |
| any other option key | scalar/boolean | none | Forwarded to the TractSeg CLI as `--key value` or a boolean flag. This is the route for `preview`, `single_output_file`, `bundle_specific_threshold`, `get_priors`, `keep_intermediate_files`, `csc_peaks`, `super_resolution`, `uncertainty`, `tract_definition`, and release-specific flags. CPU count is always the pipeline `n_cpus`. |

| pyAFQ key | Type | Default | Description |
| --- | --- | --- | --- |
| `profile` | string | `default` | `default`, `baby`, or `pediatric`. |
| `tractography_method` | string | `probabilistic` | pyAFQ tracking mode. |
| `segmentation_params` | object | null | pyAFQ segmentation parameters. |
| `cleaning_params` | object | null | pyAFQ cleaning parameters. |
| `overwrite` | boolean | `false` | Replace pyAFQ outputs. |
| `n_cpus` | integer | `1` | pyAFQ CPU budget. |

### Tract-specific analysis

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `tract_specific.enabled` | boolean | `false` | Enable bundle extraction and optional tractometry. Auto-enables MRtrix tractography. |
| `tract_specific.bundles` | list | `[]` | Bundle definitions. Each may name a TractSeg bundle or provide `name`, `include`, `exclude`, `source`, and `ends_only`. |
| `tract_specific.metrics` | list/object | `[]` | Metric images or `MODEL.METRIC` references sampled along tracts. |
| `tract_specific.streamline_statistic` | string | `mean` | Per-streamline `tcksample` statistic. |
| `tract_specific.profiles.enabled` | boolean | `false` | Generate along-tract profiles. |
| `tract_specific.profiles.nodes` | integer | `100` | Resampling points per streamline. |
| `tract_specific.track_density.enabled` | boolean | `false` | Generate a TDI for each extracted bundle. |
| `tract_specific.track_density.normalize` | boolean | `false` | Normalize TDI intensity. |
| `tract_specific.track_density.options` | object | `{}` | Extra `tckmap` options. |
| `tract_specific.connectome.enabled` | boolean | `false` | Generate a node-by-node connectome. |
| `tract_specific.connectome.nodes` | path | required | DWI-space integer node/parcellation image. |
| `tract_specific.connectome.use_sift2` | boolean | `true` | Weight edges with SIFT2 weights when available. |
| `tract_specific.connectome.statistic` | string | `sum` | Edge aggregation statistic. |
| `tract_specific.connectome.symmetric` | boolean | `true` | Symmetrize the matrix. |
| `tract_specific.connectome.zero_diagonal` | boolean | `true` | Set self-connections to zero. |
| `tract_specific.connectome.options` | object | `{}` | Extra `tck2connectome` options. |

## Standard-space normalization

The same option set is accepted by `dmri.normalization` and
`relaxometry.normalization`; dMRI defaults `driving_metric` to `FA`, while
relaxometry defaults it to `spgr_ref`.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | boolean | `false` | Enable normalization. |
| `template` | path | required | Fixed standard-space image. |
| `driving_metric` | string | modality-specific | Native image used to estimate the transform. |
| `tool` | string | `ants` | `ants`, `synthmorph`, or `robust_iterative`. |
| `space_name` / `space_entity` | string | `Standard` dMRI / `MNI` relaxometry | Output BIDS space. Legacy alias: `space`. |
| `save_transforms` | boolean | `true` | Publish transforms. Alias: `save_transform`. |
| `transform_type` | string | `SyN` | ANTs transform. |
| `include_all_metrics` | boolean | `true` | Apply the transform to every available model metric. |
| `synthmorph_model` | string/null | executable default | SynthMorph model. |
| `synthmorph_transform_ext` | string/null | inferred | Transform filename extension. |
| `synthmorph_register_args` / `synthmorph_apply_args` | string | empty | Extra registration/application arguments. |
| `synthmorph_args` | string | empty | Legacy general SynthMorph arguments. |
| `synthmorph_moving_flag` / `synthmorph_target_flag` / `synthmorph_output_flag` | string | `--mov` / `--targ` / `--o` | CLI flag names for nonstandard wrappers. |
| `skull_strip` | boolean/object | null | Strip registration images. Alias: `skull_strip_registration`. |
| `skull_strip_method` | string | null | Mask backend. Alias: `brain_extraction_method`. |
| `skull_strip_moving` / `skull_strip_fixed` | boolean | `true` | Sides to mask. |
| `skull_strip_use_gpu` | boolean | `false` | Masking GPU preference. Alias: `use_gpu`. |
| `robust_iterative.iterations` | integer | `2` | SynthMorph/ANTs refinement cycles. |
| `robust_iterative.synthmorph.enabled` | boolean | `true` | Run SynthMorph in each cycle. |
| `robust_iterative.synthmorph.model` | string | `joint` | SynthMorph model. |
| `robust_iterative.synthmorph.transform_ext` | string/null | inferred | Transform extension. |
| `robust_iterative.synthmorph.register_args` / `apply_args` | string | empty | Per-cycle SynthMorph arguments. |
| `robust_iterative.ants.enabled` | boolean | `true` | Run ANTs refinement in each cycle. |
| `robust_iterative.ants.transform_type` | string | outer value or `SyN` | Per-cycle ANTs transform. |
| `robust_iterative.ants.apply_interpolator` | string | `linear` | Per-cycle scalar interpolation. |
| `robust_iterative.ants.registration_kwargs` | object | `{}` | Additional ANTsPy registration kwargs. |
| `robust_iterative.apply.default_scalar_interpolator` | string | `linear` | Final scalar-map interpolation. |

(relaxometry-processing-and-modeling)=
## Relaxometry processing and modeling

### Preprocessing

Shared steps live under `relaxometry.preprocessing`; masking may be configured
at `relaxometry.masking` or `relaxometry.preprocessing.brain_masking`.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `preprocessing.motion_correction.enabled` | boolean | `false` | Register SPGR/SSFP volumes. |
| `...motion_correction.method` | string | `ants` | `ants` or `fsl`. |
| `...motion_correction.transform_type` | string | `Rigid` | ANTs transform. Alias: `type_of_transform`. |
| `...motion_correction.interpolation` | string | `linear` | ANTs interpolation. Alias: `interpolator`. |
| `...motion_correction.nthreads` | integer | `n_cpus` | Thread budget. Alias: `threads`. |
| `...motion_correction.dof` | integer | `6` | FSL degrees of freedom. |
| `...motion_correction.extra_args` | string | empty | Extra backend arguments. Alias: `args`. |
| `preprocessing.exclude_indices` | object | `{}` | Per-series zero-based volume indices excluded before stacking/fitting. |
| `preprocessing.spgr_reference.mode` | string | `mean` | SPGR reference selection/reduction mode. |
| `preprocessing.spgr_reference.volume_index` | integer | null | Explicit reference volume when the mode requires one. |
| `preprocessing.b1.method` | string | `afi` | `afi`, `external`, or `hifi`. |
| `preprocessing.b1.smoothing_fwhm` | number | `0` | Gaussian smoothing of the B1 map in mm; zero disables it. |
| `preprocessing.b1.registration.method` | string | `fsl` | `fsl` or `ants`. |
| `...b1.registration.transform_type` | string | `Rigid` | ANTs transform. |
| `...b1.registration.interpolation` | string | `linear` | Interpolation; alias `interpolator`. |
| `...b1.registration.moving_volume` / `fixed_volume` | integer | `reference_volume` | 4D volume used for registration. |
| `...b1.registration.reference_volume` | integer | `0` | Fallback 4D reference index. |
| `...b1.registration.nthreads` | integer | `n_cpus` | Threads; alias `threads`. |
| `...b1.registration.dof` | integer | `6` | FSL degrees of freedom. |
| `...b1.registration.cost` | string | `normmi` | FSL cost function. |

### DESPOT-family fitting

Blocks are `relaxometry.modeling.despot1`, `despot2`, `despot2fm`, and
`mcdespot`.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | boolean | `false` | Enable this model. |
| `algo` | string | model executable default | Fit algorithm; DESPOT1 defaults to `lsq`. |
| `nthreads` | integer | `n_cpus` | Fitter threads. Legacy alias: `threads`. |
| `verbose` | boolean | `false` | Verbose `qmri_fit` output. |
| `despot1.use_hifi` | boolean | `false` | Use IR-SPGR/B1-corrected DESPOT1-HIFI. |
| `mcdespot.cuda` | boolean | `false` | Use the CUDA mcDESPOT executable. |
| any other model key | scalar/list | none | Forwarded as a `qmri_fit` CLI option: booleans become flags and other values become `--key=value`. |

`despot2.mcdespot` is a deprecated compatibility switch for
`relaxometry.modeling.mcdespot.enabled`.

(atlasroi-analysis)=
## Atlas/ROI analysis

Use `dmri.analysis`, `relaxometry.analysis`, or `anat.segmentation`.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | boolean | `false` | Register atlases and extract ROI statistics. |
| `atlases` | object | `{}` | Named atlas definitions. |
| `registration_target` | string | `FA` dMRI / `reference` relaxometry | Native metric used as fixed image. Alias: `registration_metric`. |
| `registration_template` | path | per-atlas template | Override the moving template for every atlas. Alias: `atlas_template`. |
| `parameters` / `metrics` | string/list/object | all outputs | Global metrics, `MODEL:METRIC` selectors, or per-model lists. |
| `atlas_threshold` | number | null | Threshold for continuous/probabilistic atlas data. |
| `include_zero_label` | boolean | `false` | Include background label zero. |
| `background_label` | integer | `0` | Label treated as background. |
| `morphology` | string/null | null | Optional label-mask morphology operation. |
| `morphology_iterations` | integer | `1` | Number of morphology iterations. |
| `threshold` | number | `0.1` | Probabilistic mask threshold used during statistics extraction. |

Each named `atlases.<name>` object accepts:

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `labels` / `file` | path | required | 3D integer or 4D probabilistic atlas. |
| `template` | path | workflow normalization template | Image defining the atlas space/contrast. |
| `lut` | path | null | XML/TXT label lookup table. |
| `is_probabilistic` | boolean | `false` | Interpret a 4D atlas as one probability map per ROI. |
| `interpolation` | string | `genericLabel` for labels | ANTs interpolation. Use label-safe interpolation for discrete atlases. |
| `include_zero_label` | boolean | analysis setting | Per-atlas override. |
| `background_label` | integer | analysis setting | Per-atlas override. |

Legacy `atlas_file` and `atlas_labels` create one atlas named `Default`.
Statistics include mean, median, and standard deviation; probabilistic atlases
use probability weighting.

## Aggregate g-ratio analysis

The root block is `gratio`.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | boolean | `false` | Run aggregate g-ratio analysis. |
| `inputs.myelin.path` | path template | discovered mcDESPOT VFm | Myelin-volume-fraction image. Templates may include subject/session. |
| `inputs.myelin.reference` / `inputs.spgr_reference` | path | discovered | SPGR anatomical reference. |
| `inputs.axonal.path` | path | discovered NODDI ICVF | Intracellular/axonal fraction. |
| `inputs.axonal.isotropic_path` | path | paired NODDI FISO | Isotropic fraction. |
| `inputs.axonal.reference` | path | mean b0 | Diffusion-space reference. |
| `inputs.axonal.interpretation` | string | intracellular fraction | Set `axonal_volume_fraction` when the input is already AVF; otherwise AVF is `(1-FISO)*ICVF`. |
| `inputs.axon_diameter.enabled` | boolean | `false` | Enable diameter-derived outputs. |
| `inputs.axon_diameter.path` | path | null | Inner-axon diameter image. |
| `inputs.axon_diameter.units` | string | `um` | `um`/`µm` or `mm` (converted to um). |
| `registration.assume_aligned` | boolean | `false` | Skip cross-modal registration; grids must already match. |
| `registration.transform` | string | `rigid` | Registration transform. |
| `registration.tool` | string | `ants` | Registration backend. |
| `registration.interpolation` | string | `linear` | Myelin-map resampling interpolation. |
| `registration.skull_strip` | object | `{enabled: true, method: fsl}` | Registration-input masking. |
| `calibration.mode` | string | `identity` | Myelin calibration mode. |
| `calibration.slope` / `intercept` | number | null | Linear calibration coefficients when required by the mode. |
| `validity.epsilon` | number | `1e-6` | Numerical denominator tolerance. |
| `validity.clipping_tolerance` | number | `1e-6` | Allowed fraction-domain overshoot before invalidation. |
| `recommended_mask.enabled` | boolean | `true` | Generate the recommended analysis mask. |
| `recommended_mask.path` | path | subject-native WM if found | Optional external WM/probability mask. |
| `recommended_mask.prefer_subject_native_wm` | boolean | `true` | Auto-discover subject WM. |
| `recommended_mask.fiso_max` | number | `0.5` | Maximum isotropic fraction. |
| `recommended_mask.avf_min` / `fvf_min` | number | `0` | Minimum volume fractions. |
| `recommended_mask.wm_probability_min` | number | `0.5` | External WM probability threshold. |
| `recommended_mask.erosion_voxels` | integer | `0` | Binary erosion iterations. |
| `conduction.conduction_factor` | boolean | `true` | Write the dimensionless conduction factor. |
| `conduction.rushton.enabled` | boolean | `true` | Write Rushton measures. |
| `conduction.rushton.calibration_coefficient` | number | null | Optional scaling coefficient. |
| `conduction.waxman_bennett.enabled` | boolean | `true` | Write Waxman-Bennett measures. |
| `conduction.waxman_bennett.calibration_coefficient` | number | null | Optional scaling coefficient. |

(fmri-and-import-wrappers)=
## fMRI and import wrappers

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `fmriprep.enabled` | boolean | `false` | Run fMRIPrep; otherwise the current workflow routes to the HCP placeholder. |
| `fmriprep.container_path` | path | null | Local Singularity/Apptainer container image. |
| `fmriprep.docker_image` | string | null | Docker image when no local container is supplied. |
| `fmriprep.fs_license_file` | path | null | FreeSurfer license. |
| `fmriprep.custom_args` | list | `[]` | Additional fMRIPrep arguments. |
| `n_cpus` | integer | `1` | fMRIPrep process/thread count (global key). |
| `omp_nthreads` | integer | `1` | fMRIPrep OpenMP threads (top-level compatibility key). |
| `hcp_workflow_type` | string | `Volume` | HCP placeholder workflow type. The native HCP path is not yet implemented. |
| `import.dicom_dir` | path | null | DICOM directory/archive. Setting it activates automatic import unless `auto_run` is false. |
| `import.auto_run` | boolean | `true` when `dicom_dir` is set | Run import before the selected processing pipeline. |
| `import.method` | string | `dcm2bids` | `dcm2bids` or `dcm2niix`. |
| `import.subject` / `session` | string | run selection | Explicit output entities. |
| `import.dcm2bids.config_file` | path | required for dcm2bids | dcm2bids mapping file. |
| `import.dcm2niix.filename` | string | `%p_%s_%t` | Output filename pattern. |
| `import.dcm2niix.compress` | boolean | `true` | Gzip NIfTI output. |
| `import.dcm2niix.bids` | boolean | `true` | Write BIDS JSON sidecars. |
| `import.dcm2niix.extra_args` | string | empty | Additional dcm2niix arguments. |
| `import.clobber` | boolean | `false` | Replace existing import outputs. |
| `import.force_dcm2bids` | boolean | `false` | Force dcm2bids conversion. |
| `import.gradient_overrides` | object | `{}` | Match-based bval/bvec overrides. |
| `import.metadata_overrides` | object | `{}` | Match-based JSON field overrides. |
| `import.gnl_metadata.manufacturer` | string | `GE` | Scanner convention used for GNL metadata. |
| `import.gnl_metadata.dicom_dir` | path | null | DICOM source for GNL metadata recovery. |
| `import.gnl_metadata.fix_phase_encoding` | boolean | `false` | Correct phase-encoding metadata. |
| `import.gnl_metadata.phase_from` | string | `dir` | Source used to infer phase encoding. |

## Validation and compatibility notes

- Configuration YAML rejects duplicate mapping keys.
- A path required by an enabled step is validated before the external tool is
  launched whenever possible.
- `options` pass-through means that availability and accepted spelling may also
  depend on the installed backend version. Stable pipeline-owned keys are all
  listed above.
- Compatibility aliases are documented beside their canonical keys. New
  configurations should use the canonical spelling.
- Legacy top-level `do_bias_correction`, `bias_method`, `do_coregistration`,
  `coreg_method`, `do_brain_masking`, and `masking_method` are still read when
  the corresponding `dmri.preprocessing` block does not make the choice. New
  configurations should use the nested blocks because nested values win.
