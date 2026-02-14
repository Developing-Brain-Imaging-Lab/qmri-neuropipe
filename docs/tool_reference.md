# Tool Reference

This page lists the implemented tools for each workflow step and the matching configuration keys.

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
| `reference_image` | str | `t1w` | For anat coreg (`t1w`, `t2w`) |
| `options.apply_method` | str | `native` | `native`, `mrtrix` |
| `options.output_resolution` | str | `anatomical` | `anatomical`, `dwi`, `native` |
| `options.interpolation` | str | `linear` | `linear`, `nearest`, `sinc`, `cubic` |
| `options.dof` | int | 6 | FSL |
| `options.cost` | str | `normmi` | FSL |
| `options.transform_type` | str | `Rigid` | ANTs |

## Normalization (dMRI)

**Tools**
- `ants`
- `synthmorph` (FreeSurfer mri_synthmorph)

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
| `tool` | str | `ants` | `ants`, `synthmorph` |
| `transform_type` | str | `SyN` | ANTs only |
| `save_transforms` | bool | true | ANTs only |
| `include_all_metrics` | bool | true | Normalize all model outputs found |
| `synthmorph_transform_ext` | str | `.lta` | Output extension for `-t` transform |
| `synthmorph_model` | str | none | Model passed to `mri_synthmorph register -m` (e.g., `joint`, `deform`, `affine`, `rigid`) |
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
- `none` (skip)

**Config**
```yaml
dmri:
  preprocessing:
    distcorr:
      method: topup
      fallback: true
      config: /path/to/topup.cnf
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `method` | str | `none` | `topup`, `synb0`, `none` |
| `fallback` | bool | false | Allow Synb0 fallback |
| `config` | path | none | Topup config file |

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
