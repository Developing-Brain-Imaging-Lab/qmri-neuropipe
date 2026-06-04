# Relaxometry Workflow

The **Relaxometry Pipeline** (`--pipeline relax`) processes Variable Flip Angle (VFA) data — typically SPGR and SSFP sequences — to generate quantitative maps including T1, T2, M0, B1, and, in the case of mcDESPOT, Myelin Water Fraction expressed as **VFm** (Volume Fraction of the myelin compartment).

See [Tool Reference](../tool_reference.md) for the full list of tools and config keys.

## Step Method Summary

| Step | Config Key | Methods / Backends |
| --- | --- | --- |
| Reorient | `relaxometry.preprocessing.reorient` | `fsl` (`fslreorient2std`) |
| Denoising | `relaxometry.preprocessing.denoising` | `mrtrix`, `ants`, `mppca`, `patch2self`, `nlmeans`, `wavelets`, `gaussian` |
| Gibbs unringing | `relaxometry.preprocessing.degibbs` | `mrtrix`, `dipy` |
| Motion correction | `relaxometry.preprocessing.motion_correction` | `ants`, `fsl` |
| B1 mapping | `relaxometry.preprocessing.b1` | `afi`, `external`, `hifi` |
| Brain masking | `relaxometry.preprocessing.brain_masking` | `fsl`, `mrtrix`, `ants`, `freesurfer`, `synthstrip`, `hd-bet` |
| DESPOT1 fitting | `relaxometry.modeling.despot1` | `lsq` |
| DESPOT2 fitting | `relaxometry.modeling.despot2` | `lsq` |
| DESPOT2FM fitting | `relaxometry.modeling.despot2fm` | `src` |
| mcDESPOT fitting | `relaxometry.modeling.mcdespot` | `src` (CPU / CUDA) |
| Normalization | `relaxometry.normalization` | `ants`, `synthmorph` |
| Analysis | `relaxometry.analysis` | atlas registration + ROI stats |

## Model Dependency Chain

The DESPOT models have strict input dependencies:

```
SPGR images ──► DESPOT1 ──► T1 map ──►┬──► DESPOT2 ──► T2
                                        └──► DESPOT2FM ──► T2, F0
                                        └──► mcDESPOT ──► VFm, T1_fast/slow, T2_fast/slow, Tau
                                              ▲
                                         SSFP images

IR-SPGR images ──► DESPOT1-HIFI (replaces DESPOT1) ──► T1, B1
```

- **DESPOT2** and **DESPOT2FM** both require a T1 map from a completed DESPOT1 run.
- **mcDESPOT** requires both the DESPOT1 T1 map and SSFP data.
- **DESPOT1-HIFI** (`use_hifi: true`) requires IR-SPGR images and simultaneously estimates T1 and B1 — use this instead of a separate B1 map when IR-SPGR data are available.

## Workflow Steps

### 1. Preprocessing

The preprocessing stage removes artifacts and aligns all input images before model fitting.

#### 1a. Reorient

Standardizes image orientation to RAS/LAS using `fslreorient2std`.

**Config**
```yaml
relaxometry:
  preprocessing:
    reorient:
      enabled: true
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `relaxometry.preprocessing.reorient.enabled` | bool | false | Enable reorientation |

#### 1b. Denoising

Applies MP-PCA denoising to all raw input images (SPGR, SSFP, IR-SPGR) to improve SNR before fitting.

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
relaxometry:
  preprocessing:
    denoising:
      enabled: true
      method: mrtrix
      patch_radius: 2
      block_radius: 5
      mask_dilation: 2
      pca_method: eig
      model: ridge     # patch2self only
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `relaxometry.preprocessing.denoising.enabled` | bool | false | Enable denoising |
| `relaxometry.preprocessing.denoising.method` | str | `mrtrix` | `mrtrix`, `ants`, `mppca`, `patch2self`, `nlmeans`, `wavelets`, `gaussian` |
| `relaxometry.preprocessing.denoising.patch_radius` | int | 2 | MP-PCA patch radius |
| `relaxometry.preprocessing.denoising.block_radius` | int | 5 | MP-PCA block radius |
| `relaxometry.preprocessing.denoising.mask_dilation` | int | 2 | Temporary mask dilation during denoising |
| `relaxometry.preprocessing.denoising.pca_method` | str | `eig` | MP-PCA eigenvalue method |
| `relaxometry.preprocessing.denoising.model` | str | `ridge` | Patch2Self regression model |

#### 1c. Gibbs Unringing

Removes Gibbs ringing artifacts from all input series.

**Available tools**
*   `mrtrix` (mrdegibbs)
*   `dipy` (gibbs_removal)

**Config**
```yaml
relaxometry:
  preprocessing:
    degibbs:
      enabled: true
      method: mrtrix
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `relaxometry.preprocessing.degibbs.enabled` | bool | false | Enable Gibbs unringing |
| `relaxometry.preprocessing.degibbs.method` | str | `mrtrix` | `mrtrix`, `dipy` |

#### 1d. Volume Exclusion (`exclude_indices`)

Exclude specific volumes from the SPGR or SSFP series before fitting — for example, when a particular flip-angle volume is corrupted or flagged as an outlier. Indices are 0-based.

**Config**
```yaml
relaxometry:
  preprocessing:
    exclude_indices:
      spgr: "0,3"      # or a list: [0, 3]
      ssfp: [1]
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `relaxometry.preprocessing.exclude_indices.spgr` | str or list | none | SPGR volume indices to exclude (0-based) |
| `relaxometry.preprocessing.exclude_indices.ssfp` | str or list | none | SSFP volume indices to exclude (0-based) |

#### 1e. SPGR Reference Image (`spgr_reference`)

Controls how the single SPGR reference image is selected. This image serves as the fixed target for motion correction and B1 map registration.

**Config**
```yaml
relaxometry:
  preprocessing:
    spgr_reference:
      mode: mean    # mean | last | index | max_flip
      index: 0     # required when mode: index
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `relaxometry.preprocessing.spgr_reference.mode` | str | `mean` | `mean` — average of all SPGR volumes; `last` — final volume; `index` — specific 0-based volume; `max_flip` — volume with highest flip angle |
| `relaxometry.preprocessing.spgr_reference.index` | int | none | Required when `mode: index`; 0-based volume index |

#### 1f. B1 Mapping

Estimates or loads the B1 transmit field map, then registers it to the SPGR reference image. The B1 map is used to correct flip-angle errors during DESPOT fitting.

**Available tools**
*   `afi` — computes B1 from an AFI (Actual Flip-Angle Imaging) pair
*   `external` — resamples a pre-computed B1 map to SPGR reference space
*   `hifi` — B1 is estimated jointly with T1 via DESPOT1-HIFI (set `modeling.despot1.use_hifi: true`)

**Config**
```yaml
relaxometry:
  preprocessing:
    b1:
      method: afi        # afi | external | hifi
      smoothing_fwhm: 0.0
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `relaxometry.preprocessing.b1.method` | str | `afi` | `afi`, `external`, `hifi` |
| `relaxometry.preprocessing.b1.smoothing_fwhm` | float | 0.0 | Gaussian smoothing FWHM in mm; 0 = no smoothing |

#### 1g. Motion Correction

Rigidly registers every SPGR and SSFP volume to the SPGR reference image. If inputs are 4D, volumes are split, registered individually, and re-merged.

**Available tools**
*   `ants` (rigid registration via ANTs)
*   `fsl` (FLIRT)

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
| `relaxometry.preprocessing.motion_correction.enabled` | bool | false | Enable motion correction |
| `relaxometry.preprocessing.motion_correction.method` | str | `ants` | `ants`, `fsl` |

### 2. Brain Masking

Generates a binary brain mask from the motion-corrected SPGR reference image. The mask is applied to all subsequent fitting steps to restrict computation to brain voxels.

> **Note**: The pipeline always stores the brain mask as a **3D NIfTI** file. If a 4D mask is found on disk from a previous run, it is automatically squeezed to 3D at the start of the next run.

**Available tools**
*   `fsl` (BET)
*   `mrtrix` (dwi2mask)
*   `ants` (antsBrainExtraction)
*   `freesurfer` (mri_watershed)
*   `synthstrip` (mri_synthstrip)
*   `hd-bet` (HD-BET)

**Config**
```yaml
relaxometry:
  preprocessing:
    brain_masking:
      enabled: true
      method: fsl
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `relaxometry.preprocessing.brain_masking.enabled` | bool | false | Enable brain masking |
| `relaxometry.preprocessing.brain_masking.method` | str | `fsl` | `fsl`, `mrtrix`, `ants`, `freesurfer`, `synthstrip`, `hd-bet` |

### 3. Model Fitting

All models operate in the subject's native SPGR space and are restricted to brain voxels via the mask from Step 2. The `nthreads` key defaults to the global `n_cpus` setting when not explicitly set.

#### 3a. DESPOT1

Estimates T1 and M0 from SPGR images using a variable flip-angle approach. When `use_hifi: true`, IR-SPGR data are incorporated and B1 is estimated simultaneously (DESPOT1-HIFI), making a separate B1 map optional.

**Outputs**: `T1` (ms), `M0`, `B1` (only when `use_hifi: true`)

**Config**
```yaml
relaxometry:
  modeling:
    despot1:
      enabled: true
      use_hifi: false    # true requires IR-SPGR images
      algo: lsq
      nthreads: 4
      verbose: false
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `relaxometry.modeling.despot1.enabled` | bool | false | Enable DESPOT1 fitting |
| `relaxometry.modeling.despot1.use_hifi` | bool | false | Run DESPOT1-HIFI; requires IR-SPGR data; simultaneously estimates T1 and B1 |
| `relaxometry.modeling.despot1.algo` | str | `lsq` | Fitting algorithm |
| `relaxometry.modeling.despot1.nthreads` | int | global `n_cpus` | Number of CPU threads |
| `relaxometry.modeling.despot1.verbose` | bool | false | Verbose fitting output |

#### 3b. DESPOT2

Estimates single-component T2 and M0 from SSFP images using the DESPOT1 T1 map. Does not apply frequency-modulation (FM) correction.

**Requires**: DESPOT1 T1 map

**Outputs**: `T2` (ms), `M0`, `F0` (off-resonance Hz)

**Config**
```yaml
relaxometry:
  modeling:
    despot2:
      enabled: true
      algo: lsq
      nthreads: 4
      verbose: false
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `relaxometry.modeling.despot2.enabled` | bool | false | Enable DESPOT2 fitting |
| `relaxometry.modeling.despot2.algo` | str | `lsq` | Fitting algorithm |
| `relaxometry.modeling.despot2.nthreads` | int | global `n_cpus` | Number of CPU threads |
| `relaxometry.modeling.despot2.verbose` | bool | false | Verbose fitting output |

#### 3c. DESPOT2FM

Estimates T2 with explicit frequency-modulation correction, producing more accurate T2 estimates in regions with off-resonance effects.

**Requires**: DESPOT1 T1 map

**Outputs**: `T2` (ms), `M0`, `F0` (off-resonance Hz)

**Config**
```yaml
relaxometry:
  modeling:
    despot2fm:
      enabled: true
      algo: src
      nthreads: 4
      verbose: false
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `relaxometry.modeling.despot2fm.enabled` | bool | false | Enable DESPOT2FM fitting |
| `relaxometry.modeling.despot2fm.algo` | str | `src` | Fitting algorithm |
| `relaxometry.modeling.despot2fm.nthreads` | int | global `n_cpus` | Number of CPU threads |
| `relaxometry.modeling.despot2fm.verbose` | bool | false | Verbose fitting output |

#### 3d. mcDESPOT

Multi-component DESPOT fitting that separates fast (myelin-associated) and slow (free) water compartments. The primary output is **VFm** — the volume fraction of the myelin water compartment (analogous to myelin water fraction).

**Requires**: DESPOT1 T1 map and SSFP data

**Outputs**: `VFm` (%), `T1_fast` (ms), `T1_slow` (ms), `T2_fast` (ms), `T2_slow` (ms), `Tau` (exchange time, ms)

**Config**
```yaml
relaxometry:
  modeling:
    mcdespot:
      enabled: true
      cuda: false       # true = GPU-accelerated fitting
      algo: src
      nthreads: 4
      verbose: false
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `relaxometry.modeling.mcdespot.enabled` | bool | false | Enable mcDESPOT fitting |
| `relaxometry.modeling.mcdespot.cuda` | bool | false | GPU-accelerated fitting; uses `qmri_fit_mcdespot_cuda` |
| `relaxometry.modeling.mcdespot.algo` | str | `src` | Fitting algorithm |
| `relaxometry.modeling.mcdespot.nthreads` | int | global `n_cpus` | CPU threads (when `cuda: false`) |
| `relaxometry.modeling.mcdespot.verbose` | bool | false | Verbose fitting output |

### 4. Normalization

Warps all fitted quantitative maps into a standard template space (e.g. MNI152). Registration is driven by a chosen metric image so that the warp captures tissue contrast appropriate for the template.

> **Template contrast note**: Because the SPGR reference has T1w contrast, atlases distributed with FA-contrast templates (e.g. JHU-ICBM-FA) should be used with an MNI T1w template override in the analysis section (see Section 5).

**Available tools**
*   `ants` (SyN / other ANTs transforms)
*   `synthmorph`

**Config**
```yaml
relaxometry:
  normalization:
    enabled: true
    template: /path/to/MNI152_T1_1mm.nii.gz
    driving_metric: spgr_ref    # or: T1, VFm, etc.
    space_name: MNI
    space_entity: MNI
    tool: ants
    transform_type: SyN
    save_transforms: true
    include_all_metrics: true
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `relaxometry.normalization.enabled` | bool | false | Enable normalization |
| `relaxometry.normalization.template` | path | required | MNI (or other) template image |
| `relaxometry.normalization.driving_metric` | str | `spgr_ref` | Image used to drive registration; alternatives: any fitted metric name (`T1`, `VFm`, etc.) |
| `relaxometry.normalization.space_name` | str | `MNI` | BIDS `space` label used in output filenames |
| `relaxometry.normalization.space_entity` | str | `MNI` | BIDS space entity |
| `relaxometry.normalization.tool` | str | `ants` | `ants`, `synthmorph` |
| `relaxometry.normalization.transform_type` | str | `SyN` | ANTs transform type |
| `relaxometry.normalization.save_transforms` | bool | true | Save warp fields and affine to disk |
| `relaxometry.normalization.include_all_metrics` | bool | true | Warp every fitted metric to template space |

### 5. Analysis

Registers a brain atlas to the subject's relaxometry space and extracts per-ROI statistics for selected quantitative metrics. For full atlas configuration details (e.g. label maps, atlas YAML format), see [Analysis Reference](../analysis.md).

> **Registration template**: The analysis step registers atlases to the subject's SPGR reference by default. Because the SPGR reference has T1w contrast, atlas packages distributed with FA-contrast templates (e.g. `JHU-ICBM-FA-1mm.nii.gz`) must override the registration template with an MNI T1w image — otherwise the registration will fail or produce poor results. Use `registration_template` for this override.

**Config**
```yaml
relaxometry:
  analysis:
    enabled: true

    # Image in subject space used as the fixed target for atlas registration.
    # Default is "reference" (SPGR ref image). Can also be a fitted metric name.
    registration_target: reference

    # Override the per-atlas template for ALL atlases.
    # Required when atlas packages ship with FA-contrast templates but your
    # SPGR reference has T1w contrast.
    registration_template: /path/to/MNI152_T1_1mm.nii.gz

    # Metrics to extract statistics from.
    # Can be a string, list, or per-model dict.
    metrics:
      - T1
      - VFm
      - T2_fast
      - T2_slow

    atlases:
      JHU:
        labels: /path/to/JHU-ICBM-labels-1mm.nii.gz
        template: /path/to/JHU-ICBM-FA-1mm.nii.gz   # FA-contrast; override above
        lut: /path/to/JHU_labels.txt
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `relaxometry.analysis.enabled` | bool | false | Enable atlas analysis |
| `relaxometry.analysis.registration_target` | str | `reference` | Fixed image for atlas registration; `reference` = SPGR ref, or a fitted metric name |
| `relaxometry.analysis.registration_template` | path | normalization template (if enabled) | Overrides the per-atlas template key for all atlases; use an MNI T1w template when atlases ship with FA-contrast templates |
| `relaxometry.analysis.metrics` | str, list, or dict | all fitted metrics | Metrics to extract ROI statistics from |
| `relaxometry.analysis.atlases` | dict | none | Named atlas definitions; see [Analysis Reference](../analysis.md) for full format |

### 6. QC

Generates an HTML quality-control report covering preprocessing diagnostics and fitted metric distributions.

**Config**
```yaml
relaxometry:
  qc:
    enabled: true
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `relaxometry.qc.enabled` | bool | false | Generate HTML QC report |

## Output Directory Layout

```
derivatives/qmri-neuropipe/sub-XX/ses-YY/
  relax/
    sub-XX_ses-YY_desc-brain-mask.nii.gz        # 3D brain mask
    sub-XX_ses-YY_desc-spgrref_VFA.nii.gz       # SPGR reference image
    models/
      DESPOT1/
        sub-XX_ses-YY_model-DESPOT1_T1.nii.gz
        sub-XX_ses-YY_model-DESPOT1_M0.nii.gz
      DESPOT2FM/
        sub-XX_ses-YY_model-DESPOT2FM_T2.nii.gz
        sub-XX_ses-YY_model-DESPOT2FM_M0.nii.gz
        sub-XX_ses-YY_model-DESPOT2FM_F0.nii.gz
      mcDESPOT/
        sub-XX_ses-YY_model-mcDESPOT_VFm.nii.gz
        sub-XX_ses-YY_model-mcDESPOT_T1_fast.nii.gz
        sub-XX_ses-YY_model-mcDESPOT_T1_slow.nii.gz
        sub-XX_ses-YY_model-mcDESPOT_T2_fast.nii.gz
        sub-XX_ses-YY_model-mcDESPOT_T2_slow.nii.gz
        sub-XX_ses-YY_model-mcDESPOT_Tau.nii.gz
    atlases/
      sub-XX_ses-YY_desc-JHU_labels.nii.gz
    statistics/
      sub-XX_ses-YY_desc-JHU_stats.csv
      sub-XX_ses-YY_desc-roiStats_stats.csv
    normalization/
      sub-XX_ses-YY_space-MNI_model-DESPOT1_T1.nii.gz
      sub-XX_ses-YY_space-MNI_model-DESPOT2FM_T2.nii.gz
      sub-XX_ses-YY_space-MNI_model-mcDESPOT_VFm.nii.gz
      ...
```

## Resume Capability

The pipeline includes smart resume logic to avoid re-running completed steps:

- **Preprocessing**: Skipped if preprocessed outputs (`desc-preproc`) already exist.
- **B1 Mapping**: Skipped if a `TB1map` file is already present.
- **Model Fitting**: Each model is skipped independently if its output maps are already present in the output directory.
- **Normalization**: Skipped if normalized outputs already exist.

Use `rerun_from_step` to keep earlier cached outputs but force a selected step and all later relaxometry stages to run again.

```yaml
relaxometry:
  preprocessing:
    rerun_from_step: motion_correction
```

You can also set the option under `relaxometry.modeling`, `relaxometry.normalization`, or `relaxometry.analysis` when the rerun point is in those stages. Accepted aliases include `force_from_step`, `start_at_step`, and `resume_from_step`. Common relaxometry step names include `reorient`, `denoise`, `degibbs`, `gibbs`, `motion_correction`, `brain_masking`, `acqparams`, `b1`, `modeling`, `normalization`, `atlas`, and `stats`.

## Naming Conventions

All outputs follow BIDS-derivative naming conventions:

| Output | Filename pattern |
| --- | --- |
| Brain mask | `sub-XX[_ses-YY]_desc-brain-mask.nii.gz` |
| B1 map | `sub-XX[_ses-YY]_TB1map.nii.gz` |
| SPGR reference | `sub-XX[_ses-YY]_desc-spgrref_VFA.nii.gz` |
| DESPOT1 T1 | `sub-XX[_ses-YY]_model-DESPOT1_T1.nii.gz` |
| DESPOT1-HIFI T1 | `sub-XX[_ses-YY]_model-DESPOT1HIFI_T1.nii.gz` |
| DESPOT2 T2 | `sub-XX[_ses-YY]_model-DESPOT2_T2.nii.gz` |
| DESPOT2FM T2 | `sub-XX[_ses-YY]_model-DESPOT2FM_T2.nii.gz` |
| mcDESPOT VFm | `sub-XX[_ses-YY]_model-mcDESPOT_VFm.nii.gz` |
| MNI-space map | `sub-XX[_ses-YY]_space-MNI_model-<MODEL>_<metric>.nii.gz` |

## Complete Configuration Example

The following example enables DESPOT1, DESPOT2FM, and mcDESPOT fitting with full preprocessing, normalization to MNI space, and JHU atlas ROI analysis.

```yaml
relaxometry:
  preprocessing:
    reorient:
      enabled: true

    denoising:
      enabled: true
      method: mrtrix
      patch_radius: 2
      block_radius: 5

    degibbs:
      enabled: true
      method: mrtrix

    exclude_indices:
      spgr: []      # no exclusions
      ssfp: []

    spgr_reference:
      mode: mean

    b1:
      method: afi
      smoothing_fwhm: 3.0

    motion_correction:
      enabled: true
      method: ants

    brain_masking:
      enabled: true
      method: synthstrip

  modeling:
    despot1:
      enabled: true
      use_hifi: false
      algo: lsq
      nthreads: 8

    despot2:
      enabled: false    # using DESPOT2FM instead

    despot2fm:
      enabled: true
      algo: src
      nthreads: 8

    mcdespot:
      enabled: true
      cuda: false
      algo: src
      nthreads: 8

  normalization:
    enabled: true
    template: /data/templates/MNI152_T1_1mm.nii.gz
    driving_metric: spgr_ref
    space_name: MNI
    tool: ants
    transform_type: SyN
    save_transforms: true
    include_all_metrics: true

  analysis:
    enabled: true
    registration_target: reference
    # JHU ships with an FA-contrast template; override with MNI T1w so
    # registration to the T1w-contrast SPGR reference succeeds.
    registration_template: /data/templates/MNI152_T1_1mm.nii.gz
    metrics:
      - T1
      - T2
      - VFm
      - T2_fast
      - T2_slow
    atlases:
      JHU:
        labels: /data/atlases/JHU-ICBM-labels-1mm.nii.gz
        template: /data/atlases/JHU-ICBM-FA-1mm.nii.gz
        lut: /data/atlases/JHU_WhiteMatter_labels.txt

  qc:
    enabled: true
```

## Study Tracker

Per-subject processing status (queued, running, completed, failed) and output paths are recorded automatically in the study tracker database. See [Study Tracker](../study_tracker.md) for details on querying run history and re-queuing failed subjects.
