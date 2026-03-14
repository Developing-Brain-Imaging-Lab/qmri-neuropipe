# Relaxometry Workflow

The Relaxometry Workflow is designed to process Variable Flip Angle (VFA) data, typically consisting of SPGR and SSFP sequences, to generate quantitative maps such as T1, T2, M0, B1, and in the case of mcDESPOT, Myelin Water Fraction (MWF).

See [Tool Reference](../tool_reference.md) for the full list of tools and config keys.

## Workflow Steps

### 1. Preprocessing

The preprocessing stage ensures that all input images are artifact-free and aligned.

- **Denoising**: Applies MP-PCA denoising (via MRtrix3 `dwidenoise` or similar) to all raw input images (SPGR, SSFP, IR-SPGR).
- **Gibbs Ringing Correction**: Removes Gibbs ringing artifacts (via MRtrix3 `mrdegibbs`).
- **Reorientation**: Ensures all images are in logical alignment (e.g., LAS/RAS) if requested.
- **Reference Selection**: The SPGR image with the highest Flip Angle is selected as the **Reference Image** for motion correction and alignment.
- **B1 Mapping**:
    - If **AFI** (Actual Flip-Angle Imaging) data is provided:
        - Calculates the B1 map from the AFI pair.
        - Registers the B1 map to the SPGR Reference image.
    - If a pre-computed B1 map is provided:
        - It is registered/resampled to the SPGR Reference space.
- **Motion Correction**:
    - All SPGR and SSFP volumes are rigidly registered to the SPGR Reference image.
    - If inputs are 4D (multiple flip angles in one file), they are split, registered individually, and re-merged.

**Available tools**
*   Denoising: `mrtrix`, `ants`, `mppca`, `patch2self`, `nlmeans`, `wavelets`, `gaussian`
*   Gibbs: `mrtrix`, `dipy`
*   Reorient: `fsl` (fslreorient2std)
*   Motion correction: `ants`, `fsl`

**Config**
```yaml
relaxometry:
  preprocessing:
    reorient:
      enabled: true
    denoising:
      enabled: true
      method: mrtrix
    degibbs:
      enabled: true
      method: mrtrix
    motion_correction:
      enabled: true
      method: ants
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `relaxometry.preprocessing.reorient.enabled` | bool | false | Enable reorientation |
| `relaxometry.preprocessing.denoising.enabled` | bool | false | Enable denoising |
| `relaxometry.preprocessing.denoising.method` | str | `mrtrix` | `mrtrix`, `ants`, `mppca`, `patch2self`, `nlmeans`, `wavelets`, `gaussian` |
| `relaxometry.preprocessing.denoising.patch_radius` | int | 2 | MP-PCA |
| `relaxometry.preprocessing.denoising.block_radius` | int | 5 | MP-PCA |
| `relaxometry.preprocessing.denoising.mask_dilation` | int | 2 | Temporary mask dilation |
| `relaxometry.preprocessing.denoising.pca_method` | str | `eig` | MP-PCA |
| `relaxometry.preprocessing.denoising.model` | str | `ridge` | Patch2Self |
| `relaxometry.preprocessing.degibbs.enabled` | bool | false | Enable Gibbs |
| `relaxometry.preprocessing.degibbs.method` | str | `mrtrix` | `mrtrix`, `dipy` |
| `relaxometry.preprocessing.motion_correction.enabled` | bool | false | Enable motion correction |
| `relaxometry.preprocessing.motion_correction.method` | str | `ants` | `ants`, `fsl` |

### 2. Brain Masking

- **Timing**: Performed immediately after Motion Correction.
- **Input**: The motion-corrected SPGR Reference image.
- **Output**: A binary brain mask (`sub-XX_desc-brain-mask.nii.gz`).
- **Usage**: This mask is passed to all subsequent fitting steps to restrict computation to brain voxels, significantly speeding up processing and ensuring clean outputs.

**Available tools**
*   `fsl` (bet)
*   `mrtrix` (dwi2mask)
*   `ants` (antsBrainExtraction)
*   `freesurfer` (mri_watershed)
*   `synthstrip` (mri_synthstrip)
*   `hd-bet` (HD-BET)

**Config**
```yaml
relaxometry:
  masking:
    enabled: true
    method: fsl
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `relaxometry.masking.enabled` | bool | false | Enable masking |
| `relaxometry.masking.method` | str | `fsl` | `mrtrix`, `fsl`, `ants`, `freesurfer`, `synthstrip`, `hd-bet` |

### 3. Model Fitting (DESPOT)

The pipeline supports both DESPOT1 (T1 mapping) and DESPOT2 (T2 mapping), including High-Fidelity (HIFI) and multi-component (mcDESPOT) variants.

- **DESPOT1 (T1 Mapping)**:
    - **Inputs**: SPGR images (multiple flip angles).
    - **Acquisition Params JSON**: Includes `FlipAngle`, `RepetitionTime`, and `EchoTime` from the SPGR sidecars.
    - **HIFI Option**: If an IR-SPGR (Inversion Recovery) image is present, **DESPOT1-HIFI** is used to simultaneously estimate T1 and B1 (or correct B1).
    - **Outputs**: `T1map`, `M0map`, `B1map` (if HIFI).
    
- **DESPOT2 (T2/MWF Mapping)**:
    - **Inputs**: SSFP images (multiple flip angles, usually 0 and 180 phase cycling).
    - **Acquisition Params JSON**: Includes `FlipAngle`, `RepetitionTime`, `EchoTime`, and `PhaseCycling` from the SSFP sidecars.
    - **Dependencies**: Requires the T1 map from the DESPOT1 step.
    - **Standard DESPOT2**: Estimates single-component T1 and T2.
    - **mcDESPOT (Multicomponent)**: Estimates Fast and Slow water components to derive **Myelin Water Fraction (MWF)**.
    - **Outputs**: `T2map`, `MWFmap`, `Taumap` (Residence time), `OffRes` (Off-resonance frequency).

### 4. Post-Processing & Registration

- **Coregistration**: 
    - The processing is performed in the subject's native space (defined by the SPGR Reference).
    - A transformation matrix is calculated to register the SPGR Reference to the subject's high-resolution **T1w Anatomical** image (`sub-XX_T1w.nii.gz`), if provided.
- **Normalization**:
    - If a template warp (T1w -> MNI) is available, all quantitative maps are normalized to standard space.
- **ROI Stats**:
    - If a segmentation file is available (in the same space), mean values for all quantitative metrics are extracted per ROI.

## Resume Capability

The pipeline includes smart resume logic:
- **Motion Correction**: Skipped if preprocessed outputs (`desc-preproc`) already exist.
- **B1 Mapping**: Skipped if `TB1map` exists.
- **Fitting**: DESPOT1/2 fitting is skipped if the final maps (`T1map`, `T2map`) are already present in the output directory.

## Configuration

The workflow is configured via the `relaxometry` section in your config YAML:

```yaml
relaxometry:
  preprocessing:
    denoising:
      enabled: true
      method: mrtrix
    motion_correction:
      enabled: true
      method: ants
    b1:
      method: afi # or 'external'

  modeling:
    despot1:
      enabled: true
      use_hifi: true # Requires IR-SPGR
    despot2:
      enabled: true
      mcdespot: true # Enable Myelin Water Fraction mapping
```

**B1 Mapping Tools**
*   `afi` (AFI-derived B1)
*   `external` (provided B1 map)
*   `hifi` (DESPOT1-HIFI)

The generated `desc-AcqParams.json` file also carries `EchoTime` for SPGR, SSFP, and IR-SPGR inputs when that field is present in the BIDS sidecars.

**B1 Config**
```yaml
relaxometry:
  preprocessing:
    b1:
      method: afi   # afi | external | hifi
      smoothing_fwhm: 0.0
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `relaxometry.preprocessing.b1.method` | str | `afi` | `afi`, `external`, `hifi` |
| `relaxometry.preprocessing.b1.smoothing_fwhm` | float | 0.0 | Optional smoothing |

## Naming Conventions

All outputs follow BIDS-like naming conventions:
- **B1 Map**: `sub-XX[_ses-YY]_TB1map.nii.gz`
- **T1 Map**: `sub-XX[_ses-YY]_despot1_hifi_T1map.nii.gz`
- **MWF Map**: `sub-XX[_ses-YY]_despot2_fm_MWFmap.nii.gz`
- **Brain Mask**: `sub-XX[_ses-YY]_desc-brain-mask.nii.gz`
