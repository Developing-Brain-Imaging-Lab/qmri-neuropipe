# Standard Operating Procedure: Processing MRI Data with qmri-neuropipe

**SOP ID:** QNP-SOP-001  
**Version:** 1.3
**Effective date:** 2026-08-06  
**Owner:** Study PI or designated pipeline lead  
**Applies to:** Students and staff using qmri-neuropipe locally or with Apptainer/Singularity

## Table of contents

- [1. Purpose and how to use this manual](#1-purpose-and-how-to-use-this-manual)
- [2. Scope](#2-scope)
- [3. Roles and responsibilities](#3-roles-and-responsibilities)
- [4. Required inputs and access](#4-required-inputs-and-access)
- [5. Safety and data-integrity controls](#5-safety-and-data-integrity-controls)
- [6. Getting started: general workflow](#6-getting-started-general-workflow)
- [7. Anatomical processing manual](#7-anatomical-processing-manual)
- [8. Diffusion processing manual](#8-diffusion-processing-manual)
- [9. Relaxometry and mcDESPOT processing manual](#9-relaxometry-and-mcdespot-processing-manual)
- [10. Standard-space normalization manual](#10-standard-space-normalization-manual)
- [11. Tractography, tractometry, and connectome manual](#11-tractography-tractometry-and-connectome-manual)
- [12. Atlas registration and ROI extraction manual](#12-atlas-registration-and-roi-extraction-manual)
- [13. Study tracker manual](#13-study-tracker-manual)
- [14. Validation and execution](#14-validation-and-execution)
- [15. Quality control](#15-quality-control)
- [16. Cohort expansion and closeout](#16-cohort-expansion-and-closeout)
- [17. Acceptance criteria](#17-acceptance-criteria)
- [18. Deviations, failures, and reruns](#18-deviations-failures-and-reruns)
- [19. Required records and training sign-off](#19-required-records-and-training-sign-off)
- [20. References](#20-references)
- [21. Revision history](#21-revision-history)

## 1. Purpose and how to use this manual

Define a reproducible, auditable procedure for configuring, validating, running, quality-checking, and documenting qmri-neuropipe processing of BIDS-formatted anatomical MRI, diffusion MRI, and relaxometry data, including normalization, tractography, atlas/ROI analysis, mcDESPOT workflows, and cohort study tracking.

New operators should begin with Section 6, complete one supervised pilot using Sections 14–15, and then read the manual chapter for every enabled processing branch. Experienced operators may use the table of contents to jump directly to a modality or downstream-analysis chapter. Configuration examples are teaching templates; the approved study protocol and field-level configuration reference remain authoritative.

## 2. Scope

This SOP covers participant-level anatomical (`anat`), diffusion (`dmri`), and relaxometry (`relaxometry`) processing with a local qmri-neuropipe installation or the supported container wrapper. It includes standard-space normalization, diffusion tractography/tractometry, atlas registration and ROI statistics, DESPOT1/mcDESPOT fitting, and the study tracker. The SOP covers YAML preparation and routine execution, but it does not prescribe acquisition-specific scientific parameters. The study PI or pipeline lead must approve protocol choices such as distortion correction, registration, masking, normalization targets, tractography, atlas selection, gradient correction, B1 correction, and quantitative models.

## 3. Roles and responsibilities

- **Operator:** confirms authorization, protects source data, prepares the YAML, performs validation and pilot processing, reviews QC, and records the run.
- **Pipeline lead:** approves the study YAML and software/container version, resolves technical failures, and approves changes to processing logic.
- **Scientific lead/PI:** approves acquisition-specific methods and determines whether QC findings require exclusion or reprocessing.
- **Data steward:** defines storage, access, retention, and backup requirements.

## 4. Required inputs and access

- Read access to the BIDS dataset.
- Write access to dedicated output and work directories.
- A study-approved pipeline YAML.
- BIDS inputs required by the selected track: T1w/T2w for anatomical processing; DWI with `.bval`, `.bvec`, sidecar metadata, and appropriate distortion-correction inputs for diffusion processing; or SPGR plus the required SSFP, IR-SPGR, and B1/AFI inputs for the selected relaxometry models.
- Either a working local installation and required external tools, or a lab-approved `.sif` image and Apptainer/Singularity.
- FreeSurfer license, atlas, template, GNL coefficient file, model weights, or GPU access when required by enabled steps.
- Sufficient CPU, memory, runtime allocation, and disk space.
- Approved standard-space templates, label atlases, lookup tables, tract definitions, and parcellations when normalization, ROI extraction, tractometry, or connectomes are enabled.

## 5. Safety and data-integrity controls

1. Treat the BIDS input as read-only. Never write derived files into raw source folders.
2. Use distinct output and work directories. Do not set either equal to `bids_dir`.
3. Process only data for which the operator is authorized.
4. Do not store PHI, credentials, access tokens, or private keys in YAML or logs.
5. Preserve configuration, logs, reports, provenance, and software/container identifiers with the derivatives.
6. Do not delete failed outputs until logs and provenance have been reviewed and the pipeline lead authorizes cleanup.

## 6. Getting started: general workflow

The standard sequence is: inventory the BIDS data, choose local or container execution, create one version-controlled YAML for the selected pipeline, validate it, run one participant/session, complete visual and technical QC, and only then expand to the cohort. Sections 14–16 contain the execution commands and sign-off sequence.

### 6.1 Record the run identity

Before processing, create a run record containing operator, date/time, study, participant/session scope, YAML filename and revision, qmri-neuropipe version, container image path/digest if applicable, and intended output/work locations.

```bash
qmri-neuropipe --version
```

### 6.2 Verify the execution environment

For a local installation, activate the approved Python environment and verify required commands for every enabled step.

```bash
conda activate qmri
qmri-neuropipe --version
command -v flirt
command -v dwidenoise
command -v antsRegistration
```

For a container run, verify the runtime and image.

```bash
apptainer --version
ls -lh /path/to/qmri-neuropipe.sif
```

### 6.3 Verify input data and scope

Confirm `dataset_description.json`, participant/session directories, required NIfTI files, JSON sidecars, and DWI gradient files are present. Inspect the dataset:

```bash
qmri-neuropipe inspect /study/rawdata --by-session
```

Confirm participant and session labels against the inventory. Resolve missing, ambiguous, or non-BIDS inputs before continuing.

### 6.4 Prepare and review the pipeline YAML

Required top-level paths are `bids_dir` and `output_dir`; `work_dir` is strongly recommended. Specify resources, run controls, the pipeline, and approved workflow steps. Use absolute paths and consistent space indentation. Duplicate keys are prohibited. CLI overrides take precedence over YAML.

Minimum structure:

```yaml
bids_dir: /study/rawdata
output_dir: /study/derivatives/qmri-neuropipe
work_dir: /scratch/my_user/qmri-neuropipe-work
n_cpus: 8
memory_gb: 32
use_gpu: false
skip_existing: true
stop_on_error: true
pipeline: dmri

dmri:
  preprocessing:
    denoising:
      enabled: true
      method: mrtrix
  modeling:
    dti:
      enabled: true
      method: dipy
      metrics: [fa, md, ad, rd]
```

The operator and pipeline lead must confirm that enabled methods match the acquisition and scientific plan. When multiple anatomical images exist, configure strict `anat.input.t1w_match` or `anat.input.t2w_match` selectors.

#### 6.4.1 Select one processing track per run

Set the top-level `pipeline` to `anat`, `dmri`, or `relaxometry`. A study may run more than one track, but each invocation selects one pipeline. Use a separate version-controlled YAML per approved protocol when the settings differ materially.

### 6.5 Quick-start decision guide

| If the study needs… | Start with… | Then review… |
| --- | --- | --- |
| T1w/T2w preprocessing, masks, segmentation, or reconstruction | `pipeline: anat` | Sections 7 and 10–13 as enabled |
| DWI correction, tensor/FOD models, or tractography | `pipeline: dmri` | Sections 8 and 10–13 as enabled |
| SPGR/SSFP quantitative mapping or mcDESPOT | `pipeline: relaxometry` | Sections 9, 10, 12, and 13 as enabled |
| A containerized run | Section 6.2 container checks | Section 14 container validation/execution |
| A local run | Section 6.2 local checks | Section 14 local validation/execution |

Before leaving this chapter, the operator should be able to identify the selected pipeline, BIDS scope, output/work locations, execution method, enabled optional branches, and the responsible reviewer.

## 7. Anatomical processing manual

### 7.1 Purpose and required inputs

An anatomical run requires the selected T1w and any optional T2w input. Review input selection carefully when multiple acquisitions or runs exist.

Required inputs are a BIDS T1w image and sidecar; optional inputs include T2w images, a FreeSurfer license, study templates, and label atlases. When more than one candidate image exists, configure strict `anat.input.t1w_match` or `anat.input.t2w_match` entities so selection fails safely instead of choosing an unintended acquisition.

### 7.2 Recommended processing sequence

1. Confirm the selected T1w/T2w acquisition and orientation.
2. Reorient, denoise, and correct intensity nonuniformity as approved.
3. Create and visually inspect the brain mask.
4. Run optional T1w–T2w coregistration, reconstruction, or segmentation.
5. Run normalization and atlas/ROI analysis only after native-space anatomy passes QC.
6. Preserve derivatives, sidecars, logs, provenance, and QC decisions.

### 7.3 Starter YAML

```yaml
pipeline: anat

anat:
  input:
    t1w_match:
      entities:
        acq: MPRAGE
  preprocessing:
    reorient:
      enabled: true
    denoising:
      enabled: true
      method: ants
    bias_correction:
      enabled: true
      method: ants
    brain_masking:
      enabled: true
      method: synthstrip
    recon_all:
      enabled: false
      method: standard
    normalization:
      enabled: false
      method: ants
```

Enable reconstruction, normalization, segmentation, or other optional steps only when they are part of the approved protocol and their required tools, templates, and licenses are available.

### 7.4 Expected outputs and QC

Expect preprocessed anatomical images, masks, and provenance plus reconstruction, segmentation, transform, or ROI products for enabled branches. Review input selection, orientation, residual bias field, tissue contrast, mask boundaries, T1w–T2w alignment, surface/segmentation plausibility, and regional volumes. Normalization and atlas checks are detailed in Sections 10 and 12.

## 8. Diffusion processing manual

### 8.1 Purpose and required inputs

A diffusion run requires DWI NIfTI data, JSON sidecars, `.bval`, and `.bvec` files. Distortion-correction settings must match the acquired phase-encoding and field-map strategy.

Before configuring correction, inventory phase-encoding direction, total readout time, shell structure, reverse-PE or field-map inputs, gradient tables, and any gradient-nonlinearity coefficients. The scientific lead must approve how missing distortion-correction inputs are handled.

### 8.2 Recommended processing sequence

1. Verify gradients, b-values, metadata, and distortion-correction inputs.
2. Apply denoising and Gibbs correction before interpolation-heavy steps.
3. Perform susceptibility, motion, and eddy-current correction using acquisition-compatible settings.
4. Apply bias correction, coregistration, masking, and approved gradient corrections.
5. Fit DTI or advanced models and inspect maps before tractography or ROI analysis.
6. Preserve corrected gradients, model maps, QC, logs, and provenance.

### 8.3 Starter YAML

```yaml
pipeline: dmri

dmri:
  preprocessing:
    denoising:
      enabled: true
      method: mrtrix
    degibbs:
      enabled: true
      method: mrtrix
    distcorr:
      method: topup
      fallback: true
    motion_correction:
      method: eddy
    eddy:
      enabled: true
      method: eddy
    bias_correction:
      enabled: true
      method: ants
    brain_masking:
      enabled: true
      method: mrtrix
  modeling:
    dti:
      enabled: true
      method: dipy
      fit_method: WLLS
      metrics: [fa, md, ad, rd]
```

Do not copy the example distortion-correction method into a study YAML unless the input data support it. Confirm phase-encoding direction, total readout time, reverse-PE or field-map availability, gradient orientation, and any gradient-nonlinearity coefficient/metadata requirements.

### 8.4 Expected outputs and QC

Expect corrected DWI data and gradients, masks, registration products, requested model maps, logs, and provenance. Review residual susceptibility distortion, motion/outlier handling, gradient orientation, mask coverage, anatomical alignment, and spatial/quantitative plausibility of FA, MD, AD, RD, FOD, or other enabled outputs. Tractography and ROI checks are detailed in Sections 11 and 12.

## 9. Relaxometry and mcDESPOT processing manual

### 9.1 Purpose and input discovery

Relaxometry inputs are discovered from BIDS anatomical/field-map data using acquisition or description labels. SPGR files must contain `spgr` in `acq` or `desc`; SSFP files must contain `ssfp`; IR-SPGR files must identify both IR and SPGR; AFI inputs may use `afi`, and B1 maps may use a B1 suffix. Confirm discovery with a pilot run before cohort processing.

Record flip angles, repetition/echo times, SSFP phase cycling, B1 acquisition, and intentional volume exclusions. mcDESPOT requires SPGR and SSFP data, an enabled DESPOT1-derived T1 map, and a B1 map from AFI/external B1 or DESPOT1-HIFI.

### 9.2 Recommended processing sequence

1. Confirm that all SPGR, SSFP, IR-SPGR, and AFI/B1 inputs are discovered correctly.
2. Reorient and apply approved denoising/Gibbs correction.
3. Motion-correct within and across series and generate the SPGR reference.
4. Estimate/register B1 and inspect its units, scale, and coverage.
5. Create the analysis mask and run DESPOT1 before dependent models.
6. Run mcDESPOT, generate QC, and review every parameter map for failed fits.

### 9.3 Starter YAML

```yaml
pipeline: relaxometry

relaxometry:
  preprocessing:
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
      transform_type: Rigid
      nthreads: 8
    spgr_reference:
      mode: mean
    b1:
      method: afi
      smoothing_fwhm: 0.0
      registration:
        method: ants

  masking:
    enabled: true
    method: synthstrip

  modeling:
    despot1:
      enabled: true
      use_hifi: false
      algo: lsq
      nthreads: 8
    despot2:
      enabled: false
    despot2fm:
      enabled: false
    mcdespot:
      enabled: true
      cuda: false
      nthreads: 8

  qc:
    enabled: true
```

mcDESPOT requires SPGR and SSFP data, a T1 map produced by an enabled DESPOT1 fit, and a B1 map from AFI/external B1 or DESPOT1-HIFI. Set `despot1.use_hifi: true` only when IR-SPGR data are present. Set `mcdespot.cuda: true` only on an approved GPU system with the CUDA fitter available. Use `relaxometry.modeling.mcdespot.enabled`; the older `despot2.mcdespot` switch is deprecated.

Optional zero-based SPGR/SSFP volume exclusions belong under `relaxometry.preprocessing.exclude_indices`, or may be supplied temporarily with `--spgr-exclude` and `--ssfp-exclude`. Exclusions require documented QC justification.

### 9.4 Expected outputs and QC

Expect the SPGR reference, registered series/B1 products, mask, DESPOT1 T1/M0 maps, mcDESPOT parameter maps (`VFm`, `T1_fast`, `T1_slow`, `T2_fast`, `T2_slow`, and `Tau`), logs, provenance, and a QC report when enabled. Inspect completeness, units/scale, anatomy, tissue contrast, boundary artifacts, and failed-fit regions. A GPU result must be scientifically checked against the approved implementation and not accepted solely because the fitter completed.

## 10. Standard-space normalization manual

### 10.1 Purpose and safeguards

Normalization estimates a transform from a native subject image to an approved standard-space template, publishes transforms when requested, and can apply them to model metrics. Template contrast and population must match the driving image and study population. Never assume that an MNI T1w template is appropriate for FA-driven diffusion registration or that a pediatric template is appropriate for an adult cohort.

### 10.2 Configuration examples

Anatomical normalization is configured under `anat.preprocessing.normalization`:

```yaml
anat:
  preprocessing:
    normalization:
      enabled: true
      template: /atlases/MNI152_T1_1mm.nii.gz
      method: ants
      save_transforms: true
      space_entity: MNI152
      options:
        transform_type: SyN
        interpolation: linear
```

Diffusion and relaxometry normalization use modality-level blocks. Diffusion defaults to an FA driving metric; relaxometry defaults to the SPGR reference.

```yaml
dmri:
  normalization:
    enabled: true
    template: /atlases/FMRIB58_FA_1mm.nii.gz
    driving_metric: FA
    tool: ants
    transform_type: SyN
    space_name: MNI
    save_transforms: true
    include_all_metrics: true

relaxometry:
  normalization:
    enabled: true
    template: /atlases/MNI152_T1_1mm.nii.gz
    driving_metric: spgr_ref
    tool: ants
    transform_type: SyN
    space_name: MNI
    save_transforms: true
    include_all_metrics: true
```

Supported diffusion/relaxometry normalization tools include `ants`, `synthmorph`, and `robust_iterative`; use only the lab-approved method. For container runs, bind every template path or place templates beneath an existing container-visible mount. Confirm forward/inverse transform direction before reuse. Scalar metric maps normally use linear interpolation; integer label maps require label-safe interpolation such as `genericLabel` or nearest-neighbor.

### 10.3 Expected outputs and QC

Expect forward/inverse transforms when requested and standard-space driving images/metrics. Overlay native and standard images in multiple planes; inspect cortex, ventricles, cerebellum, brainstem, and field-of-view boundaries. Confirm transform direction and interpolation by data type, and reject warped labels containing non-integer interpolation artifacts.

## 11. Tractography, tractometry, and connectome manual

### 11.1 Scientific prerequisites

Tractography is configured only under `dmri.modeling.tractography`. Tracking method, diffusion model, seeding, anatomical constraints, filtering, streamline count, length limits, and bundle definitions must be fixed by the approved protocol. FOD-based algorithms such as iFOD2 require an appropriate CSD/FOD model. Anatomically constrained tractography requires an aligned anatomical image and a valid 5TT segmentation.

Verify MRtrix commands are available. TractSeg and pyAFQ require their corresponding optional software/dependency sets in a local installation; confirm availability inside the selected container before enabling them.

### 11.2 Whole-brain tractography example

```yaml
dmri:
  modeling:
    csd:
      enabled: true
      method: msmt_csd
    tractography:
      mrtrix:
        enabled: true
        algorithm: iFOD2
        select: 1000000
        cutoff: 0.06
        minlength: 10
        maxlength: 250
        act:
          enabled: true
          algorithm: fsl
          validate: true
          seed_gmwmi: true
          backtrack: true
          crop_at_gmwmi: true
        filtering:
          method: sift2
```

The one-million-streamline value above is a teaching/pilot example, not a universal default. The approved cohort protocol may require a different fixed count.

### 11.3 Bundle and along-tract analysis

Optional bundle and along-tract analysis may sample already-produced maps using `MODEL.METRIC` identifiers:

```yaml
dmri:
  modeling:
    tractography:
      tractseg:
        enabled: true
        options:
          preprocess: true
      tract_specific:
        enabled: true
        bundles:
          - {name: CST_L, source: tractseg}
          - {name: CST_R, source: tractseg}
        metrics: [DTI.FA, DTI.MD]
        streamline_statistic: mean
        profiles:
          enabled: true
          nodes: 100
        track_density:
          enabled: true
```

### 11.4 Connectome example

Connectomes require an integer node/parcellation image already aligned to DWI/tractogram space:

```yaml
tract_specific:
  enabled: true
  bundles: []
  connectome:
    enabled: true
    nodes: /path/to/subject_space-dwi_desc-nodes_dseg.nii.gz
    use_sift2: true
    statistic: sum
    symmetric: true
    zero_diagonal: true
```

Do not interpret streamline count or connection weight as a literal axon count. Tracking, filtering, seeding, parcel geometry, registration, and image quality all affect the estimate.

### 11.5 Expected outputs and QC

Expect tractograms, optional SIFT/SIFT2 weights, bundle files, along-tract profiles, track-density maps, or connectome matrices for enabled branches. Check FOD/tensor inputs, masks, 5TT/GMWMI alignment, streamline coverage/length, endpoints, empty bundles, profile coverage, parcellation alignment, symmetry/diagonal settings, and cohort outliers.

## 12. Atlas registration and ROI extraction manual

### 12.1 Purpose and configuration

Atlas/ROI analysis registers one or more atlases into native subject space and extracts mean, median, and standard deviation from selected metric maps. Configure it under `dmri.analysis`, `relaxometry.analysis`, or `anat.segmentation`. Named atlas definitions are preferred over the deprecated flat single-atlas format.

```yaml
dmri:
  analysis:
    enabled: true
    registration_target: FA
    atlases:
      JHU:
        labels: /atlases/JHU-ICBM-labels-1mm.nii.gz
        template: /atlases/JHU-ICBM-FA-1mm.nii.gz
        lut: /atlases/JHU-labels.xml
        interpolation: genericLabel
    metrics: [FA, MD, AD, RD]

relaxometry:
  analysis:
    enabled: true
    registration_target: reference
    registration_template: /atlases/MNI152_T1_1mm.nii.gz
    atlases:
      JHU:
        labels: /atlases/JHU-ICBM-labels-1mm.nii.gz
        lut: /atlases/JHU-labels.xml
        interpolation: genericLabel
    metrics:
      DESPOT1: [T1]
      mcDESPOT: [VFm, T1_fast, T1_slow, T2_fast, T2_slow, Tau]
```

The atlas template must match the contrast and space used to estimate registration. A discrete label atlas must use label-safe interpolation; linear interpolation can create invalid label values. Set `is_probabilistic: true` only for a 4D probabilistic atlas. When a LUT is available, include it so ROI outputs carry human-readable region names.

Expected outputs include registered label images beneath `atlases/`, per-atlas CSV files beneath `statistics/`, and a combined ROI statistics CSV. The standard columns include ROI name, model, metric, mean, median, and standard deviation. If no `metrics`/`parameters` filter is supplied, all available produced maps may be analyzed; explicit selection is preferred for controlled studies.

### 12.2 ROI verification and interpretation

Overlay every registered atlas on the actual registration target, checking laterality, tissue boundaries, coverage, and label integrity. Confirm the LUT, ROI count, model/metric selectors, and expected CSV columns. Treat all-zero, missing, infinite, NaN, or extreme values as review flags. ROI summaries inherit bias from preprocessing, registration, masks, partial volume, atlas definitions, and metric fitting; they are not a substitute for image-level QC.

## 13. Study tracker manual

### 13.1 Purpose and configuration

The study tracker accumulates cross-subject processing status, QC metrics, model completion, atlas/ROI metrics, and anatomical volume statistics in an Excel workbook. Each populated sheet is also exported as a cohort CSV beneath a neighboring `tracker_reports/` directory.

For a local installation, install the tracker dependency set if it is not already present: `pip install -e ".[tracker]"`.

```yaml
study_name: MyStudy

tracker:
  enabled: true
  file: /study/derivatives/qmri-neuropipe/study_tracker.xlsx
```

Set `tracker.file` explicitly for a new workbook. For a local run, use a writable host path. For a container run, use a path visible inside the container, normally `/out/study_tracker.xlsx`; non-core paths in the YAML are not automatically rewritten from host paths. The tracker workbook is the authoritative record; exported CSV files are regenerated from it.

Common sheets include `Summary`, `Processing_Status`, `Anatomical_Status`, `Diffusion_Status`, `Relaxometry_Status`, `Quality_Metrics`, per-atlas `{Atlas}_Metrics`, `Volume_Statistics`, `Processing_Details`, and `Alert_History`. Subjects/sessions are upserted so cached or targeted reruns update existing rows rather than creating duplicates.

Parallel workers serialize writes with a neighboring `.lock` file. If an interrupted process leaves a stale lock, confirm that no tracker writer is active before removing it. A lock timeout can cause a tracker update to be skipped even when image processing succeeds; review logs and confirm that every intended participant/session appears in the workbook and exported CSVs.

### 13.2 Tracker operating procedure

1. Create one study-approved workbook path and keep it stable across cohort runs.
2. Verify output permissions before the pilot and back up the workbook with the derivatives.
3. After each batch, compare intended participant/session pairs with tracker rows and derivative inventories.
4. Review status, QC, atlas, volume, processing-detail, and alert sheets as applicable.
5. Resolve missing rows or lock warnings before declaring the batch complete.
6. Treat the workbook as the authoritative tracker; regenerate cohort CSVs from it rather than editing exports manually.

## 14. Validation and execution

### 14.1 Validate the configuration

Local validation:

```bash
qmri-neuropipe run --config /study/code/qmri_dmri.yaml \
  --participant-label sub-001 --session-label ses-01 \
  --dry-run --verbose
```

Container launch check:

```bash
qmri-neuropipe container \
  --container-image /containers/qmri-neuropipe.sif \
  --config /study/code/qmri_dmri.yaml \
  --participant-label sub-001 --session-label ses-01 \
  --dry-run
```

Container pipeline validation:

```bash
qmri-neuropipe container \
  --container-image /containers/qmri-neuropipe.sif \
  --config /study/code/qmri_dmri.yaml \
  --participant-label sub-001 --session-label ses-01 \
  --extra-arg=--dry-run --extra-arg=--verbose
```

Acceptance criteria: no configuration error, expected core paths and resources, correct `anat`, `dmri`, or `relaxometry` pipeline, expected participant/session scope, and recognition of all required inputs. On a CPU-only host add `--no-nv --no-gpu`.

### 14.2 Run a pilot participant/session

Local execution:

```bash
qmri-neuropipe run --config /study/code/qmri_dmri.yaml \
  --participant-label sub-001 --session-label ses-01
```

Container execution:

```bash
qmri-neuropipe container \
  --container-image /containers/qmri-neuropipe.sif \
  --config /study/code/qmri_dmri.yaml \
  --participant-label sub-001 --session-label ses-01 \
  --n-cpus 8 --memory-gb 32
```

The wrapper mounts BIDS at `/data` read-only, output at `/out`, work at `/work`, and copied support files at `/config`. Use `--bind` for any other referenced path. Provide `--freesurfer-license` when required. A file supplied with `--gnl-coeff-file` is copied into `/config` and must be referenced there by basename in the YAML.

### 14.3 Monitor execution

Monitor terminal/log output, scheduler resource usage, disk space, and failure messages. Do not modify the YAML during an active run. Record unexpected warnings and the last completed step. If a run fails, preserve all artifacts and follow Section 8.

## 15. Quality control

### 15.1 Common technical and visual QC

Before cohort processing, verify the common items:

- expected participant/session derivatives and sidecars exist;
- logs and `provenance.json` record commands and return codes;
- report generation succeeds when enabled;
- image orientation and dimensions are plausible;
- DWI gradients align with the images;
- denoising and Gibbs correction introduce no obvious artifacts;
- distortion and motion/eddy correction are plausible;
- brain masks do not omit brain or include excessive non-brain tissue;
- anatomical-to-DWI registration is anatomically plausible;
- quantitative maps and ROI summaries are finite, spatially plausible, and complete.

Apply the track-specific QC checks below.

### 15.2 Anatomical QC

- Confirm T1w/T2w selection, orientation, bias correction, brain mask coverage, tissue contrast, and cross-contrast coregistration.
- When enabled, review normalization overlays, FreeSurfer reconstruction, segmentation labels, and regional volumes.

### 15.3 Diffusion QC

- Confirm gradient count/orientation, b-value shells, susceptibility correction, motion/eddy results, outlier handling, brain mask, and anatomical coregistration.
- Review DTI or other model maps for expected anatomy, artifacts, missing voxels, and implausible parameter values.

### 15.4 Relaxometry and mcDESPOT QC

- Confirm all expected SPGR/SSFP flip-angle and phase-cycling volumes were discovered and that any exclusions were intentional and documented.
- Review within-series motion correction, SPGR reference generation, B1/AFI registration and scale, masking, and DESPOT1 T1/M0 maps.
- For mcDESPOT, review VFm, T1-fast/T1-slow, T2-fast/T2-slow, and Tau maps for completeness, expected scale, anatomical plausibility, boundary artifacts, and failed-fit regions.
- Confirm the relaxometry QC report was produced when `relaxometry.qc.enabled: true`.

### 15.5 Normalization QC

- Overlay native and standard-space images in multiple planes; inspect cortex, ventricles, cerebellum, brainstem, and field-of-view boundaries.
- Confirm the driving image and template have compatible contrast and population, transforms are saved when required, and inverse/forward directions are labeled correctly.
- Confirm warped scalar maps remain plausible and warped discrete labels contain valid label values without linear-interpolation contamination.

### 15.6 Tractography and tractometry QC

- Confirm FOD/tensor inputs, masks, 5TT/GMWMI alignment, streamline coverage, length distribution, anatomical plausibility, and absence of gross false-positive pathways.
- Confirm SIFT/SIFT2 completion when enabled, bundle counts, along-tract profile coverage, track-density maps, and node/parcellation alignment for connectomes.
- Treat empty bundles, implausible endpoint distributions, and extreme cohort outliers as failures requiring review, not automatic exclusions.

### 15.7 Atlas/ROI analysis QC

- Overlay each native-space registered atlas on its registration target and check label integrity, laterality, tissue boundaries, and coverage.
- Confirm the atlas name, LUT, model/metric selectors, ROI count, expected CSV columns, and absence of unexpected all-zero, missing, infinite, or NaN summaries.

### 15.8 Study tracker QC

- Confirm one current row per intended participant/session, accurate completion states, expected modality/atlas sheets, and agreement with derivative files and provenance.
- Confirm the Excel workbook and cohort CSV exports are writable, readable, backed up, and free of unresolved lock warnings.

Record each QC item as pass, fail, or not applicable. A failed mandatory QC item blocks cohort expansion until reviewed by the pipeline/scientific lead.

## 16. Cohort expansion and closeout

### 16.1 Expand to the approved cohort

After pilot approval, process the approved cohort using the unchanged versioned YAML. Freeze normalization templates, atlas/LUT files, tractography parameters, bundle definitions, parcellations, ROI metric selectors, and tracker location with the protocol. For local execution, use YAML participant/session lists or the approved batch mechanism. For the container wrapper:

```bash
qmri-neuropipe container \
  --container-image /containers/qmri-neuropipe.sif \
  --config /study/code/qmri_dmri.yaml \
  --subjects 001,002,003 --sessions 01,01,01 \
  --keep-going
```

Use `--keep-going` only when independent participant failures may be collected for later review. Otherwise stop at the first failure.

### 16.2 Close out the run

Run an inventory with processing gaps, complete QC records, document exclusions and reruns, and preserve the approved YAML, settings file, logs, reports, provenance, version, and image digest with the derivative release.

```bash
qmri-neuropipe inspect /study --rawdata-dir rawdata \
  --derivatives --processing-gaps --by-session
```

Remove work files only after successful verification and according to the study retention policy.

## 17. Acceptance criteria

A processing run is complete when all intended participant/session pairs are accounted for; mandatory outputs, transforms, tractography/tractometry products, ROI tables, tracker entries, logs, reports, and provenance exist as applicable; mandatory QC items pass or have approved dispositions; configuration and software/container identifiers are archived; and exclusions, deviations, failures, and reruns are documented.

## 18. Deviations, failures, and reruns

1. Record the exact participant/session, failed step, error, command, software/image version, and YAML revision.
2. Determine whether the cause is data, configuration, resources, dependencies, permissions, or a software defect.
3. Obtain pipeline/scientific approval before changing acquisition-sensitive parameters.
4. Create a new version/commit of the YAML for any processing-logic change.
5. Prefer a targeted rerun over deleting all outputs:

```bash
qmri-neuropipe run --config /study/code/qmri_dmri.yaml \
  --participant-label sub-001 --rerun-from-step eddy
```

6. Repeat the relevant QC and document the outcome. Do not overwrite the record of the original failure.

## 19. Required records and training sign-off

### 19.1 Required records

- Run log and operator/date
- Participant/session manifest
- Approved pipeline YAML revision
- Optional container settings YAML
- qmri-neuropipe version and dependency environment or container image digest
- Command line or scheduler submission record
- Logs, reports, and `provenance.json`
- QC checklist and reviewer
- Standard-space template, atlas, LUT, tract/bundle, and parcellation identities/checksums when used
- Study tracker workbook plus exported cohort CSV reports when enabled
- Deviations, exclusions, failure dispositions, and rerun history

### 19.2 Training sign-off

A trainee is authorized for independent routine runs after demonstrating: BIDS inventory review; correct YAML editing without duplicate keys; selection of the correct anatomical, diffusion, or relaxometry pipeline; local or container dry-run validation; one supervised pilot execution; recognition of failed registration, normalization, masking, diffusion correction, tractography, atlas/ROI, B1 correction, and quantitative-map QC; correct interpretation of study-tracker warnings; preservation of logs/provenance; and correct documentation of a targeted rerun.

## 20. References

- `docs/GETTING_STARTED.md`
- `docs/configuration_reference.md`
- `docs/options_reference.md`
- `docs/tool_reference.md`
- `docs/data_organization.md`
- `docs/analysis.md`
- `docs/study_tracker.md`
- `docs/workflows/tractometry.md`
- `examples/configs/example_config.yaml`

## 21. Revision history

| Version | Date | Description |
| --- | --- | --- |
| 1.0 | 2026-08-06 | Initial SOP for local and container-based processing. |
| 1.1 | 2026-08-06 | Added anatomical, diffusion, and relaxometry processing profiles, including mcDESPOT prerequisites and QC. |
| 1.2 | 2026-08-06 | Added normalization, tractography/tractometry, atlas/ROI extraction, and study-tracker procedures and QC. |
| 1.3 | 2026-08-06 | Reorganized the SOP as a linked training manual with a general quick start, standalone modality/analysis chapters, and a navigable table of contents. |
