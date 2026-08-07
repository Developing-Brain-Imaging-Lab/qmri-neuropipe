# Getting Started with qmri-neuropipe

This page is a practical first-run guide for students and lab members processing BIDS-formatted MRI data with **qmri-neuropipe 2.x**. It covers a local installation and the Apptainer/Singularity container wrapper.

> **Recommended first run:** validate the YAML, process one participant/session, review the outputs and QC, and only then expand to a cohort.

## Table of contents

- [1. Choose how you will run](#1-choose-how-you-will-run)
- [2. Check the BIDS input](#2-check-the-bids-input)
- [3. Create the pipeline YAML](#3-create-the-pipeline-yaml)
- [4. Validate before processing](#4-validate-before-processing)
- [5. Run one participant/session](#5-run-one-participantsession)
- [6. Review the result before scaling up](#6-review-the-result-before-scaling-up)
- [7. Resume or deliberately rerun](#7-resume-or-deliberately-rerun)
- [8. Container settings file](#8-container-settings-file-optional)
- [9. Troubleshooting quick reference](#9-troubleshooting-quick-reference)
- [Detailed processing manuals](#detailed-processing-manuals)

## 1. Choose how you will run

### Local installation

Use a local installation when the required neuroimaging software is already installed and available on `PATH`. The core prerequisites are Python 3.10 or newer, FSL, MRtrix3, and ANTs. Optional steps may require additional tools or Python extras.

From the repository root:

```bash
conda create -n qmri python=3.10
conda activate qmri
pip install -e .

qmri-neuropipe --version
```

Install only the extras needed for the planned workflow, for example:

```bash
pip install -e ".[hdbet,tractseg,reporting]"
```

Check required external commands before starting. Adjust the list for the enabled YAML steps:

```bash
command -v flirt
command -v eddy
command -v dwidenoise
command -v antsRegistration
```

### Apptainer/Singularity container

Use the container when you have a lab-approved `.sif` image and Apptainer or Singularity installed. Most software dependencies are inside the image; the host still supplies the data, output storage, optional license/support files, and GPU driver access.

```bash
apptainer --version
ls -lh /path/to/qmri-neuropipe.sif
```

The wrapper binds these host locations to canonical paths inside the container:

| Host location | Container path | Access |
| --- | --- | --- |
| `bids_dir` | `/data` | read-only |
| `output_dir` | `/out` | read/write |
| `work_dir` | `/work` | read/write |
| copied config/support files | `/config` | read-only |

Core path values may remain as host paths in the shared YAML. The wrapper reads them to create the binds, then overrides them inside the container with `/data`, `/out`, and `/work`.

## 2. Check the BIDS input

At minimum, a diffusion study normally contains:

```text
/study/rawdata/
├── dataset_description.json
└── sub-001/
    └── ses-01/
        ├── anat/
        │   ├── sub-001_ses-01_T1w.nii.gz
        │   └── sub-001_ses-01_T1w.json
        ├── dwi/
        │   ├── sub-001_ses-01_dwi.nii.gz
        │   ├── sub-001_ses-01_dwi.json
        │   ├── sub-001_ses-01_dwi.bval
        │   └── sub-001_ses-01_dwi.bvec
        └── fmap/
            └── ...
```

Inspect the dataset before processing:

```bash
qmri-neuropipe inspect /study/rawdata --by-session
```

To examine raw data and existing derivatives together:

```bash
qmri-neuropipe inspect /study --rawdata-dir rawdata \
  --derivatives --processing-gaps --by-session
```

## 3. Create the pipeline YAML

Copy the starter YAML in this training set, or begin with the following conservative template:

```yaml
# Required host paths. Use absolute paths.
bids_dir: /study/rawdata
output_dir: /study/derivatives/qmri-neuropipe
work_dir: /scratch/my_user/qmri-neuropipe-work

# Cohort selection can also be supplied at the command line.
participant_label:
  - sub-001
session_label:
  - ses-01

# Resources
n_cpus: 8
memory_gb: 32
use_gpu: false

# Run behavior
skip_existing: true
stop_on_error: true
log_level: INFO
verbose: true

# Select the workflow at the top level.
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
    coregistration:
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

Treat this as a teaching template, not a universal protocol. Distortion correction, registration, masking, and modeling choices must match the acquired data and the study protocol.

### YAML rules that prevent common errors

- Use spaces, not tabs, and keep indentation consistent.
- Do not repeat a key such as `dmri:` or `preprocessing:`. Duplicate keys are rejected.
- Use YAML booleans `true` and `false`, not quoted strings.
- Prefer absolute paths. Environment variables are supported, for example `${STUDY_DIR}/rawdata`.
- Put stable study settings in YAML and use CLI flags for temporary participant or resource overrides.
- CLI values override YAML values.
- If more than one T1w or T2w exists, select one with `anat.input.t1w_match` or `anat.input.t2w_match`; zero or multiple matches cause a deliberate failure.
- Keep each protocol YAML under version control and record why non-default choices were made.

### Select the processing track

Set one top-level pipeline per invocation:

```yaml
pipeline: anat          # T1w/T2w preprocessing, masking, reconstruction, normalization
# pipeline: dmri        # DWI preprocessing and diffusion modeling
# pipeline: relaxometry # SPGR/SSFP preprocessing and DESPOT-family fitting
```

Keep the corresponding settings under the matching root key: `anat`, `dmri`, or `relaxometry`. The SOP contains reviewed starter profiles for all three tracks.

For mcDESPOT, enable DESPOT1 and mcDESPOT explicitly:

```yaml
pipeline: relaxometry

relaxometry:
  preprocessing:
    motion_correction:
      enabled: true
      method: ants
    b1:
      method: afi
  masking:
    enabled: true
    method: synthstrip
  modeling:
    despot1:
      enabled: true
      algo: lsq
    mcdespot:
      enabled: true
      cuda: false
  qc:
    enabled: true
```

mcDESPOT requires discovered SPGR and SSFP data, the T1 map from DESPOT1, and a B1 map from AFI/external B1 or DESPOT1-HIFI. Use acquisition/description labels containing `spgr`, `ssfp`, `ir` + `spgr`, or `afi` so the relaxometry workflow can classify the inputs. The deprecated `despot2.mcdespot` switch should not be used.

### Optional downstream processing

The SOP provides full examples and QC requirements for these optional branches:

- **Normalization:** `anat.preprocessing.normalization`, `dmri.normalization`, or `relaxometry.normalization`. Use an approved template whose contrast and population match the driving image, save transforms, and visually inspect standard-space overlays.
- **Tractography:** `dmri.modeling.tractography`. Match the tracking algorithm to its diffusion model; iFOD2 requires FOD/CSD data, while ACT requires aligned anatomy and a valid 5TT segmentation.
- **Atlas/ROI extraction:** `dmri.analysis`, `relaxometry.analysis`, or `anat.segmentation`. Named atlases may include `labels`, matching `template`, `lut`, and label-safe `interpolation: genericLabel`; select controlled `metrics` rather than silently analyzing every map.
- **Study tracker:** set `tracker.enabled: true` and an explicit writable `tracker.file`. The workbook accumulates modality status, QC, ROI metrics, volumes, alerts, and per-sheet cohort CSV exports.

For a container run, support paths must be container-visible. Atlas/template paths need a `--bind`, and a tracker beneath the output bind should use a path such as `/out/study_tracker.xlsx` inside the pipeline YAML.

## 4. Validate before processing

### Local validation

```bash
qmri-neuropipe run \
  --config /study/code/qmri_dmri.yaml \
  --participant-label sub-001 \
  --session-label ses-01 \
  --dry-run --verbose
```

The dry run loads and merges the configuration, validates required paths, and prints the resolved settings without processing data.

### Container launch check

This prints the Apptainer/Singularity command but does not start the container:

```bash
qmri-neuropipe container \
  --container-image /containers/qmri-neuropipe.sif \
  --config /study/code/qmri_dmri.yaml \
  --participant-label sub-001 \
  --session-label ses-01 \
  --dry-run
```

Then validate the pipeline YAML inside the container:

```bash
qmri-neuropipe container \
  --container-image /containers/qmri-neuropipe.sif \
  --config /study/code/qmri_dmri.yaml \
  --participant-label sub-001 \
  --session-label ses-01 \
  --extra-arg=--dry-run \
  --extra-arg=--verbose
```

On a CPU-only host, add `--no-nv --no-gpu`. The wrapper enables NVIDIA passthrough by default; `--no-gpu` is forwarded to the pipeline.

## 5. Run one participant/session

### Local

```bash
qmri-neuropipe run \
  --config /study/code/qmri_dmri.yaml \
  --participant-label sub-001 \
  --session-label ses-01
```

### Container

```bash
qmri-neuropipe container \
  --container-image /containers/qmri-neuropipe.sif \
  --config /study/code/qmri_dmri.yaml \
  --participant-label sub-001 \
  --session-label ses-01 \
  --n-cpus 8 \
  --memory-gb 32
```

For a FreeSurfer-dependent workflow, add:

```bash
--freesurfer-license /path/to/license.txt
```

For another absolute path referenced by the YAML, expose it explicitly. For example, a template stored at `/atlases` can be made visible at the same location:

```bash
--bind /atlases:/atlases:ro
```

If `--gnl-coeff-file /host/path/gw.dat` is used, the file is copied to `/config/gw.dat`; reference `/config/gw.dat` in the pipeline YAML.

## 6. Review the result before scaling up

Confirm all of the following:

- the terminal reports success for the requested participant/session;
- expected derivative NIfTI files and sidecars exist under the output directory;
- `provenance.json` and logs identify the commands and settings used;
- the HTML/PDF report, when enabled, opens successfully;
- brain masks, registrations, distortion correction, motion correction, gradients, and model maps pass visual QC;
- available disk space is still sufficient for the cohort and the work directory.

Re-run the dataset inventory to find processing gaps:

```bash
qmri-neuropipe inspect /study --rawdata-dir rawdata \
  --derivatives --processing-gaps --by-session
```

Only after the pilot passes QC should you process multiple participants. For a local run, place multiple labels in the YAML. For the container wrapper, use `--subjects 001,002,003` and optionally `--sessions 01,01,01`.

## 7. Resume or deliberately rerun

With `skip_existing: true`, existing completed outputs are reused. To force one step and all later steps while retaining earlier cached work:

```bash
qmri-neuropipe run --config /study/code/qmri_dmri.yaml \
  --participant-label sub-001 --rerun-from-step eddy
```

The equivalent YAML is scoped to the relevant workflow:

```yaml
dmri:
  preprocessing:
    rerun_from_step: eddy
```

Do not delete outputs as a first troubleshooting step. Preserve logs and provenance, identify the failing step, correct the cause, and use a targeted rerun.

## 8. Container settings file (optional)

The optional wrapper settings YAML stores container-launch defaults. It is separate from the pipeline YAML and does not define processing behavior.

```yaml
container:
  container_image: /containers/qmri-neuropipe.sif
  config: /study/code/qmri_dmri.yaml
  bids_dir: /study/rawdata
  output_dir: /study/derivatives/qmri-neuropipe
  work_dir: /scratch/my_user/qmri-neuropipe-work
  runtime: apptainer
  n_cpus: 8
  memory_gb: 32
  nv: false
```

Run it with:

```bash
qmri-neuropipe container --settings /study/code/qmri_container.yaml
```

Command-line flags override settings-file values, while pipeline behavior still comes from the file passed as `config`.

## 9. Troubleshooting quick reference

| Symptom | First checks |
| --- | --- |
| “No matching subject/session pairs” | Confirm BIDS folder names and labels; try both `001` and `sub-001`; run `inspect`. |
| YAML/configuration error | Check indentation, duplicate keys, accepted method names, file existence, and the dry-run output. |
| Command not found during a local run | Activate the intended environment and verify the external tool is on `PATH`. |
| Container cannot see a file | Use a path under `/data`, `/out`, `/work`, or `/config`, or add an explicit `--bind`. |
| `libcuda.so.1` or GPU runtime error | Ensure an NVIDIA host/runtime is available; otherwise use `--no-nv --no-gpu`. |
| Out of disk space | Check both output and work locations; the work directory can be much larger than final derivatives. |
| Existing output was unexpectedly skipped | Review `skip_existing`; use a scoped `rerun_from_step` when regeneration is intentional. |
| Processing fails after a tool update | Record `qmri-neuropipe --version`, image name/digest, YAML commit, and failing command from provenance. |

## Detailed processing manuals

Continue in the full SOP/manual for the processing branches enabled by the study YAML:

- [Anatomical processing](QMRI_NEUROPIPE_SOP_SOURCE.md#7-anatomical-processing-manual)
- [Diffusion processing](QMRI_NEUROPIPE_SOP_SOURCE.md#8-diffusion-processing-manual)
- [Relaxometry and mcDESPOT](QMRI_NEUROPIPE_SOP_SOURCE.md#9-relaxometry-and-mcdespot-processing-manual)
- [Standard-space normalization](QMRI_NEUROPIPE_SOP_SOURCE.md#10-standard-space-normalization-manual)
- [Tractography, tractometry, and connectomes](QMRI_NEUROPIPE_SOP_SOURCE.md#11-tractography-tractometry-and-connectome-manual)
- [Atlas registration and ROI extraction](QMRI_NEUROPIPE_SOP_SOURCE.md#12-atlas-registration-and-roi-extraction-manual)
- [Study tracker](QMRI_NEUROPIPE_SOP_SOURCE.md#13-study-tracker-manual)
- [Validation and execution](QMRI_NEUROPIPE_SOP_SOURCE.md#14-validation-and-execution)
- [Quality control](QMRI_NEUROPIPE_SOP_SOURCE.md#15-quality-control)

## Further reference

- `docs/configuration_reference.md`: task-oriented workflow configuration
- `docs/options_reference.md`: field-by-field options and accepted values
- `docs/tool_reference.md`: external tools used by each step
- `docs/data_organization.md`: expected BIDS inputs and derivative structure
- `examples/configs/example_config.yaml`: comprehensive example configuration
