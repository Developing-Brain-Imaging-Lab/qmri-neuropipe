# fMRI Workflow

The **fMRI Pipeline** (`--pipeline fmri`) provides processing for blood-oxygen-level-dependent (BOLD) functional MRI data, primarily by delegating to robust BIDS Apps like fMRIPrep, or natively handling Human Connectome Project (HCP) style fallback structures.

## Overview

The workflow scans a generic BIDS dataset for BOLD images and constructs the appropriate execution environments (Docker or Singularity) to run fMRIPrep. By using fMRIPrep as the main engine, `qmri-neuropipe` leverages the neuroimaging community's gold-standard functional preprocessing pipeline.

**Main Class**: `qmri_neuropipe.workflows.pipelines.fmri_workflow.FmriWorkflow`

## Steps

### 1. fMRIPrep Container Execution
Runs the standardized fMRI preprocessing workflow via Singularity or Docker.

**Available engines**
*   `singularity`
*   `docker`

**Config**
```yaml
fmriprep:
  enabled: true
  container_path: /path/to/fmriprep.simg  # For Singularity
  # docker_image: nipreps/fmriprep:latest # For Docker
  fs_license_file: /path/to/license.txt
  custom_args:
    - "--use-aroma"
    - "--output-spaces"
    - "MNI152NLin2009cAsym"
```

**Parameters**
| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `fmriprep.enabled` | bool | false | Enables routing to the fMRIPrep module |
| `fmriprep.container_path` | string | none | Path to the Singularity `.simg`/`.sif` image |
| `fmriprep.docker_image` | string | none | Name of the Docker image (if not using Singularity) |
| `fmriprep.fs_license_file` | string | none | Path to the FreeSurfer license file (required by fMRIPrep) |
| `fmriprep.custom_args` | list | `[]` | List of additional arguments to pass to fMRIPrep |

### 2. HCP Workflow (Fallback)
If `fmriprep.enabled` is false, the pipeline can route to a native HCP processing skeleton (`HCPfMRIStep`). *Note: This path is currently a structural placeholder for future expansion.*

**Config**
```yaml
hcp:
  enabled: true
```

## Outputs

When using `fMRIPrep`, all derivatives are written to `<output_dir>/derivatives/fmriprep/`. Reference the official [fMRIPrep documentation](https://fmriprep.org/en/stable/outputs.html) for detailed output specifications, which include:

*   Preprocessed BOLD timeseries (`*desc-preproc_bold.nii.gz`)
*   Brain masks
*   Confound regressor TSV files (`*desc-confounds_timeseries.tsv`)
*   Visual summary HTML reports (`sub-<id>.html`)
