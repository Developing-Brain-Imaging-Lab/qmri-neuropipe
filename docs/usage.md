# Usage

`qmri-neuropipe` can be run from the command line using the `qmri-neuropipe` command.

## Basic Usage

The minimal command requires the BIDS input directory and an output directory:

```bash
qmri-neuropipe --bids-dir /path/to/bids --output-dir /path/to/derivatives
```

By default, this runs the **Diffusion MRI (`dmri`)** pipeline on all subjects.

## Select Pipeline

To run the Anatomical pipeline explicitly:

```bash
qmri-neuropipe --bids-dir /path/to/bids --output-dir /path/to/derivatives --pipeline anat
```

### Available Pipelines
*   `dmri` (default): Diffusion MRI processing (includes optional anatomical preprocessing).
*   `anat`: Anatomical MRI processing (T1w/T2w preprocessing).

## Select Subjects

Process specific subjects or sessions:

```bash
# Process only sub-01
qmri-neuropipe ... --participant-label 01

# Process sub-01 and sub-02
qmri-neuropipe ... --participant-label 01 02

# Process specific session
qmri-neuropipe ... --participant-label 01 --session-label 01
```

## Configuration

You can provide a configuration file (YAML or JSON) to control pipeline parameters. CLI arguments override config file settings.

```bash
qmri-neuropipe --config config.yaml
```

### Example `config.yaml`

```yaml
bids_dir: /path/to/bids
output_dir: /path/to/derivatives
n_cpus: 8

dmri:
  preprocessing:
    denoising:
      enabled: true
      method: "mrtrix"
    distcorr:
      method: "topup"
      fallback: true
    outliers:
      enabled: true
      method: "manual"
      threshold: 0.05
    coregistration:
      enabled: true
      method: "ants"

anat:
  preprocessing:
    denoising:
      enabled: true
    brain_masking:
      enabled: true
      method: "ants"
```

## Command Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--bids-dir` | Path to BIDS dataset directory (Required) | - |
| `--output-dir` | Path to output directory (Required) | - |
| `--work-dir` | Path to working directory | `output_dir/work` |
| `--config`, `-c` | Path to YAML/JSON config file | - |
| `--pipeline` | Pipeline to run (`dmri`, `anat`) | `dmri` |
| `--participant-label`, `-p` | Participant ID(s) to process | All |
| `--session-label`, `-s` | Session ID(s) to process | All |
| `--n-cpus` | Number of CPUs to use | 1 |
| `--memory-gb` | Memory limit in GB | 8.0 |
| `--use-gpu` / `--no-gpu` | Enable GPU acceleration | False |
| `--skip-existing` | Skip already processed subjects | True |
| `--dry-run` | Validate config without running | False |

For a full list, run:
```bash
qmri-neuropipe --help
```

## Workflow Step Parameters (Quick Reference)

For full per-step defaults and options, see `docs/tool_reference.md`. Below is a compact map of the most commonly used keys.

### Diffusion (dMRI)

| Step | Key |
| --- | --- |
| Distortion correction | `dmri.preprocessing.distcorr` |
| Denoising | `dmri.preprocessing.denoising` |
| Gibbs unringing | `dmri.preprocessing.degibbs` |
| Eddy | `dmri.preprocessing.eddy` |
| Outliers | `dmri.preprocessing.outliers` |
| Bias correction | `dmri.preprocessing.bias_correction` |
| Coregistration | `dmri.preprocessing.coregistration` |
| Gradient nonlinearity | `dmri.preprocessing.grad_nonlin` |
| Brain masking | `dmri.preprocessing.brain_masking` |

**Example**
```yaml
dmri:
  preprocessing:
    distcorr:
      method: topup
      fallback: true
    denoising:
      enabled: true
      method: mppca
    degibbs:
      enabled: true
      method: mrtrix
    eddy:
      enabled: true
      method: eddy
    outliers:
      enabled: true
      method: eddy_qc
    bias_correction:
      enabled: true
      method: ants
    coregistration:
      enabled: true
      method: ants
    grad_nonlin:
      enabled: false
    brain_masking:
      enabled: true
      method: mrtrix
```

### Anatomical (anat)

| Step | Key |
| --- | --- |
| Resample | `anat.preprocessing.resample` |
| Reorient | `anat.preprocessing.reorient` |
| Denoising | `anat.preprocessing.denoising` |
| Gibbs unringing | `anat.preprocessing.degibbs` |
| Bias correction | `anat.preprocessing.bias_correction` |
| Sharpen | `anat.preprocessing.sharpen` |
| Coregistration | `anat.preprocessing.coregistration` |
| Brain masking | `anat.preprocessing.brain_masking` |
| Recon-all | `anat.preprocessing.recon_all` |
| Normalization | `anat.preprocessing.normalization` |

**Example**
```yaml
anat:
  preprocessing:
    resample:
      enabled: true
      resolution: 1.0
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
    sharpen:
      enabled: false
    coregistration:
      enabled: true
      method: fsl
      reference_image: t1w
    brain_masking:
      enabled: true
      method: ants
    recon_all:
      enabled: false
    normalization:
      enabled: false
```

### Relaxometry

| Step | Key |
| --- | --- |
| Denoising | `relaxometry.preprocessing.denoising` |
| Gibbs unringing | `relaxometry.preprocessing.degibbs` |
| Motion correction | `relaxometry.preprocessing.motion_correction` |
| B1 mapping | `relaxometry.preprocessing.b1` |
| Brain masking | `relaxometry.masking` |

**Example**
```yaml
relaxometry:
  preprocessing:
    denoising:
      enabled: true
      method: mrtrix
    degibbs:
      enabled: true
      method: mrtrix
    motion_correction:
      enabled: true
      method: ants
    b1:
      method: afi
      smoothing_fwhm: 0.0
  masking:
    enabled: true
    method: fsl
```

## Optional Extras

Install optional features via extras:

```bash
pip install -e .[all]
pip install -e .[amico]
pip install -e .[nifreeze]
pip install -e .[pyafq]
pip install -e .[tracker]
pip install -e .[reporting]
```
