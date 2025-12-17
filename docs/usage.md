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
