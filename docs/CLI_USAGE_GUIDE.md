# qmri-neuropipe: Configuration and CLI Usage Guide

## Overview

The improved CLI system provides flexible configuration through:
- **Config files** (YAML or JSON)
- **Command line arguments**
- **Intelligent merging** (CLI overrides config file values)
- **Comprehensive validation** with clear error messages

## Quick Start

### 1. Using a Config File Only

```bash
qmri-neuropipe --config config.yaml
```

### 2. Using CLI Arguments Only

```bash
qmri-neuropipe \
  --bids-dir /data/bids \
  --output-dir /data/derivatives \
  --participant-label sub-01 \
  --n-cpus 8
```

### 3. Mixed Approach (Recommended)

Use a config file for stable settings, override specific values via CLI:

```bash
qmri-neuropipe \
  --config config.yaml \
  --n-cpus 16 \
  --participant-label sub-01
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

See `docs/tool_reference.md` for per-step tool lists and config keys.

## Required Arguments

These arguments **must** be provided either via config file or command line:

| Argument | Description |
|----------|-------------|
| `--bids-dir` | Path to BIDS dataset directory |
| `--output-dir` | Path to output directory for derivatives |

If either is missing, the pipeline will exit with a clear error message showing what's needed.

## Configuration Priority

When the same parameter is specified in multiple places, priority is:

1. **Command line arguments** (highest priority)
2. **Config file values**
3. **Default values** (lowest priority)

### Example:

**config.yaml:**
```yaml
n_cpus: 4
memory_gb: 8.0
participant_label: sub-01
```

**Command:**
```bash
qmri-neuropipe --config config.yaml --n-cpus 16 --participant-label sub-02
```

**Result:**
- `n_cpus`: 16 (from CLI)
- `memory_gb`: 8.0 (from config)
- `participant_label`: sub-02 (from CLI)

## Complete Parameter Reference

### Core Paths

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `--bids-dir` | Path | **Yes** | BIDS dataset directory |
| `--output-dir` | Path | **Yes** | Output directory for derivatives |
| `--work-dir` | Path | No | Working directory (default: output_dir/work) |
| `--config`, `-c` | Path | No | Path to YAML/JSON config file |

### Pipeline Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--pipeline` | str | dmri | Pipeline to run (dmri, fmri, anat) |
| `--level` | str | participant | Analysis level (participant, group) |

### Subject/Session Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--participant-label`, `-p` | str | None | Participant ID(s) to process |
| `--session-label`, `-s` | str | None | Session ID(s) to process |

### Computational Resources

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--n-cpus` | int | 1 | Number of CPUs to use |
| `--memory-gb` | float | 8.0 | Memory limit in GB |
| `--use-gpu` / `--no-gpu` | bool | False | Enable GPU acceleration |
| `--omp-nthreads` | int | 1 | Number of OpenMP threads |

### Execution Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--skip-existing` / `--no-skip-existing` | bool | True | Skip already-processed subjects |
| `--stop-on-error` / `--continue-on-error` | bool | False | Stop on first error |
| `--skip-bids-validation` | bool | False | Skip BIDS dataset validation |
| `--dry-run` | bool | False | Validate config without running |

### Logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--log-level` | str | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `--verbose`, `-v` | bool | False | Enable verbose output |
| `--debug` | bool | False | Enable debug mode |

## Usage Examples

### Example 1: Process All Subjects with Config File

```bash
qmri-neuropipe --config config.yaml
```

**config.yaml:**
```yaml
bids_dir: /data/study/bids
output_dir: /data/study/derivatives
n_cpus: 8
memory_gb: 16.0
```

### Example 2: Process Specific Subject

```bash
qmri-neuropipe \
  --config config.yaml \
  --participant-label sub-01 \
  --session-label ses-baseline
```

### Example 3: High-Performance Processing

Override config to use more resources:

```bash
qmri-neuropipe \
  --config config.yaml \
  --n-cpus 32 \
  --memory-gb 64 \
  --use-gpu
```

### Example 4: Development/Testing

```bash
# Dry run to validate configuration
qmri-neuropipe --config config.yaml --dry-run --verbose

# Process one subject with debug output
qmri-neuropipe \
  --config config.yaml \
  --participant-label sub-01 \
  --debug \
  --verbose
```

### Example 5: No Config File

```bash
qmri-neuropipe \
  --bids-dir /data/bids \
  --output-dir /data/derivatives \
  --participant-label sub-01 \
  --n-cpus 8 \
  --memory-gb 16 \
  --verbose
```

### Example 6: Batch Processing with Different Settings

Process multiple subjects with different resource allocations:

```bash
# First batch - standard resources
qmri-neuropipe \
  --config config.yaml \
  --participant-label sub-01,sub-02,sub-03

# Second batch - high memory subjects
qmri-neuropipe \
  --config config.yaml \
  --participant-label sub-04,sub-05 \
  --memory-gb 32
```

## Configuration File Examples

### Minimal Config (YAML)

```yaml
bids_dir: /data/bids
output_dir: /data/derivatives
```

### Comprehensive Config (YAML)

See `example_config.yaml` for a fully documented configuration file with all available options.

### JSON Format

```json
{
  "bids_dir": "/data/bids",
  "output_dir": "/data/derivatives",
  "n_cpus": 8,
  "participant_label": ["sub-01", "sub-02"]
}
```

## Error Handling

The new system provides clear error messages for common issues:

### Missing Required Arguments

```bash
$ qmri-neuropipe --n-cpus 8

Configuration Error: Missing required arguments. Please provide them via config file or command line:

  --bids-dir : Input BIDS dataset directory
  --output-dir : Output directory for derivatives

Examples:
  1. Via command line:
     qmri-neuropipe --bids-dir /data/bids --output-dir /data/derivatives

  2. Via config file:
     qmri-neuropipe --config config.yaml

  3. Mixed (CLI overrides config):
     qmri-neuropipe --config config.yaml --n-cpus 16
```

### Invalid Config File

```bash
$ qmri-neuropipe --config nonexistent.yaml

Configuration Error: Configuration file not found: nonexistent.yaml
Details: Please check the path and try again.
```

### Invalid BIDS Directory

```bash
$ qmri-neuropipe --bids-dir /invalid/path --output-dir /data/out

Configuration Error: BIDS directory does not exist: /invalid/path
Details: Please check the path and ensure it's accessible.
```

## Environment Variables

Config files support environment variable expansion:

**config.yaml:**
```yaml
bids_dir: ${DATA_DIR}/bids
output_dir: ${DATA_DIR}/derivatives
work_dir: ${SCRATCH_DIR}/work
```

**Usage:**
```bash
export DATA_DIR=/mnt/storage/study01
export SCRATCH_DIR=/scratch/user
qmri-neuropipe --config config.yaml
```

## Best Practices

1. **Use config files for stable settings**: Store dataset paths, standard resource allocations, and pipeline configurations in a config file.

2. **Override via CLI for variations**: Use command line arguments for subject selection, resource adjustments, and testing.

3. **Version control your configs**: Keep config files in git to track pipeline parameter changes.

4. **Use dry-run for validation**: Always test new configurations with `--dry-run --verbose` before processing.

5. **Document custom settings**: Add comments to YAML configs explaining non-standard choices.

## Integration with Existing Code

The improved CLI is **backward compatible** with existing code:

- `PipelineConfig` class remains unchanged
- All existing config file formats work
- Pipeline classes receive the same `PipelineConfig` objects

## Troubleshooting

### Q: How do I see what configuration will be used?

A: Use `--dry-run --verbose`:

```bash
qmri-neuropipe --config config.yaml --n-cpus 16 --dry-run --verbose
```

This displays the merged configuration without running the pipeline.

### Q: Can I use both YAML and JSON?

A: Yes! The system automatically detects the format from the file extension.

### Q: What if I specify the same parameter in both config and CLI?

A: CLI always wins. This is intentional to allow easy overrides.

### Q: Can I use relative paths?

A: Yes, but absolute paths are recommended for clarity and reliability.

## Migration from Old CLI

If you're upgrading from an older version:

1. **Old style** (direct arguments):
   ```bash
   qmri-neuropipe /data/bids /data/derivatives participant
   ```

2. **New style** (named arguments):
   ```bash
   qmri-neuropipe --bids-dir /data/bids --output-dir /data/derivatives
   ```

The new style is more explicit and allows flexible ordering and config file integration.

## Getting Help

```bash
# Show all available options
qmri-neuropipe --help

# Show version
qmri-neuropipe --version
```

## Contributing

When adding new parameters:

1. Add to `PipelineConfig` dataclass in `core/config.py`
2. Add CLI option in `cli.py` `main()` function
3. Add to `cli_args` dict in `main()` for merging
4. Update example config files
5. Update this documentation
