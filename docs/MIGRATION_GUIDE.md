# Migration Guide: Implementing Improved CLI System

## Overview

This guide walks through integrating the improved CLI system into your existing qmri-neuropipe codebase.

## Changes Required

### 1. Replace `cli.py`

**File:** `qmri_neuropipe/cli.py`

**Action:** Replace the existing `cli.py` with the improved version provided in `improved_cli.py`

**Key improvements:**
- Merges config file and CLI arguments intelligently
- Validates required arguments with clear error messages
- Better error handling using existing `ConfigurationError` class
- Rich console output for better user experience
- Support for dry-run mode

### 2. Update `config.py` (Optional Enhancement)

**File:** `qmri_neuropipe/core/config.py`

The current `config.py` is already well-designed! However, you may want to add a helper method to `PipelineConfig`:

```python
@classmethod
def from_cli_args(cls, cli_args: Dict[str, Any]) -> 'PipelineConfig':
    """
    Create configuration from CLI arguments dictionary.
    
    Args:
        cli_args: Dictionary of CLI arguments
    
    Returns:
        PipelineConfig instance
    """
    # Remove None values
    cli_args = {k: v for k, v in cli_args.items() if v is not None}
    
    # Extract standard fields
    standard_fields = {}
    for field in ['bids_dir', 'output_dir', 'work_dir', 'participant_label',
                  'session_label', 'n_cpus', 'memory_gb', 'use_gpu',
                  'skip_existing', 'stop_on_error', 'log_level', 'debug', 'verbose']:
        if field in cli_args:
            standard_fields[field] = cli_args[field]
    
    config = cls(**standard_fields)
    config.validate()
    return config
```

### 3. No Changes Required

The following files work perfectly with the new system:
- `qmri_neuropipe/core/config.py` - Already supports all needed functionality
- `qmri_neuropipe/core/exceptions.py` - Already has `ConfigurationError`
- `qmri_neuropipe/io/data_loader.py` - Receives same config object
- `workflows/pipelines/dmri.py` - Receives same config object

## Installation Steps

### Step 1: Backup Current Files

```bash
cd /path/to/qmri-neuropipe
cp qmri_neuropipe/cli.py qmri_neuropipe/cli.py.backup
```

### Step 2: Install Improved CLI

```bash
# Copy the improved CLI
cp improved_cli.py qmri_neuropipe/cli.py
```

### Step 3: Add Example Configs to Repository

```bash
# Create examples directory
mkdir -p examples/configs

# Copy example configs
cp example_config.yaml examples/configs/
cp example_config.json examples/configs/

# Create a minimal example
cat > examples/configs/minimal.yaml << EOF
# Minimal configuration - required fields only
bids_dir: /path/to/bids
output_dir: /path/to/derivatives
EOF
```

### Step 4: Update Documentation

```bash
# Copy usage guide
cp CLI_USAGE_GUIDE.md docs/

# Update README
cat >> README.md << EOF

## Configuration

See [CLI Usage Guide](docs/CLI_USAGE_GUIDE.md) for detailed information on:
- Configuration file formats
- Command line arguments
- Merging strategies
- Usage examples

### Quick Start

\`\`\`bash
# Using config file
qmri-neuropipe --config config.yaml

# Using CLI arguments
qmri-neuropipe --bids-dir /data/bids --output-dir /data/derivatives

# Mixed approach (recommended)
qmri-neuropipe --config config.yaml --n-cpus 16 --participant-label sub-01
\`\`\`
EOF
```

### Step 5: Test the Installation

```bash
# Test with example config
python test_cli_config.py

# Test actual CLI (help)
python -m qmri_neuropipe.cli --help

# Test dry run with example config
python -m qmri_neuropipe.cli \
  --config examples/configs/example_config.yaml \
  --dry-run \
  --verbose
```

## Validation Checklist

- [ ] Backup original files created
- [ ] New CLI installed
- [ ] Example configs in place
- [ ] Help message displays correctly
- [ ] Config file loading works (YAML)
- [ ] Config file loading works (JSON)
- [ ] CLI-only mode works
- [ ] Mixed mode (config + CLI) works
- [ ] CLI overrides config values correctly
- [ ] Required argument validation works
- [ ] Error messages are clear
- [ ] Dry-run mode works
- [ ] Verbose output displays config
- [ ] Existing pipelines receive correct config

## Testing Examples

### Test 1: Help System

```bash
python -m qmri_neuropipe.cli --help
```

Expected: Comprehensive help message with all options

### Test 2: Missing Required Arguments

```bash
python -m qmri_neuropipe.cli --n-cpus 8
```

Expected: Clear error message listing required arguments

### Test 3: Config File Only

```bash
# First create test dirs
mkdir -p /tmp/test_bids /tmp/test_output

# Update example config paths
cat > /tmp/test_config.yaml << EOF
bids_dir: /tmp/test_bids
output_dir: /tmp/test_output
n_cpus: 4
EOF

python -m qmri_neuropipe.cli --config /tmp/test_config.yaml --dry-run --verbose
```

Expected: Displays merged configuration, no errors

### Test 4: CLI Override

```bash
python -m qmri_neuropipe.cli \
  --config /tmp/test_config.yaml \
  --n-cpus 16 \
  --participant-label sub-01 \
  --dry-run \
  --verbose
```

Expected: Configuration shows n_cpus=16 (overridden from 4)

### Test 5: Invalid Config File

```bash
python -m qmri_neuropipe.cli --config /nonexistent/config.yaml
```

Expected: Clear error message about missing file

## Troubleshooting

### Issue: Import Error

**Problem:**
```
ImportError: cannot import name 'ConfigurationError' from 'qmri_neuropipe.core.exceptions'
```

**Solution:**
Ensure your `core/exceptions.py` has the `ConfigurationError` class. It should already be there based on the files you shared.

### Issue: Typer Not Found

**Problem:**
```
ModuleNotFoundError: No module named 'typer'
```

**Solution:**
```bash
pip install typer rich
```

### Issue: Pipeline Import Error

**Problem:**
```
ModuleNotFoundError: No module named 'workflows.pipelines.dmri'
```

**Solution:**
Update the import in `cli.py`:
```python
# Change this:
from workflows.pipelines.dmri import DMRIPipeline

# To this:
from qmri_neuropipe.workflows.pipelines.dmri import DMRIPipeline
```

## Rollback Procedure

If you need to rollback to the original CLI:

```bash
# Restore backup
cp qmri_neuropipe/cli.py.backup qmri_neuropipe/cli.py

# Verify
python -m qmri_neuropipe.cli --help
```

## Advanced Usage

### Custom Validation Rules

You can extend validation in `cli.py`:

```python
def validate_required_arguments(config: PipelineConfig) -> None:
    """Extended validation with custom rules"""
    # Original validation
    # ... existing code ...
    
    # Add custom validation
    if config.use_gpu and config.n_cpus < 4:
        console.print(
            "[yellow]Warning:[/yellow] GPU mode typically requires at least 4 CPUs"
        )
    
    if config.memory_gb < 8:
        console.print(
            "[yellow]Warning:[/yellow] Less than 8GB memory may cause issues"
        )
```

### Pipeline-Specific Validation

Add validation for specific pipelines:

```python
def validate_pipeline_config(config: PipelineConfig, pipeline: str) -> None:
    """Validate pipeline-specific requirements"""
    if pipeline == 'dmri':
        # Check for dmri-specific config
        if not config.get('dmri.preprocessing.eddy.enabled'):
            console.print(
                "[yellow]Warning:[/yellow] Eddy correction is disabled"
            )
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Test CLI

on: [push, pull_request]

jobs:
  test-cli:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest
      - name: Test CLI help
        run: |
          python -m qmri_neuropipe.cli --help
      - name: Test config validation
        run: |
          python test_cli_config.py
```

## Next Steps

After successful integration:

1. **Update user documentation** with new examples
2. **Create training materials** for lab members
3. **Add unit tests** for CLI functions
4. **Consider adding** shell completion scripts
5. **Monitor user feedback** and iterate

## Questions?

If you encounter issues during migration:

1. Check the test script output: `python test_cli_config.py`
2. Verify all imports are correct
3. Ensure config file paths are absolute
4. Check file permissions on config files
5. Review error messages carefully - they're designed to be helpful!

## Summary

The improved CLI system provides:
- ✅ Flexible configuration (file + CLI)
- ✅ Clear validation and error messages  
- ✅ Backward compatibility
- ✅ Better user experience
- ✅ Minimal code changes required

Your existing `PipelineConfig` and exception handling are excellent and work perfectly with this system!
