# qmri-neuropipe

Robust neuroimaging pipeline for BIDS datasets.

## Overview

`qmri-neuropipe` provides configurable pipelines for processing MRI data, supporting both anatomical and diffusion workflows. It is designed to be robust and easy to use with BIDS formatted datasets.

## Installation

```bash
pip install -e .
```

## Containers

GPU-enabled tools such as `eddy_cuda` require the host NVIDIA driver libraries at
runtime. Those are not reliably solved by baking `libcuda.so.1` into the image.
Use the container with GPU passthrough enabled:

```bash
# Apptainer
apptainer exec --nv qneuro.sif eddy_cuda

# Docker
docker run --gpus all ...
```

If `eddy_cuda` reports `libcuda.so.1: cannot open shared object file`, the usual
cause is that the container was launched without `--nv` (Apptainer) or without
the NVIDIA runtime / `--gpus all` (Docker).

The container also provides bind-target directories:

```text
/data
/output
/code
```

### Optional Extras

Install optional features as needed:

```bash
# Everything
pip install -e .[all]

# Specific feature sets
pip install -e .[amico]
pip install -e .[nifreeze]
pip install -e .[pyafq]
pip install -e .[tracker]
pip install -e .[reporting]
```

## Documentation

Full documentation is available in the [docs/](docs/) directory.
