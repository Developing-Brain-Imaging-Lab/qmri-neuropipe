# qmri-neuropipe

Robust neuroimaging pipeline for BIDS datasets.

## Overview

`qmri-neuropipe` provides configurable pipelines for processing MRI data, supporting both anatomical and diffusion workflows. It is designed to be robust and easy to use with BIDS formatted datasets.

## Installation

```bash
pip install -e .
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
