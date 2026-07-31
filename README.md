# qmri-neuropipe

Robust neuroimaging pipeline for BIDS datasets.

## Overview

`qmri-neuropipe` provides configurable pipelines for processing MRI data, supporting both anatomical and diffusion workflows. It is designed to be robust and easy to use with BIDS formatted datasets.

**Supported modalities:**

- **Structural (T1w/T2w)**: denoising, bias correction, brain masking, coregistration, normalization, FreeSurfer recon-all, SuperSynth segmentation + volume extraction
- **Diffusion (dMRI)**: full preprocessing (eddy, topup, DRBUDDI), model fitting (DTI, DKI, NODDI, FWE-DTI, NEXI, MAP-MRI, SANDI, CSD), atlas-based ROI statistics
- **Relaxometry (DESPOT/mcDESPOT)**: SPGR/SSFP/IR-SPGR preprocessing, DESPOT1/2FM/mcDESPOT fitting, VFm/T1/T2 maps, atlas-based ROI statistics
- **fMRI**: fMRIPrep or HCP pipeline wrapper

## Installation

```bash
pip install -e .
```

The default installation includes the released dmipy 2.1 CPU fitting stack.
Install a JAX execution backend explicitly when vectorized fitting is needed:

```bash
# JAX on CPU
pip install -e ".[dmipy-jax]"

# JAX with CUDA 12 support
pip install -e ".[dmipy-cuda12]"

# Optional legacy CSD/visualization optimizer dependencies
pip install -e ".[dmipy-legacy]"
```

### Quick Start

```bash
# Run the full pipeline on your BIDS dataset
qmri-neuropipe run \
  --bids-dir /data/bids \
  --output-dir /data/derivatives \
  --config config.yaml \
  --participant-label 001 002 003
```

## Containers

Run the pipeline through an Apptainer/Singularity image with the same pipeline
options used by the native command:

```bash
qmri-neuropipe container \
  --container-image /path/to/qmri-neuropipe.sif \
  --config config.yaml \
  --participant-label 001 \
  --session-label 01
```

`bids_dir`, `output_dir`, and `work_dir` may live in `config.yaml`; the wrapper
mounts those host directories at the same absolute paths inside the container.
This allows the same config to be used for native and container runs. You may
instead supply `--bids-dir`, `--output-dir`, or `--work-dir` to override the
config.

Any other absolute path referenced by the pipeline config must also be visible
inside the container. Add it with `--bind /host/path:/host/path` when the
container runtime does not expose it automatically.

The optional `--settings` YAML/JSON file contains defaults for the container
wrapper—such as `container_image`, runtime, extra binds, and host-path or
resource overrides. It is not a second pipeline configuration. Command-line
options override settings, and pipeline behavior remains in the file passed to
`--config`.

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
pip install -e .[dev]
pip install -e .[amico]
pip install -e .[nifreeze]
pip install -e .[pyafq]
pip install -e .[synb0]
pip install -e .[tracker]
pip install -e .[reporting]
```

The development extra installs the test and lint tools. Run the project checks
with:

```bash
pytest
ruff check src/qmri_neuropipe
```

The project version has a single source of truth in
`qmri_neuropipe.__version__`. Package metadata, the CLI, and the documentation
all read that value, so a release requires changing it only once.

## Documentation

Full documentation is available in the [docs/](docs/) directory.

Users upgrading from the legacy dmipy backend should read the
[dmipy 2.x migration guide](docs/dmipy2_migration.md). The complete backend,
model registry, solver, GPU, and gradient-nonlinearity reference is in
[dmipy2.md](docs/dmipy2.md).

For cached reruns, set `rerun_from_step` under the relevant workflow scope
(`dmri.preprocessing`, `dmri.modeling`, `anat.preprocessing`, or
`relaxometry.preprocessing`) to force that step and all later steps while
leaving earlier cached outputs intact.

## License

Copyright © the qmri-neuropipe contributors.

qmri-neuropipe is licensed under the GNU Affero General Public License,
version 3 only (`AGPL-3.0-only`). See [LICENSE](LICENSE). If a modified version
is made available as a network service, its users must be offered the
Corresponding Source for the version they are using.

The dmipy 2.x fitting backend is separately distributed under
`AGPL-3.0-only OR LicenseRef-Commercial`. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependency notices.
