# Migrating from dmipy 1.x to dmipy 2.x

This guide covers the user-visible changes introduced by the dmipy 2.x backend.
The detailed backend and model reference is available in
[dmipy 2.x backend](dmipy2.md).

## Licensing

qmri-neuropipe is now distributed under `AGPL-3.0-only`. The open-source
integration uses `dmipy-fit` and `dmipy-sim` under their AGPL terms. Review
[the project license](../LICENSE) and
the repository's `THIRD_PARTY_NOTICES.md` before distributing modified software
or providing it as a network service.

## Create a new environment

The dmipy 2.x backend requires NumPy 2.x and should be installed in a new
environment rather than over an environment containing the legacy dmipy stack.

```bash
conda create -n qmri-dmipy2 python=3.12
conda activate qmri-dmipy2

git clone \
  https://github.com/Developing-Brain-Imaging-Lab/qmri-neuropipe.git
cd qmri-neuropipe

# Native CPU solvers
pip install -e .

# Or JAX on CPU
pip install -e ".[dmipy-jax]"

# Or JAX with CUDA 12 support
pip install -e ".[dmipy-cuda12]"
```

Do not install the old `dmipy` Python package alongside this backend.
dmipy 2.x exposes its analytical API through the `dmipy_fit` namespace.

## Existing model commands

The model-specific commands remain available:

```text
qmri-tools fit-noddi
qmri-tools fit-sandi
qmri-tools fit-microglia
```

They retain their model-specific options but now use the shared dmipy 2.x
runtime and BIDS derivative writer. Output filenames therefore differ from
legacy runs. Downstream scripts should discover outputs from BIDS entities and
metric suffixes rather than hard-coded legacy filenames.

Every image has a JSON sidecar containing its complete dmipy parameter name,
model configuration, solver, resolved device, dmipy-fit version, and units.

## Registry-driven models

Use `fit-dmipy` for an allow-listed reference model:

```bash
qmri-tools dmipy-models
qmri-tools dmipy-model-info --model sandi

qmri-tools fit-dmipy \
  --model sandi \
  --input sub-01_dwi.nii.gz \
  --bval sub-01_dwi.bval \
  --bvec sub-01_dwi.bvec \
  --mask sub-01_mask.nii.gz \
  --delta sub-01_small_delta.txt \
  --big-delta sub-01_big_delta.txt \
  --solver jax \
  --device gpu \
  --batch-size 4000 \
  --output-dir sandi-dmipy2
```

The integrated pipeline can fit several registry models in one run:

```yaml
dmri:
  modeling:
    dmipy:
      models:
        - name: noddi
          solver: jax
          device: gpu
          gpu_device: 0
          solver_options:
            batch_size: 4000
        - name: verdict
          solver: brute2fine
```

Do not enable NODDI, SANDI, or microglia in both this registry list and its
legacy model-specific configuration block.

## Acquisition timing

Models requiring a PGSE waveform now receive two independent files:

- `delta`: gradient pulse duration in seconds;
- `Delta`: gradient separation in seconds.

Each file may contain one value or one value per DWI measurement. Multi-echo
models additionally require an echo-time file in seconds. Store `delta` and
`Delta` on the pipeline `DWIFile`, or provide explicit file paths in the model
configuration.

## JAX and GPU execution

JAX CPU and GPU use the same solver interface. Use an explicit GPU request when
silent CPU fallback would be undesirable:

```bash
qmri-tools fit-noddi \
  --input sub-01_dwi.nii.gz \
  --bval sub-01_dwi.bval \
  --bvec sub-01_dwi.bvec \
  --mask sub-01_mask.nii.gz \
  --output-dir noddi-jax \
  --solver jax \
  --device gpu \
  --gpu-device 0 \
  --batch-size 4000
```

`--batch-size` controls optimizer batches inside the single JAX process; it
does not split the input into multiprocessing chunks. Startup may include JAX
tracing and compilation. Periodic heartbeat messages distinguish compilation
or long optimizer work from a stalled job.

Use a private persistent cache with `--jax-cache-dir` to reuse compilations.
Within a scheduler or container, `--gpu-device` indexes the GPUs already made
visible to the job.

## Gradient-nonlinearity correction

Compatible JAX full-signal models can apply a voxel-specific gradient tensor
without reverting to one fit per voxel. Supply the tensor with
`--grad-nonlin`, or make it available through the integrated modeling context.

Spherical-mean models, arbitrary waveform/OGSE schemes, and unsupported JAX
forward models retain an exact fallback or are rejected explicitly. The
dedicated historical spherical-mean SANDI command does not use the accelerated
GNL route.

## NEXI

The existing `fit-nexi` command continues to use the separate `nexi` package.
The dmipy registry model is selected with `fit-dmipy --model nexi`.

Use a native dmipy solver for the registry NEXI model. dmipy-fit 2.1's released
NEXI JAX implementation expects obsolete parameters and is blocked to prevent
scientifically inconsistent results.

## Cached and interrupted runs

Registry-driven fits write `dmipy-completion.json` only after every expected
image and sidecar succeeds. The pipeline skips a fit only when this manifest
matches the current inputs, dmipy-fit version, model factory, solver options,
and complete derivative set.

Changing timing, gradients, mask, GNL tensor, solver, or factory options causes
a rerun. An interrupted or partially serialized run also reruns automatically.

## Pre-merge validation

The branch has passed:

- construction and finite-signal simulation for every registered model;
- native/JAX forward parity for representative model families;
- native/JAX recovery of known synthetic parameters;
- the full qmri-neuropipe unit and characterization suite;
- a real-data NODDI JAX GPU run completed in approximately 18 minutes with
  plausible complete outputs.
