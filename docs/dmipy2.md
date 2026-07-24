# dmipy 2.x backend

qmri-neuropipe uses the analytical `dmipy-fit` engine from dmipy 2.x. The
`dmipy` distribution is an umbrella package in 2.x and does not provide an
importable `dmipy` Python namespace.

## Supported release

The current compatibility window is:

```text
dmipy-fit >= 2.1, < 2.2
numpy >= 2.0, < 3
```

The upper bound is intentional. The dmipy website can document code newer than
the latest PyPI release, so each new minor release must pass the model-contract
and numerical parity tests before the bound is raised.

## Installation profiles

```bash
# Native CPU solvers: brute2fine and MIX
pip install -e .

# JAX vectorized fitting on CPU
pip install -e ".[dmipy-jax]"

# JAX vectorized fitting with CUDA 12
pip install -e ".[dmipy-cuda12]"

# Optional legacy CSD and visualization dependencies
pip install -e ".[dmipy-legacy]"
```

CUDA fitting requires a working host CUDA 12 driver/runtime. Container runs
must expose the GPU, for example with Apptainer `--nv` or Docker `--gpus all`.

## Solver and device options

The model-specific commands accept:

```text
--solver brute2fine|mix|jax
--device auto|cpu|gpu
--gpu-device N
--grid-samples N
--orientation-samples N
--maxiter N
--batch-size N
--jax-cache-dir PATH
--jax-log-compiles
--heartbeat-interval SECONDS
```

`device=gpu` requires `solver=jax`. Native solvers use CPU multiprocessing.
JAX fitting uses dmipy-fit's internal voxel vectorization and does not create
one JAX runtime per qmri-neuropipe worker.

`--batch-size` controls dmipy's internal JAX optimizer batches. The single
pipeline worker still receives the complete masked dataset so that the model
and compiled program are not recreated for every batch. Before fitting,
qmri-neuropipe reports both the complete voxel count and the expected number of
optimizer batches. With the JAX solver, `--nthreads` is the CPU-thread allowance
for model setup and compilation; it does not create additional GPU workers.

Use `--gpu-device` to restrict JAX to one CUDA device. The index is interpreted
within the visibility established by the scheduler or container. For example,
if Slurm exposes one assigned GPU as logical device 0, use `--gpu-device 0`
even when that GPU has a different host-level index. Explicit GPU requests fail
immediately when JAX cannot find a usable GPU; they never silently fall back to
CPU.

JAX compilation can be reused between runs by selecting a private cache
directory:

```bash
qmri-tools fit-noddi \
  --input dwi.nii.gz \
  --bval dwi.bval \
  --bvec dwi.bvec \
  --mask mask.nii.gz \
  --output-dir noddi-jax \
  --solver jax \
  --device gpu \
  --gpu-device 0 \
  --batch-size 4000 \
  --jax-cache-dir /path/to/private/scratch/qmri-jax-cache
```

The cache directory must not be writable by untrusted users. Use
`--jax-log-compiles` when diagnosing tracing or compilation delays. JAX and
dmipy progress remain visible, and qmri-neuropipe prints a liveness heartbeat
every 30 seconds by default; change this with `--heartbeat-interval`.

Gradient-nonlinearity correction currently requires a distinct acquisition
scheme for each voxel. That path remains per-voxel even with the JAX solver and
will not realize normal whole-mask GPU throughput.

## Model registry

`qmri_neuropipe.interfaces.dmipy_backend.MODEL_REGISTRY` provides stable,
allow-listed identifiers for the published dmipy-fit 2.1 reference factories.
It includes Gaussian/tensor, NODDI/SMT, axon-diameter, soma, exchange,
time-dependent, and multi-TE model families.

The existing NODDI, SANDI, and microglia workflows remain compatibility
wrappers with their established derivative names. New generic model workflow
and acquisition-waveform support will be added after CPU parity of these
existing outputs is established.

## Provenance

dmipy-generated sidecars record:

- `FittingSoftware=dmipy-fit`
- the installed dmipy-fit version
- solver
- requested device
- selected CUDA device index, when specified
- resolved execution backend
- visible JAX devices, when applicable
- persistent compilation-cache directory and compile-logging state

Released containers must identify the exact qmri-neuropipe tag or Git commit
whose source corresponds to the image.
