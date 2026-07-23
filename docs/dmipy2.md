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
--grid-samples N
--orientation-samples N
--maxiter N
--batch-size N
```

`device=gpu` requires `solver=jax`. Native solvers use CPU multiprocessing.
JAX fitting uses dmipy-fit's internal voxel vectorization and does not create
one JAX runtime per qmri-neuropipe worker.

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
- resolved execution backend
- visible JAX devices, when applicable

Released containers must identify the exact qmri-neuropipe tag or Git commit
whose source corresponds to the image.
