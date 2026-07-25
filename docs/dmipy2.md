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

dmipy-fit 2.1.0 performs its final nested-volume-fraction conversion with one
small JAX dispatch per voxel. On whole-brain masks this post-processing step can
take substantially longer than the GPU optimization after the progress bar
reaches 100%. qmri-neuropipe detects that released implementation inside each
JAX worker and substitutes the mathematically equivalent NumPy conversion.
Future dmipy versions that no longer contain the per-voxel JAX call are not
modified.

dmipy outputs use a model-independent BIDS-style derivative writer. Input
entities such as subject, session, acquisition, run, and space are preserved;
the input preprocessing description is removed and a stable `model-<Model>`
entity is added.

Curated output aliases use a compact metric suffix. For example, NODDI writes:

```text
sub-01_ses-01_model-NODDI_ODI.nii.gz
sub-01_ses-01_model-NODDI_ICVF.nii.gz
sub-01_ses-01_model-NODDI_EXVF.nii.gz
sub-01_ses-01_model-NODDI_FISO.nii.gz
```

Each image has a matching JSON sidecar containing the dmipy version, solver,
resolved execution device, model configuration, metric description, and units.
The NODDI, SANDI, and microglia interfaces all use this shared writer.

The same writer supports every reference model in the dmipy registry, as well
as custom models. Parameters with no curated alias use a deterministic fallback
that is valid for BIDS entities:

```text
sub-01_model-BallAndStick_desc-BundleModel1C1Stick1LambdaPar_parameter.nii.gz
```

The complete, unsanitized dmipy parameter name is retained in the JSON sidecar
under `Parameter`, so the filename conversion is lossless. Vector-valued
parameters are stored as 4D NIfTI images and their component count is recorded
as `ParameterCardinality`. `write_dmipy_fit_result` can write all
`fitted_parameters` returned by any model fitted through the shared
`fit_model` API; model-specific interfaces can pass derived maps to
`write_dmipy_derivatives`.

Output serialization is shared, but acquisition construction remains
model-specific. The model registry records requirements such as `delta`,
`Delta`, and echo time, and includes those requirements in each sidecar.

## Generic reference-model command

`fit-dmipy` fits any allow-listed dmipy-fit 2.1 reference model and sends every
fitted parameter through the shared derivative writer. Models based on PGSE
timing receive small-delta and big-Delta from separate files:

```bash
qmri-tools fit-dmipy \
  --model nexi \
  --input sub-01_dwi.nii.gz \
  --bval sub-01_dwi.bval \
  --bvec sub-01_dwi.bvec \
  --mask sub-01_mask.nii.gz \
  --delta sub-01_small_delta.txt \
  --big-delta sub-01_big_delta.txt \
  --solver jax \
  --device gpu \
  --gpu-device 0 \
  --batch-size 4000 \
  --output-dir nexi-dmipy
```

Each timing file may contain one value to broadcast to all measurements or one
value per DWI volume. Values are in seconds. dmipy's NEXI and Kärger models
calculate the effective diffusion time as `Delta - delta/3`, so both files are
required rather than a single derived diffusion-time file. Use filenames that
differ by more than capitalization so they remain distinct on case-insensitive
filesystems. Multi-echo models add an independent `--te` file, also in seconds.

The existing `fit-nexi` command uses the separate `nexi` package and retains
its noise-map-based interface. Use `fit-dmipy --model nexi` to select the
dmipy-fit 2.x reference model.

### Microglia and astrocyte activation model

`microglia` is a project-maintained dmipy-fit 2.x model based on
Garcia-Hernandez et al. (2022). It combines:

- a Watson-dispersed stick and zeppelin bundle;
- a fitted small restricted sphere for microglial cell bodies;
- a fitted large restricted sphere for astrocytes;
- an isotropic free-water compartment.

Run the default registered model through the generic interface:

```bash
qmri-tools fit-dmipy \
  --model microglia \
  --input sub-01_dwi.nii.gz \
  --bval sub-01_dwi.bval \
  --bvec sub-01_dwi.bvec \
  --mask sub-01_mask.nii.gz \
  --delta sub-01_small_delta.txt \
  --big-delta sub-01_big_delta.txt \
  --output-dir microglia-dmipy
```

The dedicated `fit-microglia` command remains available when custom sphere
diameter bounds, initial diameters, or fixed diffusivities are needed. Both
commands use the same registered model factory and publish the paper-facing
stick, extracellular, tissue, sphere-radius, dispersion, and orientation maps.
The generic command also accepts `--grad-nonlin` when `--solver jax` is used.

dmipy-fit 2.1 exposes optional surface-relaxivity parameters on its restricted
sphere implementation. The diffusion-only microglia model fixes both to zero,
because they cannot be identified without a relaxation-sensitive acquisition.

Gradient-nonlinearity correction uses a distinct acquisition scheme for each
voxel. With `--solver jax`, qmri-neuropipe now passes those schemes through a
vectorized loss and optimizer, retaining `--batch-size` batching on JAX CPU or
GPU. The nominal scheme is used only for the global brute-grid initializer;
every LBFGS-B objective and gradient evaluation uses the voxel-corrected
b-values, directions, q-values, and gradient strengths.

This accelerated path supports full-signal PGSE models, including NODDI,
SANDI, and microglia models that have dmipy JAX forward support. It does
not apply to arbitrary waveform/OGSE schemes or spherical-mean models, because
GNL changes the acquisition directions and shell membership at each voxel.
NODDI with an external voxelwise FISO constraint and SMT-NODDI retain the
existing exact per-voxel path.

The oriented Kärger/NEXI model is also excluded in dmipy-fit 2.1: its released
JAX forward expects obsolete parameter names and does not reproduce the native
relaxation model. qmri-neuropipe raises a clear error instead of silently using
an approximate objective.

Example:

```bash
qmri-tools fit-dmipy \
  --model sandi \
  --input sub-01_dwi.nii.gz \
  --bval sub-01_dwi.bval \
  --bvec sub-01_dwi.bvec \
  --mask sub-01_mask.nii.gz \
  --delta sub-01_small_delta.txt \
  --big-delta sub-01_big_delta.txt \
  --grad-nonlin sub-01_desc-gradnonlin_tensor.nii.gz \
  --solver jax \
  --device gpu \
  --batch-size 4000 \
  --output-dir sandi-jax-gnl
```

## Model registry

`qmri_neuropipe.interfaces.dmipy_backend.MODEL_REGISTRY` provides stable,
allow-listed identifiers for the published dmipy-fit 2.1 reference factories
and project-maintained models such as `microglia`. It includes
Gaussian/tensor, NODDI/SMT, axon-diameter, soma, exchange, time-dependent, and
multi-TE model families.

Inspect the registry without running a fit:

```bash
qmri-tools dmipy-models
qmri-tools dmipy-models --format json
qmri-tools dmipy-model-info --model nexi
```

`dmipy-model-info` resolves the factory and reports parameter names,
cardinalities, physical bounds, output aliases, acquisition requirements, and
references. Add `--probe` to either command to construct the selected models
and simulate deterministic signals using an in-range validation parameter set:

```bash
qmri-tools dmipy-models --probe
qmri-tools dmipy-model-info --model microglia --probe
```

The solver list in this report describes the shared dmipy fitting interfaces;
it is not by itself a numerical-validation claim. The automated capability
matrix separately constructs and simulates every registry model. When JAX is
installed, representative Gaussian, dispersion, exchange, glial, and multi-TE
forward models are compared with their native NumPy implementations. Optimizer
recovery is tested independently so forward-model parity is not confused with
fitting convergence.

The existing NODDI, SANDI, and microglia workflows remain model-specific
compatibility wrappers and publish their established metrics through the shared
writer. A generic model execution CLI and acquisition-waveform support are
separate from this output layer.

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
