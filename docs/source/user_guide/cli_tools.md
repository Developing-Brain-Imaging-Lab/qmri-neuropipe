# Command Line Tools

qmri-neuropipe provides a suite of standalone command-line tools for running specific modeling or processing steps directly on NIfTI files, without requiring a full BIDS dataset structure.

These tools are accessed via the `qmri-tools` command.

## General Usage

```bash
qmri-tools [COMMAND] [OPTIONS]
```

To see available commands:
```bash
qmri-tools --help
```

---

## Model Fitting Tools

### 1. DTI Fitting (`fit-dti`)

Fit a Diffusion Tensor Imaging (DTI) model to a DWI dataset.

**Usage:**
```bash
qmri-tools fit-dti -i <dwi.nii.gz> -o <output_dir> --bval <dwi.bval> --bvec <dwi.bvec>
```

**Options:**
- `-i, --input`: Input DWI NIfTI file (required).
- `-o, --output-dir`: Output directory (required).
- `--bval`: Path to bval file (required).
- `--bvec`: Path to bvec file (required).
- `-m, --mask`: Path to brain mask (optional but recommended).
- `--method`: Fitting method. Options: `WLLS` (default), `OLS`, `NLLS`, `RESTORE`.
- `--nthreads`: Number of parallel threads (default: 1).
- `--smoothing`: Optional Gaussian smoothing sigma (or FWHM) to apply before fitting.
- `--metric`: List of metrics to save. Default: `fa`, `md`, `ad`, `rd`, `color_fa`, `tensor`.
- `--grad-nonlin`: Path to gradient nonlinearity tensor file for voxel-wise correction.

---

### 2. DKI Fitting (`fit-dki`)

Fit a Diffusion Kurtosis Imaging (DKI) model.

**Usage:**
```bash
qmri-tools fit-dki -i <dwi.nii.gz> -o <output_dir> ...
```

**Options:**
- Standard inputs (`input`, `output-dir`, `bval`, `bvec`, `mask`, `nthreads`).
- `--mean-signal`: Enable Mean Signal DKI (MSDKI) for better robustness at high b-values.
- `--metric`: Default: `mk`, `ak`, `rk`, `fa`, `md`.
- `--grad-nonlin`: Path to gradient nonlinearity tensor file.

---

### 3. NODDI Fitting (`fit-noddi`)

Fit the Neurite Orientation Dispersion and Density Imaging (NODDI) model. This tool supports both `dmipy` (default, highly configurable) and `amico` (faster) backends.

**Usage:**
```bash
qmri-tools fit-noddi -i <dwi.nii.gz> -o <output_dir> ...
```

**Key Options:**
- `--backend`: `dmipy` (default) or `amico`.
- `--parallel-diff`: Intrinsic parallel diffusivity (default: 1.7e-9).
- `--iso-diff`: Isotropic diffusivity (default: 3.0e-9).

**Advanced Config (Dmipy backend):**
- `--solver`: Optimization solver. Default: `brute2fine`. Other options supported by Dmipy.
- `--distribution`: Orientation distribution function. Default: `Watson`. Option: `Bingham`.
- `--model-type`: `standard` (default) or `smt` (Spherical Mean Technique variant).

**Example:**
```bash
# Fit NODDI using Bingham distribution and SMT model
qmri-tools fit-noddi \
    -i dwi.nii.gz \
    -o noddi_out \
    --bval dwi.bval \
    --bvec dwi.bvec \
    --backend dmipy \
    --distribution Bingham \
    --model-type smt
```

---

### 4. MAP-MRI Fitting (`fit-mapmri`)

Fit the Mean Apparent Propagator (MAP) MRI model.

**Usage:**
```bash
qmri-tools fit-mapmri -i <dwi.nii.gz> -o <output_dir> ...
```

**Options:**
- `--laplacian/--no-laplacian`: Use Laplacian regularization (default: True).
- `--positivity/--no-positivity`: Enforce positivity constraint (default: True).
- `--metric`: Default: `rtop`, `rtap`, `rtpp`, `qiv`, `msd`.
- `--grad-nonlin`: Path to gradient nonlinearity tensor file.

