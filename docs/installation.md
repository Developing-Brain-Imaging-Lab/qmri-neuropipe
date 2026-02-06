# Installation

`qmri-neuropipe` is a Python-based processing pipeline. It requires **Python 3.9+** and several external neuroimaging tools.

## Prerequisites

The following tools must be installed and accessible in your system path:
*   [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki) (e.g., `flirt`, `topup`, `eddy`)
*   [MRtrix3](https://www.mrtrix.org/) (e.g., `dwidenoise`, `mrdegibbs`)
*   [ANTs](http://stnava.github.io/ANTs/) (e.g., `N4BiasFieldCorrection`, `antsRegistration`)

## Setting up the Environment

It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts. You can use either **Conda** or Python's built-in **venv**.

### Option 1: Using Conda (Recommended)

1.  **Install Miniforge or Anaconda**: [Download Miniforge](https://github.com/conda-forge/miniforge) (lighter) or [Anaconda](https://www.anaconda.com/products/distribution).
2.  **Create a new environment**:
    ```bash
    conda create -n qmri python=3.10
    ```
3.  **Activate the environment**:
    ```bash
    conda activate qmri
    ```
4.  **Install the pipeline**:
    ```bash
    # Ensure you are in the repository root
    pip install -e .
    ```

### Option 2: Using venv

1.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    ```
2.  **Activate the environment**:
    *   **Linux/macOS**:
        ```bash
        source venv/bin/activate
        ```
    *   **Windows**:
        ```bash
        venv\Scripts\activate
        ```
3.  **Install the pipeline**:
    ```bash
    pip install -e .
    ```

## Installation Options

*   **Editable Install (Development)**:
    Use `pip install -e .` to install the package in editable mode. Changes to the source code will be immediately reflected without re-installing.
*   **Standard Install**:
    Use `pip install .` for a static installation.

## Optional Extras

Optional features are provided via extras. Install the ones you need:

```bash
# Everything
pip install -e .[all]

# Feature-specific
pip install -e .[amico]
pip install -e .[nifreeze]
pip install -e .[pyafq]
pip install -e .[tracker]
pip install -e .[reporting]
```

## Verify Installation

Check that the CLI tool is available:
```bash
qmri-neuropipe --version
```
