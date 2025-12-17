# Installation

`qmri-neuropipe` is a Python-based processing pipeline. It requires **Python 3.9+** and several external neuroimaging tools.

## Prerequisites

The following tools must be installed and accessible in your system path:
*   [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki) (e.g., `flirt`, `topup`, `eddy`)
*   [MRtrix3](https://www.mrtrix.org/) (e.g., `dwidenoise`, `mrdegibbs`)
*   [ANTs](http://stnava.github.io/ANTs/) (e.g., `N4BiasFieldCorrection`, `antsRegistration`)

## Python Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Developing-Brain-Imaging-Lab/qmri-neuropipe.git
    cd qmri-neuropipe
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the package:**
    ```bash
    pip install .
    ```

    Or for development (editable mode):
    ```bash
    pip install -e .
    ```

## Verify Installation

Check that the CLI tool is available:
```bash
qmri-neuropipe --version
```
