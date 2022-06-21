# qmri-neuropipe

Remove carriage returns (`\r`) from the file.

It has four different pipelines (anatomical, diffusion, relax, segmentation).

Currently the starting point is nifti files organized according to BIDS.

# Required software (basic installation)
- CMake
- Anaconda
- FSL
- AFNI
- HD-BET

# To install and activate `qmri` conda-environment:
 ```
 git clone https://github.com/Developing-Brain-Imaging-Lab/qmri-neuropipe.git qmri-neuropipe --recurse-submodules
 conda env create -f environment.yml
 conda activate qmri
 ```

# Python Packages Used
- ANTS
- ANTSPyNET
- MRtrix3
- DIPY
- DMIPY
- DMRI-AMICO
- TractSeg

# Basic use cases
## DTI
```./qmri-neuropipe.py --load_json testing/dti-proc-config.json --subject 10021 --session 01```
## NODDI
## CSD
## Connectomes

# TODOs
- help pages for each specific pipeline
- starting point with dicoms
