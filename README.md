# qmri-neuropipe

Remove carriage returns (`\r`) from the file.
Install bids (`pip install bids`)

It has four different pipelines (anatomical, diffusion, mcdespot, segmentation).

Currently the starting point is nifti files organized according to BIDS.

# Required software (basic installation)
- Anaconda
- FSL
- MRtrix3
- AFNI
- HD-BET

# To install and activate `qmri` conda-environment:
 ```
 conda env create -f environment.yml
 conda activate qmri
 ```

# Python Packages Used
- ANTS
- ANTSPyNET
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
