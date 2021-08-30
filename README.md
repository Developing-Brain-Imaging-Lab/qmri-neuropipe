# qmri-neuropipe

Remove carriage returns (`\r`) from the file.
Install bids (`pip install bids`)

It has four different pipelines (anatomical, diffusion, mcdespot, segmentation).

Currently the starting point is nifti files organized according to BIDS.

# Required software (basic installation)
- FSL
- MRtrix3
- ANTS
- ANTSPyNET
- Anaconda
- DIPY
- DMIPY
- AMICO
- AFNI
- HD-BET
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
