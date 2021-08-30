# qmri-neuropipe

Remove carriage returns (`\r`) from the file.
Install bids (`pip install bids`)

It has four different pipelines (anatomical, diffusion, mcdespot, segmentation).

Currently the starting point is nifti files organized according to BIDS.

```./qmri-neuropipe.py --load_json testing/dti-proc-config.json --subject 10021 --session 01```


# TODOs
- help pages for each specific pipeline
- starting point with dicoms
