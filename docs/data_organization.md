# Data Organization

`qmri-neuropipe` adheres to the [BIDS (Brain Imaging Data Structure)](https://bids.neuroimaging.io/) standard for inputs and produces structured derivatives.

## Input: BIDS Dataset

The pipeline expects a valid BIDS dataset.

If multiple anatomical acquisitions exist, they may all be present in `anat/`.
You can choose a specific `T1w` or `T2w` in the config using
`anat.input.t1w_match` / `anat.input.t2w_match`.

### Required Structure
```text
/path/to/bids/
├── dataset_description.json
├── sub-01/
│   ├── ses-01/
│   │   ├── anat/
│   │   │   ├── sub-01_ses-01_T1w.nii.gz
│   │   │   ├── sub-01_ses-01_T1w.json
│   │   │   ├── sub-01_ses-01_T2w.nii.gz  (Optional)
│   │   │   └── sub-01_ses-01_T2w.json
│   │   ├── dwi/
│   │   │   ├── sub-01_ses-01_dwi.nii.gz
│   │   │   ├── sub-01_ses-01_dwi.json
│   │   │   ├── sub-01_ses-01_dwi.bval
│   │   │   └── sub-01_ses-01_dwi.bvec
│   │   └── fmap/ (Optional for Distortion Correction)
│   │       ├── sub-01_ses-01_dir-AP_epi.nii.gz
│   │       └── sub-01_ses-01_dir-PA_epi.nii.gz
└── ...
```

## Output: Derivatives

Outputs are saved to the specified `--output-dir`. A `derivatives/qmri-neuropipe` folder is created if not already present (though currently the pipeline writes directly to the output dir structure if specified as such).

### Structure
```text
/path/to/derivatives/
├── sub-01/
│   ├── ses-01/
│   │   ├── anat/
│   │   │   ├── sub-01_ses-01_desc-preproc_T1w.nii.gz
│   │   │   ├── sub-01_ses-01_desc-preproc_T1w.json
│   │   │   ├── sub-01_ses-01_desc-preproc_mask.nii.gz
│   │   │   └── ...
│   │   ├── dwi/
│   │   │   ├── sub-01_ses-01_desc-preproc_dwi.nii.gz
│   │   │   ├── sub-01_ses-01_desc-preproc_dwi.bval
│   │   │   ├── sub-01_ses-01_desc-preproc_dwi.bvec
│   │   │   ├── sub-01_ses-01_desc-brain_mask.nii.gz
│   │   │   ├── qc/
│   │   │   └── ...
│   │   └── report.html  (Pipeline Report)
└── ...
```

### Working Directory
Intermediate files are stored in the work directory (default: `output_dir/work` or specified via `--work-dir`). These can be massive and should be cleaned up after successful processing if space is a concern.
