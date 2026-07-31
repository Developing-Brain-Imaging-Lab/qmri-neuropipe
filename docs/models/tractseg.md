# TractSeg Bundle Segmentation

For every accepted field and default, see {ref}`Tractography and tract analysis <tractography-and-tract-analysis>`.

TractSeg is a deep learning-based tool for automated bundle segmentation and tractography. `qmri-neuropipe` integrates TractSeg with enhanced preprocessing workflows and flexible bundle selection.

TractSeg bundle masks and MRtrix streamline tractography are separate outputs.
Enable MRtrix tractography when `.tck` streamlines, SIFT/SIFT2, tractometry, or
connectomes are required. TractSeg masks can then constrain named-tract
extraction; see [Extracting Specific Tracts](../workflows/tract_extraction.md).

## Features

- **Automated MNI Registration**: The pipeline handles registration to MNI space (1.25mm ISO) for segmentation and **automatically inverse-warps** the bundle masks back to native space.
    - **Manual Workflow (Preferred)**: Uses the pipeline's own normalization transforms (e.g. from `ants` or `synthmorph`) for better consistency and to avoid redundant registration.
    - **Internal Fallback**: If pipeline transforms are not available, it safely falls back to TractSeg's internal `--preprocess` mechanism.

## Configuration

Add the `tractseg` configuration under `dmri: modeling: tractography`:

```yaml
dmri:
  modeling:
    tractography:
      tractseg:
        enabled: true
        options:
          preprocess: true             # Register to MNI for segmentation, warp back to native
          bundles:                     # Optional: List of specific bundles (default is all)
            - "AF_L"
            - "AF_R"
            - "CST_L"
            - "CST_R"
            - "CC_1"
          bundle_specific_threshold: true
          super_resolution: false
          gpu: true
```

## Outputs

TractSeg outputs are saved in the `TractSeg/` directory within your modeling derivatives:

- `bundle_masks/`: Individual NIfTI masks for each segmented bundle in **native space** (if `preprocess: true`).
- `peaks.nii.gz`: Input peaks used for segmentation (generated from CSD FOD if available).

## Requirements

TractSeg requires a GPU for optimal performance. Ensure the `CUDA_VISIBLE_DEVICES` environment variable is set or passed via the `--use-gpu` flag.
