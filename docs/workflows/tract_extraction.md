# Extracting Specific Tracts

`tract_specific` extracts named streamline bundles from the whole-brain MRtrix
tractogram. Definitions can use TractSeg masks or explicit inclusion and
exclusion ROIs. Tract extraction therefore requires `tractography.mrtrix`.

## Extract TractSeg bundles

```yaml
dmri:
  modeling:
    tractography:
      mrtrix:
        enabled: true
        algorithm: iFOD2
      tractseg:
        enabled: true
        options:
          preprocess: true
          bundles: [CST_L, CST_R, AF_L, AF_R]
      tract_specific:
        enabled: true
        bundles:
          - {name: CST_L, source: tractseg}
          - {name: CST_R, source: tractseg}
          - {name: AF_L, source: tractseg}
          - {name: AF_R, source: tractseg}
```

The requested name must match a mask filename produced by TractSeg. The masks
must be in diffusion/native space; the standard TractSeg integration performs
the inverse warp when preprocessing is enabled.

## Custom ROI definitions

```yaml
tract_specific:
  enabled: true
  bundles:
    - name: custom_motor
      source: roi
      include:
        - /path/to/motor_cortex.nii.gz
        - /path/to/cerebral_peduncle.nii.gz
      exclude:
        - /path/to/exclusion.nii.gz
      ends_only: false
```

Each inclusion mask must be traversed. Exclusion masks remove traversing
streamlines. `ends_only: true` applies the inclusion test to streamline
endpoints. Advanced `tckedit` arguments can be supplied under `options`.

Extracted tractograms are saved as BIDS-style derivatives such as
`MRtrix/tract_specific/bundles/sub-01_space-dwi_desc-CSTL_tractography.tck`,
with a matching JSON sidecar, and recorded under
`context["tractography"]["bundles"]`.

If a tract is unexpectedly empty, verify image orientation and alignment,
relax overly restrictive ROI combinations, and inspect the whole-brain
tractogram and masks together in `mrview`.
