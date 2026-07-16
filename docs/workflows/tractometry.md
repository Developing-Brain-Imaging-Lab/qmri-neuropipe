# Tractometry and Connectomes

qmri-neuropipe supports both voxel-mask ROI statistics and streamline-based
tractometry. The existing statistics workflow summarizes values inside bundle
masks. `tract_specific` uses MRtrix `tcksample` to sample maps along the
streamlines belonging to each extracted tract.

## Streamline tractometry

```yaml
dmri:
  modeling:
    dti:
      enabled: true
      metrics: [fa, md]
    noddi:
      enabled: true
    tractography:
      tract_specific:
        enabled: true
        bundles:
          - {name: CST_L, source: tractseg}
          - {name: CST_R, source: tractseg}
        metrics:
          - DTI.FA
          - DTI.MD
          - NODDI.NDI
          - NODDI.ODI
        streamline_statistic: mean
        profiles:
          enabled: true
          nodes: 100
        track_density:
          enabled: true
```

Metric names use `MODEL.METRIC` and refer to maps already produced by the
modeling workflow. A mapping may supply an arbitrary image:

```yaml
metrics:
  - myelin: /path/to/dwi-space_myelin_map.nii.gz
```

For each tract and metric, the pipeline writes per-streamline samples and a
`sub-*_space-dwi_desc-tractometry_stats.tsv` table containing the number of valid streamlines, mean,
median, and standard deviation. Optional track-density images are written next
to the table. Along-tract profiles first resample every streamline to the
configured number of nodes and write node-wise mean, median, and standard
deviation to `sub-*_space-dwi_desc-alongtract_profiles.tsv`.

## Connectomes

```yaml
tract_specific:
  enabled: true
  bundles: []
  connectome:
    enabled: true
    nodes: /path/to/dwi-space_parcellation.nii.gz
    use_sift2: true
    statistic: sum
    symmetric: true
    zero_diagonal: true
```

The node image must contain integer parcel labels and be aligned to the
tractogram. With `use_sift2: true`, the whole-brain SIFT2 weights are passed to
`tck2connectome`. The matrix is saved as
`MRtrix/tract_specific/sub-*_space-dwi_desc-connectome_connectivity.tsv`.

Do not interpret streamline count as a literal axon count. Tracking parameters,
parcel geometry, filtering, and data quality affect connectivity estimates;
keep these settings fixed across participants and retain the generated
provenance.
