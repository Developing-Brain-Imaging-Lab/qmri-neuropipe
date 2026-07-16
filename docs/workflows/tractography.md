# MRtrix Tractography

The diffusion modeling workflow can generate whole-brain MRtrix3 tractograms
from a CSD fibre-orientation distribution (FOD) or an MRtrix-format tensor.
Probabilistic `iFOD2` tracking with anatomically constrained tractography (ACT)
and SIFT2 is the recommended general-purpose configuration.

## Recommended ACT workflow

```yaml
dmri:
  modeling:
    csd:
      enabled: true
      method: msmt_csd
    tractography:
      mrtrix:
        enabled: true
        algorithm: iFOD2
        select: 10000000
        cutoff: 0.06
        minlength: 10
        maxlength: 250
        act:
          enabled: true
          algorithm: fsl
          validate: true
          seed_gmwmi: true
          backtrack: true
          crop_at_gmwmi: true
        filtering:
          method: sift2
```

When ACT is enabled, qmri-neuropipe first uses `context["5tt"]` if the
anatomical workflow provided one. Otherwise it runs `5ttgen` on the
preprocessed T1w image. The result is checked with `5ttcheck`, and
`5tt2gmwmi` creates the grey-matter/white-matter interface seed image. ACT
fails clearly if neither a 5TT nor an anatomical image is available.

An existing five-tissue image can be supplied explicitly:

```yaml
act:
  enabled: true
  five_tt: /path/to/dwi-space_5tt.nii.gz
  validate: true
```

All masks and tissue images must be aligned to the tracking image.

## Algorithms

| Algorithm | Input | Type |
| --- | --- | --- |
| `iFOD2` | CSD FOD | Probabilistic; recommended |
| `SD_STREAM` | CSD FOD | Deterministic |
| `Tensor_Det` | MRtrix tensor | Deterministic |
| `Tensor_Prob` | MRtrix tensor | Probabilistic |

FOD algorithms automatically enable CSD fitting. Tensor algorithms
automatically enable DTI fitting and request an MRtrix-format tensor.

## Tracking without ACT

```yaml
tractography:
  mrtrix:
    enabled: true
    algorithm: SD_STREAM
    select: 1000000
    act:
      enabled: false
    filtering:
      method: none
```

Optional `seed_image`, `include`, and `exclude` values can constrain tracking.
Additional documented MRtrix flags can be passed through `options`.

## SIFT and SIFT2

Set `filtering.method` to `none`, `sift`, or `sift2`. SIFT produces a filtered
tractogram; SIFT2 preserves the tractogram and writes one weight per streamline.
Those weights are used automatically for connectomes when requested.

## Outputs

Outputs are written below `MRtrix/tractography/`:

```text
sub-01_ses-01_space-dwi_desc-wholebrainiFOD2_tractography.tck
sub-01_ses-01_space-dwi_desc-wholebrainiFOD2_tractography.json
sub-01_ses-01_space-dwi_desc-wholebrainiFOD2SIFT_tractography.tck  # SIFT only
sub-01_ses-01_space-dwi_desc-sift2_weights.tsv                     # SIFT2 only
```

Because tractography does not yet have a finalized stable BIDS specification,
these filenames follow the general BIDS-Derivatives convention for
non-standardized outputs: source entities are retained, `space-dwi` declares
the coordinate system, `desc-*` distinguishes the product, and JSON sidecars
record the algorithm, source FOD/tensor, ACT state, requested streamline count,
and relevant tissue image. This convention can be revised when the tractography
BIDS extension is finalized.

The structured paths are also available to later steps under
`context["tractography"]`. See [Extracting Specific Tracts](tract_extraction.md)
and [Tractometry and Connectomes](tractometry.md).

## Requirements and troubleshooting

MRtrix3 must be available on `PATH`. The `fsl` 5TT backend also requires FSL;
other `5ttgen` backends may require FreeSurfer. A missing FOD usually means CSD
did not complete or its output was not restored from cache. A missing 5TT error
means ACT was requested without usable anatomical inputs.
