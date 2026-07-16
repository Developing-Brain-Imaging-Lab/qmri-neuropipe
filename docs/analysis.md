# Atlas Registration and ROI Statistics

The **analysis module** (`lib/common/analysis.py`) provides two pipeline steps — `AtlasRegistrationStep` and `StatsExtractionStep` — that are shared across the diffusion, relaxometry, and anatomical workflows. `AtlasRegistrationStep` warps one or more MNI-space atlas label volumes into native subject space using ANTs, leveraging the same normalization transforms already computed during the pipeline run. `StatsExtractionStep` then samples the registered atlas labels against native-space metric maps to produce per-ROI summary statistics (mean, median, standard deviation).

Both steps are enabled via the `analysis` sub-key of the relevant modality block (`dmri.analysis`, `relaxometry.analysis`, or `anat.segmentation`) and run at the end of each modality's processing chain. Outputs land alongside the model/fit results: registered atlas images are written to an `atlases/` subdirectory and statistics CSVs to a `statistics/` subdirectory.

## Enabling Analysis

```yaml
dmri:          # or relaxometry: / anat:
  analysis:
    enabled: true
    atlases:
      JHU:
        labels: /path/to/JHU-ICBM-labels-1mm.nii.gz
        template: /path/to/JHU-ICBM-FA-1mm.nii.gz
      XTRACT:
        labels: /path/to/xtract_labels.nii.gz
        template: /path/to/MNI152_T1_1mm.nii.gz
      MyCustomAtlas:
        labels: /path/to/custom_atlas.nii.gz
        is_probabilistic: false
```

## Atlas Configuration

Each entry under `atlases` is a named atlas definition. The key (e.g. `JHU`, `XTRACT`) is used as a label in output filenames and in the study tracker.

### Per-Atlas Keys

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `labels` / `file` | path | required | Atlas label NIfTI (`.nii.gz`) in MNI or atlas template space |
| `template` | path | normalization template | Template image the atlas was built from; used as the fixed image for registration. **Must match the atlas contrast** (e.g. FA for JHU-ICBM-FA, T1w for MNI152_T1) |
| `lut` | path | — | Label lookup table (XML or TXT) mapping integer label IDs to region names; used to annotate output CSV rows |
| `is_probabilistic` | bool | `false` | Treat the atlas as probabilistic (4D); per-ROI statistics use a weighted mean |
| `interpolation` | str | `genericLabel` | Interpolation mode for label warping: `genericLabel` (recommended for discrete labels), `linear`, `nearestNeighbor` |

> **Note:** `genericLabel` interpolation is strongly preferred for integer-valued label atlases. Using `linear` on a label image will introduce partial-volume values that do not correspond to any defined region.

## Registration Target and Template Override

Atlas registration requires two native-space inputs: a **fixed image** (the subject's native-space reference) and a **template** (the image the atlas was originally built from, used as the moving image during registration).

### Registration Target

`registration_target` selects which native-space image is used as the fixed image when registering each atlas. Acceptable values are metric names produced by the current modality.

| Modality | Default `registration_target` |
| --- | --- |
| Diffusion | `FA` |
| Relaxometry | `reference` (the SPGR reference image) |
| Anatomical | T1w brain |

### Template Override

`registration_template` overrides the per-atlas `template` key for **all** atlases at once. This is particularly important when atlases built from FA contrast (e.g. `JHU-ICBM-FA`) are being registered for a relaxometry pipeline that uses a T1w reference — the registration must use a matching T1w template on the atlas side.

```yaml
relaxometry:
  analysis:
    enabled: true
    registration_target: reference        # use the SPGR reference as the fixed image
    registration_template: /path/to/MNI152_T1_1mm.nii.gz   # override ALL per-atlas templates
    atlases:
      JHU:
        labels: /path/to/JHU-ICBM-labels-1mm.nii.gz
        # template key here would be ignored — registration_template takes precedence
      XTRACT:
        labels: /path/to/xtract_labels.nii.gz
```

> **Note:** `atlas_template` is accepted as an alias for `registration_template`. If both are provided, `registration_template` takes precedence.

## Metrics to Extract

The `parameters` (or `metrics`) key controls which native-space metric maps are sampled during statistics extraction. It accepts several formats:

**Comma-separated string** — extract the named metrics from all available models:
```yaml
relaxometry:
  analysis:
    metrics: "T1, VFm, T2"
```

**List** — equivalent list form:
```yaml
dmri:
  analysis:
    metrics: [FA, MD]
```

**Per-model dict** — restrict extraction to specific metrics per model:
```yaml
relaxometry:
  analysis:
    metrics:
      DESPOT1: [T1]
      mcDESPOT: [VFm, T2m, MWF]
```

**Model:metric syntax** — colon-separated strings in a list:
```yaml
relaxometry:
  analysis:
    metrics:
      - DESPOT1:T1
      - mcDESPOT:VFm
```

If `metrics` / `parameters` is omitted, statistics are extracted from all metric maps produced by all enabled models.

## Orientation Handling

The step automatically applies `nib.as_closest_canonical()` to both metric maps and atlas files before processing. This resolves axis-ordering differences that can arise between fitting tools and reference images when images are stored in non-RAS orientations. 4D metric files (e.g. mcDESPOT outputs) are handled by extracting volume 0 as the scalar map prior to sampling.

## Options Reference

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Enable atlas registration and statistics extraction |
| `atlases` | dict | — | Named atlas definitions (see [Per-Atlas Keys](#per-atlas-keys)) |
| `registration_target` | str | `FA` (dmri) / `reference` (relaxometry) | Name of the native-space metric used as the fixed image for registration |
| `registration_template` | path | per-atlas `template` | Override template image for all atlases |
| `atlas_template` | path | — | Alias for `registration_template` |
| `parameters` / `metrics` | str, list, or dict | all available | Metric maps to extract statistics from |
| `atlas_threshold` | float | — | Intensity threshold applied to continuous or probabilistic atlases before label extraction |
| `include_zero_label` | bool | `false` | Include label 0 (background) in per-ROI statistics output |
| `background_label` | int | `0` | Integer index treated as background; excluded unless `include_zero_label` is true |

## Output Files

Outputs are written into subdirectories alongside the model fit results for the subject/session.

| File | Description |
| --- | --- |
| `atlases/sub-XX_ses-YY_desc-{AtlasName}_labels.nii.gz` | Atlas label volume registered to native subject space |
| `statistics/sub-XX_ses-YY_desc-{AtlasName}_stats.csv` | Per-ROI statistics for a single atlas (columns: `roi_name`, `model`, `metric`, `mean`, `median`, `std`) |
| `statistics/sub-XX_ses-YY_desc-roiStats_stats.csv` | Combined CSV concatenating rows across all atlases for the subject |

> **Note:** If a label lookup table (`lut`) is supplied, the `roi_name` column contains human-readable region names. Without a LUT, `roi_name` contains the integer label index.

## Legacy Single-Atlas Format

For backward compatibility, a single atlas can be configured using flat keys instead of the `atlases` dict. This format is deprecated and will not support multiple atlases.

```yaml
dmri:
  analysis:
    enabled: true
    atlas_file: /path/to/atlas.nii.gz
    atlas_labels: /path/to/labels.txt
    metrics: [FA, MD]
```

The `atlas_file` / `atlas_labels` keys are translated internally to a single entry in the `atlases` dict named `Default`.

## Full Configuration Example

```yaml
dmri:
  analysis:
    enabled: true
    registration_target: FA
    atlases:
      JHU:
        labels: /data/atlases/JHU-ICBM-labels-1mm.nii.gz
        template: /data/atlases/JHU-ICBM-FA-1mm.nii.gz
        lut: /data/atlases/JHU-labels.xml
        interpolation: genericLabel
      XTRACT:
        labels: /data/atlases/xtract_labels.nii.gz
        template: /data/atlases/MNI152_T1_1mm.nii.gz
    metrics: [FA, MD, AD, RD]

relaxometry:
  analysis:
    enabled: true
    registration_target: reference
    registration_template: /data/atlases/MNI152_T1_1mm.nii.gz
    atlases:
      JHU:
        labels: /data/atlases/JHU-ICBM-labels-1mm.nii.gz
        lut: /data/atlases/JHU-labels.xml
    metrics:
      DESPOT1: [T1]
      mcDESPOT: [VFm, T2m, MWF]
```
# Tract statistics and streamline tractometry

Bundle masks from TractSeg can be passed through the standard ROI statistics
workflow for voxel-weighted summaries. This is different from streamline
tractometry, which samples a metric along each extracted streamline. Enable
`dmri.modeling.tractography.tract_specific` for streamline sampling,
track-density maps, and connectomes. See
[Tractometry and Connectomes](workflows/tractometry.md).
