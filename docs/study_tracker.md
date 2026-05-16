# Study Tracker

The **study tracking system** (`lib/common/tracker.py`, `lib/common/tracking.py`) maintains a multi-sheet Excel workbook that accumulates processing status, QC metrics, and ROI statistics across all subjects and sessions as the pipeline runs. When `tracker.enabled: true`, the workbook is updated after each subject completes — new subjects are appended and existing rows are upserted in place, so the file always reflects the current state of the full cohort.

After every forced (end-of-subject) save, the tracker automatically exports each populated sheet as an individual CSV into a `tracker_reports/` directory next to the Excel file. These CSVs accumulate across runs and are intended for easy cross-subject analysis without opening Excel.

## Enabling the Tracker

```yaml
tracker:
  enabled: true
  file: /data/derivatives/study_tracker.xlsx   # optional; defaults to output_dir/study_tracker.xlsx

study_name: MyStudy   # optional study identifier written to the Summary sheet
```

**Parameters**

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `tracker.enabled` | bool | `false` | Enable the study tracker |
| `tracker.file` | path | `{output_dir}/study_tracker.xlsx` | Absolute path to the Excel workbook |
| `study_name` | str | — | Optional identifier written to the Summary sheet header |

## Excel Sheet Layout

The workbook contains one sheet per data category. Sheets are created on first write and omitted if no data is available for a given modality.

| Sheet | Contents |
| --- | --- |
| `Summary` | High-level completion rates per modality across all subjects |
| `Subject_Metadata` | Demographics and scan metadata per subject/session |
| `Processing_Status` | Overall pipeline status per subject/session |
| `Anatomical_Status` | Step-by-step anatomical preprocessing status |
| `Diffusion_Status` | Diffusion preprocessing, model fits, and atlas registration status |
| `Relaxometry_Status` | Relaxometry preprocessing, model fits, and atlas registration status |
| `Quality_Metrics` | DWI motion (absolute/relative mm), SNR, outlier counts per subject |
| `{Atlas}_Metrics` | Per-atlas ROI metrics in wide format; one sheet per atlas (e.g. `JHU_Metrics`) |
| `Volume_Statistics` | Anatomical region volumes (mm³) from FreeSurfer or SuperSynth |
| `Processing_Details` | Fine-grained step-by-step parameters and tool versions |
| `Alert_History` | Processing alerts and warnings with timestamps |

> **Note:** Atlas metric sheets are named dynamically from the atlas key defined in the config (e.g. an atlas named `XTRACT` produces an `XTRACT_Metrics` sheet). If more than one atlas is configured, each gets its own sheet.

## Cohort CSV Export

After every end-of-subject save, the tracker exports each populated sheet as a CSV into `tracker_reports/` alongside the Excel file. The directory is created automatically if it does not exist.

Example exported files:
- `tracker_reports/Summary.csv`
- `tracker_reports/Relaxometry_Status.csv`
- `tracker_reports/JHU_Metrics.csv`
- `tracker_reports/Subject_Metadata.csv`
- `tracker_reports/Quality_Metrics.csv`

New subjects are appended via an Excel UPSERT before re-export, so each CSV always contains the full cohort rather than only the most recent subject.

> **Note:** CSVs are intended for downstream analysis in Python, R, or spreadsheet applications. The Excel file remains the authoritative source; CSVs are regenerated from it on each run.

## What Gets Recorded Per Modality

### Diffusion

The `Diffusion_Status` sheet records:

- Preprocessing step statuses: `Denoising`, `Gibbs_Ringing`, `Eddy_Correction`, `Topup`, `SynB0`, `Coregistration`
- Model fit statuses: DTI, DKI, NODDI, and any other enabled models
- Atlas registration and statistics extraction status per atlas
- Overall subject completion status

The `Quality_Metrics` sheet records:

- DWI motion: absolute and relative displacement (mm) from Eddy QC
- SNR estimate per b-value shell
- Outlier slice counts and percentages per b-value shell

### Relaxometry

The `Relaxometry_Status` sheet records:

- Preprocessing step statuses: `Denoising`, `Gibbs_Correction`, `Motion_Correction`, `B1_Mapping_Method`
- Model fit statuses: `DESPOT1`, `DESPOT2FM`, `mcDESPOT`
- Atlas registration and statistics extraction status per atlas
- Overall subject completion status

### Anatomical

The `Anatomical_Status` sheet records:

- Step statuses: `Denoising`, `Bias_Correction`, `Brain_Masking`, `Segmentation`, `Coregistration`, `Reorienting`
- Segmentation method used: `FreeSurfer`, `FSL_Anat`, or `SuperSynth`
- Normalization status

## Volume Statistics

If SuperSynth `compute_volumes: true` or FreeSurfer `recon-all` is enabled, anatomical region volumes (mm³) are written to the `Volume_Statistics` sheet. Each row corresponds to one region for one subject/session, with a `method` column indicating `freesurfer` or `supersynth`.

```yaml
anat:
  segmentation:
    method: supersynth
    compute_volumes: true   # triggers Volume_Statistics entries
```

## Self-Healing / Re-Run Behavior

The tracker automatically detects existing outputs on disk when a step was cached and skipped during a re-run. Steps that were completed in a prior run are recorded with their correct completion status even if they did not execute in the current run. This means the tracker remains accurate across incremental or partial re-runs without requiring a full reprocessing of the cohort.

## Concurrency

When multiple subjects are processed in parallel, the tracker uses a file-based lock (`study_tracker.xlsx.lock`) to serialize writes to the Excel workbook. Each subject acquires the lock, performs an upsert of its rows, releases the lock, and then exports CSVs. Subjects that cannot acquire the lock within the timeout period log a warning and skip the tracker update for that run.

> **Note:** The lock file is created next to the Excel file and removed after each write. If the pipeline is interrupted while holding the lock, the `.lock` file may need to be manually deleted before the next run.

## Full Configuration Example

```yaml
tracker:
  enabled: true
  file: /data/derivatives/MyStudy/study_tracker.xlsx

study_name: MyStudy

dmri:
  analysis:
    enabled: true
    atlases:
      JHU:
        labels: /data/atlases/JHU-ICBM-labels-1mm.nii.gz
        template: /data/atlases/JHU-ICBM-FA-1mm.nii.gz

relaxometry:
  analysis:
    enabled: true
    registration_target: reference
    registration_template: /data/atlases/MNI152_T1_1mm.nii.gz
    atlases:
      JHU:
        labels: /data/atlases/JHU-ICBM-labels-1mm.nii.gz

anat:
  segmentation:
    method: supersynth
    compute_volumes: true
```

With this configuration the tracker will produce an Excel workbook containing `Diffusion_Status`, `Relaxometry_Status`, `Anatomical_Status`, `JHU_Metrics`, `Volume_Statistics`, and the other standard sheets, with matching CSVs exported to `/data/derivatives/MyStudy/tracker_reports/` after each subject.
