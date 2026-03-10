# Native GE Gradient Nonlinearity

This backend adds a native GE gradient nonlinearity workflow that does not require raw DICOM access during preprocessing or modeling.

## Summary

1. During import, qmri-neuropipe converts DICOMs with `dcm2niix` or `dcm2bids`.
2. Optionally, an import-time gradient override step replaces incorrect DICOM-derived `.bval`/`.bvec` files with curated tables.
3. A GE metadata enrichment step reads representative source DICOMs once and appends a `GradientNonlinearityCorrection` block to each DWI JSON sidecar.
4. During dMRI processing, the `native_ge` backend:
   - computes the GNL tensor in the native/raw acquisition geometry
   - extracts native and final mean b0 images
   - rigidly registers native b0 to final b0
   - applies that rigid transform to the native GNL tensor

This matches the current project policy: use a rigid native-to-final mapping because the GNL field varies smoothly.

## Required Config

### Import

```yaml
pipeline: dmri
bids_dir: /path/to/bids/dataset
output_dir: /path/to/output/derivatives
work_dir: /path/to/output/work

import:
  auto_run: true
  dicom_dir: /path/to/source_dicoms/sub-01/ses-01
  subject: "01"
  session: "01"
  method: dcm2bids
  dcm2bids:
    config_file: /path/to/bids/code/dcm2bids_config.json
  gradient_overrides:
    enabled: true
    require_both: true
    stop_on_mismatch: true
    rules:
      - match:
          entities:
            dir: AP
            run: "01"
        bval: /path/to/correct/AP_run01.bval
        bvec: /path/to/correct/AP_run01.bvec
      - match:
          json_fields:
            PhaseEncodingDirection: i-
            SeriesDescription: dw500_pe0_out
        bval: /path/to/correct/pe0_out.bval
        bvec: /path/to/correct/pe0_out.bvec
  gnl_metadata:
    enabled: true
    manufacturer: GE
```

### dMRI preprocessing/modeling

```yaml
dmri:
  preprocessing:
    grad_nonlin:
      enabled: true
      method: native_ge
      coeff_file: /path/to/gw_coils_magnus.dat

  modeling:
    grad_nonlin:
      enabled: true
      method: native_ge
      coeff_file: /path/to/gw_coils_magnus.dat
```

## Sidecar Metadata

The import enrichment step writes:

```json
{
  "GradientNonlinearityCorrection": {
    "Manufacturer": "GE",
    "Method": "native_ge",
    "Source": "dicom_import",
    "IsocenterOffsetScannerRASmm": [x, y, z],
    "Derivation": {
      "PDBKeys": ["SLOC1", "ELOC1", "FOVCNT1", "FOVCNT2"],
      "PDBCenterScannerRASRelativeToIsocenterMm": [x, y, z],
      "NativeGeometryConvention": "make-L_ge_eval_frame"
    }
  }
}
```

If gradient overrides are enabled, the import step also writes:

```json
{
  "GradientTableOverride": {
    "Applied": true,
    "Source": "import.gradient_overrides",
    "MatchingRule": {
      "entities": {
        "dir": "AP",
        "run": "01"
      }
    },
    "ReplacementBval": "/path/to/correct/AP_run01.bval",
    "ReplacementBvec": "/path/to/correct/AP_run01.bvec"
  }
}
```

## Example

See [dmri_native_ge_gnl_example.yaml](/Users/deaniii/Developer/code/repos/qmri-neuropipe/src/qmri_neuropipe/examples/configs/dmri_native_ge_gnl_example.yaml).

## Import Command

Example import invocation for a GE dataset:

```bash
qmri-neuropipe import \
  --config /path/to/config.yaml \
  --dicom-dir /path/to/source_dicoms/sub-01/ses-01 \
  --output-dir /path/to/bids/dataset \
  --subject 01 \
  --session 01
```

## One-Command DICOM To Processing

If your config contains `import.dicom_dir`, qmri-neuropipe can import first and then continue directly into the dMRI pipeline:

```bash
qmri-neuropipe --config /path/to/config.yaml --pipeline dmri
```

Behavior:

1. If `import.auto_run` is omitted or `true`, qmri-neuropipe runs the import workflow first.
2. Imported NIfTI/BIDS outputs are written to `bids_dir`.
3. The normal preprocessing and modeling pipeline then runs against that BIDS dataset.

Notes:

- `bids_dir` is the import target BIDS dataset.
- `output_dir` remains the derivatives / processing output directory.
- For `dcm2bids`, automatic import requires exactly one subject; set `import.subject` and optionally `import.session` in the config, or provide a single `participant_label` / `session_label`.

How this works:

1. `--dicom-dir` tells the import workflow which source DICOM directory to scan and convert.
2. `dcm2bids` or `dcm2niix` converts that directory.
3. If `import.gnl_metadata.enabled: true`, qmri-neuropipe scans the same `--dicom-dir` for GE metadata and writes the derived isocenter offset into each matching DWI JSON sidecar.
4. If `import.gradient_overrides.enabled: true`, qmri-neuropipe matches imported DWI sidecars against the configured rules and replaces the generated `.bval`/`.bvec` files before preprocessing.
   Matching can use either BIDS entities or JSON metadata fields such as `PhaseEncodingDirection` and `SeriesDescription`.

Archive inputs are also supported:

- pass a single `.tgz` or `.tar.gz` file directly to `--dicom-dir`, or
- pass a directory containing one or more `.tgz`/`.tar.gz` archives

The import workflow will extract those archives into the configured work directory before conversion and GE metadata enrichment.

## Notes

- `native_ge` currently requires `numba`.
- The GE coefficient file must be supplied with `coeff_file`.
- If native and final grids already match, no rigid mapping step is applied.
- The rigid mapping is estimated from mean b0 images, not from full nonlinear distortion fields.
- The import step derives `IsocenterOffsetScannerRASmm` from the converted native NIfTI geometry so it matches the `make-L.py` convention rather than storing raw PDB center coordinates directly.
- Gradient overrides are best applied during import rather than by manual file edits afterward, because the sidecar provenance records which replacement tables were used.
- If `dcm2bids` emits `run-01`, `run-02`, etc. instead of `dir-AP`, `dir-PA`, match gradient overrides with `json_fields` or fix the `dcm2bids` config so it assigns distinct `dir` or `acq` entities.
