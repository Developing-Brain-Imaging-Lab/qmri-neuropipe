# Native GE Gradient Nonlinearity

This backend adds a native GE gradient nonlinearity workflow that does not require raw DICOM access during preprocessing or modeling.

## Summary

1. During import, qmri-neuropipe converts DICOMs with `dcm2niix` or `dcm2bids`.
2. A GE metadata enrichment step reads representative source DICOMs once and appends a `GradientNonlinearityCorrection` block to each DWI JSON sidecar.
3. During dMRI processing, the `native_ge` backend:
   - computes the GNL tensor in the native/raw acquisition geometry
   - extracts native and final mean b0 images
   - rigidly registers native b0 to final b0
   - applies that rigid transform to the native GNL tensor
   - reorients tensor components using the rigid rotation

This matches the current project policy: use a rigid native-to-final mapping because the GNL field varies smoothly.

## Required Config

### Import

```yaml
import:
  method: dcm2bids
  dcm2bids:
    config_file: /path/to/bids/code/dcm2bids_config.json
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
      "PDBKeys": ["SLOC1", "ELOC1", "FOVCNT1", "FOVCNT2"]
    }
  }
}
```

## Example

See [dmri_native_ge_gnl_example.yaml](/Users/deaniii/Developer/code/repos/qmri-neuropipe/src/qmri_neuropipe/examples/configs/dmri_native_ge_gnl_example.yaml).

## Notes

- `native_ge` currently requires `numba`.
- The GE coefficient file must be supplied with `coeff_file`.
- If native and final grids already match, no rigid mapping step is applied.
- The rigid mapping is estimated from mean b0 images, not from full nonlinear distortion fields.
