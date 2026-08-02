# Diffusion motion and eddy-current correction

The diffusion preprocessing workflow accepts a unified
`dmri.preprocessing.motion_correction.method` setting for individual motion
backends, plus a top-level `dmri.preprocessing.tortoise_v4` stream for the
integrated TORTOISE preprocessing chain:

- `eddy`: existing FSL eddy backend. Set `reference_selection.enabled: true`
  to pass the native optimal b0 index through `--ref_scan_no`.
- `tortoise_v4`: TORTOISEV4 `TORTOISEProcess`. Earlier neuropipe outputs are
  passed explicitly with `--up_data`, `--ub`, and `--uv`; JSON files are staged
  beside their matching NIfTI files as required by v4. Corrected TORTOISE
  `.bvals/.bvecs` are returned as a normal `DWIFile`.
- `ants`: rigid volume motion correction with finite-strain b-vector rotation.
- `ants_native`: experimental local motion/eddy backend using ANTs affine
  volume registration and optional overlapping-slice-group-to-volume rigid
  refinement. The affine rotation component is extracted with polar
  decomposition and converted from ANTs LPS physical coordinates to NIfTI
  voxel coordinates before rotating FSL bvecs.

The native b0 selector is TORTOISE-inspired. It finds all volumes at or below
`b0_threshold`, aligns their foreground centers, evaluates pairwise local
squared normalized cross-correlation, writes the best pair average, and uses
one member of that pair as the zero-based reference index. Selection metrics
are written to `optimal_b0_selection.json`.

For slice-to-volume ANTs correction, a single 3xN FSL bvec table can represent
only the global volume rotation. The backend therefore also writes
`*_slicewise.bvec.npy` with shape `(volume, slice, xyz)` and records its path in
the JSON sidecar. Conventional downstream models continue to receive the
globally rotated FSL bvecs; slice-aware/voxelwise models can consume the
slicewise table.

TORTOISEV4 is incompatible with TORTOISEV3. This integration targets the v4
executables `TORTOISEProcess` and `TORTOISEProcess_cuda`.

## Complete TORTOISEV4 workflow

The adapter can let TORTOISE own the full preprocessing and final-data chain:

```yaml
dmri:
  preprocessing:
    tortoise_v4:
      denoising: for_final
      gibbs: true
      drift: linear
      correction_mode: quadratic
      slice_to_volume: true
      repol: true
      synb0_repol_policy: disable
      niter: 3
      nthreads: 8
      epi: DRBUDDI
      use_reverse_pe: true
      coregistration_to_anatomy:
        enabled: true
        reference: auto
        output_resolution: anatomical
      output_data_combination: JacConcat
      output_signal_redist_method: LSR
```

The presence of the top-level block enables the stream unless `enabled: false`
is set. The older `motion_correction.method: tortoise_v4` plus nested
`motion_correction.tortoise_v4` form remains a compatibility alias.

`nthreads` overrides the pipeline-wide `n_cpus` value for TORTOISE only. The
runner exports OpenMP, ITK, MKL, and OpenBLAS limits, including the hard
`OMP_THREAD_LIMIT` ceiling for TORTOISE's internal OpenMP configuration.

When TORTOISE owns a stage, neuropipe skips its duplicate denoising, Gibbs,
resampling, distortion-correction, and anatomical-coregistration steps. Native
opposite-PE series remain separate and are passed as `--up_data/--down_data`.
For `epi: DRBUDDI`, native reverse-PE input is inferred when neither reverse-PE
nor Synb0 behavior is specified explicitly.

For a single acquired PE series, a generated Synb0 can be passed as the
DRBUDDI down image:

```yaml
dmri:
  preprocessing:
    tortoise_v4:
      synb0:
        anatomical_input: auto
        registration: direct
        registration_backend: ants
```

The presence of `synb0` enables it unless `enabled: false` is set, implies
`epi: DRBUDDI`, and automatically uses `JacSep`, returning only the corrected
acquired up series to downstream modeling rather than concatenating synthetic
b0 volumes.

TORTOISE V4.1.1 segfaults in its final WLLS outlier model when `repol` is used
with a b0-only Synb0 down series. The adapter therefore defaults
`synb0_repol_policy` to `disable`, logs the change, and records requested versus
applied outlier replacement in the output JSON. `error` fails before execution;
`allow` preserves the raw TORTOISE behavior for testing newer releases. Native
reverse-PE DWI pairs continue to support `repol: true` normally.

`epi: T2Wreg` instead performs TORTOISE's b0-to-undistorted-T2W susceptibility
correction. `t2w_fallback.source: auto` (the default) prefers an acquired T2w;
when none exists, it runs `mri_super_synth` on the selected T1w or other
anatomical scan and converts the synthesized T2w to NIfTI. Set the source to
`synthesized` to prefer SuperSynth even when an acquired T2w exists, or to
`acquired` to prohibit synthesis.

For `epi: DRBUDDI`, set `t2w_fallback.use_for_drbuddi: true` to pass the same
selected T2w as TORTOISE's structural input. This works with native reverse-PE
data and with a Synb0-generated down image. Explicit `structural_file` remains
the highest-priority override.

`coregistration_to_anatomy.output_resolution: native` passes the original DWI
voxel sizes and matrix dimensions to TORTOISE. `anatomical` passes the selected
anatomical reference's voxel sizes, matrix dimensions, and orientation. The
final-grid reference is selected independently from the T2w used for EPI
correction. Set `coregistration_to_anatomy.reference: synthesized` to create or
reuse the SuperSynth T2w for TORTOISE's `--reorientation` target and, with
`output_resolution: anatomical`, its exact output grid.

TORTOISEV4 does not expose a direct input for an FSL topup field or a
conventional fieldmap in Hz; use DRBUDDI, Synb0-as-reverse-PE, or T2Wreg.
