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

All images passed to TORTOISEProcess are staged as uncompressed `.nii` files,
including up/down DWI data, structural images, and the reorientation target;
the original `.nii.gz` files are not modified. TORTOISE is also given an
uncompressed output path. After it finishes successfully, neuropipe atomically
gzips and validates the image as a 4D NIfTI before exposing the final `.nii.gz`
to downstream steps. This avoids TORTOISE V4 compressed-NIfTI compatibility and
output-content/filename-extension mismatches.

The desired voxel-axis convention is explicit and defaults to RAS:

```yaml
dmri:
  preprocessing:
    reorient:
      enabled: true
      orientation: RAS  # Any valid code, for example LAS, LPS, or RPI
```

`target_orientation` is accepted as an alias for `orientation`. This is a
lossless axis permutation/flip based on the NIfTI affine, similar in purpose to
`fslreorient2std`; it is not registration or normalization to a template.

When reorientation and the TORTOISE stream are both enabled, neuropipe keeps
the acquired 4D DWI orientation for TORTOISE input and delegates the requested
orientation to TORTOISE's final-output `--reorientation` reference. TORTOISE
assumes that `SliceTiming` describes NIfTI axis 3 during
input processing; permuting the acquired slice axis before TORTOISE can make it
pad or crop the DWI to the number of timing entries. The final reference is a
3D b0 derived from the input and permuted/flipped without resampling, so its
matrix, resolution, affine, and field of view remain data-driven. Neuropipe
also rejects an input timing/axis mismatch before TORTOISE can crop it.

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
        registration_backend: synthmorph
        # SynthMorph uses rigid T1w-to-DWI and affine T1w-to-MNI models by
        # default. Optional global or stage-specific linear overrides:
        # synthmorph_model: affine
        # synthmorph_rigid_model: rigid
        # synthmorph_affine_model: affine
        # synthmorph_register_args: ""
```

`registration_backend` accepts `fsl`, `ants`, or `synthmorph`
(`mri_synthmorph` is also accepted as an alias). SynthMorph produces explicit
forward and inverse linear transforms, which are converted to ITK format before
the existing Synb0 transform chains are applied. Deformable SynthMorph models
are intentionally rejected here because Synb0 requires composable linear
T1w-to-DWI and T1w-to-MNI transforms.

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
data and with a Synb0-generated down image. If no acquired or synthesized T2w
is available, DRBUDDI uses the selected T1w instead; if neither structural is
available, correction continues without the optional structural input.
Explicit `structural_file` remains the highest-priority override. `T2Wreg`
remains strict and still requires an undistorted T2-weighted image.

Any acquired or synthesized anatomy passed to TORTOISE can be skull-stripped
privately with `structural_brain_masking.enabled: true`. `method` accepts the
pipeline's structural masking backends, including `synthstrip`, and `apply_to`
can be `structural`, `reorientation`, `all`, or a list. The source image remains
unchanged, image geometry is verified after masking, and the generated brain
image and mask are recorded in the processing context. For example:

```yaml
structural_brain_masking:
  enabled: true
  method: synthstrip
  apply_to: [structural, reorientation]
```

`coregistration_to_anatomy.output_resolution: native` passes the original DWI
voxel sizes and matrix dimensions to TORTOISE. `anatomical` passes the selected
anatomical reference's voxel sizes, matrix dimensions, and orientation. The
final-grid reference is selected independently from the T2w used for EPI
correction when an explicit reference is requested. With `reference: auto`, the
acquired or synthesized T2w selected for TORTOISE EPI correction is preferred
for `--reorientation`; only when no selected T2w exists does it fall back to the
other anatomical inputs. Set `reference: synthesized` to require creation or
reuse of the SuperSynth T2w. With `output_resolution: anatomical`, the selected
reference also defines the exact output grid.

TORTOISEV4 does not expose a direct input for an FSL topup field or a
conventional fieldmap in Hz; use DRBUDDI, Synb0-as-reverse-PE, or T2Wreg.

## Post-TORTOISE Synb0 and Topup

For a single acquired phase-encoding direction, neuropipe can let TORTOISE own
denoising, Gibbs correction, motion/eddy correction, slice-to-volume motion,
and outlier replacement, then run Synb0 and FSL Topup afterward:

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
      epi: off

    distcorr:
      method: synb0
      application: post_tortoise
      apply_method: jac
      synb0:
        registration_backend: ants
```

The resulting order is `TORTOISEProcess`, Synb0 estimation from the corrected
DWI, `topup`, and `applytopup`. The acquired b0 is deliberately the first Topup
input and `applytopup` uses acquisition row 1. TORTOISE's corrected b-values and
b-vectors are retained because susceptibility unwarping does not add a rigid
gradient rotation.

This mode performs a second interpolation during `applytopup`. It requires
`tortoise_v4.epi: off`, exactly one post-TORTOISE DWI stream, and an anatomical
image for Synb0. Native reverse-PE input is not yet supported in this mode; it
requires separately processing and retaining aligned up/down TORTOISE streams
before Topup estimation and application.
