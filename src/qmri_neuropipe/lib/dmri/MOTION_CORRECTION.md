# Diffusion motion and eddy-current correction

The diffusion preprocessing workflow accepts a unified
`dmri.preprocessing.motion_correction.method` setting:

- `eddy`: existing FSL eddy backend. Set `reference_selection.enabled: true`
  to pass the native optimal b0 index through `--ref_scan_no`.
- `tortoise_v4`: TORTOISEV4 `TORTOISEProcess`. Earlier neuropipe outputs are
  passed explicitly with `--up_data`, `--ub`, `--uv`, and `--up_json`;
  denoising, Gibbs, and EPI correction are disabled here to avoid duplicating
  earlier pipeline stages. Corrected TORTOISE `.bvals/.bvecs` are returned as a
  normal `DWIFile`.
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
