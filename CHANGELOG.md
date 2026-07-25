# Changelog

All notable changes to qmri-neuropipe are documented here.

## Unreleased

### dmipy 2.x backend

- Replaced the legacy `dmipy>=1.0` integration with the supported
  `dmipy-fit>=2.1,<2.2` analytical fitting backend.
- Added installation profiles for native CPU, JAX CPU, and CUDA 12 JAX
  execution.
- Added `--solver`, `--device`, `--gpu-device`, optimizer batching, persistent
  JAX compilation caching, compile diagnostics, and liveness heartbeats.
- Added a registry-driven `fit-dmipy` command and integrated pipeline support
  for running multiple allow-listed dmipy models.
- Added model inspection commands and construction, simulation, forward-parity,
  and synthetic optimizer-recovery merge gates.
- Migrated NODDI, SANDI, and microglia fitting through shared dmipy model
  factories and execution infrastructure.
- Added the Garcia-Hernandez et al. microglia/astrocyte activation model.
- Added separate small-delta, big-Delta, and echo-time acquisition inputs for
  timing-sensitive and multi-echo models.
- Added voxel-parallel gradient-nonlinearity correction for compatible JAX
  full-signal models.
- Added model-independent BIDS-style derivative names and JSON provenance
  sidecars.
- Added atomic completion manifests that invalidate incomplete, modified, or
  configuration-mismatched registry fits.
- Explicitly disabled dmipy-fit 2.1 NEXI JAX fitting because its released JAX
  parameterization is inconsistent with the native model.

### Licensing

- Adopted `AGPL-3.0-only` for qmri-neuropipe.
- Added third-party notices for the dual-licensed dmipy 2.x packages used under
  their AGPL terms.
