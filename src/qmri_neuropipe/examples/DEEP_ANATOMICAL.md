# Optional FastSurfer and DeepPrep anatomy

Enable `anat.fastsurfer.enabled` and/or `anat.deepprep.enabled` in a normal
qmri-neuropipe configuration, and run the anatomical pipeline as usual. See
[the configuration example](configs/deep_anatomical.yaml). Both default to
disabled. They can run alongside existing recon-all, but each maintains its own
output tree. Neither replaces the native-space preprocessed T1w, brain mask, or
FreeSurfer context used by other modalities.

## Installation and execution

These are external tools, not new mandatory Python dependencies. Install
FastSurfer separately and expose `run_fastsurfer.sh` on PATH, or set `executable`
to its absolute path. A native DeepPrep installation must expose a compatible
`deepprep-docker` launcher; the official container is the recommended route.
Install Docker, Apptainer, or Singularity for the corresponding runtime and set
`image` explicitly. Docker takes an image tag/digest; Apptainer/Singularity can
take a local SIF or a supported image URI. Pin the release you validate. The
example uses the documented DeepPrep 25.1.0 CLI contract.

FastSurfer `mode: segmentation` runs the FastSurferCNN/VINN segmentation via
`--seg_only`. `mode: full` (default) includes recon-surf and morphometry.
Full FastSurfer and DeepPrep require an existing FreeSurfer license supplied as
`fs_license`. Segmentation-only FastSurfer does not require a license.

`device` defaults to `cpu`, or `auto` when the global `use_gpu` is true. An
explicit device wins. GPU containers use Docker `--gpus all` or Apptainer/
Singularity `--nv`, and need a compatible NVIDIA host setup. Native FastSurfer
can use `mps` on supported Macs. DeepPrep device names follow its own CLI
(`cpu`, `auto`, or GPU index), not FastSurfer's `cuda:N` syntax.

Both tools receive global `n_cpus`; DeepPrep also receives `memory_gb`, rounded
up to an integer. Full commands are captured by the existing execution and
provenance machinery. No shell interpolation is used for paths.

## Inputs and sessions

The default `input: raw` uses exactly one T1w from the workflow's selected
`t1w_files`. Use `anat.input.t1w_match` to resolve multiple acquisitions/runs.
T2w-only input is an error for an enabled backend. The existing anatomical
selector also accepts UNIT1 images as T1w candidates, but users should validate
such inputs for the external tool.

`input: preprocessed` selects `preprocessed_t1w` instead. This must still be a
full-head T1w suitable for the backend, not a skull-stripped or template-space
image. The input must be a nonempty NIfTI file.

DeepPrep always runs `--anat_only`. A small BIDS dataset containing a real copy
of the selected T1w, its available sidecar, and dataset description is staged in
`<subject/session anatomical work directory>/deepprep_input`. Its internal subject
has no session entity. Each original subject/session has a separate output root,
so DeepPrep cannot merge timepoints or inadvertently process other scans or BOLD.
The completion manifest records the original source path.

## Outputs and workflow context

With the default output root:

| Backend | Subject directory | Context keys |
|---|---|---|
| FastSurfer | `<output_dir>/fastsurfer/sub-ID[_ses-SESSION]` | `fastsurfer_dir`, `fastsurfer_subject_dir`, `fastsurfer_outputs` |
| DeepPrep | `<output_dir>/deepprep/sub-ID[_ses-SESSION]/Recon/sub-ID` | `deepprep_dir`, `deepprep_subject_dir`, `deepprep_outputs` |

Set a backend's `output_dir` to override its root. The `*_outputs` dictionaries
expose segmentation paths and, in full mode, bilateral white/pial surfaces,
thickness, spherical registration, and regional statistics. Files remain in the
backend's native FreeSurfer-style layout. These are not automatically converted
to BIDS surface files, resampled to qMRI, or ingested as recon-all tracker volumes.
DeepPrep's additional reports and derivatives stay in its output root.

## Completion, reruns, and failures

Reuse requires a successful qmri-neuropipe completion manifest, matching source
path/size/mtime and configuration/resources, and nonempty required output files.
Cached anatomical preprocessing cannot hide an incomplete enabled backend.
Changing the image file or configuration invalidates reuse. Replacing installed
models in place without changing configuration is not automatically detected;
force a rerun in that case.

Use `anat.preprocessing.rerun_from_step: fastsurfer` or `deepprep` for the
existing step rerun mechanism. Global `force` or `skip_existing: false` also
reruns enabled backends. To avoid external caches silently reusing outdated
products, a rerun preserves an existing backend subject directory (the full
session-specific root for DeepPrep) beside it as `*.previous-<unique-id>` and
starts a clean output tree. Backups can consume substantial disk space; remove
them after reviewing the new output. A failed run never writes a completion
manifest. Errors appear in the anatomical context/report and respect
`stop_on_error`.

The tests exercise CLI construction, session isolation, caching, reruns, missing
outputs, and failure propagation using mocked external execution. Real model
inference, image geometry, and biological accuracy require a local smoke test
and anatomical QC. Toddler suitability should be evaluated separately.

## Upstream references

- [FastSurfer CLI](https://deep-mi.org/FastSurfer/dev/scripts/RUN_FASTSURFER.html)
- [FastSurfer containers](https://deep-mi.org/FastSurfer/stable/overview/CONTAINERS.html)
- [DeepPrep CLI](https://deepprep.readthedocs.io/en/latest/usage_local.html)
- [DeepPrep outputs](https://deepprep.readthedocs.io/en/latest/outputs.html)
