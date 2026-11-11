# Population templates

Create a cohort JSON object mapping identifiers to one 3D NIfTI each. Paths
are relative to the JSON file or absolute. Use the same contrast or parameter
map for each entry.

```json
{
  "sub-01": "sub-01_FA.nii.gz",
  "sub-02": "sub-02_FA.nii.gz"
}
```

```sh
qmri-tools build-template cohort.json fa-template --image-type FA --method ants --iterations 4
qmri-tools apply-template-transform fa-template/transforms.json sub-01 sub-01_MD.nii.gz sub-01_space-cohort_MD.nii.gz
```

The output directory must be new. By default, the builder resamples the cohort
into a common world-space grid spanning all input fields of view, with median
voxel spacing. It writes `initial_mean.nii.gz` using equal weights for all
images that cover each voxel. Header alignment determines how well the images
overlap in this initial mean. Use `--initial-template` to supply a starting
image and its grid. Each round registers the original subject images to the
current template and takes their voxelwise mean. The grid stays fixed and the
workflow does not perform ANTs-style average-warp shape recentering. After averaging,
the workflow fits a final transform from each original image to the finished
template. The manifest records those transforms in application order.

Additional maps must already be spatially aligned with the subject's source
image. Use `--nearest` for discrete labels.

- `ants` uses ANTsPy SyN by default.
- `fireants` uses an affine initialization followed by greedy deformable
  registration. Install FireANTs separately. The default device is CUDA and
  application uses ANTsPy.
- `synthmorph` (also `mri_synthmorph`) uses FreeSurfer's joint model by default.
  Ensure the command and its model weights are available. Choose a backend
  appropriate to the contrast.

Use `--registration-options` with a JSON object for backend settings, such as
`{"transform_type": "SyN"}` for ANTs or
`{"device": "cuda", "scales": [4, 2, 1]}` for FireANTs. FireANTs also
accepts `affine_iterations` and `iterations` schedules matching the scales.

For individual anatomical normalization, set
`preprocessing.normalization.method` to `fireants`, enable normalization, and
point `preprocessing.normalization.template` to the desired reference NIfTI.
Set FireANTs options under `preprocessing.normalization.options`.
