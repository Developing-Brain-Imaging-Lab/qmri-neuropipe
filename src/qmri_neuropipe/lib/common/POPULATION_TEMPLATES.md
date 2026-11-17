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
current template and takes their voxelwise mean. FireANTs can also recenter
that mean using the inverse average displacement field. The sampling grid
stays fixed. After averaging, the workflow fits a final transform from each
original image to the finished template. The manifest records those
transforms in application order.

Additional maps must already be spatially aligned with the subject's source
image. Use `--nearest` for discrete labels.

- `ants` uses ANTsPy SyN by default.
- `fireants` uses an affine initialization followed by greedy deformable
  registration. Install FireANTs separately. The default device is CUDA and
  application uses ANTsPy.
- `synthmorph` (also `mri_synthmorph`) uses FreeSurfer's joint model by default.
  Ensure the command and its model weights are available. Choose a backend
  appropriate to the contrast.

Use `--registration-options path/to/options.json` to load a JSON object
of backend settings. For ANTs, an example is `{"transform_type": "SyN"}`.
For FireANTs, the following JSON shows the defaults.

```json
{
  "device": "cuda",
  "scales": [4, 2, 1],
  "affine_iterations": [200, 100, 50],
  "iterations": [200, 100, 25],
  "interpolation": "linear",
  "shape_averaging": true
}
```

Both optimizer schedules must have one entry per scale, in coarse-to-fine
order. JSON `iterations` controls the deformable optimizer for each image
registration. CLI `--iterations` controls the number of cohort averaging
rounds. Interpolation is used when resampling the moving image.

For cohorts with differing head poses, add a rigid seed alignment pass with
any template backend:

```sh
qmri-tools build-template cohort.json fa-template --image-type FA --method fireants \
  --initial-registration rigid --iterations 4 --registration-options fireants-options.json
```

The pass aligns each input to the starting mean or the image provided by
`--initial-template`. It saves the resulting mean as
`initial_aligned_mean.nii.gz` before deformable refinement. Use
`--initial-registration affine` to allow scale and shear as well.
The `fsl` (FNIRT), `fnirt`, and `mmorf` (also `mmorph`) backends use
FLIRT for this pass. ANTs, SynthMorph, and FireANTs use their own linear
registration modes. Set `initial_options` to a nested JSON object of
settings for the linear backend, such as `{"initial_options": {"cost": "normmi"}}`
for FLIRT. For FireANTs, `initial_iterations` also sets the per-scale
schedule and defaults to `affine_iterations`.

Each refinement round registers the original images directly to the current
mean. For FireANTs, set `"shape_averaging": false` for intensity-only averaging.

For individual anatomical normalization, set
`preprocessing.normalization.method` to `fireants`, enable normalization, and
point `preprocessing.normalization.template` to the desired reference NIfTI.
Set FireANTs options under `preprocessing.normalization.options`.
