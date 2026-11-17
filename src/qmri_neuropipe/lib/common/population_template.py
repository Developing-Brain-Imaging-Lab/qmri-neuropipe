"""Iterative population templates and reusable forward image transforms."""

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from ...interfaces import ants, freesurfer, fsl
from .image_registration import estimate_and_resample, register_images


def _cohort_mean(sources, output):
    lower_bounds = []
    upper_bounds = []
    voxel_sizes = []
    for source in sources:
        image = nib.as_closest_canonical(nib.load(source))
        corners = np.array(list(np.ndindex(2, 2, 2))) * (np.array(image.shape) - 1)
        world = nib.affines.apply_affine(image.affine, corners)
        lower_bounds.append(world.min(axis=0))
        upper_bounds.append(world.max(axis=0))
        voxel_sizes.append(nib.affines.voxel_sizes(image.affine))

    spacing = np.median(voxel_sizes, axis=0)
    lower = np.min(lower_bounds, axis=0)
    upper = np.max(upper_bounds, axis=0)
    shape = tuple(np.ceil((upper - lower) / spacing).astype(int) + 1)
    affine = np.diag([*spacing, 1.0])
    affine[:3, 3] = lower
    total = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.int32)
    for source in sources:
        image = nib.load(source)
        data = image.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError(f"Expected finite intensities in cohort image: {source}")
        resampled = resample_from_to(
            nib.Nifti1Image(data, image.affine), (shape, affine), order=1, cval=np.nan,
        )
        data = np.asarray(resampled.dataobj)
        covered = np.isfinite(data)
        total[covered] += data[covered]
        count += covered
    np.divide(total, count, out=total, where=count > 0)
    nib.save(nib.Nifti1Image(total.astype(np.float32), affine), output)


def _recenter_fireants_mean(mean, output, displacement, geometry, work_dir, nthreads):
    """Resample the intensity mean through the inverse mean displacement."""
    import ants as antspy

    field = antspy.from_numpy(displacement.astype(np.float32), has_components=True, **geometry)
    inverse = antspy.invert_displacement_field(
        field, field * -1, enforce_boundary_condition=False,
    )
    if not np.isfinite(inverse.numpy()).all():
        raise ValueError("FireANTs shape averaging produced a nonfinite inverse displacement.")
    inverse_path = work_dir / "inverse_mean_warp.nii.gz"
    antspy.image_write(inverse, str(inverse_path))
    ants.apply_transforms(mean, mean, output, [inverse_path], nthreads=nthreads)


def build_population_template(
    subjects: dict[str, Path],
    output_dir: Path,
    *,
    image_type: str,
    method: str = "ants",
    iterations: int = 4,
    initial_template: Path | None = None,
    initial_registration: str = "none",
    options: dict | None = None,
    nthreads: int = 1,
) -> Path:
    """Alternate registration and averaging on the seed grid, then fit final warps.

    Subjects are keyed by cohort identifier. All inputs must be scalar 3D images
    of the selected contrast or parameter map. Returns the transform manifest.
    """
    method = method.lower()
    method = {"fsl": "fnirt", "mmorph": "mmorf", "mri_synthmorph": "synthmorph"}.get(method, method)
    if method not in {"ants", "fireants", "synthmorph", "fnirt", "mmorf"}:
        raise ValueError(f"Unsupported population registration method: {method}")
    if iterations < 1 or len(subjects) < 2 or not image_type.strip():
        raise ValueError("Provide at least two subjects, a nonempty image type, and one or more iterations.")
    initial_registration = initial_registration.lower()
    if initial_registration not in {"none", "rigid", "affine"}:
        raise ValueError("Choose none, rigid, or affine for initial registration.")
    registration_options = dict(options or {})
    initial_options = registration_options.pop("initial_options", {})
    if not isinstance(initial_options, dict):
        raise ValueError("initial_options must be a JSON object.")
    if initial_options and initial_registration == "none":
        raise ValueError("Set --initial-registration to rigid or affine to use initial_options.")
    initial_method = "flirt" if method in {"fnirt", "mmorf"} else method
    shape_averaging = False
    if method == "fireants":
        shape_averaging = registration_options.pop("shape_averaging", True)
        if not isinstance(shape_averaging, bool):
            raise ValueError("FireANTs shape_averaging must be a JSON boolean.")
        initial_defaults = {
            key: registration_options[key]
            for key in ("device", "scales", "affine_iterations", "interpolation")
            if key in registration_options
        }
        if "initial_iterations" in registration_options:
            if initial_registration == "none":
                raise ValueError("Set --initial-registration to rigid or affine to use initial_iterations.")
            initial_defaults["iterations"] = registration_options.pop("initial_iterations")
        initial_options = {**initial_defaults, **initial_options}
        if shape_averaging:
            import ants as antspy

    sources = {key: Path(path).resolve() for key, path in sorted(subjects.items())}
    if len(set(sources.values())) != len(sources):
        raise ValueError("Provide a distinct source image for each cohort identifier.")
    seed = Path(initial_template).resolve() if initial_template else None
    inputs = list(sources.values())
    for source in [*inputs, *([seed] if seed else [])]:
        if len(nib.load(source).shape) != 3:
            raise ValueError(f"Expected a scalar 3D image: {source}")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    template = output_dir / "template.nii.gz"
    if seed is None:
        seed = output_dir / "initial_mean.nii.gz"
        _cohort_mean(inputs, seed)
    reference = nib.load(seed)
    fixed = seed
    with TemporaryDirectory(prefix="iterations-", dir=output_dir) as scratch:
        first_round = 0 if initial_registration == "none" else -1
        for iteration in range(first_round, iterations):
            initializing = iteration == -1
            update_shape = shape_averaging and not initializing
            total = np.zeros(reference.shape, dtype=np.float64)
            displacement_sum = np.zeros((*reference.shape, 3), dtype=np.float64) if update_shape else None
            round_dir = Path(scratch) / str(iteration)
            for index, source in enumerate(sources.values()):
                subject_dir = round_dir / str(index)
                if initializing:
                    warped, transforms = register_images(
                        fixed, source, subject_dir, method=initial_method,
                        transform=initial_registration, options=initial_options, nthreads=nthreads,
                    )
                else:
                    warped, transforms = estimate_and_resample(
                        fixed, source, subject_dir, method, registration_options, nthreads,
                    )
                image = nib.load(warped)
                if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-4):
                    raise ValueError(f"Registration output grid differs from the seed grid: {warped}")
                data = image.get_fdata(dtype=np.float32)
                if not np.isfinite(data).all():
                    raise ValueError(f"Registration produced nonfinite intensities: {source}")
                total += data
                if displacement_sum is not None:
                    warp_path, = transforms
                    if not np.allclose(nib.load(warp_path).affine, reference.affine, atol=1e-4):
                        raise ValueError(f"FireANTs warp grid differs from the seed grid: {warp_path}")
                    field = antspy.image_read(str(warp_path))
                    displacement = field.numpy()
                    if displacement.shape != displacement_sum.shape or not np.isfinite(displacement).all():
                        raise ValueError(f"Expected a finite 3D FireANTs displacement field: {warp_path}")
                    displacement_sum += displacement
            if initializing:
                fixed = output_dir / "initial_aligned_mean.nii.gz"
            else:
                fixed = template if iteration == iterations - 1 else round_dir / "mean.nii.gz"
            mean = round_dir / "intensity_mean.nii.gz" if update_shape else fixed
            nib.save(nib.Nifti1Image((total / len(sources)).astype(np.float32), reference.affine), mean)
            if displacement_sum is not None:
                _recenter_fireants_mean(
                    mean, fixed, displacement_sum / len(sources),
                    dict(origin=field.origin, spacing=field.spacing, direction=field.direction),
                    round_dir, nthreads,
                )
            # Keep only the current mean and release intermediate transforms.
            for path in round_dir.iterdir():
                if path.is_dir():
                    shutil.rmtree(path)

    records = {}
    for index, (key, source) in enumerate(sources.items()):
        _, transforms = estimate_and_resample(
            template, source, output_dir / "subjects" / f"{index:04d}", method, registration_options, nthreads,
        )
        if any(not path.is_file() for path in transforms) or not transforms:
            raise RuntimeError(f"Registration did not produce a forward transform for {key}.")
        records[key] = {
            "source": str(source),
            "transforms": [str(path.relative_to(output_dir)) for path in transforms],
        }

    manifest = output_dir / "transforms.json"
    manifest.write_text(json.dumps({
        "template": template.name,
        "image_type": image_type,
        "method": method,
        "iterations": iterations,
        **({"shape_averaging": shape_averaging} if method == "fireants" else {}),
        **({"initial_registration": initial_registration} if initial_registration != "none" else {}),
        "seed": str(seed),
        "subjects": records,
    }, indent=2) + "\n")
    return manifest


def apply_population_transform(
    manifest: Path,
    subject: str,
    moving: Path,
    output: Path,
    *,
    nearest: bool = False,
    nthreads: int = 1,
) -> Path:
    """Warp a scalar image already aligned to the subject's registration image."""
    manifest = Path(manifest)
    data = json.loads(manifest.read_text())
    transforms = [manifest.parent / path for path in data["subjects"][subject]["transforms"]]
    template = manifest.parent / data["template"]
    if data["method"] in {"fnirt", "mmorf", "mmorph"}:
        return fsl.applywarp(moving, template, output, warp=transforms[0],
                             interp="nn" if nearest else "trilinear", force=True)
    if data["method"] == "synthmorph":
        return freesurfer.mri_synthmorph_apply(
            moving, template, transforms[0], output,
            extra_args="-m nearest" if nearest else "", overwrite=True,
        )
    return ants.apply_transforms(
        template, moving, output, transforms, invert_transforms=[False] * len(transforms),
        interpolator="nearestNeighbor" if nearest else "linear", nthreads=nthreads,
    )
