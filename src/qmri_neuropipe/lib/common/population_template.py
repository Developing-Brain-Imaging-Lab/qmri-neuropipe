"""Iterative population templates and reusable forward image transforms."""

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from ...interfaces import ants, fireants, freesurfer, fsl


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


def _register(fixed, moving, directory, method, options, nthreads):
    directory.mkdir(parents=True)
    warped = directory / "warped.nii.gz"
    options = dict(options)
    if method in {"fnirt", "mmorf", "mmorph"}:
        transform = fsl.nonlinear_registration(fixed, moving, directory / "warp.nii.gz",
            method=method, options=options, nthreads=nthreads)
        fsl.applywarp(moving, fixed, warped, warp=transform, force=True)
        return warped, [transform]
    if method == "ants":
        options.setdefault("transform_type", "SyN")
        if "interpolation" in options:
            options["interpolator"] = options.pop("interpolation")
        warped, transforms = ants.registration(
            fixed, moving, directory / "registration_", nthreads=nthreads, **options,
        )
        return Path(warped), [Path(transform) for transform in transforms]
    if method == "fireants":
        interpolation = options.pop("interpolation", "linear")
        transform = fireants.registration(fixed, moving, directory / "warp.nii.gz", **options)
        ants.apply_transforms(
            fixed, moving, warped, [transform], interpolator=interpolation, nthreads=nthreads,
        )
    else:
        model = options.pop("model", "joint")
        extension = freesurfer.synthmorph_transform_extension(model)
        transform = freesurfer.mri_synthmorph_register(
            moving, fixed, directory / f"transform{extension}", output_image=warped,
            model=model, **options,
        )
    return warped, [transform]


def build_population_template(
    subjects: dict[str, Path],
    output_dir: Path,
    *,
    image_type: str,
    method: str = "ants",
    iterations: int = 4,
    initial_template: Path | None = None,
    options: dict | None = None,
    nthreads: int = 1,
) -> Path:
    """Alternate registration and averaging on the seed grid, then fit final warps.

    Subjects are keyed by cohort identifier. All inputs must be scalar 3D images
    of the selected contrast or parameter map. Returns the transform manifest.
    """
    method = method.lower()
    if method == "mri_synthmorph":
        method = "synthmorph"
    if method not in {"ants", "fireants", "synthmorph", "fnirt", "mmorf", "mmorph"}:
        raise ValueError(f"Unsupported population registration method: {method}")
    if iterations < 1 or len(subjects) < 2 or not image_type.strip():
        raise ValueError("Provide at least two subjects, a nonempty image type, and one or more iterations.")

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
        for iteration in range(iterations):
            total = np.zeros(reference.shape, dtype=np.float64)
            round_dir = Path(scratch) / str(iteration)
            for index, source in enumerate(sources.values()):
                warped, _ = _register(fixed, source, round_dir / str(index), method, options or {}, nthreads)
                image = nib.load(warped)
                if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-4):
                    raise ValueError(f"Registration output grid differs from the seed grid: {warped}")
                data = image.get_fdata(dtype=np.float32)
                if not np.isfinite(data).all():
                    raise ValueError(f"Registration produced nonfinite intensities: {source}")
                total += data
            fixed = template if iteration == iterations - 1 else round_dir / "mean.nii.gz"
            nib.save(nib.Nifti1Image((total / len(sources)).astype(np.float32), reference.affine), fixed)
            # Keep only the current mean and release intermediate transforms.
            for path in round_dir.iterdir():
                if path.is_dir():
                    shutil.rmtree(path)

    records = {}
    for index, (key, source) in enumerate(sources.items()):
        _, transforms = _register(
            template, source, output_dir / "subjects" / f"{index:04d}", method, options or {}, nthreads,
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
