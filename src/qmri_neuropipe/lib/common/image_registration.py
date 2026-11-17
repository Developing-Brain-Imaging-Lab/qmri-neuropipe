"""Registration between two scalar 3D images."""

from pathlib import Path

from ...interfaces import ants, fireants, freesurfer, fsl


def estimate_and_resample(fixed, moving, directory, method, options, nthreads, *, fireants_transform="nonlinear"):
    directory.mkdir(parents=True)
    warped = directory / "warped.nii.gz"
    options = dict(options)
    if method in {"fnirt", "mmorf", "mmorph"}:
        interpolation = options.pop("interpolation", "spline")
        transform = fsl.nonlinear_registration(fixed, moving, directory / "warp.nii.gz",
            method=method, options=options, nthreads=nthreads)
        fsl.applywarp(moving, fixed, warped, warp=transform, interp=interpolation, force=True)
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
        filename = "warp.nii.gz" if fireants_transform == "nonlinear" else "transform.mat"
        options["transform_type"] = fireants_transform
        transform = fireants.registration(fixed, moving, directory / filename, **options)
        ants.apply_transforms(
            fixed, moving, warped, [transform], interpolator=interpolation, nthreads=nthreads,
        )
    elif method == "synthmorph":
        model = options.pop("model", "joint")
        extension = freesurfer.synthmorph_transform_extension(model)
        transform = freesurfer.mri_synthmorph_register(
            moving, fixed, directory / f"transform{extension}", output_image=warped,
            model=model, **options,
        )
    else:
        raise ValueError(f"Unsupported registration method: {method}")
    return warped, [transform]


_TRANSFORMS = {
    "ants": ("rigid", "affine", "nonlinear"),
    "fsl": ("rigid", "affine", "nonlinear"),
    "flirt": ("rigid", "affine"),
    "fnirt": ("nonlinear",),
    "mmorf": ("nonlinear",),
    "fireants": ("rigid", "affine", "nonlinear"),
    "synthmorph": ("rigid", "affine", "nonlinear"),
    "freesurfer": ("rigid", "affine"),
}


def register_images(
    fixed: Path,
    moving: Path,
    output_dir: Path,
    *,
    method: str,
    transform: str,
    options: dict | None = None,
    nthreads: int = 1,
) -> tuple[Path, list[Path]]:
    """Resample moving to the fixed grid and return the image and forward transforms."""
    import nibabel as nib

    method = method.lower()
    method = {"mmorph": "mmorf", "mri_synthmorph": "synthmorph"}.get(method, method)
    transform = transform.lower()
    if method not in _TRANSFORMS:
        raise ValueError(f"Unknown registration method {method!r}. Choose from {', '.join(_TRANSFORMS)}.")
    if transform not in _TRANSFORMS[method]:
        raise ValueError(f"{method} supports these transforms: {', '.join(_TRANSFORMS[method])}.")
    fixed, moving = fixed.resolve(), moving.resolve()
    output_dir = output_dir.resolve()
    for image in (fixed, moving):
        if len(nib.load(image).shape) != 3:
            raise ValueError(f"Expected a scalar 3D image: {image}")

    options = dict(options or {})
    if method == "ants":
        options["transform_type"] = {"rigid": "Rigid", "affine": "Affine", "nonlinear": "SyN"}[transform]
    elif method == "synthmorph":
        options["model"] = "joint" if transform == "nonlinear" else transform
    elif method == "fsl" and transform == "nonlinear":
        method = "fnirt"

    if method in {"fsl", "flirt", "freesurfer"}:
        dof = 6 if transform == "rigid" else 12
        interpolation = options.pop("interpolation", "trilinear")
        if method == "freesurfer" and options:
            raise ValueError(f"Unsupported FreeSurfer options: {', '.join(options)}.")
        if "dof" in options:
            raise ValueError("Use --transform to select the number of degrees of freedom.")
        output_dir.mkdir(parents=True)
        warped = output_dir / "warped.nii.gz"
        matrix = output_dir / "transform.mat"
        if method == "freesurfer":
            lta = freesurfer.mri_coreg(
                moving, fixed, output_dir / "transform.lta", dof=dof, nthreads=nthreads,
            )
            freesurfer.lta_to_fsl(lta, matrix, src=moving, trg=fixed)
            fsl.applywarp(moving, fixed, warped, premat=matrix, interp=interpolation)
            transforms = [lta]
        else:
            fsl.flirt(
                moving, fixed, warped, omat=matrix, dof=dof,
                cost=options.pop("cost", "normmi"),
                extra_opts={**options, "interp": interpolation},
            )
            transforms = [matrix]
    else:
        warped, transforms = estimate_and_resample(
            fixed, moving, output_dir, method, options, nthreads,
            fireants_transform=transform,
        )

    for path in [warped, *transforms]:
        if not path.is_file():
            raise RuntimeError(f"{method} did not create expected registration output: {path}")
    return warped, transforms
