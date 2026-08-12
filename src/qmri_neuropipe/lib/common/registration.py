"""
Registration module for generic coregistration.

Supports coregistration of any two images using ANTs, FSL, or FreeSurfer.
"""

from pathlib import Path
from typing import Optional, Literal, Dict, Any
import logging
import shutil
import nibabel as nib
import numpy as np

from ...core import BaseProcessingStep, ValidationError, ProcessingError
from ...core.run import run_cmd
from ...core.types import DWIFile, ImageFile
from ...interfaces import ants, fsl, freesurfer, c3d, mrtrix
from ...core.utils import resolve_freesurfer_subjects_dir
from ...io.bids import build_bids_name
from ...core.utils import check_nifti_integrity
from .json_metadata import copy_json_with_metadata
from .spatial_transforms import write_transform_chain_to_sidecar


_SKULL_STRIP_OPTION_KEYS = {
    "skull_strip",
    "skull_strip_registration",
    "skullstrip",
    "skullstrip_registration",
    "brain_extract_registration",
    "brain_extraction",
}
_SKULL_STRIP_AUX_OPTION_KEYS = {
    "skull_strip_enabled",
    "skullstrip_enabled",
    "skull_strip_method",
    "skullstrip_method",
    "brain_extraction_method",
    "skull_strip_moving",
    "skull_strip_fixed",
    "skull_strip_use_gpu",
    "skull_strip_mask_input",
}
_ALL_SKULL_STRIP_OPTION_KEYS = _SKULL_STRIP_OPTION_KEYS | _SKULL_STRIP_AUX_OPTION_KEYS


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}
    return bool(value)


def _normalize_skull_strip_method(method: Any) -> str:
    value = str(method or "fsl").strip().lower()
    aliases = {
        "bet": "fsl",
        "fsl-bet": "fsl",
        "fsl_bet": "fsl",
        "hdbet": "hd-bet",
        "hd_bet": "hd-bet",
        "synth-strip": "synthstrip",
        "mri_synthstrip": "synthstrip",
    }
    return aliases.get(value, value)


def _preserve_header_only_bvec(
    input_bvec: Path,
    output_dir: Path,
    entities: Dict[str, Any],
    description: str,
) -> Path:
    """Copy unchanged image-space gradients for header-only registration."""
    output_bvec = output_dir / build_bids_name(
        {**entities, "desc": description},
        suffix="bvec",
        extension=".bvec",
    )
    shutil.copy(input_bvec, output_bvec)
    return output_bvec


def _flatten_registration_options(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    flattened = dict(options or {})
    nested = flattened.pop("options", None)
    if isinstance(nested, dict):
        merged = dict(nested)
        merged.update(flattened)
        return merged
    return flattened


def _bbregister_contrast(
    options: Dict[str, Any],
    *,
    is_dwi: bool,
    target_modality: str,
) -> str:
    """Resolve bbregister contrast from the moving image modality."""
    explicit = options.get("contrast_type")
    if explicit:
        return str(explicit).strip().lower()
    if is_dwi:
        return "dti"
    if target_modality == "T2w":
        return "t2"
    return "t1"


def _mrtrix_coregistration_grid_options(
    output_grid_ref: Path,
    output_resolution: str,
) -> Dict[str, Path]:
    """Return MRtrix grid options without reslicing native-space DWI data."""
    grid_options = {"strides": Path(output_grid_ref)}
    if str(output_resolution).lower() == "anatomical":
        grid_options["template"] = Path(output_grid_ref)
    return grid_options


def _first_config_value(config, keys, default=None):
    for key in keys:
        value = config.get(key) if hasattr(config, "get") else None
        if value not in (None, ""):
            return value
    return default


def _build_multivariate_extras(options: Dict[str, Any]) -> Optional[list[tuple]]:
    fixed_extras = options.get("registration_fixed_extras") or []
    moving_extras = options.get("registration_moving_extras") or []
    if not fixed_extras or not moving_extras:
        return None
    if len(fixed_extras) != len(moving_extras):
        raise ProcessingError(
            "registration_fixed_extras and registration_moving_extras must have the same length."
        )

    metric = options.get("multivariate_metric", "Mattes")
    weight = float(options.get("multivariate_weight", 0.5))
    sampling = int(options.get("multivariate_sampling", 32))
    return [
        (metric, fixed, moving, weight, sampling)
        for fixed, moving in zip(fixed_extras, moving_extras)
    ]


def _registration_skull_strip_config(options: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    options = _flatten_registration_options(options)
    raw = None
    for key in _SKULL_STRIP_OPTION_KEYS:
        if key in options:
            raw = options.get(key)
            break

    if raw is None:
        enabled = options.get("skull_strip_enabled")
        if enabled is None:
            enabled = options.get("skullstrip_enabled")
        if enabled is None:
            return None
        raw = enabled

    if isinstance(raw, dict):
        cfg = dict(raw)
        enabled = cfg.get("enabled", True)
    else:
        enabled = raw
        cfg = {}

    if not _normalize_bool(enabled):
        return None

    method = (
        cfg.get("method")
        or options.get("skull_strip_method")
        or options.get("skullstrip_method")
        or options.get("brain_extraction_method")
        or "fsl"
    )
    cfg["method"] = _normalize_skull_strip_method(method)
    cfg["strip_moving"] = _normalize_bool(cfg.get("strip_moving", options.get("skull_strip_moving", True)))
    cfg["strip_fixed"] = _normalize_bool(cfg.get("strip_fixed", options.get("skull_strip_fixed", True)))
    cfg["use_gpu"] = cfg.get("use_gpu", options.get("skull_strip_use_gpu", options.get("use_gpu", False)))
    cfg["mask_input"] = cfg.get("mask_input", options.get("skull_strip_mask_input", "b0"))
    return cfg


def _strip_for_registration(
    config,
    logger: logging.Logger,
    image_path: Path,
    output_dir: Path,
    label: str,
    skull_cfg: Dict[str, Any],
    nthreads: int,
    force: bool = False,
) -> Path:
    """Return a skull-stripped copy for transform estimation only."""
    from .mask import BrainMaskingStep

    image_path = Path(image_path)
    if not image_path.exists():
        raise ProcessingError(f"Cannot skull-strip missing registration {label} image: {image_path}")

    strip_dir = output_dir / "registration_skullstrip" / label
    strip_dir.mkdir(parents=True, exist_ok=True)
    method = str(skull_cfg.get("method", "fsl"))
    logger.info(
        f"Skull-stripping registration {label} image with {method}: {image_path.name}"
    )
    step = BrainMaskingStep(
        config=config,
        logger=logger,
        provenance=None,
        method=method,
        nthreads=nthreads,
        apply_mask=True,
        mask_input=str(skull_cfg.get("mask_input", "b0")),
        use_gpu=skull_cfg.get("use_gpu"),
    )
    result = step.run(image_path, strip_dir, force=force, nthreads=nthreads)
    stripped_path = Path(result.img if hasattr(result, "img") else result)
    if not stripped_path.exists():
        raise ProcessingError(
            f"Skull stripping did not produce a registration {label} image: {stripped_path}"
        )
    return stripped_path


def prepare_registration_images(
    config,
    logger: logging.Logger,
    moving: Path,
    fixed: Path,
    output_dir: Path,
    options: Optional[Dict[str, Any]],
    nthreads: int,
    force: bool = False,
) -> tuple[Path, Path, bool]:
    """
    Optionally skull-strip moving/fixed images for transform estimation.

    The returned paths must only be used to estimate the transform. Downstream
    transform application should still use the original moving image.
    """
    skull_cfg = _registration_skull_strip_config(options)
    moving_for_reg = Path(moving)
    fixed_for_reg = Path(fixed)
    if not skull_cfg:
        return moving_for_reg, fixed_for_reg, False

    changed = False
    if skull_cfg.get("strip_moving", True):
        moving_for_reg = _strip_for_registration(
            config, logger, moving_for_reg, output_dir, "moving", skull_cfg, nthreads, force=force
        )
        changed = True
    if skull_cfg.get("strip_fixed", True):
        fixed_for_reg = _strip_for_registration(
            config, logger, fixed_for_reg, output_dir, "fixed", skull_cfg, nthreads, force=force
        )
        changed = True
    return moving_for_reg, fixed_for_reg, changed


def prepare_multivariate_registration_images(
    config,
    logger: logging.Logger,
    fixed_images: list[Path],
    moving_images: list[Path],
    output_dir: Path,
    options: Optional[Dict[str, Any]],
    nthreads: int,
    force: bool = False,
) -> tuple[list[Path], list[Path]]:
    """Apply the registration-only skull-strip policy to extra metric pairs."""
    if len(fixed_images) != len(moving_images):
        raise ProcessingError(
            "Multivariate fixed and moving registration images must have "
            "matching lengths."
        )

    prepared_fixed: list[Path] = []
    prepared_moving: list[Path] = []
    for index, (fixed, moving) in enumerate(zip(fixed_images, moving_images)):
        moving_reg, fixed_reg, _ = prepare_registration_images(
            config,
            logger,
            Path(moving),
            Path(fixed),
            Path(output_dir) / "registration_multivariate" / str(index),
            options,
            nthreads,
            force=force,
        )
        prepared_fixed.append(fixed_reg)
        prepared_moving.append(moving_reg)
    return prepared_fixed, prepared_moving


def _ensure_fsl_registration_nifti(
    image_path: Path,
    output_dir: Path,
    label: str,
    logger: logging.Logger,
    force: bool = False,
) -> Path:
    """Convert MGH/MGZ registration inputs to NIfTI for reliable FSL I/O."""
    image_path = Path(image_path)
    if image_path.suffix.lower() not in {".mgh", ".mgz"}:
        return image_path

    output_path = output_dir / f"fsl_{label}_{image_path.stem}.nii.gz"
    if output_path.exists() and not force and check_nifti_integrity(output_path):
        return output_path

    logger.info(
        f"Converting FSL registration {label} from {image_path.suffix} to NIfTI: "
        f"{output_path.name}"
    )
    try:
        source = nib.load(str(image_path))
        converted = nib.Nifti1Image.from_image(source)
        nib.save(converted, str(output_path))
    except Exception as exc:
        raise ProcessingError(
            f"Could not convert FSL registration {label} image to NIfTI: "
            f"{image_path}"
        ) from exc

    if not check_nifti_integrity(output_path):
        raise ProcessingError(
            f"FSL registration {label} conversion produced an invalid NIfTI: "
            f"{output_path}"
        )
    return output_path


def _coregistration_output_reference(
    input_image: Path,
    registration_target: Path,
    application_fixed: Optional[Path],
    output_resolution: str,
) -> Path:
    """Resolve the grid used when applying a coregistration transform."""
    if str(output_resolution).lower() in {"dwi", "native"}:
        return Path(input_image)
    if application_fixed is not None:
        return Path(application_fixed)
    return Path(registration_target)


def _spatial_grids_match(first: Path, second: Path, atol: float = 1e-5) -> bool:
    """Return whether two NIfTI images use the same 3D lattice and affine."""
    first_image = nib.load(str(first))
    second_image = nib.load(str(second))
    return (
        first_image.shape[:3] == second_image.shape[:3]
        and np.allclose(first_image.affine, second_image.affine, rtol=1e-5, atol=atol)
    )


def _mrtrix_header_alignment_matches(
    output_file: Path,
    input_file: Path,
    transform_file: Path,
    atol: float = 1e-5,
) -> bool:
    """Return whether MRtrix applied a linear transform without regridding.

    Without ``-template``, ``mrtransform -linear`` retains the input voxel
    array and composes the inverse of MRtrix's fixed-to-moving transform into
    the image header.  This deliberately changes the affine, so native output
    must not be compared with the input using :func:`_spatial_grids_match`.
    """
    try:
        output_image = nib.load(str(output_file))
        input_image = nib.load(str(input_file))
        transform = np.asarray(np.loadtxt(transform_file), dtype=float)
    except (OSError, ValueError):
        return False
    if output_image.shape != input_image.shape:
        return False

    if transform.shape == (3, 4):
        transform = np.vstack([transform, [0.0, 0.0, 0.0, 1.0]])
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        return False

    try:
        expected_affine = np.linalg.inv(transform) @ input_image.affine
    except np.linalg.LinAlgError:
        return False
    return np.allclose(
        output_image.affine,
        expected_affine,
        rtol=1e-5,
        atol=atol,
    )


def _ants_affine_to_ras_matrix(transform_file: Path) -> np.ndarray:
    """Read an ANTs/ITK affine and return its moving-to-fixed RAS matrix.

    ANTs stores the affine used by image resampling as a fixed-to-moving
    (output-to-input) pull transform.  Header-only registration needs the
    opposite, moving-to-fixed mapping before composing it with the input
    voxel-to-world affine.
    """
    import ants as antspy

    transform = antspy.read_transform(str(transform_file))
    parameters = np.asarray(transform.parameters, dtype=float)
    fixed_parameters = np.asarray(transform.fixed_parameters, dtype=float)
    if parameters.size != 12 or fixed_parameters.size < 3:
        raise ProcessingError(
            "Header-only coregistration requires a 3D affine ANTs transform; "
            f"got {parameters.size} parameters from {transform_file}."
        )

    linear = parameters[:9].reshape(3, 3)
    if not np.allclose(linear.T @ linear, np.eye(3), rtol=1e-5, atol=1e-5) or not np.isclose(
        np.linalg.det(linear), 1.0, rtol=1e-5, atol=1e-5
    ):
        raise ProcessingError(
            "Header-only coregistration received a transform containing scale, shear, "
            f"or reflection: {transform_file}"
        )
    translation = parameters[9:12]
    center = fixed_parameters[:3]
    itk_lps = np.eye(4, dtype=float)
    itk_lps[:3, :3] = linear
    itk_lps[:3, 3] = translation + center - linear @ center
    lps_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    fixed_to_moving_ras = lps_ras @ itk_lps @ lps_ras
    return np.linalg.inv(fixed_to_moving_ras)


def _write_header_registered_image(
    input_file: Path,
    output_file: Path,
    world_transform: np.ndarray,
) -> Path:
    """Copy voxel data unchanged and compose a rigid transform into qform/sform."""
    source = nib.load(str(input_file))
    new_affine = np.asarray(world_transform, dtype=float) @ source.affine
    output = nib.Nifti1Image(np.asanyarray(source.dataobj), new_affine, source.header.copy())
    qform_code = int(source.header["qform_code"]) or 1
    sform_code = int(source.header["sform_code"]) or 1
    output.set_qform(new_affine, code=qform_code)
    output.set_sform(new_affine, code=sform_code)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, str(output_file))
    return Path(output_file)


def _apply_mrtrix_header_transform(
    input_file: Path,
    output_file: Path,
    itk_transform: Path,
    mrtrix_transform: Path,
    nthreads: int,
) -> np.ndarray:
    """Apply a rigid ITK transform to an image header without reslicing data."""
    # mrtransform -linear uses the reverse (output/fixed -> input/moving)
    # scanner-space convention. ANTsPy writes a binary ITK .mat that cannot be
    # consumed by transformconvert's text-only itk_import, so decode it and
    # write the required MRtrix matrix explicitly.
    moving_to_fixed = _ants_affine_to_ras_matrix(itk_transform)
    fixed_to_moving = np.linalg.inv(moving_to_fixed)
    Path(mrtrix_transform).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(mrtrix_transform, fixed_to_moving, fmt="%.15g")
    mrtrix.mrtransform(
        in_file=input_file,
        out_file=output_file,
        linear_transform=mrtrix_transform,
        template=None,
        interp=None,
        nthreads=nthreads,
        force=True,
    )
    return (
        nib.load(str(output_file)).affine
        @ np.linalg.inv(nib.load(str(input_file)).affine)
    )


def _candidate_freesurfer_subject_ids(context: Optional[dict], input_image=None, options: Optional[Dict[str, Any]] = None):
    options = options or {}
    explicit = options.get("subject_id") or options.get("fs_subject_id") or options.get("freesurfer_subject_id")
    if explicit:
        yield str(explicit)

    if context:
        for key in ("freesurfer_subject_id", "fs_subject_id"):
            if context.get(key):
                yield str(context[key])

    sub = context.get("subject") if context else None
    ses = context.get("session") if context else None
    if not sub and hasattr(input_image, "entities"):
        sub = input_image.entities.get("sub")
    if not ses and hasattr(input_image, "entities"):
        ses = input_image.entities.get("ses")

    if sub:
        sub = str(sub)
        values = []
        if not sub.startswith("sub-"):
            values.append(f"sub-{sub}")
        values.append(sub)
        for value in values:
            if ses and "_ses-" not in value:
                yield f"{value}_ses-{ses}"
            yield value


def _resolve_freesurfer_subject(config, context: Optional[dict], input_image=None, options: Optional[Dict[str, Any]] = None):
    options = options or {}

    if context and context.get("freesurfer_dir"):
        fs_subject_dir = Path(context["freesurfer_dir"])
        return fs_subject_dir.parent, fs_subject_dir.name

    subjects_dir = (
        options.get("subjects_dir")
        or options.get("freesurfer_subjects_dir")
        or _first_config_value(
            config,
            (
                "dmri.preprocessing.coregistration.subjects_dir",
                "dmri.preprocessing.coregistration.options.subjects_dir",
                "dmri.preprocessing.coregistration.freesurfer_subjects_dir",
                "anat.preprocessing.recon_all.subjects_dir",
                "anat.recon_all.subjects_dir",
                "freesurfer.subjects_dir",
            ),
        )
    )
    if subjects_dir:
        subjects_dir = Path(subjects_dir)
    else:
        try:
            subjects_dir = resolve_freesurfer_subjects_dir(config)
        except ValueError:
            subjects_dir = None

    if not subjects_dir:
        return None, None

    seen = set()
    for subject_id in _candidate_freesurfer_subject_ids(context, input_image, options):
        if subject_id in seen:
            continue
        seen.add(subject_id)
        if (subjects_dir / subject_id).exists():
            return subjects_dir, subject_id

    # Return the most specific candidate for a clearer downstream error even if
    # the directory does not exist yet.
    candidates = list(_candidate_freesurfer_subject_ids(context, input_image, options))
    return subjects_dir, candidates[-1] if candidates else None


def _resolve_freesurfer_anatomical_mgz(subject_dir: Path) -> Optional[Path]:
    """Return the best FreeSurfer anatomical MGZ for registration reference."""
    mri_dir = Path(subject_dir) / "mri"
    for name in ("orig.mgz", "native.mgz"):
        candidate = mri_dir / name
        if candidate.exists():
            return candidate
    return None


def _ensure_freesurfer_orig_mgz(subject_dir: Path, logger: logging.Logger) -> Optional[Path]:
    """Ensure bbregister can read mri/orig.mgz for recon-all-clinical subjects."""
    mri_dir = Path(subject_dir) / "mri"
    orig = mri_dir / "orig.mgz"
    native = mri_dir / "native.mgz"
    if orig.exists():
        return orig
    if not native.exists():
        return None

    try:
        orig.symlink_to(native.name)
        logger.info(f"Created FreeSurfer orig.mgz symlink to native.mgz for {subject_dir.name}.")
    except Exception as e:
        logger.warning(
            f"Could not symlink FreeSurfer orig.mgz to native.mgz ({e}); copying native.mgz instead."
        )
        shutil.copy2(native, orig)
    return orig


class NonlinearRegistrationStep(BaseProcessingStep):
    """
    Nonlinear registration (SyN) to a template.
    output: warped image and warp field.
    """
    
    def __init__(
        self,
        config,
        logger=None,
        provenance=None,
        template: Optional[Path] = None,
        method: str = "ants",
        options: Optional[Dict[str, Any]] = None,
        save_transforms: bool = True,
        space_entity: Optional[str] = None,
    ):
        super().__init__(config, logger, provenance)
        self.template = template # If None, must be in config or passed in run
        self.method = str(method or "ants").lower()
        self.options = _flatten_registration_options(options)
        self.save_transforms = save_transforms
        self.space_entity = str(space_entity or "Standard")

    def run(self, first_arg, output_dir: Path, template: Optional[Path]=None, **kwargs) -> Any:
        context, input_image = self.unpack_input(first_arg)
        if not input_image:
             raise ValidationError("No input image for nonlinear registration.")

        target = template or self.template or self.config.get("template")
        if not target:
             # Default to MNI?
             # For now require explicit template
             self.logger.warning("No template specified for NonlinearRegistration. Skipping.")
             return context if context else input_image
             
        target = Path(target)
        if not target.exists():
               raise ValidationError(f"Template not found: {target}")

        output_dir = self.get_step_output_dir(output_dir)
        entities = input_image.entities.copy() if hasattr(input_image, 'entities') else {}
        
        # New name
        # MNI or template space?
        # Usually 'space-MNI152NLin2009cAsym' etc.
        # Let's assume generic 'space-template' or just update desc.
        # "anat_proc" usually produces '..._space-MNI..._desc-preproc_T1w.nii.gz'
        
        # We'll rely on config to enforce naming if strict, else generic:
        new_desc = "std" # standardized space
        entities['space'] = kwargs.get("space_entity", self.space_entity)
        
        output_img = output_dir / build_bids_name({**entities, "desc": "norm"})
        output_transform = output_dir / build_bids_name({**entities, "desc": "norm", "suffix": "transform"})
        method = str(kwargs.get("method", self.method) or "ants").lower()
        options = dict(self.options)
        options.update(_flatten_registration_options(kwargs.get("options", {}) or {}))
        nthreads = kwargs.get("nthreads", getattr(self.config, "n_cpus", 1))

        transform_ref: Optional[Path] = None
        transform_kind = method
        expected_transform = output_transform
        if method == "fsl":
            expected_transform = output_dir / build_bids_name(
                {**entities, "desc": "norm", "suffix": "transform"},
                extension=".mat",
            )
        transform_ref = expected_transform
        
        if output_img.exists() and expected_transform.exists() and not kwargs.get('force', False):
             self.logger.info(f"Skipping nonlinear registration (exists): {output_img}")
        else:
             in_p = self._extract_path(input_image)
             moving_for_reg = in_p
             fixed_for_reg = target
             moving_for_reg, fixed_for_reg, registration_inputs_stripped = prepare_registration_images(
                 self.config,
                 self.logger,
                 moving_for_reg,
                 fixed_for_reg,
                 output_dir,
                 options,
                 nthreads,
                 force=kwargs.get("force", False),
             )
             if method == "ants":
                 transform_type = str(options.get("transform_type", "SyN"))
                 interpolator = str(options.get("interpolation", "linear"))
                 warped, transforms = ants.registration(
                     fixed_file=fixed_for_reg,
                     moving_file=moving_for_reg,
                     out_prefix=output_transform,
                     transform_type=transform_type,
                     interpolator=interpolator,
                     nthreads=nthreads,
                     **{
                         k: v for k, v in options.items()
                         if k not in {"transform_type", "interpolation"} | _ALL_SKULL_STRIP_OPTION_KEYS
                     }
                 )

                 if registration_inputs_stripped:
                     ants.apply_transforms(
                         fixed_file=fixed_for_reg,
                         moving_file=in_p,
                         out_file=output_img,
                         transforms=transforms,
                         interpolator=interpolator,
                         nthreads=nthreads,
                     )
                 else:
                     # ANTs outputs:
                     # prefixWarped.nii.gz -> output_img
                     # prefix1Warp.nii.gz -> forward warp
                     # prefix0GenericAffine.mat -> affine
                     import shutil
                     warped = output_transform.with_suffix("").parent / (output_transform.name + "Warped.nii.gz")
                     if warped.exists():
                         shutil.copy(warped, output_img)
                     elif warped and Path(warped).exists():
                         shutil.copy(warped, output_img)
                     else:
                         self.logger.warning("ANTs SyN completed but warped image not found?")
             elif method == "fsl":
                 output_mat = expected_transform
                 flirt_out = output_img
                 if registration_inputs_stripped:
                     flirt_out = output_dir / f"{output_img.stem}_registration_estimate.nii.gz"
                 flirt_opts: Dict[str, Any] = {}
                 if "interpolation" in options:
                     flirt_opts["interp"] = options["interpolation"]
                 for key, value in options.items():
                     if key in {"dof", "cost", "interpolation", "output_resolution"} | _ALL_SKULL_STRIP_OPTION_KEYS:
                         continue
                     flirt_opts[key] = value

                 fsl.flirt(
                     in_file=moving_for_reg,
                     ref_file=fixed_for_reg,
                     out_file=flirt_out,
                     omat=output_mat,
                     dof=int(options.get("dof", 6)),
                     cost=str(options.get("cost", "normmi")),
                     extra_opts=flirt_opts or None,
                 )
                 if registration_inputs_stripped:
                     fsl.flirt(
                         in_file=in_p,
                         ref_file=fixed_for_reg,
                         out_file=output_img,
                         extra_args=f"-applyxfm -init {output_mat} -interp {options.get('interpolation', 'trilinear')}",
                     )
             else:
                 raise ValidationError(f"Unsupported normalization method: {method}")

        result = ImageFile(img=output_img, entities=entities)
        
        if context:
             context["current_image"] = result
             if transform_ref is not None:
                 context["template_transform"] = transform_ref
                 context["template_transform_type"] = transform_kind
             return context
        return result


class CoregistrationStep(BaseProcessingStep):
    """
    Coregistration step (Moving <-> Fixed).
    
    Methods:
    - 'ants': Uses ANTs registration (Rigid).
    - 'fsl': Uses FSL FLIRT.
    - 'freesurfer': Uses FreeSurfer bbregister (for B0/T1w scenarios mostly).
    """
    
    def __init__(
        self,
        config,
        logger: Optional[logging.Logger] = None,
        provenance = None,
        method: Literal['ants', 'fsl', 'freesurfer'] = 'ants',
        options: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config, logger, provenance)
        self.method = method
        self.options = _flatten_registration_options(options)
        self.logger.info(f"Initialized CoregistrationStep with method: {method}")

    def validate_inputs(self, first_arg, **kwargs) -> None:
        pass

    def validate_outputs(self, result) -> None:
        pass

    def run(self, first_arg, output_dir: Path, target: Optional[Path]=None, options: Optional[Dict[str, Any]]=None, target_modality: str = "T1w", **kwargs) -> Any:
        
        # Unpack input
        context, input_image = self.unpack_input(first_arg)
        
        if input_image is None:
             raise ValidationError("No input image provided for coregistration.")

        if hasattr(input_image, 'img'):
            in_path = input_image.img
            # Check if this is DWIFile (or has bval attribute) to preserve type later
            is_dwi = isinstance(input_image, DWIFile) or (hasattr(input_image, 'bval') and input_image.bval is not None)
            entities = input_image.entities.copy() if hasattr(input_image, 'entities') else {}
        else:
            in_path = self._extract_path(input_image)
            is_dwi = False
            entities = {}

        if not in_path or not in_path.exists():
             raise ValidationError(f"Input image not found: {in_path}")

        # Try to infer target from context if not provided
        if not target and context:
             # Heuristic: look for files matching modality
             if target_modality == "T2w":
                  t2w = context.get('t2w_files', [])
                  if t2w: 
                       target = t2w[0].img
                  elif context.get('t1w_files'):
                       target = context.get('t1w_files')[0].img
             
             if not target:
                  t1w = context.get('t1w_files', [])
                  if t1w: 
                       target = t1w[0].img
                  elif context.get('t2w_files'):
                       target = context.get('t2w_files')[0].img
             # Add other lookups here if needed

        if not target:
             raise ProcessingError("CoregistrationStep requires a target image (reference).")

        target_path = self._extract_path(target)
        if not target_path.exists():
            raise ProcessingError(f"Coregistration target (reference) image not found: {target_path}")

        merged_options = dict(self.options)
        merged_options.update(_flatten_registration_options(options))
        options = merged_options
        fs_subjects_dir = None
        fs_subject_id = None
        fs_reference_mgz = None
        fs_registration_target = None
        uses_arbitrary_registration_pair = bool(
            options.get("registration_moving") and options.get("registration_fixed")
        )
        if self.method == "freesurfer" and not uses_arbitrary_registration_pair:
            fs_subjects_dir, fs_subject_id = _resolve_freesurfer_subject(
                self.config,
                context,
                input_image=input_image,
                options=options,
            )
            if not fs_subjects_dir or not fs_subject_id:
                raise ProcessingError(
                    "FreeSurfer coregistration requires a resolvable SUBJECTS_DIR and subject ID. "
                    "Set dmri.preprocessing.coregistration.options.subjects_dir or run anatomical recon-all first."
                )

            if fs_subjects_dir and fs_subject_id:
                fs_subject_dir = fs_subjects_dir / fs_subject_id
                fs_reference_mgz = _resolve_freesurfer_anatomical_mgz(fs_subject_dir)
                if fs_reference_mgz is None:
                    self.logger.warning(
                        f"FreeSurfer anatomical reference not found under {fs_subject_dir / 'mri'} "
                        "(checked orig.mgz, native.mgz). Falling back to target image."
                    )
                elif fs_reference_mgz.name == "native.mgz":
                    self.logger.info(
                        f"FreeSurfer orig.mgz not found for {fs_subject_id}; using recon-all-clinical native.mgz."
                    )
                self.logger.info(f"Using FreeSurfer subject for bbregister: {fs_subject_id} (SUBJECTS_DIR={fs_subjects_dir})")

        output_dir = self.get_step_output_dir(output_dir)
        
        apply_method = options.get('apply_method', 'native').lower() # 'native' or 'mrtrix'
        application_mode = str(options.get("application_mode", "resample")).lower()
        if application_mode not in {"resample", "header"}:
            raise ProcessingError(
                f"Unknown coregistration application_mode={application_mode!r}; "
                "expected 'resample' or 'header'."
            )
        
        from ...core.utils import get_nifti_stem
        input_stem = get_nifti_stem(in_path)
        
        new_desc = "coreg"
        # BIDS name generic approach often appends entities to a base or rebuilds.
        # "input_image.entities" is usually safer if available.
        # But if we just want to ensure NO double ext:
        # build_bids_name handles it if we pass clean entities.
        # If build_bids_name uses suffix, it appends ".extension" (usually).
        
        # Let's trust build_bids_name but ensure previous ".nii" isn't lingering in some entity value?
        # Usually entities are clean.
        
        output_img = output_dir / build_bids_name({**entities, "desc": new_desc})
        output_bvec = output_img.with_suffix("").with_suffix(".bvec")
        output_bval = output_img.with_suffix("").with_suffix(".bval")
        
        # For transform filename:
        transform_name_full = build_bids_name({**entities, "desc": "coreg", "suffix": "transform"})
        
        # No extensions for base transform path
        output_transform = output_dir / get_nifti_stem(transform_name_full)
        
        # Define standard transform paths for both calculation and application
        output_mat = output_transform.with_suffix(".mat")
        mrtrix_transform = output_dir / "transform_mrtrix.txt"
        reg_lta = output_dir / "bbregister.lta" # for FreeSurfer

        # Determine nthreads
        nthreads = kwargs.get('nthreads', self.config.n_cpus)

        if self.method == "freesurfer" and fs_reference_mgz is not None:
            fs_registration_target = output_dir / f"freesurfer_{fs_reference_mgz.stem}.nii.gz"
            if not fs_registration_target.exists() or kwargs.get("force", False):
                self.logger.info(
                    f"Converting FreeSurfer {fs_reference_mgz.name} to NIfTI for registration reference..."
                )
                freesurfer.mri_convert(fs_reference_mgz, fs_registration_target)

        # --- PRE-REGISTRATION: Extract Reference for Calculation/Application ---
        # We extract this even if should_run is False, as it may be needed for mask transform conversion
        moving_for_reg = in_path
        if is_dwi and not options.get("registration_moving"):
            # Modality-specific reference extraction
            if target_modality in ["T1w", "T2w"]:
                self.logger.info(f"Target is {target_modality}: Extracting and averaging non-b0 volumes for coregistration reference...")
                avg_dwi_path = output_dir / "temp_avg_dwi_ref.nii.gz"
                if not avg_dwi_path.exists() or kwargs.get('force', False):
                    mrtrix.dwiextract(input_image, avg_dwi_path, no_bzero=True, nthreads=nthreads, force=True)
                    mrtrix.mrmath(avg_dwi_path, "mean", avg_dwi_path, axis=3, nthreads=nthreads, force=True)
                moving_for_reg = avg_dwi_path
            else:
                self.logger.info(f"Target is {target_modality}: Extracting and averaging b0 volumes for coregistration reference...")
                avg_b0_path = output_dir / "temp_avg_b0_ref.nii.gz"
                if not avg_b0_path.exists() or kwargs.get('force', False):
                    try:
                        mrtrix.dwiextract(input_image, avg_b0_path, bzero=True, nthreads=nthreads, force=True)
                        mrtrix.mrmath(avg_b0_path, "mean", avg_b0_path, axis=3, nthreads=nthreads, force=True)
                    except Exception as e:
                        self.logger.warning(f"MRtrix extraction failed: {e}. Falling back to total series mean via fslmaths.")
                        run_cmd(f"fslmaths {in_path} -Tmean {avg_b0_path}", label="calculate_total_mean_ref")
                moving_for_reg = avg_b0_path
        
        # --- GRID CONSOLIDATION: Resample target if native resolution requested ---
        options = options or {}
        out_res = options.get('output_resolution', 'anatomical').lower()
        if application_mode == "header":
            transform_type = str(options.get("transform_type", "Rigid"))
            if self.method != "ants":
                raise ProcessingError("Header-only coregistration currently requires method: ants.")
            if transform_type.lower() != "rigid":
                raise ProcessingError(
                    "Header-only coregistration requires transform_type: Rigid; "
                    f"got {transform_type!r}."
                )
            if out_res not in {"dwi", "native"}:
                raise ProcessingError(
                    "Header-only coregistration preserves the DWI grid and therefore "
                    "requires output_resolution: native (or dwi). Use "
                    "application_mode: resample for anatomical-resolution output."
                )
        registration_target = fs_registration_target or target
        application_fixed = Path(options["application_fixed"]) if options.get("application_fixed") else None
        if application_fixed and not application_fixed.exists():
            raise ProcessingError(f"Transform application reference not found: {application_fixed}")
        registration_moving = Path(options["registration_moving"]) if options.get("registration_moving") else None
        if registration_moving:
            if not registration_moving.exists():
                raise ProcessingError(f"Registration moving image not found: {registration_moving}")
            self.logger.info(f"Using alternate moving image for transform estimation: {registration_moving.name}")
            moving_for_reg = registration_moving

        registration_fixed = Path(options["registration_fixed"]) if options.get("registration_fixed") else None
        if registration_fixed:
            if not registration_fixed.exists():
                raise ProcessingError(f"Registration fixed image not found: {registration_fixed}")
            self.logger.info(f"Using alternate fixed image for transform estimation: {registration_fixed.name}")
            registration_target = registration_fixed
            target_path = registration_fixed
            target = registration_fixed

        if out_res in {'dwi', 'native'}:
             self.logger.info(
                 "Native output mode: estimating registration against the original "
                 f"full-resolution {target_modality}; output-grid handling is deferred "
                 "until transform application."
             )

        # The output geometry is distinct from the images used to estimate the
        # transform. Resampling back to the native grid uses the exact input
        # DWI lattice. MRtrix native application instead retains the voxel
        # array and moves that lattice into anatomical space via its header.
        output_grid_ref = _coregistration_output_reference(
            in_path,
            registration_target,
            application_fixed,
            out_res,
        )
        mrtrix_native_header_alignment = (
            apply_method == "mrtrix"
            and application_mode != "header"
            and out_res in {"dwi", "native"}
        )
        if out_res in {"dwi", "native"}:
            if mrtrix_native_header_alignment:
                self.logger.info(
                    "Native MRtrix mode: retaining the DWI voxel array and applying "
                    "the registration through its image header."
                )
            else:
                self.logger.info(
                    "Native resolution mode: applying the transform on the exact input "
                    f"DWI grid ({output_grid_ref.name})."
                )

        # Skip main coregistration if output exists and is valid
        should_run = True
        if output_img.exists() and not kwargs.get('force', False):
             # 0. Check Integrity
             if not check_nifti_integrity(output_img):
                  self.logger.warning(f"Output file corrupted: {output_img}. Re-running.")
                  should_run = True
             elif is_dwi and getattr(input_image, "bvec", None) and input_image.bvec.exists() and not output_bvec.exists():
                  self.logger.info(f"Coregistered DWI exists but rotated bvec is missing: {output_bvec}. Re-running.")
                  should_run = True
             elif is_dwi and getattr(input_image, "bval", None) and input_image.bval.exists() and not output_bval.exists():
                  self.logger.info(f"Coregistered DWI exists but bval sidecar is missing: {output_bval}. Re-running.")
                  should_run = True
             else:
                 # Check timestamps: if input is newer than output, we MUST re-run
                 in_mtime = in_path.stat().st_mtime
                 out_mtime = output_img.stat().st_mtime
                 
                 # Check dimensions consistency (especially for Outlier Removal re-runs)
                 dims_consistent = True
                 try:
                     in_shape = nib.load(in_path).shape
                     out_shape = nib.load(output_img).shape
                     # For 4D DWI, check 4th dim. For 3D, check all?
                     # Main issue is num volumes.
                     if len(in_shape) != len(out_shape):
                         dims_consistent = False
                         self.logger.info(f"Dimension mismatch (Rank: {len(in_shape)} vs {len(out_shape)}). Re-running.")
                     elif len(in_shape) == 4 and len(out_shape) == 4:
                         if in_shape[3] != out_shape[3]:
                             dims_consistent = False
                             self.logger.info(f"Dimension mismatch (In: {in_shape[3]}, Out: {out_shape[3]}). Re-running.")
                     expected_image = nib.load(str(output_grid_ref))
                     expected_spatial_shape = expected_image.shape[:3]
                     if mrtrix_native_header_alignment:
                         grid_matches = (
                             mrtrix_transform.exists()
                             and _mrtrix_header_alignment_matches(
                                 output_img,
                                 in_path,
                                 mrtrix_transform,
                             )
                         )
                     else:
                         grid_matches = (
                             out_shape[:3] == expected_spatial_shape
                             and (
                                 application_mode == "header"
                                 or np.allclose(
                                     nib.load(str(output_img)).affine,
                                     expected_image.affine,
                                     rtol=1e-5,
                                     atol=1e-5,
                                 )
                             )
                         )
                     if not grid_matches:
                         dims_consistent = False
                         self.logger.info(
                             "Coregistration output geometry mismatch "
                             f"(output={out_shape[:3]}, expected={expected_spatial_shape}). Re-running."
                         )
                 except Exception as e:
                     # If read fails, assume incompatible/bad output
                     self.logger.warning(f"Could not check dimensions: {e}. Re-running.")
                     dims_consistent = False
    
                 if not dims_consistent or in_mtime > out_mtime:
                     if in_mtime > out_mtime:
                          self.logger.info(f"Input ({in_path.name}) is newer than output. Re-running coregistration.")
                          self.logger.info(f"Debug: Input mtime={in_mtime}, Output mtime={out_mtime}, Diff={in_mtime-out_mtime:.2f}s")
                     else:
                          self.logger.info("Dimension mismatch (or read error). Re-running coregistration.")
                     should_run = True
                 else:
                     self.logger.info(f"Skipping coregistration (output exists and is up-to-date): {output_img}")
                     self.logger.debug(f"Debug: Input mtime={in_mtime}, Output mtime={out_mtime}, In Shape={in_shape}, Out Shape={out_shape}")
                     should_run = False
                 
        # Get nthreads from kwargs or config
        nthreads = kwargs.get('nthreads', self.config.n_cpus)
        
        mrtrix_rotated_bvecs = None # Track if MRTrix handled bvecs
        
        if should_run:
            self.logger.info(f"Running {self.method} coregistration with {nthreads} threads...")
            self.logger.info(f"Application method: {apply_method}")

            registration_inputs_stripped = False
            if self.method in {"ants", "fsl"}:
                moving_for_reg, registration_target, registration_inputs_stripped = prepare_registration_images(
                    self.config,
                    self.logger,
                    moving_for_reg,
                    registration_target,
                    output_dir,
                    options,
                    nthreads,
                    force=kwargs.get("force", False),
                )
                fixed_extras = options.get("registration_fixed_extras") or []
                moving_extras = options.get("registration_moving_extras") or []
                if fixed_extras or moving_extras:
                    prepared_fixed_extras, prepared_moving_extras = (
                        prepare_multivariate_registration_images(
                            self.config,
                            self.logger,
                            [Path(path) for path in fixed_extras],
                            [Path(path) for path in moving_extras],
                            output_dir,
                            options,
                            nthreads,
                            force=kwargs.get("force", False),
                        )
                    )
                    options["registration_fixed_extras"] = prepared_fixed_extras
                    options["registration_moving_extras"] = prepared_moving_extras
            if self.method == "fsl":
                moving_for_reg = _ensure_fsl_registration_nifti(
                    moving_for_reg,
                    output_dir,
                    "moving",
                    self.logger,
                    force=kwargs.get("force", False),
                )
                registration_target = _ensure_fsl_registration_nifti(
                    registration_target,
                    output_dir,
                    "fixed",
                    self.logger,
                    force=kwargs.get("force", False),
                )

            # --- Registration Options Processing ---
            dof = options.get("dof", 6)
            cost = options.get("cost", "normmi")
            
            # Known args to exclude from extra_opts
            known_args = [
                'dof', 'cost', 'extra_args', 'output_resolution', 'interpolation',
                'enabled', 'reference_image', 'method', 'wm_seg_method',
                'apply_method', 'application_mode', 'transform_type', 'registration_fixed',
                'registration_moving', 'application_fixed', 'registration_fixed_extras',
                'registration_moving_extras', 'multivariate_metric',
                'multivariate_weight', 'multivariate_sampling',
                'supersynth_registration', 'supersynth_input',
                'supersynth_mode', 'supersynth_device',
                'supersynth_sharpen_synths', 'supersynth_b0_threshold', 'force',
            ] + list(_ALL_SKULL_STRIP_OPTION_KEYS)
            fsl_opts = {k: v for k, v in options.items() if k not in known_args}

            # Setup BBR if requested
            if cost == 'bbr':
                try:
                    from ..anat.segmentation import generate_wm_segmentation
                    use_gpu = getattr(self.config, 'use_gpu', False)
                    wm_seg = generate_wm_segmentation(
                        target, 
                        output_dir, 
                        method=options.get('wm_seg_method', 'fast'), 
                        nthreads=nthreads,
                        gpu=use_gpu
                    )
                    fsl_opts['wmseg'] = wm_seg
                except Exception as e:
                    self.logger.warning(f"BBR setup failed: {e}. Falling back to default cost function.")
                    cost = "normmi"

            try:
                # --- Logic Branch: Application Method ---
                if apply_method == 'mrtrix' and application_mode != "header":
                    # MRTrix-based Coregistration (handles 4D and gradients correctly)
                    res_val = options.get('output_resolution', 'anatomical').lower()
                    self.logger.info(f"Executing Coregistration via MRtrix (Resolution: {res_val})...")
                    
                    # 1. Calculate Registration
                    transform_file = None
                    transform_type = 'flirt' 
                    
                    if self.method == 'freesurfer':
                         transform_file = output_mat
                         if registration_moving and registration_fixed:
                             freesurfer.mri_coreg(
                                 moving_file=moving_for_reg,
                                 reference_file=registration_target,
                                 out_lta=reg_lta,
                                 dof=int(dof),
                                 nthreads=nthreads,
                                 force=bool(kwargs.get("force", False)),
                             )
                             freesurfer.lta_to_fsl(
                                 reg_lta,
                                 transform_file,
                                 src=moving_for_reg,
                                 trg=registration_target,
                                 force=bool(kwargs.get("force", False)),
                             )
                         else:
                             if not fs_subjects_dir or not fs_subject_id:
                                 raise ProcessingError(
                                     "Unable to resolve FreeSurfer subject directory for bbregister."
                                 )
                             fs_subject_dir = fs_subjects_dir / fs_subject_id
                             if not _ensure_freesurfer_orig_mgz(fs_subject_dir, self.logger):
                                 raise ProcessingError(
                                     f"FreeSurfer bbregister requires {fs_subject_dir / 'mri' / 'orig.mgz'} "
                                     "or recon-all-clinical native.mgz."
                                 )

                             fs_contrast = _bbregister_contrast(
                                 options,
                                 is_dwi=is_dwi,
                                 target_modality=target_modality,
                             )

                             freesurfer.bbregister(
                                 in_file=moving_for_reg,
                                 target_file=fs_subject_id,
                                 out_reg_file=reg_lta,
                                 contrast_type=fs_contrast,
                                 fsl_mat_out=transform_file,
                                 subjects_dir=fs_subjects_dir
                             )

                             if not transform_file.exists():
                                  raise ProcessingError(
                                       f"FreeSurfer bbregister did not produce FSL matrix transform: {transform_file}"
                                  )
                         
                    elif self.method == 'ants':
                         transform_file = output_dir / "coreg_dwi_to_anat_0GenericAffine.mat"
                         out_prefix = output_dir / "coreg_dwi_to_anat_"
                         
                         ants.registration(
                             fixed_file=registration_target,
                             moving_file=moving_for_reg,
                             out_prefix=out_prefix,
                             transform_type='Rigid',
                             nthreads=nthreads
                         )
                         
                         if not transform_file.exists():
                              transform_file = Path(str(out_prefix) + "0GenericAffine.mat")
                         
                         if not transform_file.exists():
                              raise ProcessingError(f"ANTs registration failed to produce transform: {transform_file}")
                           
                         # Convert ANTs -> FSL for consistency
                         fsl_mat = output_dir / "coreg_dwi_to_anat_fsl.mat"
                         c3d.ants2fsl(registration_target, moving_for_reg, transform_file, fsl_mat)
                         transform_file = fsl_mat
                         
                    else:
                         transform_file = output_dir / "coreg_dwi_to_anat.mat"
                         temp_reg_out = output_dir / "temp_flirt_calc.nii.gz"
                         fsl.flirt(
                             in_file=moving_for_reg,
                             ref_file=registration_target,
                             out_file=temp_reg_out,
                             omat=transform_file,
                             dof=dof,
                             cost=cost,
                             extra_opts=fsl_opts
                         )
                         if temp_reg_out.exists(): temp_reg_out.unlink()

                    # 2. Apply via MRtrix
                    temp_mif_in = output_dir / "temp_input.mif"
                    
                    # Ensure Gradients are embedded in the MIF
                    conv_kwargs = {'in_file': in_path, 'out_file': temp_mif_in, 'nthreads': nthreads, 'force': True}
                    if is_dwi:
                         bvec_in = getattr(input_image, 'bvec', None)
                         bval_in = getattr(input_image, 'bval', None)
                         
                         if not bvec_in or not bval_in:
                             # Try sidecars based on filename
                             candidate_bvec = in_path.with_suffix("").with_suffix(".bvec")
                             # Reliable way:
                             base_path = str(in_path).split(".nii")[0]
                             candidate_bvec = Path(base_path + ".bvec")
                             candidate_bval = Path(base_path + ".bval")

                             if candidate_bvec.exists() and candidate_bval.exists():
                                 bvec_in = candidate_bvec
                                 bval_in = candidate_bval
                         
                         if bvec_in and bval_in:
                             conv_kwargs['in_bvec'] = bvec_in
                             conv_kwargs['in_bval'] = bval_in
                             
                    mrtrix.mrconvert(**conv_kwargs)
                    
                    mrtrix_transform = output_dir / "transform_mrtrix.txt"
                    mrtrix.transformconvert(
                        in_transform=transform_file,
                        out_mrtrix_transform=mrtrix_transform,
                        operation="flirt_import",
                        ref_image=registration_target,
                        in_image=moving_for_reg, 
                        force=True
                    )
                    
                    # Apply transform with STRIDES logic
                    temp_mif_out = output_dir / "temp_output.mif"
                    mrtrix_interp = options.get("interpolation", "linear").lower()
                    # Map standard interp names to mrtrix
                    if mrtrix_interp == 'linear': mrtrix_interp = 'linear'
                    elif mrtrix_interp == 'nearest': mrtrix_interp = 'nearest'
                    elif mrtrix_interp == 'sinc': mrtrix_interp = 'sinc'
                    elif mrtrix_interp == 'cubic': mrtrix_interp = 'cubic'

                    mt_kwargs = {
                        'in_file': temp_mif_in,
                        'out_file': temp_mif_out,
                        'linear_transform': mrtrix_transform,
                        'interp': mrtrix_interp,
                        'nthreads': nthreads,
                        'force': True
                    }
                    
                    # Without -template MRtrix retains the voxel array and composes
                    # the linear transform into the header. Anatomical output still
                    # uses -template and is resliced onto the requested target grid.
                    mt_kwargs.update(
                        _mrtrix_coregistration_grid_options(output_grid_ref, out_res)
                    )

                    mrtrix.mrtransform(**mt_kwargs)
                    
                    # Export to NIfTI
                    out_bvec = output_bvec
                    out_bval = output_bval
                    
                    mrtrix.mrconvert(
                        in_file=temp_mif_out,
                        out_file=output_img,
                        export_grad_fsl=(out_bvec, out_bval),
                        nthreads=nthreads,
                        force=True
                    )
                    
                    if temp_mif_in.exists(): temp_mif_in.unlink()
                    if temp_mif_out.exists(): temp_mif_out.unlink()
                    
                    mrtrix_rotated_bvecs = out_bvec

                else:
                    # --- STANDARD LOGIC ---
                    if self.method == 'ants':
                        transform_type = options.get("transform_type", "Rigid")
                        warped, prefix = ants.registration(
                            fixed_file=registration_target, 
                            moving_file=moving_for_reg, 
                            out_prefix=output_transform, 
                            transform_type=transform_type,
                            nthreads=nthreads,
                            multivariate_extras=_build_multivariate_extras(options),
                            **{k:v for k,v in options.items() if k not in known_args}
                        )
    
                        if is_dwi or registration_moving or registration_inputs_stripped:
                             if application_mode == "header":
                                 affine_transforms = [Path(t) for t in prefix if str(t).endswith(".mat")]
                                 if len(affine_transforms) != 1:
                                     raise ProcessingError(
                                         "Header-only coregistration expected exactly one rigid affine transform, "
                                         f"got: {prefix}"
                                     )
                                 # Delegate ITK convention conversion and header
                                 # composition to MRtrix. With -linear and no
                                 # -template, mrtransform changes the image header
                                 # transform without reslicing voxel data.
                                 header_mrtrix_transform = output_dir / "header_transform_mrtrix.txt"
                                 header_world_transform = _apply_mrtrix_header_transform(
                                     in_path,
                                     output_img,
                                     affine_transforms[0],
                                     header_mrtrix_transform,
                                     nthreads,
                                 )
                             else:
                                 apply_kwargs = {
                                     "fixed_file": output_grid_ref,
                                     "moving_file": in_path,
                                     "out_file": output_img,
                                     "transforms": prefix,
                                     "interpolator": options.get("interpolation", "linear"),
                                     "nthreads": nthreads,
                                 }
                                 if is_dwi:
                                     apply_kwargs["imagetype"] = 3
                                 ants.apply_transforms(**apply_kwargs)
                        elif warped and Path(warped).exists():
                             shutil.copy(warped, output_img)

                    elif self.method == 'fsl':
                        output_mat = output_transform.with_suffix(".mat")
                        estimate_mat = output_mat
                        flirt_out = output_img
                        if registration_moving:
                            estimate_mat = output_dir / f"{output_transform.name}_synthetic.mat"
                            flirt_out = output_dir / f"{output_transform.name}_synthetic.nii.gz"
                        if registration_inputs_stripped and not is_dwi:
                            flirt_out = output_dir / "temp_flirt_calc.nii.gz"
                        
                        # Ensure we clean up before running to force execution
                        if output_img.exists(): output_img.unlink()
                        if output_mat.exists(): output_mat.unlink()
                        if estimate_mat != output_mat and estimate_mat.exists():
                            estimate_mat.unlink()
                        
                        self.logger.info(f"DEBUG: Calling fsl.flirt with in={moving_for_reg}, ref={target}, out={output_img}, cost={cost}, dof={dof}")
                        fsl.flirt(
                            in_file=moving_for_reg, 
                            ref_file=registration_target, 
                            out_file=flirt_out, 
                            omat=estimate_mat,
                            dof=dof, 
                            cost=cost, 
                            extra_opts=fsl_opts
                        )
                        
                        if is_dwi:
                            apply_ref = output_grid_ref
                            if estimate_mat != output_mat:
                                itk_transform = output_dir / f"{output_transform.name}_synthetic_itk.txt"
                                c3d.fsl2ants(
                                    registration_target,
                                    moving_for_reg,
                                    estimate_mat,
                                    itk_transform,
                                )
                                c3d.ants2fsl(
                                    apply_ref,
                                    in_path,
                                    itk_transform,
                                    output_mat,
                                )
                            self.logger.info(f"Applying 4D transform to full DWI series using FSL (interp={options.get('interpolation', 'trilinear')})...")
                            fsl.apply_xfm_4d(
                                in_file=in_path, 
                                ref_file=apply_ref,
                                out_file=output_img, 
                                mat=output_mat,
                                interp=options.get("interpolation", "trilinear")
                            )
                        elif registration_moving:
                            apply_ref = output_grid_ref
                            itk_transform = (
                                output_dir
                                / f"{output_transform.name}_synthetic_itk.txt"
                            )
                            c3d.fsl2ants(
                                registration_target,
                                moving_for_reg,
                                estimate_mat,
                                itk_transform,
                            )
                            c3d.ants2fsl(
                                apply_ref,
                                in_path,
                                itk_transform,
                                output_mat,
                            )
                            fsl.flirt(
                                in_file=in_path,
                                ref_file=apply_ref,
                                out_file=output_img,
                                extra_args=(
                                    f"-applyxfm -init {output_mat} -interp "
                                    f"{options.get('interpolation', 'trilinear')}"
                                ),
                            )
                        elif registration_inputs_stripped:
                            fsl.flirt(
                                in_file=in_path,
                                ref_file=output_grid_ref,
                                out_file=output_img,
                                extra_args=f"-applyxfm -init {output_mat} -interp {options.get('interpolation', 'trilinear')}",
                            )
                    
                    elif self.method == 'freesurfer':
                        output_dat = output_transform.with_suffix(".dat")
                        if registration_moving and registration_fixed:
                            freesurfer.mri_coreg(
                                moving_file=moving_for_reg,
                                reference_file=registration_target,
                                out_lta=reg_lta,
                                dof=int(dof),
                                nthreads=nthreads,
                                force=bool(kwargs.get("force", False)),
                            )
                            apply_ref = output_grid_ref
                            freesurfer.lta_to_fsl(
                                reg_lta,
                                output_mat,
                                src=in_path,
                                trg=apply_ref,
                                force=bool(kwargs.get("force", False)),
                            )
                        else:
                            fs_contrast = _bbregister_contrast(
                                options,
                                is_dwi=is_dwi,
                                target_modality=target_modality,
                            )
                            if not fs_subjects_dir or not fs_subject_id:
                                raise ProcessingError(
                                    "Unable to resolve FreeSurfer subject directory for bbregister."
                                )
                            freesurfer.bbregister(
                                in_file=moving_for_reg,
                                target_file=fs_subject_id,
                                out_reg_file=output_dat,
                                contrast_type=fs_contrast,
                                fsl_mat_out=output_mat,
                                subjects_dir=fs_subjects_dir,
                            )
                        if is_dwi:
                            apply_ref = output_grid_ref
                            fsl.apply_xfm_4d(
                                in_file=in_path,
                                ref_file=apply_ref,
                                out_file=output_img,
                                mat=output_mat,
                                interp=options.get("interpolation", "trilinear")
                            )
                        else:
                            fsl.flirt(
                                in_file=in_path,
                                ref_file=output_grid_ref,
                                out_file=output_img,
                                extra_args=f"-applyxfm -init {output_mat} -interp {options.get('interpolation', 'trilinear')}",
                            )

                    else:
                         raise ValueError(f"Unknown coregistration method: {self.method}")

            except Exception as e:
                 raise ProcessingError(f"Coregistration failed: {e}", step_name="coregistration") from e

            # --- Rotation Logic ---
            if is_dwi:
                 rotated_bvecs = input_image.bvec
                 
                 if application_mode == "header":
                     self.logger.info(
                         "Header-only coregistration leaves voxel data and image-space b-vectors unchanged."
                     )
                     if input_image.bvec and Path(input_image.bvec).exists():
                         preserved_bvecs = _preserve_header_only_bvec(
                             Path(input_image.bvec),
                             output_dir,
                             entities,
                             new_desc,
                         )
                         rotated_bvecs = preserved_bvecs
                 elif apply_method == 'mrtrix' and mrtrix_rotated_bvecs:
                     self.logger.info("Using b-vectors rotated by MRTrix.")
                     rotated_bvecs = mrtrix_rotated_bvecs
                 else:
                     if self.method == 'fsl':
                         mat_file = output_transform.with_suffix(".mat")
                         if hasattr(input_image, 'bvec') and input_image.bvec and mat_file.exists():
                             new_bvec_path = output_dir / build_bids_name({**entities, "desc": new_desc}, suffix="bvec", extension=".bvec")
                             try:
                                 fsl.rotate_bvecs(input_image.bvec, mat_file, new_bvec_path)
                                 rotated_bvecs = new_bvec_path
                             except Exception as e:
                                 raise ProcessingError(f"FSL b-vector rotation failed: {e}") from e

                     elif self.method == 'freesurfer':
                         mat_file = output_transform.with_suffix(".mat")
                         if c3d.is_valid_fsl_affine(mat_file) and hasattr(input_image, 'bvec') and input_image.bvec and input_image.bvec.exists():
                             new_bvec_path = output_dir / build_bids_name({**entities, "desc": new_desc}, suffix="bvec", extension=".bvec")
                             try:
                                 fsl.rotate_bvecs(input_image.bvec, mat_file, new_bvec_path)
                                 rotated_bvecs = new_bvec_path
                             except Exception as e:
                                 raise ProcessingError(f"FreeSurfer b-vector rotation failed: {e}") from e
                     
                     elif self.method == 'ants':
                         # Search for ANTs affine if consistent
                         ants_affine = None
                         transform_list = locals().get('prefix', [])
                         for t in transform_list:
                             if str(t).endswith(".mat"):
                                 ants_affine = Path(t)
                                 break
                         
                         if not ants_affine:
                               potential_mat = Path(str(output_transform) + "0GenericAffine.mat")
                               if potential_mat.exists():
                                    ants_affine = potential_mat
                         
                         if ants_affine and ants_affine.exists():
                             fsl_mat = output_dir / build_bids_name({**entities, "desc": new_desc}, suffix="fsl_affine", extension=".mat")
                             try:
                                 ref_moving = locals().get('moving_for_reg')
                                 if not ref_moving: ref_moving = output_dir / "temp_b0_ref.nii.gz" if is_dwi else in_path
                                 
                                 if ref_moving.exists():
                                     if not c3d.is_valid_fsl_affine(fsl_mat):
                                         rotation_ref = output_grid_ref
                                         rotation_src = ref_moving
                                         if is_dwi:
                                             rotation_src = in_path
                                         c3d.ants2fsl(
                                             rotation_ref,
                                             rotation_src,
                                             ants_affine,
                                             fsl_mat,
                                         )
                                     
                                     if c3d.is_valid_fsl_affine(fsl_mat) and hasattr(input_image, 'bvec') and input_image.bvec and input_image.bvec.exists():
                                         new_bvec_path = output_dir / build_bids_name({**entities, "desc": new_desc}, suffix="bvec", extension=".bvec")
                                         fsl.rotate_bvecs(input_image.bvec, fsl_mat, new_bvec_path)
                                         rotated_bvecs = new_bvec_path

                             except Exception as e:
                                 raise ProcessingError(f"ANTs b-vector rotation failed: {e}") from e

        # --- Result Construction ---
        if is_dwi:
             final_bvec = locals().get('rotated_bvecs')
             if not final_bvec:
                 base_name = output_img.name
                 for ext in ['.nii.gz', '.nii']:
                     if base_name.endswith(ext):
                         base_name = base_name[:-len(ext)]
                         break
                 bvec_cand = output_dir / (base_name + ".bvec")
                 if bvec_cand.exists(): final_bvec = bvec_cand
                 elif input_image.bvec and input_image.bvec.exists():
                      shutil.copy(input_image.bvec, bvec_cand)
                      final_bvec = bvec_cand
             
             final_bval = None
             if input_image.bval and input_image.bval.exists():
                 base_name = output_img.name
                 for ext in ['.nii.gz', '.nii']:
                     if base_name.endswith(ext):
                         base_name = base_name[:-len(ext)]
                         break
                 bval_path = output_dir / (base_name + ".bval")
                 if not bval_path.exists(): shutil.copy(input_image.bval, bval_path)
                 final_bval = bval_path

             final_Delta = None
             if getattr(input_image, "Delta", None) and input_image.Delta.exists():
                 base_name = output_img.name
                 for ext in ['.nii.gz', '.nii']:
                     if base_name.endswith(ext):
                         base_name = base_name[:-len(ext)]
                         break
                 Delta_path = output_dir / (base_name + ".bigdelta")
                 if not Delta_path.exists(): shutil.copy(input_image.Delta, Delta_path)
                 final_Delta = Delta_path

             final_delta = None
             if getattr(input_image, "delta", None) and input_image.delta.exists():
                 base_name = output_img.name
                 for ext in ['.nii.gz', '.nii']:
                     if base_name.endswith(ext):
                         base_name = base_name[:-len(ext)]
                         break
                 delta_path = output_dir / (base_name + ".delta")
                 if not delta_path.exists(): shutil.copy(input_image.delta, delta_path)
                 final_delta = delta_path
             
             final_json = None
             if input_image.json and input_image.json.exists():
                 base_name = output_img.name
                 for ext in ['.nii.gz', '.nii']:
                     if base_name.endswith(ext):
                         base_name = base_name[:-len(ext)]
                         break
                 json_path = output_dir / (base_name + ".json")
                 final_json = copy_json_with_metadata(input_image.json, json_path)

             result = DWIFile(
                 img=output_img,
                 bvec=final_bvec,
                 bval=final_bval,
                 Delta=final_Delta,
                 delta=final_delta,
                 entities=entities,
                 json=final_json,
             )
        else:
             result = ImageFile(img=output_img, entities=entities)

        if not output_img.exists():
             raise ProcessingError(f"Coregistration step finished but output not found: {output_img}")
             
        if not check_nifti_integrity(output_img):
             raise ProcessingError(f"Coregistration step finished but output is corrupt/truncated: {output_img}")

        if out_res in {"dwi", "native"} and application_mode != "header":
             if mrtrix_native_header_alignment:
                 native_geometry_matches = (
                     mrtrix_transform.exists()
                     and _mrtrix_header_alignment_matches(
                         output_img,
                         in_path,
                         mrtrix_transform,
                     )
                 )
                 failure_detail = (
                     "MRtrix native coregistration did not preserve the input voxel "
                     "array while applying the expected header transform"
                 )
             else:
                 native_geometry_matches = _spatial_grids_match(output_img, in_path)
                 failure_detail = (
                     "Native coregistration output does not preserve the exact input DWI grid"
                 )
             if not native_geometry_matches:
                 raise ProcessingError(
                     f"{failure_detail}: "
                     f"input={nib.load(str(in_path)).shape[:3]}, "
                     f"output={nib.load(str(output_img)).shape[:3]}"
                 )
        if application_mode == "header":
             header_output = nib.load(str(output_img))
             header_input = nib.load(str(in_path))
             if (
                 header_output.shape != header_input.shape
                 or not np.allclose(
                     nib.affines.voxel_sizes(header_output.affine),
                     nib.affines.voxel_sizes(header_input.affine),
                     rtol=1e-5,
                     atol=1e-5,
                 )
             ):
                 raise ProcessingError(
                     "Header-only coregistration changed the DWI matrix or voxel sizes."
                 )
             if "header_world_transform" not in locals():
                 header_world_transform = header_output.affine @ np.linalg.inv(header_input.affine)

        # Dimension Verification
        try:
             chk_img = nib.load(output_img)
             self.logger.info(f"Coregistration output dimensions: {chk_img.shape}")
             if is_dwi and len(chk_img.shape) < 4:
                  self.logger.warning(f"CRITICAL: Coregistration produced a 3D output for a DWI series. DTI/DKI fitting will fail.")
        except Exception as e:
             self.logger.warning(f"Could not verify output dimensions: {e}")

        # --- Mask Handling (Fix for Dimension Mismatch) ---
        if context is not None and context.get("current_mask"):
            mask_input = context["current_mask"]
            mask_in_path = self._extract_path(mask_input)
            
            mask_entities = mask_input.entities.copy() if hasattr(mask_input, 'entities') else {}
            mask_entities['desc'] = new_desc
            mask_out_path_str = str(output_dir / build_bids_name({**mask_entities, "suffix": "mask"}))
            
            if not mask_out_path_str.endswith(".nii.gz") and not mask_out_path_str.endswith(".nii"):
                mask_out_path = Path(mask_out_path_str + ".nii.gz")
            else:
                mask_out_path = Path(mask_out_path_str)

            # Optimization: If output resolution is anatomical, try to find/use the structural mask 
            # instead of resampling the DWI mask.
            is_anatomical = options.get('output_resolution', 'anatomical').lower() == 'anatomical'
            struct_mask = context.get('structural_mask')
            
            if is_anatomical and not struct_mask:
                # Try to find structural mask near target
                target_img_path = Path(target_path)
                parent = target_img_path.parent
                # Pattern match based on target name
                t_stem = target_img_path.name.split(".nii")[0]
                # Common patterns: sub-01_T1w_mask.nii.gz, sub-01_desc-brain_mask.nii.gz
                potential_masks = [
                    parent / (t_stem + "_mask.nii.gz"),
                    parent / (t_stem.replace("_T1w", "_desc-brain_mask")), # sub-01_desc-brain_mask
                    parent / (t_stem.replace("_T1w", "_desc-brain_mask.nii.gz")),
                    parent / (t_stem + ".mask.nii.gz")
                ]
                for pm in potential_masks:
                    if pm.exists():
                        struct_mask = pm
                        self.logger.info(f"Automatically identified structural mask for anatomical space: {pm.name}")
                        break

            if is_anatomical and struct_mask:
                self.logger.info(f"Using structural mask for anatomical space: {Path(struct_mask).name}")
                shutil.copy(struct_mask, mask_out_path)
                mask_should_run = False
            else:
                # Fallback to resampling
                # Heuristic: If image is anatomical (usually >100 slices) and mask is native (usually ~60), trigger resampling
                mask_should_run = should_run
                if not mask_out_path.exists():
                    mask_should_run = True
                elif mask_out_path.exists():
                    try:
                        m_img = nib.load(mask_out_path)
                        # Use explicit target image for comparison if available, or the output image
                        ref_shape = chk_img.shape[:3] if 'chk_img' in locals() else nib.load(output_img).shape[:3]
                        
                        if m_img.shape != ref_shape:
                            self.logger.info(f"Existing mask {mask_out_path.name} shape {m_img.shape} mismatches image {ref_shape}. Re-resampling.")
                            mask_should_run = True
                    except Exception as e:
                        self.logger.warning(f"Could not verify mask dimensions: {e}")
                        mask_should_run = True

            if mask_should_run:
                self.logger.info(f"Applying coregistration transform to mask: {mask_in_path.name}")
                try:
                    if application_mode == "header":
                        if "header_world_transform" not in locals():
                            raise ProcessingError(
                                "Header-only mask update requires the rigid header transform."
                            )
                        _write_header_registered_image(
                            mask_in_path,
                            mask_out_path,
                            header_world_transform,
                        )
                    # Determine which transform to use
                    # Priority 1: MRTrix transform if applying via MRTrix
                    elif apply_method == 'mrtrix':
                        if not mrtrix_transform.exists() and output_mat.exists():
                             self.logger.info("Converting existing FSL transform to MRTrix for mask application...")
                             mrtrix.transformconvert(output_mat, mrtrix_transform, operation="flirt_import", ref_image=registration_target, in_image=moving_for_reg, force=True)
                        
                        if mrtrix_transform.exists():
                            mask_grid_ref = output_img if self.method == "freesurfer" or not is_anatomical else registration_target
                            mask_kwargs = {
                                "in_file": mask_in_path,
                                "out_file": mask_out_path,
                                "linear_transform": mrtrix_transform,
                                "interp": "nearest",
                                "nthreads": nthreads,
                                "force": True,
                            }
                            if self.method != "freesurfer" or not is_anatomical:
                                # Native output omits -template and updates the mask
                                # header without interpolation, just like the DWI.
                                # Anatomical output retains template-based reslicing.
                                mask_kwargs.update(
                                    _mrtrix_coregistration_grid_options(
                                        mask_grid_ref,
                                        out_res,
                                    )
                                )
                            mrtrix.mrtransform(**mask_kwargs)
                        else:
                             self.logger.warning(f"MRTrix transform not found. Falling back to FSL for mask.")
                             if output_mat.exists():
                                 fsl.flirt(in_file=mask_in_path, ref_file=output_grid_ref, out_file=mask_out_path, extra_opts={"applyxfm": True, "init": output_mat, "interp": "nearestneighbour"})
                             
                    elif output_mat.exists():
                        # Default FSL application
                        fsl.flirt(
                            in_file=mask_in_path,
                            ref_file=output_grid_ref,
                            out_file=mask_out_path,
                            extra_opts={
                                "applyxfm": True,
                                "init": output_mat,
                                "interp": "nearestneighbour"
                            }
                        )
                    else:
                        self.logger.warning("Could not identify transform to apply to mask. Mask might be misaligned.")
                except Exception as e:
                    self.logger.warning(f"Failed to apply coregistration to mask: {e}")
            
            if mask_out_path.exists():
                context["current_mask"] = ImageFile(img=mask_out_path, entities=mask_entities)

        if context is not None:
             spatial_transform = None
             if is_dwi:
                  transform_list = []
                  ants_prefix = locals().get('prefix', [])
                  if ants_prefix:
                       transform_list = [str(t) for t in ants_prefix if Path(t).exists()]
                  affine_path = None
                  transform_file_local = locals().get('transform_file')
                  if transform_file_local and Path(transform_file_local).exists():
                       affine_path = Path(transform_file_local)
                  elif output_mat.exists():
                       affine_path = output_mat

                  if transform_list or affine_path:
                       spatial_transform = {
                            "type": "linear",
                            "registration_method": self.method,
                            "apply_method": apply_method,
                            "application_mode": application_mode,
                            "usable_for_gnl_mapping": True,
                            "transforms": transform_list,
                            "fsl_affine": str(affine_path) if affine_path else None,
                            "moving_reference": str(moving_for_reg) if Path(moving_for_reg).exists() else None,
                            "fixed_reference": str(target) if Path(target).exists() else None,
                       }

             context["current_image"] = result
             if spatial_transform is not None:
                  context["spatial_transform"] = spatial_transform
                  write_transform_chain_to_sidecar(getattr(result, "json", None), [spatial_transform])
                  gnl_transforms = context.setdefault("gnl_transform_map", {})
                  gnl_transforms[str(in_path)] = spatial_transform
             
             return context
             
        return result
