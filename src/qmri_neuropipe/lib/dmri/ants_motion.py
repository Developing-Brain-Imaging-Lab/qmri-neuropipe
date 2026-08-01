"""ANTs-based diffusion motion correction with gradient reorientation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from ...core import BaseProcessingStep, ProcessingError, ValidationError
from ...core.types import DWIFile
from ...io.bids import build_bids_name
from .b0_reference import b0_candidate_indices, load_bvals, select_optimal_b0


def closest_rotation(linear: np.ndarray) -> np.ndarray:
    """Return the proper rotation component of an affine linear matrix."""
    matrix = np.asarray(linear, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 matrix, got {matrix.shape}")
    left, _, right_t = np.linalg.svd(matrix)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    return rotation


def normalize_bvec_table(values: np.ndarray) -> tuple[np.ndarray, bool]:
    """Normalize a bvec table to 3xN and report whether it was transposed."""
    table = np.asarray(values, dtype=float)
    if table.ndim == 1:
        if table.size % 3:
            raise ValidationError("A one-dimensional bvec table must contain 3*N values")
        table = table.reshape(3, -1)
        return table, False
    if table.ndim != 2:
        raise ValidationError(f"Invalid bvec table shape: {table.shape}")
    if table.shape[0] == 3:
        return table, False
    if table.shape[1] == 3:
        return table.T, True
    raise ValidationError(f"A bvec table must be 3xN or Nx3, got {table.shape}")


def rotate_bvec_table(bvecs: np.ndarray, rotations: list[np.ndarray]) -> np.ndarray:
    """Apply one motion rotation per volume and preserve zero vectors."""
    table, _ = normalize_bvec_table(bvecs)
    if table.shape[1] != len(rotations):
        raise ValidationError(
            f"Received {len(rotations)} rotations for {table.shape[1]} diffusion vectors"
        )
    result = np.zeros_like(table, dtype=float)
    for index, rotation in enumerate(rotations):
        vector = np.asarray(rotation, dtype=float) @ table[:, index]
        norm = np.linalg.norm(vector)
        if norm > 1e-8:
            vector /= norm
        result[:, index] = vector
    return result


def ants_physical_rotation_to_voxel(rotation: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Convert an ANTs LPS physical rotation to FSL/NIfTI voxel axes."""
    linear = np.asarray(affine, dtype=float)[:3, :3]
    scales = np.linalg.norm(linear, axis=0)
    if np.any(scales <= 0):
        raise ValidationError("NIfTI affine has a degenerate spatial axis")
    voxel_to_ras = linear / scales
    lps_to_ras = np.diag([-1.0, -1.0, 1.0])
    rotation_ras = lps_to_ras @ np.asarray(rotation, dtype=float) @ lps_to_ras
    return closest_rotation(np.linalg.inv(voxel_to_ras) @ rotation_ras @ voxel_to_ras)


def _affine_linear_from_ants(transform_path: Path) -> np.ndarray:
    import ants

    transform = ants.read_transform(str(transform_path))
    parameters = np.asarray(transform.parameters, dtype=float)
    if parameters.size < 9:
        raise ProcessingError(f"Cannot extract a 3D rotation from {transform_path}")
    return parameters[:9].reshape(3, 3)


def _write_sidecar(source: DWIFile, result: DWIFile, metadata: dict) -> Path:
    payload = {}
    if source.json and Path(source.json).exists():
        try:
            payload = json.loads(Path(source.json).read_text())
        except (OSError, json.JSONDecodeError):
            pass
    payload.update(metadata)
    destination = Path(str(result.img).split(".nii", 1)[0] + ".json")
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return destination


class AntsDiffusionMotionCorrectionStep(BaseProcessingStep):
    """Register each DWI volume with ANTs and rotate its diffusion vector.

    ``mode='motion'`` estimates rigid transforms. ``mode='motion_eddy'`` uses
    affine registrations, with the proper rotation extracted via polar
    decomposition for b-vector reorientation. Optional slice-wise refinement
    performs intra-volume 2D registrations after the global 3D correction.
    """

    def __init__(
        self,
        config,
        logger: Optional[logging.Logger] = None,
        provenance=None,
        *,
        mode: str = "motion",
        slice_to_volume: bool = False,
        reference_selection: Optional[dict] = None,
        transform_type: Optional[str] = None,
        interpolator: str = "linear",
        registration_options: Optional[dict] = None,
    ):
        super().__init__(config, logger, provenance)
        if mode not in {"motion", "motion_eddy"}:
            raise ValueError("ANTs diffusion correction mode must be 'motion' or 'motion_eddy'")
        self.mode = mode
        self.slice_to_volume = bool(slice_to_volume)
        self.reference_selection = dict(reference_selection or {})
        self.transform_type = transform_type or ("Rigid" if mode == "motion" else "Affine")
        self.interpolator = interpolator
        self.registration_options = dict(registration_options or {})
        self.slice_registration_options = self.registration_options.pop(
            "slice_registration_options", {}
        )
        self.slice_group_radius = int(self.slice_registration_options.pop("group_radius", 1))

    def validate_inputs(self, first_arg, output_dir: Path, **kwargs) -> None:
        _, image = self.unpack_input(first_arg)
        if not isinstance(image, DWIFile):
            raise ValidationError("ANTs diffusion correction requires a DWIFile")
        for path in (image.img, image.bval, image.bvec):
            if not path or not Path(path).exists():
                raise ValidationError(f"Missing ANTs diffusion correction input: {path}")
        shape = nib.load(str(image.img)).shape
        if len(shape) != 4:
            raise ValidationError(f"ANTs diffusion correction requires 4D data, got {shape}")

    def validate_outputs(self, result) -> None:
        image = result.get("current_image") if isinstance(result, dict) else result
        if not isinstance(image, DWIFile):
            raise ProcessingError("ANTs diffusion correction did not return a DWIFile")
        for path in (image.img, image.bval, image.bvec):
            if not path or not Path(path).exists():
                raise ProcessingError(f"Missing ANTs diffusion correction output: {path}")

    def run(self, first_arg, output_dir: Path, **kwargs):
        context, input_dwi = self.unpack_input(first_arg)
        step_dir = self.get_step_output_dir(output_dir)
        descriptor = "antsmotioneddy" if self.mode == "motion_eddy" else "antsmotion"
        entities = {**input_dwi.entities, "desc": descriptor}
        output_img = step_dir / build_bids_name(entities)
        output_base = Path(str(output_img).split(".nii", 1)[0])
        output_bvec = Path(f"{output_base}.bvec")
        output_bval = Path(f"{output_base}.bval")
        output_json = Path(f"{output_base}.json")
        output_slice_bvec = Path(f"{output_base}_slicewise.bvec.npy")
        force = bool(kwargs.get("force", False))
        if all(path.exists() for path in (output_img, output_bvec, output_bval)) and not force:
            result = DWIFile(
                entities=entities,
                img=output_img,
                json=output_json if output_json.exists() else input_dwi.json,
                bval=output_bval,
                bvec=output_bvec,
                Delta=getattr(input_dwi, "Delta", None),
                delta=getattr(input_dwi, "delta", None),
            )
            if output_slice_bvec.exists():
                setattr(result, "slice_bvec", output_slice_bvec)
            return self._return_result(context, result)

        preferred_index = self.reference_selection.get("index")
        if not self.reference_selection.get("enabled", True) and preferred_index is None:
            preferred_index = int(
                b0_candidate_indices(
                    load_bvals(Path(input_dwi.bval)),
                    float(self.reference_selection.get("b0_threshold", 50.0)),
                )[0]
            )
        selection = select_optimal_b0(
            input_dwi,
            step_dir / "b0_reference",
            threshold=float(self.reference_selection.get("b0_threshold", 50.0)),
            local_radius=int(self.reference_selection.get("local_radius", 3)),
            preferred_index=preferred_index,
            force=force,
        )
        try:
            import ants
        except ImportError as exc:
            raise ProcessingError(
                "ANTs diffusion correction requires antspyx (the 'ants' Python package)"
            ) from exc

        source = nib.load(str(input_dwi.img))
        data = np.asanyarray(source.dataobj)
        work_dir = step_dir / "volume_registrations"
        work_dir.mkdir(parents=True, exist_ok=True)
        fixed = ants.image_read(str(selection.pair_average_image))
        corrected = np.empty(data.shape, dtype=np.float32)
        rotations: list[np.ndarray] = []
        slice_rotations: list[list[np.ndarray]] = []
        transform_files: list[list[str]] = []
        nthreads = int(kwargs.get("nthreads", self.config.n_cpus))

        import os
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)
        for volume_index in range(data.shape[3]):
            moving_path = work_dir / f"volume_{volume_index:04d}.nii.gz"
            nib.save(
                nib.Nifti1Image(np.asarray(data[..., volume_index], dtype=np.float32), source.affine),
                moving_path,
            )
            moving = ants.image_read(str(moving_path))
            prefix = work_dir / f"volume_{volume_index:04d}_"
            registration = ants.registration(
                fixed=fixed,
                moving=moving,
                type_of_transform=self.transform_type,
                outprefix=str(prefix),
                **self.registration_options,
            )
            affine_transforms = [Path(path) for path in registration["fwdtransforms"] if str(path).endswith(".mat")]
            rotation = (
                ants_physical_rotation_to_voxel(
                    closest_rotation(_affine_linear_from_ants(affine_transforms[-1])),
                    source.affine,
                )
                if affine_transforms
                else np.eye(3)
            )
            rotations.append(rotation)
            warped_image = registration["warpedmovout"]
            warped = warped_image.numpy()
            volume_slice_rotations: list[np.ndarray] = []
            if self.slice_to_volume:
                warped, volume_slice_rotations = self._slice_refinement(
                    ants,
                    fixed,
                    warped_image,
                    source.affine,
                    work_dir,
                    volume_index,
                )
            corrected[..., volume_index] = warped
            slice_rotations.append(volume_slice_rotations)
            transform_files.append([str(path) for path in registration["fwdtransforms"]])

        output_header = source.header.copy()
        output_header.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(corrected, source.affine, output_header), output_img)
        bvecs = np.loadtxt(input_dwi.bvec)
        rotated_bvecs = rotate_bvec_table(bvecs, rotations)
        np.savetxt(output_bvec, rotated_bvecs, fmt="%.10f")
        output_bval.write_text(Path(input_dwi.bval).read_text())
        slice_bvec_file = None
        if self.slice_to_volume:
            # A single FSL bvec cannot encode intra-volume rotations. Persist
            # the effective vector for every volume/slice so slice-aware or
            # voxelwise models can consume the complete correction.
            slice_bvecs = np.empty((data.shape[3], data.shape[2], 3), dtype=np.float64)
            for volume_index in range(data.shape[3]):
                global_vector = rotated_bvecs[:, volume_index]
                for slice_index, slice_rotation in enumerate(slice_rotations[volume_index]):
                    vector = slice_rotation @ global_vector
                    norm = np.linalg.norm(vector)
                    slice_bvecs[volume_index, slice_index] = vector / norm if norm > 1e-8 else vector
            slice_bvec_file = output_slice_bvec
            np.save(slice_bvec_file, slice_bvecs)
        result = DWIFile(
            entities=entities,
            img=output_img,
            json=None,
            bval=output_bval,
            bvec=output_bvec,
            Delta=getattr(input_dwi, "Delta", None),
            delta=getattr(input_dwi, "delta", None),
        )
        if slice_bvec_file:
            setattr(result, "slice_bvec", slice_bvec_file)
        result.json = _write_sidecar(
            input_dwi,
            result,
            {
                "MotionCorrection": "ANTs",
                "EddyCurrentCorrection": "ANTs affine registration" if self.mode == "motion_eddy" else False,
                "SliceToVolumeCorrection": self.slice_to_volume,
                "MotionCorrectionReferenceVolume": selection.index,
                "MotionCorrectionReferenceImage": str(selection.pair_average_image),
                "DiffusionGradientRotation": "finite-strain/polar decomposition of volume transform",
                "SliceWiseDiffusionGradientFile": str(slice_bvec_file) if slice_bvec_file else None,
                "VolumeTransformFiles": transform_files,
            },
        )
        return self._return_result(context, result, selection)

    def _slice_refinement(
        self,
        ants_module,
        reference,
        moving,
        affine: np.ndarray,
        work_dir: Path,
        volume_index: int,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Register overlapping 3D slice groups into the reference volume."""
        refined = np.asarray(moving.numpy(), dtype=np.float32).copy()
        shape = refined.shape
        options = dict(self.slice_registration_options)
        rotations: list[np.ndarray] = []
        for slice_index in range(shape[2]):
            start = max(0, slice_index - self.slice_group_radius)
            stop = min(shape[2], slice_index + self.slice_group_radius + 1)
            mask_array = np.zeros(shape, dtype=np.uint8)
            mask_array[..., start:stop] = 1
            fixed_mask = reference.new_image_like(mask_array)
            moving_mask = moving.new_image_like(mask_array)
            result = ants_module.registration(
                fixed=reference,
                moving=moving,
                type_of_transform="Rigid",
                mask=fixed_mask,
                moving_mask=moving_mask,
                mask_all_stages=True,
                outprefix=str(work_dir / f"volume_{volume_index:04d}_slice_{slice_index:04d}_"),
                **options,
            )
            refined[..., slice_index] = result["warpedmovout"].numpy()[..., slice_index]
            affine_transforms = [
                Path(path) for path in result["fwdtransforms"] if str(path).endswith(".mat")
            ]
            rotation = (
                ants_physical_rotation_to_voxel(
                    closest_rotation(_affine_linear_from_ants(affine_transforms[-1])), affine
                )
                if affine_transforms
                else np.eye(3)
            )
            rotations.append(rotation)
        return refined, rotations

    def _return_result(self, context, result: DWIFile, selection=None):
        if context is None:
            return result
        context["current_image"] = result
        context.setdefault("preprocessed_dwis", []).append(result)
        if selection:
            context["b0_reference_selection"] = selection
            context["motion_reference"] = selection.pair_average_image
        context["spatial_transform"] = {
            "type": "motion_eddy_correction" if self.mode == "motion_eddy" else "motion_correction",
            "method": "ants_native" if self.mode == "motion_eddy" else "ants",
            "slice_to_volume": self.slice_to_volume,
            "usable_for_gnl_mapping": False,
        }
        return context


__all__ = [
    "AntsDiffusionMotionCorrectionStep",
    "ants_physical_rotation_to_voxel",
    "closest_rotation",
    "normalize_bvec_table",
    "rotate_bvec_table",
]
