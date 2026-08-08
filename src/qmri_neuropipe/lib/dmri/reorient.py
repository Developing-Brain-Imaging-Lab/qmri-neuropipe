"""
dMRI Reorientation step.
"""
from pathlib import Path
from typing import Any
import json
import shutil
import nibabel as nib
import numpy as np

from ...core import BaseProcessingStep, ValidationError
from ...core.caching import all_outputs_exist, reuse_enabled
from ...core.types import DWIFile
from ...io.bids import build_bids_name
from ...io.dmri.bids import (
    infer_phase_encoding_direction,
    phase_encoding_direction_to_vector,
    phase_encoding_transform_matrix,
    phase_encoding_vector_to_direction,
    transform_acqparams_file,
    transform_phase_encoding_direction,
)
from ..common.json_metadata import copy_json_with_metadata
from ..common.spatial_transforms import write_transform_chain_to_sidecar


def normalize_target_orientation(value: Any) -> str:
    """Validate and normalize a three-letter NIfTI voxel-axis orientation."""
    orientation = str(value or "RAS").strip().upper()
    if len(orientation) != 3:
        raise ValidationError(
            f"dMRI reorientation must be a three-letter axis code, got {value!r}"
        )
    axis_groups = ({"L", "R"}, {"A", "P"}, {"S", "I"})
    if any(sum(code in group for code in orientation) != 1 for group in axis_groups):
        raise ValidationError(
            "dMRI reorientation must contain exactly one left/right, one "
            f"anterior/posterior, and one superior/inferior axis, got {orientation!r}"
        )
    return orientation


def reorient_image_to_orientation(
    image: nib.spatialimages.SpatialImage,
    target_orientation: str,
) -> nib.Nifti1Image:
    """Permute/flip image axes to a requested orientation without interpolation."""
    target = normalize_target_orientation(target_orientation)
    source_ornt = nib.orientations.io_orientation(image.affine)
    target_ornt = nib.orientations.axcodes2ornt(tuple(target))
    transform = nib.orientations.ornt_transform(source_ornt, target_ornt)
    data = nib.orientations.apply_orientation(np.asanyarray(image.dataobj), transform)
    affine = image.affine @ nib.orientations.inv_ornt_aff(transform, image.shape[:3])
    header = image.header.copy()
    return nib.Nifti1Image(data, affine, header)


class DMRIReorientStep(BaseProcessingStep):
    """
    Reorient a dMRI to a requested voxel-axis convention without interpolation.

    The image axes, b-vectors, and axis-dependent BIDS metadata are transformed
    together. RAS is the default, but any valid three-letter orientation is
    supported.
    """
    
    def __init__(
        self,
        config,
        logger=None,
        provenance=None,
        target_orientation: str = "RAS",
    ):
        super().__init__(config, logger, provenance)
        self.target_orientation = normalize_target_orientation(target_orientation)
        self.method = f"NIfTI axis permutation/flip ({self.target_orientation})"

    def run(self, first_arg, output_dir: Path, **kwargs) -> Any:
        context, input_image = self.unpack_input(first_arg)
        
        # Global Execution Mode (Context)
        if context:
            dwi_files = context.get("dwi_files", [])
            processed_files = []
            
            step_output_dir = self.get_step_output_dir(output_dir)
            
            for dwi in dwi_files:
                res = self._process_single(dwi, step_output_dir, **kwargs)
                processed_files.append(res)

            self._update_context_phase_encoding(
                context,
                processed_files,
                step_output_dir,
            )
            context["dwi_files"] = processed_files
            return context
            
        # Single Execution Mode
        else:
             if not input_image:
                 raise ValidationError("No input image for dMRI reorientation.")
             step_output_dir = self.get_step_output_dir(output_dir)
             return self._process_single(input_image, step_output_dir, **kwargs)

    def _process_single(self, input_image, output_dir: Path, **kwargs) -> Any:
        # Check if we have gradients (crucial for dMRI reorient)
        in_bvec = getattr(input_image, 'bvec', None)
        in_bval = getattr(input_image, 'bval', None)
        
        if not in_bvec or not in_bvec.exists():
             self.logger.warning(f"No bvec file found for {input_image.img.name}. Reorientation might be unsafe if gradients are not rotated.")
             
        entities = input_image.entities.copy() if hasattr(input_image, 'entities') else {}
        
        # New DESC
        desc = entities.get('desc', '')
        new_desc = f"{desc}reor" if desc else "reor"
        entities['desc'] = new_desc
        
        # Construct output filenames
        # DWI
        out_name = build_bids_name(entities)
        if not out_name.endswith(".nii.gz"): out_name += ".nii.gz"
        out_path = output_dir / out_name
        
        # Gradients
        out_bvec_path = out_path.with_suffix("").with_suffix(".bvec")
        out_bval_path = out_path.with_suffix("").with_suffix(".bval")
        
        # If output exists and skip_existing, assume done
        if reuse_enabled(
            self.config,
            explicit_force=kwargs.get("force", False),
            force_keys=(),
        ) and all_outputs_exist((out_path, out_bvec_path)):
             try:
                 cached_image = nib.load(out_path)
                 cached_orientation = "".join(nib.aff2axcodes(cached_image.affine))
                 if cached_orientation != self.target_orientation:
                     raise ValueError(
                         f"expected {self.target_orientation}, got {cached_orientation}"
                     )
             except Exception as e:
                 self.logger.warning(
                     f"Existing reoriented DWI is invalid ({out_path.name}): {e}. Re-running."
                 )
                 try:
                     out_path.unlink(missing_ok=True)
                 except Exception:
                     pass
             else:
                 self.logger.info(f"Skipping dMRI Reorientation (exists): {out_path.name}")
                 
                 result = DWIFile(
                     img=out_path,
                     bval=out_bval_path if out_bval_path.exists() else None,
                     bvec=out_bvec_path if out_bvec_path.exists() else None,
                     json=out_path.with_suffix("").with_suffix(".json"),
                     Delta=getattr(input_image, "Delta", None),
                     delta=getattr(input_image, "delta", None),
                     entities=entities
                 )
                 self._update_output_phase_encoding(input_image, result)
                 return result
        
        source_image = nib.load(str(input_image.img))
        reoriented_image = reorient_image_to_orientation(
            source_image,
            self.target_orientation,
        )
        nib.save(reoriented_image, out_path)
        if in_bvec:
            bvecs = np.asarray(np.loadtxt(in_bvec), dtype=float)
            if bvecs.ndim == 1:
                bvecs = bvecs.reshape(3, 1)
            if bvecs.shape[0] != 3 and bvecs.shape[1] == 3:
                bvecs = bvecs.T
            if bvecs.shape[0] != 3:
                raise ValidationError(f"Expected a 3xN b-vector table, got {bvecs.shape}")
            voxel_transform = phase_encoding_transform_matrix(
                source_image.affine,
                reoriented_image.affine,
            )
            np.savetxt(out_bvec_path, voxel_transform @ bvecs, fmt="%.10f")
        if in_bval:
            shutil.copy2(in_bval, out_bval_path)

        out_json = out_path.with_suffix("").with_suffix(".json")
        copy_json_with_metadata(getattr(input_image, "json", None), out_json)

        # Create new DWIFile object
        result = DWIFile(
            img=out_path,
            bval=out_bval_path if out_bval_path.exists() else None,
            bvec=out_bvec_path if out_bvec_path.exists() else None,
            json=out_json,
            Delta=getattr(input_image, "Delta", None),
            delta=getattr(input_image, "delta", None),
            entities=entities
        )
        spatial_transform = {
            "type": "reorient_header",
            "method": "nifti_axis_permutation_flip",
            "usable_for_gnl_mapping": False,
            "target_orientation": self.target_orientation,
            "bvecs_rotated": bool(in_bvec),
        }
        write_transform_chain_to_sidecar(result.json, [spatial_transform])
        setattr(result, "spatial_transform", spatial_transform)
        self._update_output_phase_encoding(input_image, result)
        return result

    def _update_output_phase_encoding(self, source: DWIFile, target: DWIFile) -> None:
        """Update axis-dependent BIDS metadata after reorienting the voxel grid."""
        source_image = nib.load(str(source.img))
        target_image = nib.load(str(target.img))
        transform = phase_encoding_transform_matrix(source_image.affine, target_image.affine)
        setattr(target, "phase_encoding_transform", transform)

        source_payload = {}
        if source.json and Path(source.json).exists():
            try:
                source_payload = json.loads(Path(source.json).read_text())
            except (OSError, json.JSONDecodeError):
                source_payload = {}

        target_json = Path(target.json) if target.json else target.img.with_suffix("").with_suffix(".json")
        payload = {}
        if target_json.exists():
            payload = json.loads(target_json.read_text())
        if "SliceTiming" in source_payload:
            payload.setdefault("SliceTiming", source_payload["SliceTiming"])

        direction = infer_phase_encoding_direction(source)
        if direction:
            transformed_direction = transform_phase_encoding_direction(
                direction,
                source_image.affine,
                target_image.affine,
            )
            payload["PhaseEncodingDirection"] = transformed_direction
            self.logger.info(
                f"Updated PhaseEncodingDirection for {target.img.name}: "
                f"{direction} -> {transformed_direction}"
            )

        slice_direction = source_payload.get("SliceEncodingDirection")
        if not slice_direction:
            slice_timing = source_payload.get("SliceTiming")
            if isinstance(slice_timing, list):
                matching_axes = [
                    axis
                    for axis, size in enumerate(source_image.shape[:3])
                    if size == len(slice_timing)
                ]
                if len(matching_axes) == 1:
                    slice_direction = "ijk"[matching_axes[0]]
                    self.logger.info(
                        "Inferred SliceEncodingDirection=%s from %d SliceTiming "
                        "entries before reorientation",
                        slice_direction,
                        len(slice_timing),
                    )

        if slice_direction:
            try:
                transformed_slice_direction = transform_phase_encoding_direction(
                    slice_direction,
                    source_image.affine,
                    target_image.affine,
                )
            except ValueError as exc:
                raise ValidationError(
                    "Cannot transform SliceEncodingDirection after dMRI "
                    f"reorientation: {slice_direction!r}"
                ) from exc
            payload["SliceEncodingDirection"] = transformed_slice_direction
            self.logger.info(
                f"Updated SliceEncodingDirection for {target.img.name}: "
                f"{slice_direction} -> {transformed_slice_direction}"
            )

        target_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        target.json = target_json

    def _update_context_phase_encoding(
        self,
        context: dict,
        target_dwis: list[DWIFile],
        output_dir: Path,
    ) -> None:
        transforms = [
            getattr(target, "phase_encoding_transform", None)
            for target in target_dwis
        ]
        transforms = [transform for transform in transforms if transform is not None]
        if not transforms:
            return

        transform = transforms[0]
        if any(not np.array_equal(transform, candidate) for candidate in transforms[1:]):
            if context.get("acqp") or context.get("merged_acqp"):
                raise ValidationError(
                    "Cannot transform one shared acqparams file because DWI inputs were "
                    "reoriented with different axis permutations. Merge the inputs first."
                )
            return

        acqp_sources = []
        for key in ("acqp", "merged_acqp"):
            value = context.get(key)
            if value and Path(value) not in acqp_sources:
                acqp_sources.append(Path(value))
        for group in context.get("topup_groups", []):
            if isinstance(group, dict) and group.get("acqp"):
                path = Path(group["acqp"])
                if path not in acqp_sources:
                    acqp_sources.append(path)

        transformed_paths = {}
        for index, source in enumerate(acqp_sources):
            suffix = "" if index == 0 else f"_{index + 1}"
            destination = output_dir / f"acqparams{suffix}.txt"
            transformed_paths[source] = transform_acqparams_file(
                source,
                destination,
                transform,
            )
            self.logger.info(f"Reoriented acquisition parameters: {source} -> {destination}")

        for key in ("acqp", "merged_acqp"):
            value = context.get(key)
            if value and Path(value) in transformed_paths:
                context[key] = transformed_paths[Path(value)]
        for group in context.get("topup_groups", []):
            if isinstance(group, dict) and group.get("acqp"):
                source = Path(group["acqp"])
                if source in transformed_paths:
                    group["acqp"] = transformed_paths[source]

        source_info = context.get("merge_source_info", [])
        for item in source_info:
            direction = item.get("phase_encoding_direction")
            if direction:
                vector = transform @ phase_encoding_direction_to_vector(direction)
                item["phase_encoding_direction"] = phase_encoding_vector_to_direction(vector)
