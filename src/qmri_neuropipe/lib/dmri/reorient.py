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
from ...core.types import DWIFile
from ...interfaces import mrtrix
from ...io.bids import build_bids_name, get_entities_from_path
from ...io.dmri.bids import (
    infer_phase_encoding_direction,
    phase_encoding_direction_to_vector,
    phase_encoding_transform_matrix,
    phase_encoding_vector_to_direction,
    transform_acqparams_file,
    transform_phase_encoding_direction,
)
from ..common.json_metadata import copy_json_with_metadata, write_sanitized_json_copy
from ..common.spatial_transforms import write_transform_chain_to_sidecar

class DMRIReorientStep(BaseProcessingStep):
    """
    Reorient dMRI image to standard orientation (RAS, stride 1,2,3,4) using mrconvert.
    Crucially, this also rotates the b-vectors to match the new image orientation.
    """
    
    def __init__(self, config, logger=None, provenance=None):
        super().__init__(config, logger, provenance)
        self.method = "mrconvert (stride 1,2,3,4)"

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
        
        # Run mrconvert
        # Standardize stride to 1,2,3,4 (x,y,z,vol) which is usually RAS or close to standard convention
        stride = "1,2,3,4"
        
        # If output exists and skip_existing, assume done
        if self.config.get("skip_existing") and out_path.exists() and out_bvec_path.exists() and not kwargs.get('force', False):
             try:
                 _ = nib.load(out_path)
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
                     entities=entities
                 )
                 self._update_output_phase_encoding(input_image, result)
                 return result
        
        sanitized_json_import = None
        if getattr(input_image, "json", None):
            sanitized_json_import = output_dir / f"{out_path.stem}.mrtrix_import.json"
            write_sanitized_json_copy(input_image.json, sanitized_json_import)

        mrtrix.mrconvert(
                 in_file=input_image.img,
                 out_file=out_path,
                 stride=stride,
                 in_bvec=in_bvec,
                 in_bval=in_bval,
                 export_grad_fsl=(out_bvec_path, out_bval_path) if in_bvec else None,
                 json_import=sanitized_json_import,
                 json_export=out_path.with_suffix("").with_suffix(".json"), # Export sidecar
                 nthreads=self.config.get("n_cpus", 1),
                 force=True # We checked above
             )
        if sanitized_json_import:
            sanitized_json_import.unlink(missing_ok=True)

        out_json = out_path.with_suffix("").with_suffix(".json")
        copy_json_with_metadata(getattr(input_image, "json", None), out_json)

        # Create new DWIFile object
        result = DWIFile(
            img=out_path,
            bval=out_bval_path if out_bval_path.exists() else None,
            bvec=out_bvec_path if out_bvec_path.exists() else None,
            json=out_json,
            entities=entities
        )
        spatial_transform = {
            "type": "reorient_header",
            "method": "mrtrix_mrconvert_stride",
            "usable_for_gnl_mapping": False,
            "stride": stride,
            "bvecs_rotated": bool(in_bvec),
        }
        write_transform_chain_to_sidecar(result.json, [spatial_transform])
        setattr(result, "spatial_transform", spatial_transform)
        self._update_output_phase_encoding(input_image, result)
        return result

    def _update_output_phase_encoding(self, source: DWIFile, target: DWIFile) -> None:
        """Update the target sidecar and retain the voxel-axis transform for context files."""
        source_image = nib.load(str(source.img))
        target_image = nib.load(str(target.img))
        transform = phase_encoding_transform_matrix(source_image.affine, target_image.affine)
        setattr(target, "phase_encoding_transform", transform)

        direction = infer_phase_encoding_direction(source)
        if not direction:
            return

        transformed_direction = transform_phase_encoding_direction(
            direction,
            source_image.affine,
            target_image.affine,
        )
        target_json = Path(target.json) if target.json else target.img.with_suffix("").with_suffix(".json")
        payload = {}
        if target_json.exists():
            payload = json.loads(target_json.read_text())
        payload["PhaseEncodingDirection"] = transformed_direction
        target_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        target.json = target_json
        self.logger.info(
            f"Updated PhaseEncodingDirection for {target.img.name}: "
            f"{direction} -> {transformed_direction}"
        )

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
