"""
dMRI Reorientation step.
"""
from pathlib import Path
from typing import Any
import shutil

from ...core import BaseProcessingStep, ValidationError
from ...core.types import DWIFile
from ...interfaces import mrtrix
from ...io.bids import build_bids_name, get_entities_from_path

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
                res = self._process_single(dwi, step_output_dir)
                processed_files.append(res)
                
            context["dwi_files"] = processed_files
            return context
            
        # Single Execution Mode
        else:
             if not input_image:
                 raise ValidationError("No input image for dMRI reorientation.")
             step_output_dir = self.get_step_output_dir(output_dir)
             return self._process_single(input_image, step_output_dir)

    def _process_single(self, input_image, output_dir: Path) -> Any:
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
        if self.config.get("skip_existing") and out_path.exists() and out_bvec_path.exists():
             self.logger.info(f"Skipping dMRI Reorientation (exists): {out_path.name}")
        else:
             mrtrix.mrconvert(
                 in_file=input_image.img,
                 out_file=out_path,
                 stride=stride,
                 in_bvec=in_bvec,
                 in_bval=in_bval,
                 export_grad_fsl=(out_bvec_path, out_bval_path) if in_bvec else None,
                 json_import=input_image.json if getattr(input_image, 'json', None) else None,
                 json_export=out_path.with_suffix("").with_suffix(".json"), # Export sidecar
                 nthreads=self.config.get("n_cpus", 1),
                 force=True # We checked above
             )

        # Create new DWIFile object
        result = DWIFile(
            img=out_path,
            bval=out_bval_path if out_bval_path.exists() else None,
            bvec=out_bvec_path if out_bvec_path.exists() else None,
            json=out_path.with_suffix("").with_suffix(".json"),
            entities=entities
        )
        return result
