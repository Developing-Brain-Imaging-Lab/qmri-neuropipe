"""
Resampling step.
"""

from pathlib import Path
from typing import Optional, Tuple, Any
import logging

from ...core import BaseProcessingStep, ValidationError
from ...core.types import ImageLike, DWIFile, ImageFile
from ...interfaces import freesurfer
from ...io.bids import build_bids_name

class ResampleStep(BaseProcessingStep):
    """
    Resample image to a new resolution (isotropic or specific).
    Uses `mri_convert` (FreeSurfer) or other tools if implemented.
    """
    
    def __init__(self, config, logger=None, provenance=None, resolution: Optional[float] = None):
        super().__init__(config, logger, provenance)
        self.resolution = resolution or config.get("resolution")

    def run(self, first_arg, output_dir: Path, **kwargs) -> Any:
        context, input_image = self.unpack_input(first_arg)
        if not input_image:
             raise ValidationError("No input image for resampling.")
             
        # Check config resolution
        res = self.resolution or self.config.get("resolution")
        if not res:
             self.logger.info("No resolution specified. Skipping ResampleStep.")
             return context if context else input_image

        output_dir = self.get_step_output_dir(output_dir)
        entities = input_image.entities.copy() if hasattr(input_image, 'entities') else {}
        
        # New name
        desc = entities.get('desc', '')
        new_desc = f"{desc}resmpl" if desc else "resmpl"
        entities['desc'] = new_desc
        
        output_img = output_dir / build_bids_name({**entities, "suffix": "T1w"}) # Assuming T1w/T2w suffix is preserved or we should check?
        # Actually better to preserve original suffix.
        orig_suffix = entities.get('suffix', 'T1w')
        output_img = output_dir / build_bids_name({**entities, "suffix": orig_suffix})

        if output_img.exists():
             self.logger.info(f"Skipping resampling (exists): {output_img}")
        else:
             in_p = self._extract_path(input_image)
             # Use mri_convert
             # Logic: mri_convert -vs x x x
             # The mri_convert wrapper in freesurfer.py needs to support extra args or we assume isometric
             # Currently mri_convert wrapper doesn't support extra args.
             # We might need to update wrapper or call run_cmd directly.
             # Let's call run_cmd for now or update wrapper.
             # Actually let's assume `freesurfer.mri_convert` is flexible? 
             # It accepts (in, out). 
             
             # I'll use run_cmd here for flexibility or just trust I can pass args to existing wrapper?
             # No, existing wrapper is rigid.
             
             from ...core.run import run_cmd
             
             # Handle CSV resolution
             if isinstance(res, str) and ',' in res:
                  # Parse "0.8,0.8,0.8" -> "0.8 0.8 0.8"
                  res_args = res.replace(",", " ")
             else:
                  # Assume isotropic float or single value
                  res_args = f"{res} {res} {res}"
                  
             cmd = f"mri_convert {in_p} {output_img} -vs {res_args}"
             run_cmd(cmd, label="mri_convert_resample")

        # Preserve input type (DWIFile vs ImageFile)
        if isinstance(input_image, DWIFile):
             result = DWIFile(
                  img=output_img,
                  entities=entities,
                  bval=input_image.bval,
                  bvec=input_image.bvec,
                  json=input_image.json
             )
        else:
             result = ImageFile(img=output_img, entities=entities)
        
        if context:
             context["current_image"] = result
             return context
        return result
