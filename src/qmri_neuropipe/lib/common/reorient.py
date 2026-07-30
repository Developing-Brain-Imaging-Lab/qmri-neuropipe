"""
Reorientation step.
"""

from pathlib import Path
from typing import Any

from ...core import BaseProcessingStep, ValidationError
from ...core.types import ImageFile
from ...interfaces import fsl
from ...io.bids import build_bids_name
from .json_metadata import copy_json_with_metadata

class ReorientStep(BaseProcessingStep):
    """
    Reorient image to standard orientation (MNI) using fslreorient2std.
    """
    
    def __init__(self, config, logger=None, provenance=None):
        super().__init__(config, logger, provenance)

    def run(self, first_arg, output_dir: Path, **kwargs) -> Any:
        context, input_image = self.unpack_input(first_arg)
        if not input_image:
             raise ValidationError("No input image for reorientation.")

        output_dir = self.get_step_output_dir(output_dir)
        entities = input_image.entities.copy() if hasattr(input_image, 'entities') else {}
        
        # New name
        # fslreorient2std usually doesn't change content much but changes header.
        # Often we overwrite or mark as reoriented.
        # "anat_proc" usually does this early.
        # Let's append desc-reorient?
        desc = entities.get('desc', '')
        new_desc = f"{desc}reor" if desc else "reor"
        entities['desc'] = new_desc
        
        orig_suffix = entities.get('suffix', 'T1w')
        output_img = output_dir / build_bids_name({**entities, "suffix": orig_suffix})
        output_json = output_img.with_suffix("").with_suffix(".json")

        if output_img.exists() and not kwargs.get('force', False):
             self.logger.info(f"Skipping reorientation (exists): {output_img}")
        else:
             fsl.reorient2std(input_image, output_img)
        copy_json_with_metadata(getattr(input_image, "json", None), output_json)

        result_json = output_json if output_json.exists() else getattr(input_image, "json", None)
        result = ImageFile(img=output_img, entities=entities, json=result_json)
        
        if context:
             context["current_image"] = result
             return context
        return result
