"""
Image Sharpening Step
"""
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from ...core import BaseProcessingStep, ValidationError
from ...core.types import ImageFile
from ...interfaces import ants
from ...io.bids import build_bids_name

class SharpeningStep(BaseProcessingStep):
    """
    Step to sharpen image using ANTs iMath SharpenImage.
    """
    
    def __init__(self, config, logger: Optional[logging.Logger] = None, provenance=None, method='ants'):
        super().__init__(config, logger, provenance)
        self.method = method

    def run(self, first_arg, output_dir: Path, options: Optional[Dict[str, Any]]=None, **kwargs) -> Any:
        context, input_image = self.unpack_input(first_arg)
        
        if not input_image:
             raise ValidationError("No input image for sharpening.")
             
        in_path = self._extract_path(input_image)
        output_dir = self.get_step_output_dir(output_dir)
        
        entities = input_image.entities.copy() if hasattr(input_image, 'entities') else {}
        
        # Output naming
        # Typically distinct derivatives. Let's use desc="sharp".
        
        entities['desc'] = 'sharp'
        output_img = output_dir / build_bids_name(entities)
        
        options = options or {}
        
        if self.method == 'ants':
             self.logger.info(f"Sharpening image with ANTs iMath: {in_path}")
             # Usage: iMath Image Sharpen InputImage 
             ants.iMath(in_path, output_img, operation='Sharpen', **options)
             
        else:
             self.logger.warning(f"Sharpening method {self.method} not implemented.")
             return context if context else input_image
             
        result = ImageFile(img=output_img, entities=entities)
        
        if context:
             context['current_image'] = result
             return context
        return result
