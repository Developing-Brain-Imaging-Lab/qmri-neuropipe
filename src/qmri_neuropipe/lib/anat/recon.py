"""
FreeSurfer Recon-all step.
"""

from pathlib import Path
from typing import Optional, Any
import logging

from ...core import BaseProcessingStep, ValidationError
from ...core.types import ImageLike, ImageFile
from ...interfaces import freesurfer
from ...io.bids import build_bids_name, extract_bids_entities

class ReconAllStep(BaseProcessingStep):
    """
    Run FreeSurfer recon-all.
    """
    
    def __init__(self, config, logger=None, provenance=None):
        super().__init__(config, logger, provenance)
        # Look up config in anat.preprocessing.recon_all
        anat_cfg = config.get("anat", {}).get("preprocessing", {})
        self.recon_config = anat_cfg.get("recon_all", {})
        self.enabled = self.recon_config.get("enabled", False)
        self.args = self.recon_config.get("args", "-all")

    def run(self, first_arg, output_dir: Path, **kwargs) -> Any:
        # Check explicit enable
        if not self.enabled:
             self.logger.info("ReconAllStep disabled (via config check in run).")
             return first_arg

        context, input_image = self.unpack_input(first_arg)
        if not input_image:
             raise ValidationError("No input image for recon-all.")

        # Subject ID logic
        # Either from context['subject'] or entities['sub']
        sub = context.get('subject') if context else None
        if not sub and hasattr(input_image, 'entities'):
             sub = input_image.entities.get('sub')
        
        if not sub:
             self.logger.warning("Could not determine subject ID for recon-all. Using 'subject'.")
             sub = "subject"
             
        ses = context.get('session') if context else input_image.entities.get('ses')
        
        fs_sub_id = f"sub-{sub}"
        if ses:
             fs_sub_id += f"_ses-{ses}"

        # Output dir for freesurfer subjects
        # Typically "derivatives/freesurfer"
        # We can put it inside output_dir provided (which is typically work dir or derivatives root)
        # We want it to be persistent.
        # Let's assume output_dir passed in run() is the working directory for this subject.
        # We might want a dedicated freesurfer output dir.
        
        # We can use config.output_dir / "derivatives" / "freesurfer"?
        # Or just output_dir / "freesurfer"
        
        fs_dir = self.config.output_dir / "derivatives" / "freesurfer"
        fs_dir.mkdir(parents=True, exist_ok=True)
        
        # Run recon-all
        freesurfer.recon_all(
            in_file=input_image,
            subject_id=fs_sub_id,
            subjects_dir=fs_dir,
            openmp=self.n_threads,
            extra_args=self.args
        )
        
        # Return what? 
        # Typically recon-all doesn't change the "current image" flow for subsequent ANTs registration 
        # unless we want to register the FS output (norm.mgz)?
        # Usually we continue with the T1w.
        # So we just return inputs unchanged but updated context if we want to store FS location?
        
        if context:
             context["freesurfer_dir"] = fs_dir / fs_sub_id
             return context
             
        return input_image
