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
        
        # Check global use_freesurfer flag (or nested in generic options)
        # Checking: anat.use_freesurfer OR anat.preprocessing.use_freesurfer
        self.use_freesurfer = config.get("anat", {}).get("use_freesurfer", False) or \
                              config.get("anat", {}).get("preprocessing", {}).get("use_freesurfer", False)

    def run(self, first_arg, output_dir: Path, **kwargs) -> Any:
        # Check explicit enable OR use_freesurfer
        if not self.enabled and not self.use_freesurfer:
             self.logger.info("ReconAllStep disabled.")
             return first_arg

        context, input_image = self.unpack_input(first_arg)
        
        # Need input image to determine subject/session IF running recon-all
        # If strictly using existing FS, we might only need subject ID, but input_image helps.
        
        # Subject ID logic
        sub = context.get('subject') if context else None
        if not sub and hasattr(input_image, 'entities'):
             sub = input_image.entities.get('sub')
        
        if not sub:
             self.logger.warning("Could not determine subject ID for recon-all. Using 'subject'.")
             sub = "subject"
             
        ses = context.get('session') if context else getattr(input_image, 'entities', {}).get('ses')
        
        fs_sub_id = f"sub-{sub}"
        if ses:
             fs_sub_id += f"_ses-{ses}"

        # FS Directory
        # Use bids_dir/derivatives/freesurfer to share across pipelines
        fs_dir = self.config.bids_dir / "derivatives" / "freesurfer"
        subj_dir = fs_dir / fs_sub_id
        
        # Check force run
        if kwargs.get('force', False) and subj_dir.exists():
             self.logger.info(f"Recon-all force run: Removing existing subject dir {subj_dir}")
             import shutil
             shutil.rmtree(subj_dir)
             
        fs_dir.mkdir(parents=True, exist_ok=True)
        
        # Check integrity of key FS output
        brain_mgz = subj_dir / "mri" / "brain.mgz"
        fs_complete = brain_mgz.exists()
        
        if not fs_complete:
             if not input_image:
                 raise ValidationError("FreeSurfer output missing and no input image provided to run recon-all.")
                 
             self.logger.info(f"Running FreeSurfer recon-all for {fs_sub_id}...")
             freesurfer.recon_all(
                in_file=input_image,
                subject_id=fs_sub_id,
                subjects_dir=fs_dir,
                openmp=kwargs.get("nthreads", 8), # Default to reasonable threads
                extra_args=self.args
             )
        else:
             self.logger.info(f"Using existing FreeSurfer output for {fs_sub_id}")

        # Post-Processing / Injection
        if context:
             context["freesurfer_dir"] = subj_dir
             
        if self.use_freesurfer:
             # Convert brain.mgz -> NIfTI
             # Use output_dir (which is step-specific or workflow passed)
             # Ideally we want this in "anat/preproc"? 
             # But this step is likely added to the workflow.
             # We should save to the `output_dir` provided to run().
             
             fs_out_dir = output_dir # This might be .../recon or just .../anat depending on workflow
             # If step specific dir logic is used, it might be nested.
             # Let's trust output_dir.
             
             out_name = build_bids_name({
                 'sub': sub, 'ses': ses, 'desc': 'preproc', 'suffix': 'T1w'
             })
             
             t1w_nii = fs_out_dir / f"{out_name}.nii.gz"
             mask_nii = fs_out_dir / f"{out_name.replace('T1w', 'mask')}.nii.gz"
             
             # Convert brain.mgz
             self.logger.info(f"Converting FS brain.mgz to {t1w_nii.name}")
             freesurfer.mri_convert(brain_mgz, t1w_nii)
             
             # Create mask
             # 1. Binarize brain.mgz (since it's skull stripped, >0 is brain)
             self.logger.info(f"Creating brain mask from FS output...")
             freesurfer.mri_binarize(brain_mgz, mask_nii, min_val=1)
             
             # Define new image objects
             entities = {'sub': sub, 'ses': ses, 'desc': 'preproc', 'suffix': 'T1w'}
             new_img = ImageFile(path=t1w_nii, entities=entities)
             
             mask_entities = {'sub': sub, 'ses': ses, 'desc': 'preproc', 'suffix': 'mask'}
             new_mask = ImageFile(path=mask_nii, entities=mask_entities)

             # Update Context if available
             if context:
                 context['current_image'] = new_img
                 context['brain_mask'] = new_mask
                 
                 self.logger.info("Updated pipeline context to use FreeSurfer-derived structural images.")
                 return context
             else:
                 # Standalone mode: return the new image directly
                 return new_img
                  
        # Fallback if not using FS as primary but just running it
        return context if context else input_image
