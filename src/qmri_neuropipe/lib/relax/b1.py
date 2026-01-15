
from pathlib import Path
from typing import Optional, Union, Dict
import nibabel as nib
import numpy as np
import shutil

from ...core import BaseProcessingStep
from ...core.types import ImageFile
from ...core.utils import ensure_dir
from ...io.bids import build_bids_name
from ...interfaces import ants, fsl # Assuming existence

class B1MappingStep(BaseProcessingStep):
    """
    Handles B1 Map preparation.
    1. If AFI: Registers AFI Reference to SPGR Reference, applies transform to B1 Map.
    2. If External: Registers/Resamples to SPGR Reference.
    """
    
    def __init__(self, config, logger, provenance, method="afi"):
        super().__init__(config, logger, provenance)
        self.method = method # 'afi', 'external', 'hifi'
        
    def run(self, 
            b1_image: ImageFile, 
            reference_image: ImageFile, 
            output_dir: Path, 
            force: bool = False,
            b1_ref_image: Optional[ImageFile] = None
           ) -> ImageFile:
           
        output_dir = ensure_dir(output_dir)
        
        ents = dict(b1_image.entities)
        ents['desc'] = 'preproc' # Aligned B1
        ents['space'] = getattr(reference_image.entities, 'get', lambda k,d: d)('space', 'native') 
        # Inherit space from ref?
        
        out_name = build_bids_name(ents)
        out_path = output_dir / out_name
        
        if out_path.exists() and not force:
             self.logger.info(f"Skipping B1 Alignment (Exists): {out_name}")
             return ImageFile(img=out_path, entities=ents)
             
        self.logger.info(f"Aligning B1 Map ({self.method}) to {reference_image.img.name}")
        
        if self.method == 'afi':
            if not b1_ref_image:
                 # AFI usually comes as a pair or 4D. 
                 # If separate, user must provide ref.
                 # If 4D, maybe b1_image is the map, and we assume ref was separated?
                 self.logger.warning("AFI B1 correction requested but no B1 Reference image provided. Assuming direct alignment of B1 map.")
                 # Fallback: Register B1 map itself to SPGR? (Risky due to contrast diffs)
                 moving = b1_image.img
            else:
                 moving = b1_ref_image.img
                 
            # 1. Register B1 Ref -> SPGR Ref
            # Use Rigid (intra-subject)
            # ANTs Quick
            prefix = str(output_dir / "b1_to_spgr_")
            
            # TODO: Refactor this generic registration call
            # For now simplified:
            from ...interfaces.fsl import flirt, applywarp
            
            # Calculate transform
            mat_file = output_dir / "b1_to_spgr.mat"
            flirt(in_file=moving, ref_file=reference_image.img, out_file=output_dir / "b1_ref_aligned.nii.gz", omat=mat_file, dof=6)
            
            # Apply to B1 Map
            self.logger.info("Applying transform to B1 Map")
            # Usually B1 map in same space as B1 Ref.
            from ...interfaces.fsl import applywarp # actually flirt -applyxfm
            from ...interfaces.fsl import resample_to_image # uses applyxfm usesqform - wait.
            # We want applyxfm with init.
            
            # Using FLIRT applyxfm
            # flirt -in <b1> -ref <spgr> -out <out> -init <mat> -applyxfm
            cmd = f"flirt -in {b1_image.img} -ref {reference_image.img} -out {out_path} -init {mat_file} -applyxfm"
            from ...core.run import run_cmd
            run_cmd(cmd, label="apply_b1_transform")
            
        elif self.method == 'external':
             # Just resample to match grid if needed, assuming already aligned? 
             # Or Register? 
             self.logger.info("Resampling External B1 to Reference Grid")
             from ...interfaces.fsl import resample_to_image
             resample_to_image(source_file=b1_image.img, reference_file=reference_image.img, out_file=out_path)
             
        elif self.method == 'hifi':
             pass # Handled by fitting binary directly usually? 
             
        return ImageFile(img=out_path, entities=ents)
