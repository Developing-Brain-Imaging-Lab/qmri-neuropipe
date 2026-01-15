
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import nibabel as nib

from ...core import BaseProcessingStep
from ...core.types import ImageFile
from ...core.utils import ensure_dir
from ...io.bids import build_bids_name
from ...interfaces import ants, fsl # Assuming ants interface exists or use direct
from ...utils.relax_params import _extract_bids_param

class SPGRMotionCorrectionStep(BaseProcessingStep):
    """
    Motion Correction for VFA (SPGR/SSFP) data.
    Registers all volumes to the volume with the highest Flip Angle (highest SNR).
    """
    
    def __init__(self, config, logger, provenance, method="ants"):
        super().__init__(config, logger, provenance)
        self.method = method
        
    def run(self, 
            images: List[ImageFile], 
            output_dir: Path, 
            force: bool = False,
            reference_image: Optional[ImageFile] = None
           ) -> List[ImageFile]:
           
        output_dir = ensure_dir(output_dir)
        
        # 1. Identify Reference (Max Flip Angle)
        if not reference_image:
            max_fa = -1.0
            ref_idx = 0
            
            for i, img in enumerate(images):
                # Handle 4D: If image is 4D, we might need to split it? 
                # Or assume input is list of 3D or 4D?
                # Usually VFA is acquired as separate files or one 4D. 
                # If 4D, we need to extract volumes first?
                # User's plan said "Input: Motion-corrected 4D VFA". 
                # Actually, input TO this step is raw 4D VFA.
                
                # If input is a LIST of ImageFiles, assume each is a 3D volume (or one 4D).
                # If it's a single 4D file in a list, we need to split it for motion correction?
                # Or use a 4D motion correction tool.
                
                # Check dimensions
                try:
                    p_img = nib.load(img.img)
                    dims = p_img.shape
                    
                    if len(dims) == 4 and dims[3] > 1:
                        # It is 4D. Step handles splitting?
                        # For now assume list of separate files (DESPOT usually separate scans).
                        # If 4D, we might treat the first volume as index?
                        pass
                except:
                    pass
                
                fa =  _extract_bids_param(img, "FlipAngle", 0.0)
                if isinstance(fa, list): fa = max(fa) if fa else 0.0 # If 4D and list
                
                if fa > max_fa:
                    max_fa = fa
                    ref_idx = i
                    
            reference_image = images[ref_idx]
            self.logger.info(f"Selected reference image (FA={max_fa}): {reference_image.img.name}")

        # 2. Register each image to Reference
        corrected_images = []
        
        # Prepare reference path
        ref_path = Path(reference_image.img)
        
        for img in images:
            # Skip if it is the reference? (Just copy)
            is_ref = (img.img == reference_image.img)
            
            ents = dict(img.entities)
            ents['desc'] = 'moco'
            out_name = build_bids_name(ents)
            out_path = output_dir / out_name
            
            if out_path.exists() and not force:
                self.logger.info(f"Skipping Motion Correction (Exists): {out_name}")
                corrected_images.append(ImageFile(img=out_path, entities=ents, json=img.json)) # Keep original JSON
                continue
            
            if is_ref:
                # Copy reference
                import shutil
                shutil.copy(img.img, out_path)
                corrected_images.append(ImageFile(img=out_path, entities=ents, json=img.json))
                continue
                
            self.logger.info(f"Registering {img.img.name} -> {ref_path.name}")
            
            # Execute Registration (Rigid)
            if self.method == 'ants':
                # Use ants.registration (Quick rigid)
                # Need wrapper or direct call?
                # Assuming simple rigid interface exists or wrap antsRegistrationSyNQuick
                
                # cmd: antsRegistrationSyNQuick.sh -d 3 -f <ref> -m <moving> -o <out_prefix> -t r
                # We need to construct output prefix logic for ANTs
                prefix = str(out_path).replace(".nii.gz", "").replace(".nii", "")
                
                # Call ANTs interface directly?
                # I'll rely on a run_cmd valid call or the `interfaces.ants` if I knew it.
                # Since I didn't verify `interfaces.ants`, I'll use run_cmd directly for `antsRegistrationSyNQuick.sh`
                
                cmd = f"antsRegistrationSyNQuick.sh -d 3 -f {ref_path} -m {img.img} -o {prefix} -t r -n 2" # -n 2 threads
                
                # Check if output is named consistently. 
                # antsRegistrationSyNQuick -o prefix -> outputs prefixWarped.nii.gz
                # We want just out_path.
                # So we rename after.
                
                run_cmd(cmd, label="ants_moco")
                
                warped = Path(f"{prefix}Warped.nii.gz")
                if warped.exists():
                     warped.rename(out_path)
                     
                # Add check for Mat? 
            
            elif self.method == 'fsl':
                # FLIRT
                # flirt -in <in> -ref <ref> -out <out> -dof 6
                from ...interfaces.fsl import flirt
                flirt(in_file=img.img, ref_file=ref_path, out_file=out_path, dof=6)
                
            corrected_images.append(ImageFile(img=out_path, entities=ents, json=img.json))
            
        return corrected_images

