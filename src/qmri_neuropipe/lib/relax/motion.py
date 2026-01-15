
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
    

    def __init__(self, config, logger, provenance, method="ants", options: dict = None):
        super().__init__(config, logger, provenance)
        self.method = method
        self.options = options or {}
        
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
                # Custom options for ANTs
                # Default logic: Rigid (-t r)
                # If options provided, override construction or append?
                # Supporting basic flags: 'transform_type' (t), 'threads' (n)
                # Or raw args string?
                # User asked for "pass along fsl flirt or ants options"
                
                transform = self.options.get('transform_type', 'r') # default rigid
                threads = self.options.get('threads', 4) # default 4
                
                # Construct base cmd
                prefix = str(out_path).replace(".nii.gz", "").replace(".nii", "")
                
                cmd = f"antsRegistrationSyNQuick.sh -d 3 -f {ref_path} -m {img.img} -o {prefix} -t {transform} -n {threads}" 
                
                # Append extra raw args if needed?
                # e.g. options={'args': '-x Mask.nii.gz'}
                if 'args' in self.options:
                     cmd += f" {self.options['args']}"
                
                run_cmd(cmd, label="ants_moco")
                
                warped = Path(f"{prefix}Warped.nii.gz")
                if warped.exists():
                     warped.rename(out_path)
                     
            elif self.method == 'fsl':
                # FLIRT
                from ...interfaces.fsl import flirt
                # Map options dict to flirt args
                # supported: dof, cost, searchcost, interp...
                
                flirt_kwargs = {
                    'in_file': img.img,
                    'ref_file': ref_path,
                    'out_file': out_path,
                    'dof': self.options.get('dof', 6), # Default 6 (Rigid)
                }
                
                # Optional args supported by wrapper or we pass as raw?
                # Our wrapper usually accepts specific args.
                # If wrapper doesn't support 'cost', we might need to modify wrapper or use run_cmd?
                # Let's check wrapper signature... we assumed it exists.
                # Standard pattern: kwargs passed to run_cmd or built?
                # Let's just pass self.options content that matches FLIRT flags if possible.
                
                # Basic ones
                if 'cost' in self.options: flirt_kwargs['cost'] = self.options['cost']
                if 'bins' in self.options: flirt_kwargs['bins'] = self.options['bins']
                if 'searchcost' in self.options: flirt_kwargs['searchcost'] = self.options['searchcost']
                if 'interp' in self.options: flirt_kwargs['interp'] = self.options['interp']
                
                flirt(**flirt_kwargs)
                
            corrected_images.append(ImageFile(img=out_path, entities=ents, json=img.json))
            
        return corrected_images

