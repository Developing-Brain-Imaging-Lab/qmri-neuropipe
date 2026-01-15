
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
        processed_outputs = []

        # 1. Identify Reference (Max Flip Angle)
        if not reference_image:
             max_fa = -1.0
             ref_img_candidate = None
             for img in images:
                 fa = _extract_bids_param(img, "FlipAngle", 0.0)
                 if isinstance(fa, list): fa = max(fa) if fa else 0.0
                 if float(fa) > max_fa:
                     max_fa = float(fa)
                     ref_img_candidate = img
             
             if not ref_img_candidate:
                 ref_img_candidate = images[0]
                 
             reference_image = ref_img_candidate
             self.logger.info(f"Selected reference image (FA={max_fa}): {reference_image.img.name}")
             
        # Ensure Reference is 3D
        ref_path = Path(reference_image.img)
        try:
            ref_nii = nib.load(ref_path)
            if len(ref_nii.shape) == 4 and ref_nii.shape[3] > 1:
                 # Extract vol 0 as temp ref
                 temp_ref = output_dir / "temp_ref.nii.gz"
                 # Using fslroi directly via run_cmd
                 cmd = f"fslroi {ref_path} {temp_ref} 0 1"
                 run_cmd(cmd, label="fslroi_ref")
                 ref_path = temp_ref
        except Exception as e:
            self.logger.warning(f"Could not check dimensions of ref: {e}")

        # 2. Process Inputs
        for img in images:
            # Check if 4D
            is_4d = False
            try:
                nii = nib.load(img.img)
                if len(nii.shape) == 4 and nii.shape[3] > 1:
                    is_4d = True
            except:
                pass
            
            # Prepare Output Name
            ents = dict(img.entities)
            
            # Naming Logic: Force desc to {Modality}preproc
            # e.g. sub-01_acq-spgr_desc-SPGRpreproc_VFA.nii.gz
            acq_label = ents.get('acq', '').upper()
            if not acq_label: acq_label = "Moco" # Fallback
            
            new_desc = f"{acq_label}preproc"
            ents['desc'] = new_desc
            
            out_name = build_bids_name(ents)
            out_path = output_dir / out_name
            
            # Check if exists and valid
            if out_path.exists() and not force:
                try: 
                    check = nib.load(out_path)
                    if is_4d and (len(check.shape) != 4 or check.shape[3] < 2):
                         self.logger.warning(f"Existing output {out_name} appears truncated. Re-running.")
                    else:
                         self.logger.info(f"Skipping Motion Correction (Exists): {out_name}")
                         processed_outputs.append(ImageFile(img=out_path, entities=ents, json=img.json))
                         continue
                except:
                    pass 

            self.logger.info(f"Processing Motion Correction for: {img.img.name}")
            
            if is_4d:
                from ...interfaces.fsl import split, merge
                self.logger.info(f"  Input is 4D. Splitting and registering {nii.shape[3]} volumes...")
                
                split_dir = output_dir / f"temp_split_{img.img.stem}"
                split_dir.mkdir(exist_ok=True)
                split_prefix = split_dir / "vol"
                
                vols = split(img.img, split_prefix)
                
                corrected_vols = []
                for i, vol in enumerate(vols):
                    vol_out = split_dir / f"vol{i:04d}_moco.nii.gz"
                    self._register(vol, ref_path, vol_out)
                    corrected_vols.append(vol_out)
                    
                merge(corrected_vols, out_path, dimension='t')
                
                import shutil
                shutil.rmtree(split_dir)
                
            else:
                self._register(img.img, ref_path, out_path)
                
            processed_outputs.append(ImageFile(img=out_path, entities=ents, json=img.json))
            
        return processed_outputs

    def _register(self, in_file, ref_file, out_file):
        """Helper to run registration."""
        if self.method == 'ants':
             transform = self.options.get('transform_type', 'r') 
             threads = self.options.get('threads', 4)
             prefix = str(out_file).replace(".nii.gz", "").replace(".nii", "")
             cmd = f"antsRegistrationSyNQuick.sh -d 3 -f {ref_file} -m {in_file} -o {prefix} -t {transform} -n {threads}"
             if 'args' in self.options: cmd += f" {self.options['args']}"
             run_cmd(cmd, label="ants_moco")
             warped = Path(f"{prefix}Warped.nii.gz")
             if warped.exists(): warped.rename(out_file)
             
        elif self.method == 'fsl':
             from ...interfaces.fsl import flirt
             flirt_kwargs = {
                 'in_file': in_file, 'ref_file': ref_file, 'out_file': out_file,
                 'dof': self.options.get('dof', 6)
             }
             for k in ['cost', 'bins', 'searchcost', 'interp']:
                 if k in self.options: flirt_kwargs[k] = self.options[k]
             flirt(**flirt_kwargs)


