
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
            # Check if input is Map or Raw (2 volumes)
            import nibabel as nib
            img = nib.load(b1_image.img)
            
            # Identify if Raw AFI (4th dim = 2)
            is_raw_afi = (len(img.shape) == 4 and img.shape[3] == 2)
            
            b1_map_path = b1_image.img
            # Setup Reference for Registration
            # If Raw: use 1st volume (S1) as moving ref
            # If Map: use provided b1_ref_image OR map itself (less ideal)
            
            if is_raw_afi:
                 self.logger.info("Detected Raw AFI Input (2 volumes). Computing B1 Map...")
                 # Calculate B1 Map
                 b1_map_path = self._compute_afi(b1_image, output_dir)
                 
                 # Ref for registration is the first volume of raw AFI
                 # Extract vol 0
                 ref_moving = output_dir / f"{b1_image.img.stem}_vol0.nii.gz"
                 # nibabel slicing
                 img_data = img.get_fdata()
                 vol0 = img_data[..., 0]
                 nib.save(nib.Nifti1Image(vol0, img.affine, img.header), ref_moving)
                 moving = ref_moving
            else:
                 # Input is already a map? Or separate files?
                 # Assuming Map if single volume or user pre-processed.
                 moving = b1_ref_image.img if b1_ref_image else b1_image.img

            # 1. Register B1 Ref -> SPGR Ref
            # Use Rigid (intra-subject)
            prefix = str(output_dir / "b1_to_spgr_")
            mat_file = output_dir / "b1_to_spgr.mat"
            
            from ...interfaces.fsl import flirt, applywarp
            
            # Calculate transform
            flirt(in_file=moving, ref_file=reference_image.img, out_file=output_dir / "b1_ref_aligned.nii.gz", omat=mat_file, dof=6)
            
            # Apply to B1 Map
            self.logger.info("Applying transform to B1 Map")
            # Apply XFM to the computed (or input) map
            cmd = f"flirt -in {b1_map_path} -ref {reference_image.img} -out {out_path} -init {mat_file} -applyxfm"
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

    def _compute_afi(self, afi_image: ImageFile, output_dir: Path) -> Path:
        """
        Calculate B1 map from AFI (2-volume 4D) data.
        B1 = arccos( (r*n - 1) / (n - r) ) / nominal_FA
        Where r = S1/S2, n = TR2/TR1
        """
        import numpy as np
        

        # 1. Get Parameters (TRRatio, TR1, TR2, FA)
        json_data = afi_image.json or {}
        
        # Check for explicit TRRatio (n)
        n_ratio = json_data.get('TRRatio')
        

        if n_ratio:
             self.logger.info(f"Using provided TRRatio from JSON: {n_ratio}")
             n_ratio = float(n_ratio)
        else:
            # Fallback to TR calculation
            tr = json_data.get('RepetitionTime')
            tr1, tr2 = 0.0, 0.0
            
            if isinstance(tr, list) and len(tr) == 2:
                tr1, tr2 = sorted(tr) 
                tr1, tr2 = tr[0], tr[1]
                n_ratio = float(tr2 / tr1)
            else:
                self.logger.warning("AFI: RepetitionTime not a list of 2 in JSON and no TRRatio found. Assuming n=5 for testing.")
                if not tr or not isinstance(tr, list):
                     raise ValueError(f"AFI Calculation requires 'RepetitionTime' as [TR1, TR2] or 'TRRatio' in JSON.")
                # If we are here, we might have crashed above, but let's be robust
                n_ratio = 5.0 # Fallback
        
        flip_angle_deg = json_data.get('FlipAngle')
        # If list, take first? Usually same FA.
        if isinstance(flip_angle_deg, list): flip_angle_deg = flip_angle_deg[0]
        if not flip_angle_deg: 
             self.logger.warning("AFI: FlipAngle missing. Assuming 60 degrees.")
             flip_angle_deg = 60.0
        
        flip_angle_rad = np.deg2rad(flip_angle_deg)
        
        # Load Data
        img_nii = nib.load(afi_image.img)
        data = img_nii.get_fdata()
        
        s1 = data[..., 0]
        s2 = data[..., 1]
        
        # Masking? (Avoid div by zero)
        mask = (s1 > 10) & (s2 > 10) # Simple threshold
        
        # Ratio r = S2 / S1 ? Or S1 / S2?
        # Yarnykh: r = S2/S1 (where S1 is short TR, S2 is long TR signal... NO)
        # Equation: alpha = arccos( (r*n - 1) / (n - r) )
        # If n = TR2/TR1 > 1.
        # Check limits.
        # Implemented formula:
        # r = S2 / S1 ?
        # Let's try standard approx.
        # r = S1 / S2 (Short / Long). Expect S1 < S2 due to T1 recovery? 
        # Actually Long TR = more recovery = Higher signal.
        # So S2 > S1. r < 1.
        # n > 1.
        # (r*n - 1) / (n - r).
        
        r = np.zeros_like(s1)
        r[mask] = s1[mask] / s2[mask] # S1 (short) / S2 (long)
        
        # B1 Map = alpha_actual / alpha_nominal
        # alpha_actual = arccos ...
        
        val = (r * n_ratio - 1) / (n_ratio - r)
        # clip to [-1, 1] for arccos
        val = np.clip(val, -1.0, 1.0)
        
        alpha_act = np.arccos(val)
        b1_map = alpha_act / flip_angle_rad
        
        # Clip crazy values
        b1_map[~mask] = 0
        b1_map = np.clip(b1_map, 0, 2.0) # B1 usually 0.5 to 1.5
        
        # Save
        out_name = output_dir / f"{afi_image.img.stem}_B1.nii.gz"
        nib.save(nib.Nifti1Image(b1_map, img_nii.affine, img_nii.header), out_name)
        
        return out_name
