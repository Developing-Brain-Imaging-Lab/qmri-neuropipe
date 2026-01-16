
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
    
    def __init__(self, config, logger, provenance, method="afi", smoothing_fwhm: float = 2.0):
        super().__init__(config, logger, provenance)
        self.method = method # 'afi', 'external', 'hifi'
        self.smoothing_fwhm = float(smoothing_fwhm if smoothing_fwhm is not None else 0.0)
        
    def run(self, 
            b1_image: ImageFile, 
            reference_image: ImageFile, 
            output_dir: Path, 
            force: bool = False,
            b1_ref_image: Optional[ImageFile] = None
           ) -> ImageFile:
           
        output_dir = ensure_dir(output_dir)
        
        ents = dict(b1_image.entities)
        # Final Output Name: sub-XX_TB1map.nii.gz
        # Filter entities to minimal set (ensure sub/ses keys exist)
        minimal_ents = {}
        # Normalize to short keys for build_bids_name
        if 'subject' in ents: minimal_ents['sub'] = ents['subject']
        if 'sub' in ents: minimal_ents['sub'] = ents['sub']
        if 'session' in ents: minimal_ents['ses'] = ents['session']
        if 'ses' in ents: minimal_ents['ses'] = ents['ses']

        minimal_ents['suffix'] = 'TB1map' # Force suffix
        
        out_name = build_bids_name(minimal_ents)
        out_path = output_dir / out_name
        
        if out_path.exists() and not force:
             self.logger.info(f"Skipping B1 Alignment (Exists): {out_name}")
             return ImageFile(img=out_path, entities=ents)
             
        self.logger.info(f"Aligning B1 Map ({self.method}) to {reference_image.img.name}")
        

        if self.method == 'afi':
            # Check if input is Map or Raw (2 volumes)
            import nibabel as nib
            img = nib.load(b1_image.img)
            
            # Identify if Raw AFI (4th dim exists and > 1)
            is_raw_afi = (len(img.shape) == 4 and img.shape[3] >= 2)
            
            b1_map_path = b1_image.img # Default if already map
            
            if is_raw_afi:
                 self.logger.info("Detected Raw AFI Input (4D). Aligning before computation...")
                 
                 # 1. Split 4D
                 from ...interfaces.fsl import split, merge, flirt, applywarp
                 
                 # Temp dir for split
                 tmp_split = output_dir / "tmp_afi_split"
                 tmp_split.mkdir(exist_ok=True)
                 
                 split_base = tmp_split / "vol"
                 vols = split(b1_image.img, split_base)
                 
                 if len(vols) < 2:
                     raise ValueError("AFI input seems to be 4D but found less than 2 volumes.")
                     
                 # 2. Register Vol 0 (S1) to Reference
                 # Use Rigid (6 DOF)
                 vol0 = vols[0]
                 mat_file = output_dir / "afi_to_spgr.mat"
                 
                 if not mat_file.exists() or force:
                      flirt(in_file=vol0, ref_file=reference_image.img, out_file=tmp_split / "vol0_aligned_ref.nii.gz", omat=mat_file, dof=6)
                 
                 # 3. Apply Transform to ALL volumes independently
                 aligned_vols = []
                 for i, v in enumerate(vols):
                     out_v = tmp_split / f"vol{i}_aligned.nii.gz"
                     # applywarp or flirt -applyxfm
                     cmd = f"flirt -in {v} -ref {reference_image.img} -out {out_v} -init {mat_file} -applyxfm"
                     from ...core.run import run_cmd
                     run_cmd(cmd, label=f"apply_afi_prop_{i}")
                     aligned_vols.append(out_v)
                     
                 # 4. Merge back to 4D
                 aligned_afi_path = output_dir / f"{b1_image.img.stem}_aligned.nii.gz"
                 merge(aligned_vols, aligned_afi_path, dimension='t')
                 
                 # Cleanup split
                 import shutil
                 shutil.rmtree(tmp_split)
                 
                 # Update B1 Image object to point to aligned data
                 aligned_afi_img = ImageFile(img=aligned_afi_path, entities=b1_image.entities, json=b1_image.json)
                 
                 # 5. Compute Map from Aligned Data
                 b1_map_path = self._compute_afi(aligned_afi_img, output_dir)
                 
                 # Move result to final if needed? _compute_afi returns path to generated map
                 # Ensure it goes to out_path
                 if b1_map_path != out_path:
                      import shutil
                      shutil.move(b1_map_path, out_path)
                      
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
        B1 = arccos( (r*n - 1) / (n - r) ) / nominal_FA   (Standard)
        Here using r = S2/S1 (Long/Short).
        Derived: val = (n - r) / (n*r - 1)
        """
        import numpy as np
        import scipy.ndimage as nd
        
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
            if isinstance(tr, list) and len(tr) == 2:
                # TR1 is Short, TR2 is Long usually? 
                # Standards: TR1 = 20ms, TR2 = 100ms -> n=5.
                tr_sorted = sorted(tr)
                n_ratio = float(tr_sorted[1] / tr_sorted[0])
            else:
                self.logger.warning("AFI: RepetitionTime not a list of 2 in JSON and no TRRatio found. Assuming n=5 for testing.")
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
        
        # Check dim
        if data.shape[3] < 2:
             raise ValueError("AFI data must have at least 2 volumes.")
             
        s1 = data[..., 0]
        s2 = data[..., 1]
        
        # Smoothing (if requested)
        if self.smoothing_fwhm > 0:
            self.logger.info(f"Applying Gaussian Smoothing (FWHM={self.smoothing_fwhm}mm)")
            
            # Convert FWHM to Sigma (voxels)
            # sigma = FWHM / 2.355
            # sigma_vox = sigma / vox_size
            vox_sizes = img_nii.header.get_zooms()[:3]
            sigmas = [(self.smoothing_fwhm / 2.355) / v for v in vox_sizes]
            
            s1 = nd.gaussian_filter(s1, sigma=sigmas)
            s2 = nd.gaussian_filter(s2, sigma=sigmas)
            

        
        r = s2 / s1
        r[r>1] = 1

        arg = (r*n_ratio - 1)/(n_ratio-r)
        arg[arg>1] = 1
        arg[arg<-1] = -1
        
        alpha_act = np.arccos(arg)
        b1_map = alpha_act / flip_angle_rad
        
        # Clip crazy values
        b1_map = np.clip(b1_map, 0, 2.0) # B1 usually 0.5 to 1.5
        
        # Save
        # Save Intermediate: sub-XX_desc-preproc_TB1AFI
        # Use entities from input
        afi_ents = dict(afi_image.entities)
        # Keep sub/ses, add/overwrite desc/suffix
        afi_inter_ents = {}
        if 'subject' in afi_ents: afi_inter_ents['sub'] = afi_ents['subject']
        if 'sub' in afi_ents: afi_inter_ents['sub'] = afi_ents['sub']
        if 'session' in afi_ents: afi_inter_ents['ses'] = afi_ents['session']
        if 'ses' in afi_ents: afi_inter_ents['ses'] = afi_ents['ses']
        
        afi_inter_ents['desc'] = 'preproc'
        afi_inter_ents['suffix'] = 'TB1AFI'
        
        out_name = output_dir / build_bids_name(afi_inter_ents)
        nib.save(nib.Nifti1Image(b1_map, img_nii.affine, img_nii.header), out_name)
        
        return out_name
