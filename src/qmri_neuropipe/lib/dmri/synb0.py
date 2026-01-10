"""
Synb0 Distortion Correction module (Deep Learning).

Uses dipy.nn.tf.synb0 to estimate an undistorted (or reverse PE) b0 signal
from a distorted b0 and a T1w image.
"""

from pathlib import Path
from typing import Optional, Literal, Tuple
import numpy as np
import nibabel as nib
import logging
import json

from ...core import BaseProcessingStep, ValidationError, ProcessingError
from ...core.run import run_cmd
from ...core.types import ImageFile, DWIFile, ImageLike
from ...interfaces import dipy, freesurfer, fsl, c3d, ants
from ..common.mask import mask_brain    
from ...io.bids import build_bids_name

class Synb0EstimationStep(BaseProcessingStep):
    """
    Synb0 estimation step.
    
    Generates a synthetic b0 image (representing the reverse phase encoding direction)
    using a Deep Learning model (DIPY Synb0).
    
    This synthetic b0 is then paired with the real b0 to form a Topup config,
    allowing Topup to estimate the susceptibility field.
    
    Attributes:
        method: 'dipy-dl'
    """
    
    def __init__(
        self,
        config,
        logger: Optional[logging.Logger] = None,
        provenance = None,
    ):
        super().__init__(config, logger, provenance)
        self.method = 'dipy-dl'
        self.logger.info(f"Initialized Synb0 estimation (Deep Learning).")

    def validate_inputs(self, first_arg, output_dir: Path, **kwargs) -> None:
        context, _ = self.unpack_input(first_arg)
        if context is None:
             raise ValidationError("Synb0EstimationStep requires pipeline context.")
        
        if not context.get("t1w_files"):
            self.logger.warning("Synb0 estimation requires T1w images in context.")
        
        dwi_files = context.get("dwi_files", [])
        if not dwi_files:
            raise ValidationError("No DWI files found for Synb0 estimation.")

    def run(self, first_arg, output_dir: Path, **kwargs) -> dict:
        """
        Run Synb0 estimation.
        """
        context, _ = self.unpack_input(first_arg)
        if context is None:
            raise ProcessingError("Synb0EstimationStep must run in pipeline context mode.")
        dwi_files: list[DWIFile] = context.get("dwi_files", [])
        t1w_files = context.get("t1w_files", [])
        
        if not t1w_files:
             self.logger.warning("Skipping Synb0 estimation: No T1w found.")
             return context
             
        # Use first T1w
        t1w_path = t1w_files[0].img
        
        # We need to generate a synthetic b0 for each distinct acquisition group? 
        # Or usually just one per session if they share geometry?
        # Let's assume we do it for the first DWI file and assume others share distortion if valid.
        # Ideally, we should check for different PE directions.
        
        # For simplicity, pick the first DWI as the "forward" b0 source.
        input_dwi = dwi_files[0]
        real_json = input_dwi.json
        
        output_dir = self.get_step_output_dir(output_dir)
        
        # 1. Check if outputs exist
        syn_b0_path = output_dir / "syn_b0_desc-synthetic.nii.gz"
        syn_b0_native_path = output_dir / "syn_b0_native.nii.gz"
        syn_json_path = syn_b0_path.with_suffix(".json")
        b0_path = output_dir / "real_b0.nii.gz"
        dummy_bval_path = output_dir / "b0.bval"
        
        if syn_b0_path.exists() and b0_path.exists() and syn_b0_native_path.exists() and syn_json_path.exists() and dummy_bval_path.exists() and not kwargs.get('force', False):
            self.logger.info(f"Skipping Synb0 estimation (outputs exist): {syn_b0_path}")
        else:
            # Extract real b0
            img = nib.load(str(input_dwi.img))
            data = img.get_fdata()
            
            # Use bvals to find b0
            b0_vol = data[..., 0] # Default
            if input_dwi.bval and input_dwi.bval.exists():
                 try:
                     bvals = np.loadtxt(str(input_dwi.bval))
                     if bvals.ndim == 2: bvals = bvals[0]
                     b0_indices = np.where(bvals < 50)[0]
                     if len(b0_indices) > 0:
                         b0_vol = data[..., b0_indices]
                 except Exception:
                     pass
            
            if b0_vol.ndim == 4:
                b0_vol = b0_vol.mean(axis=-1)
            # Force 4D
            if b0_vol.ndim == 3:
                b0_vol = b0_vol[..., np.newaxis]
            nib.Nifti1Image(b0_vol, img.affine, img.header).to_filename(b0_path)
    
            #Preprocess T1w
            #Normalize T1w
            t1w_mgz = output_dir / "t1w.mgz"
            freesurfer.mri_convert(in_file=t1w_files[0].img, out_file=t1w_mgz)
    
            t1w_n3 = output_dir / "t1w_n3.mgz"
            freesurfer.mri_nu_correct(in_file=t1w_mgz, out_file=t1w_n3)
    
            t1w_norm = output_dir / "t1w_norm.mgz"
            freesurfer.mri_normalize(in_file=t1w_n3, out_file=t1w_norm)
    
            t1w_norm_nii = output_dir / "t1w_norm.nii.gz"
            freesurfer.mri_convert(in_file=t1w_norm, out_file=t1w_norm_nii)
    
            #Skull-strip T1w
            t1w_brain, t1w_mask = mask_brain(
                input_image=t1w_norm_nii,
                output_dir=output_dir,
                method="synthstrip",
                return_mask=True
            )
            
            # Get paths
            t1w_brain_path = t1w_brain.img
            # t1w_mask_path = t1w_mask.img
    
            #Register T1 to b0 (rigid)
            t1w_brain_reg = output_dir / "t1w_brain_reg.nii.gz"
            t1w_2_dwi_fslmat = output_dir / "t1w_2_dwi.mat"
            t1w_2_dwi_antsmat = output_dir / "t1w_2_dwi.txt"
    
            # Standardized call
            _, t1w_2_dwi_fslmat = fsl.flirt(in_file=t1w_brain_path,
                                           ref_file=b0_path,
                                           out_file=t1w_brain_reg,
                                           omat=t1w_2_dwi_fslmat,
                                           cost="normmi",
                                           dof=6,
                                           extra_args="-usesqform -searchrx -180 180 -searchry -180 180 -searchrz -180 180")
            #Convert FSL to ANTS
            c3d.fsl2ants(ref_file=b0_path, # Target is DWI
                         in_file=t1w_brain_path, # Moving is T1w
                         transform_file=t1w_2_dwi_fslmat,
                         out_file=t1w_2_dwi_antsmat)

            # Pre-invert FSL matrix (DWI -> T1w) for ANTs
            # (ANTsPy fails to invert ITK files from c3d on the fly)
            dwi_2_t1w_fslmat = output_dir / "dwi_2_t1w.mat"
            fsl.convert_xfm(in_file=t1w_2_dwi_fslmat, out_file=dwi_2_t1w_fslmat, inverse=True)
            
            # Convert Inverse FSL to ITK (DWI -> T1w)
            dwi_2_t1w_antsmat = output_dir / "dwi_2_t1w.txt"
            c3d.fsl2ants(ref_file=t1w_brain_path, # Target is T1w
                         in_file=b0_path,        # Moving is DWI
                         transform_file=dwi_2_t1w_fslmat,
                         out_file=dwi_2_t1w_antsmat)
    
            #REGISTER T1 to Atlas
            mni_atlas_img = Path(__file__).parent / "data" / "mni_icbm152_t1_tal_nlin_asym_09c_2_5.nii.gz"
            t1w_mni = output_dir / "t1w_mni.nii.gz"
            t1w_2_mni_fslmat = output_dir / "t1w_2_mni.mat"
            t1w_2_mni_antsmat = output_dir / "t1w_2_mni.txt"
            mni_2_t1w_fslmat = output_dir / "mni_2_t1w.mat"
            mni_2_t1w_antsmat = output_dir / "mni_2_t1w.txt"
            
            _, t1w_2_mni_fslmat = fsl.flirt(in_file=t1w_norm_nii,
                                            ref_file=mni_atlas_img,
                                            out_file=t1w_mni, # This should be the output image, not the matrix
                                            omat=t1w_2_mni_fslmat,
                                            cost="normmi",
                                            dof=12,
                                            extra_args="-usesqform")
            fsl.convert_xfm(in_file=t1w_2_mni_fslmat, out_file=mni_2_t1w_fslmat, inverse=True)

            c3d.fsl2ants(ref_file=mni_atlas_img,
                         in_file=t1w_brain_path,
                         transform_file=t1w_2_mni_fslmat,
                         out_file=t1w_2_mni_antsmat)

            c3d.fsl2ants(ref_file=t1w_brain_path,
                         in_file=mni_atlas_img,
                         transform_file=mni_2_t1w_fslmat,
                         out_file=mni_2_t1w_antsmat)
    
           
            #Apply linear registration to normalized T1w to get into Atlas space
            t1w_norm_atlas = output_dir / "t1w_norm_mni.nii.gz"
            b0_in_mni = output_dir / "b0_in_mni.nii.gz"
    
            ants.apply_transforms(fixed_file=mni_atlas_img,
                                  moving_file=t1w_norm_nii,
                                  out_file=t1w_norm_atlas,
                                  transforms=[t1w_2_mni_antsmat],
                                  invert_transforms=[False],
                                  interpolator="bSpline")
                                  
            #Apply series of registrations to dwi
            # Chain: DWI -> T1w -> MNI
            ants.apply_transforms(fixed_file=mni_atlas_img,
                                  moving_file=b0_path,
                                  out_file=b0_in_mni,
                                  transforms=[t1w_2_mni_antsmat, dwi_2_t1w_antsmat],
                                  invert_transforms=[False, False],
                                  interpolator="bSpline")   
                                  
    
            # 2. Run Synb0 Estimation (Real b0 + T1w -> Synthetic Reverse b0)
            # Run in isolated subprocess to avoid TF/Torch conflicts
            try:
                import sys
                import os
                
                # Check for GPUs
                if self.config.gpu_ids is not None:
                    gpus = self.config.gpu_ids
                    if isinstance(gpus, int):
                        gpus = [gpus]
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
                    self.logger.info(f"Setting CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
                
                self.logger.info(f"Running Synb0 estimation (isolated process) using {t1w_path.name}...")
                
                # Create temporary script for isolation
                script_path = output_dir / "run_synb0_isolated.py"
                # Use raw strings and proper escaping
                script_content = f"""
import os
import sys
import logging

# CRITICAL: Prevent PyTorch from loading to avoid TF/Torch conflict (SegFault).
# Even if not used directly, dipy.data might try to import it.
try:
    sys.modules['torch'] = None
    sys.modules['dipy.nn.torch'] = None
    sys.modules['dipy.nn.torch.deepn4'] = None
except Exception: 
    pass

# Ensure we can import modules from the environment
try:
    print("DEBUG: Importing tensorflow...", file=sys.stderr, flush=True)
    import tensorflow as tf
    print(f"DEBUG: TensorFlow Version: {{tf.__version__}}", file=sys.stderr, flush=True)
    
    print("DEBUG: Importing dipy...", file=sys.stderr, flush=True)
    import dipy
    print(f"DEBUG: DIPY Version: {{dipy.__version__}}", file=sys.stderr, flush=True)

    print("DEBUG: Importing dipy.nn.tf.synb0...", file=sys.stderr, flush=True)
    import dipy.nn.tf.synb0  # Force TF import here in isolation
    from dipy.nn.tf.synb0 import Synb0
    print("DEBUG: Import successful.", file=sys.stderr, flush=True)

except ImportError as e:
    print(f"DEBUG: Import failed: {{e}}", file=sys.stderr, flush=True)
    # Fallback for newer DIPY versions if structure changed, or try generic
    print("Warning: dipy.nn.tf.synb0 not found, trying dipy.nn.synb0", file=sys.stderr, flush=True)
    import dipy.nn.synb0 as synb0_mod
    Synb0 = synb0_mod.Synb0

import nibabel as nib
import numpy as np

def run_synb0(b0_file, t1_file, out_file):
    print(f"Loading files: {{b0_file}}, {{t1_file}}", file=sys.stderr, flush=True)
    b0_img = nib.load(b0_file)
    t1_img = nib.load(t1_file)
    
    b0_data = b0_img.get_fdata()
    t1_data = t1_img.get_fdata()
    
    if b0_data.ndim == 4:
        b0_data = b0_data[..., 0]
        
    print("Initializing model...", file=sys.stderr, flush=True)
    # Initialize model (False = not verbose? or parallel? Synb0 init arg is 'verbose')
    try:
        SyNb0 = Synb0(verbose=False)
        print("Model initialized.", file=sys.stderr, flush=True)
    except TypeError:
        print("Retrying init without verbose arg...", file=sys.stderr, flush=True)
        SyNb0 = Synb0() 
        print("Model initialized (fallback).", file=sys.stderr, flush=True)

    print("Predicting...", file=sys.stderr, flush=True)
    # Use config-like checks if needed, but hardcoded for now
    rev_b0_data = SyNb0.predict(b0_data, t1_data)
    print("Prediction done.", file=sys.stderr, flush=True)
    
    print(f"Saving to {{out_file}}", file=sys.stderr, flush=True)
    nib.Nifti1Image(rev_b0_data, b0_img.affine, b0_img.header).to_filename(out_file)

if __name__ == "__main__":
    try:
        run_synb0(
            r"{str(b0_in_mni)}",
            r"{str(t1w_norm_atlas)}",
            r"{str(syn_b0_path)}"
        )
        print("Synb0 prediction complete.", file=sys.stderr, flush=True)
    except Exception as e:
        # Check for OOM or other massive errors
        print(f"Error: {{e}}", file=sys.stderr, flush=True)
        sys.exit(1)
"""
                with open(script_path, "w") as f:
                    f.write(script_content)
                    
                # Execute
                cmd = f"{sys.executable} {str(script_path)}"
                run_cmd(cmd)
                
            except Exception as e:
                 raise ProcessingError(f"Synb0 estimation failed: {e}")
                 
            #Now inverse warp the synthetic to native b0 space
            # Chain: Synthetic -> T1w -> DWI 
            # syn_b0_native_path declared above
            ants.apply_transforms(fixed_file=b0_path,
                                  moving_file=syn_b0_path,
                                  out_file=syn_b0_native_path,
                                  transforms=[t1w_2_dwi_antsmat, mni_2_t1w_antsmat],
                                  invert_transforms=[False, False],
                                  interpolator="bSpline")

            #Warp T1w mask to DWI space
            t1w_mask_2_dwi = output_dir / "t1w_mask_2_dwi.nii.gz"
            ants.apply_transforms(fixed_file=b0_path,
                                  moving_file=t1w_mask,
                                  out_file=t1w_mask_2_dwi,
                                  transforms=[t1w_2_dwi_antsmat],
                                  invert_transforms=[False],
                                  interpolator="nearestNeighbor")


            #Apply mask to synthetic and real b0
            #run_cmd(f"fslmaths {syn_b0_native_path} -mas {t1w_mask_2_dwi} {syn_b0_native_path}")
            #run_cmd(f"fslmaths {b0_path} -mas {t1w_mask_2_dwi} {b0_path}")
            
            #Force 4D for syn_b0_native
            img_syn = nib.load(str(syn_b0_native_path))
            if img_syn.ndim == 3:
                 nib.Nifti1Image(img_syn.get_fdata()[..., np.newaxis], img_syn.affine, img_syn.header).to_filename(syn_b0_native_path)
            
            #Force 4D for b0_path (real b0)
            img_b0 = nib.load(str(b0_path))
            if img_b0.ndim == 3:
                 nib.Nifti1Image(img_b0.get_fdata()[..., np.newaxis], img_b0.affine, img_b0.header).to_filename(b0_path)

            
            

            real_meta = {}
            if real_json and real_json.exists():
                with open(real_json, "r") as f:
                    real_meta = json.load(f)
            
            real_pe = real_meta.get("PhaseEncodingDirection", "j-")
            
            # Invert PE
            # if real_pe.endswith("-"):
            #     syn_pe = real_pe.rstrip("-")
            # else:
            #     syn_pe = real_pe + "-"
                
            syn_meta = {
                "PhaseEncodingDirection": real_pe,
                "TotalReadoutTime": 0.0, # For synthetic b0
                "Synthesized": True
            }
            with open(syn_json_path, "w") as f:
                json.dump(syn_meta, f)
                       # Create dummy bval for single b0
            # dummy_bval_path defined above
            with open(dummy_bval_path, "w") as f:
                f.write("0\n")

        syn_dwi_file = DWIFile(
            entities={**input_dwi.entities, "desc": "synthetic"},
            img=syn_b0_native_path,
            json=syn_json_path,
            bval=dummy_bval_path,
            bvec=input_dwi.bvec  # Optional
        )
        
        # 4. Create a Topup Group
        # [Real b0, Synthetic b0]
        # We need to represent the real b0 as a DWIFile too if we want to add it to the group?
        # The input_dwi is the whole 4D file. We extracted real_b0.
        # Let's wrap real_b0 as well to be clean.
        real_b0_obj = DWIFile(
            entities={**input_dwi.entities, "desc": "realb0"},
            img=b0_path,
            json=input_dwi.json, # Shares metadata
            bval=dummy_bval_path,
            bvec=None
        )
        
        new_group = {
            "inputs": [real_b0_obj, syn_dwi_file],
            "targets": dwi_files
        }
        
        # Add to context
        # Check if 'topup_groups' exists. 
        topup_groups = context.get("topup_groups", [])
        topup_groups.append(new_group)
        context["topup_groups"] = topup_groups
        
        # Also need to ensure the main DWI is mapped to this group for applying corrections?
        # TopupStep usually handles mapping based on matching acquisition params.
        # We might need to manually update topup_map in TopupStep or ensure parameters match.
        # If we copy TotalReadoutTime, it should match.
        
        self.logger.info("Synb0 synthetic image generated and added to Topup groups.")
        return context
