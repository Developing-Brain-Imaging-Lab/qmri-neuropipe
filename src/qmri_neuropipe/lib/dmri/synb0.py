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
import gzip

from ...core import BaseProcessingStep, ValidationError, ProcessingError
from ...core.run import run_cmd
from ...core.types import ImageFile, DWIFile, ImageLike
from ...interfaces import dipy, freesurfer, fsl, c3d, ants
from ..common.mask import mask_brain    
from ...io.bids import build_bids_name
import multiprocessing
import sys
import traceback

def _run_synb0_worker(in_file, t1_file, out_file, gpu_ids=None):
    """
    Worker function to run Synb0 estimation in a separate process.
    This ensures GPU memory is released when the process terminates.
    """
    try:
        # Set CUDA_VISIBLE_DEVICES if provided
        if gpu_ids is not None:
             import os
             gpus = gpu_ids
             if isinstance(gpus, int):
                 gpus = [gpus]
             os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
             print(f"Synb0 Worker: Setting CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        
        # Import inside the worker process to avoid initializing TF in main process
        from ...interfaces import dipy
        
        print(f"Synb0 Worker: Running estimation...")
        dipy.synb0_estimation(
            in_file=in_file,
            t1_file=t1_file,
            out_file=out_file
        )
        print("Synb0 Worker: Finished successfully.")
        
    except Exception:
        print("Synb0 Worker failed with error:")
        traceback.print_exc()
        sys.exit(1)

def _validate_nifti(path: Path, logger: logging.Logger, label: str) -> None:
    if not path.exists():
        raise ProcessingError(f"{label} does not exist: {path}")
    if path.stat().st_size == 0:
        raise ProcessingError(f"{label} is empty (0 bytes): {path}")
    if path.suffix == ".gz":
        try:
            with gzip.open(path, "rb") as f:
                _ = f.read(2)
        except Exception as e:
            raise ProcessingError(f"{label} is not valid gzip: {path} ({e})")

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
        self.synb0_cfg = config.get("dmri.preprocessing.distcorr.synb0", {}) or {}
        self.logger.info(f"Initialized Synb0 estimation (Deep Learning).")

    def _select_anatomical_reference(self, context: dict, output_dir: Path, force: bool = False) -> Optional[ImageFile]:
        """
        Select the anatomical image used by Synb0.

        Defaults to the existing behavior (first T1w in context). When
        ``dmri.preprocessing.distcorr.synb0.t1w_source`` is ``supersynth`` or
        ``prefer_supersynth``, generate or reuse a SuperSynth T1w image from the
        configured anatomical input.
        """
        t1w_files = context.get("t1w_files", [])
        t2w_files = context.get("t2w_files", [])
        source = str(self.synb0_cfg.get("t1w_source", "raw")).lower()

        def _image_file(path: Path, desc: str = "synthT1w") -> ImageFile:
            entities = {
                "sub": context.get("subject"),
                "ses": context.get("session"),
                "desc": desc,
                "suffix": "T1w",
            }
            return ImageFile(
                entities={k: v for k, v in entities.items() if v},
                img=Path(path),
                json=None,
            )

        outputs = context.get("super_synth_outputs", {}) or {}
        existing_synth = outputs.get("synth_t1w")
        if source in {"supersynth", "prefer_supersynth"} and existing_synth:
            synth_path = Path(existing_synth)
            if synth_path.exists():
                self.logger.info(f"Using existing SuperSynth T1w for Synb0: {synth_path}")
                return _image_file(synth_path)

        if source in {"supersynth", "prefer_supersynth"}:
            preference = str(self.synb0_cfg.get("supersynth_input", "auto")).lower()
            anat_input = None
            if preference == "t2w":
                anat_input = t2w_files[0] if t2w_files else None
            elif preference == "t1w":
                anat_input = t1w_files[0] if t1w_files else None
            else:
                anat_input = t1w_files[0] if t1w_files else (t2w_files[0] if t2w_files else None)

            if anat_input is None:
                if source == "supersynth":
                    return None
            else:
                ss_dir = output_dir / "supersynth"
                synth_path = ss_dir / "T1w.nii.gz"
                if not synth_path.exists() or force:
                    self.logger.info(f"Generating SuperSynth T1w for Synb0 from {anat_input.img.name}")
                    freesurfer.mri_super_synth(
                        in_file=anat_input.img,
                        out_dir=ss_dir,
                        mode=self.synb0_cfg.get(
                            "supersynth_mode",
                            self.config.get("anat.super_synth.mode", "invivo"),
                        ),
                        threads=getattr(self.config, "n_cpus", -1),
                        device=self.synb0_cfg.get(
                            "supersynth_device",
                            self.config.get("anat.super_synth.device"),
                        ),
                        sharpen_synths=bool(self.synb0_cfg.get(
                            "supersynth_sharpen_synths",
                            self.config.get("anat.super_synth.sharpen_synths", False),
                        )),
                        overwrite=force,
                    )
                _validate_nifti(synth_path, self.logger, "SuperSynth T1w")
                context.setdefault("super_synth_outputs", {})["synth_t1w"] = synth_path
                context["synb0_t1w_source"] = "supersynth"
                return _image_file(synth_path)

        if t1w_files:
            context["synb0_t1w_source"] = "t1w"
            return t1w_files[0]
        return None

    def validate_inputs(self, first_arg, output_dir: Path, **kwargs) -> None:
        context, _ = self.unpack_input(first_arg)
        if context is None:
             raise ValidationError("Synb0EstimationStep requires pipeline context.")
        
        has_t1w = bool(context.get("t1w_files"))
        has_t2w = bool(context.get("t2w_files"))
        source = str(self.synb0_cfg.get("t1w_source", "raw")).lower()
        if not has_t1w and not (source in {"supersynth", "prefer_supersynth"} and has_t2w):
            self.logger.warning("Synb0 estimation requires T1w images in context, or T2w with SuperSynth enabled for Synb0.")
        
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
        output_dir = self.get_step_output_dir(output_dir)

        t1w_ref = self._select_anatomical_reference(
            context,
            output_dir,
            force=bool(kwargs.get("force", False)),
        )
        if t1w_ref is None:
             self.logger.warning("Skipping Synb0 estimation: No usable anatomical reference found.")
             return context

        t1w_path = t1w_ref.img
        
        # We need to generate a synthetic b0 for each distinct acquisition group? 
        # Or usually just one per session if they share geometry?
        # Let's assume we do it for the first DWI file and assume others share distortion if valid.
        # Ideally, we should check for different PE directions.
        
        # For simplicity, pick the first valid DWI as the "forward" b0 source.
        input_dwi = None
        for candidate in dwi_files:
            try:
                _validate_nifti(candidate.img, self.logger, "Input DWI")
                _ = nib.load(str(candidate.img))
            except Exception as e:
                self.logger.warning(f"Skipping invalid DWI for Synb0: {candidate.img} ({e})")
                continue
            input_dwi = candidate
            break

        if input_dwi is None:
            raise ProcessingError("No valid DWI files found for Synb0 estimation.")

        real_json = input_dwi.json
        
        should_skip = False
        # 1. Check if outputs exist
        syn_b0_path = output_dir / "syn_b0_desc-synthetic.nii.gz"
        syn_b0_native_path = output_dir / "syn_b0_native.nii.gz"
        syn_json_path = syn_b0_path.with_suffix(".json")
        b0_path = output_dir / "real_b0.nii.gz"
        dummy_bval_path = output_dir / "b0.bval"
        
        if syn_b0_path.exists() and b0_path.exists() and syn_b0_native_path.exists() and syn_json_path.exists() and dummy_bval_path.exists() and not kwargs.get('force', False):
            # Check timestamps
            out_mtime = syn_b0_path.stat().st_mtime
            t1_mtime = t1w_path.stat().st_mtime
            dwi_mtime = input_dwi.img.stat().st_mtime
            
            if t1_mtime > out_mtime or dwi_mtime > out_mtime:
                 self.logger.info(f"Synb0 inputs (T1 or DWI) are newer than output. Re-running.")
            else:
                 self.logger.info(f"Skipping Synb0 estimation (outputs exist and are up-to-date): {syn_b0_path}")
                 should_skip = True
        
        if not should_skip:
            # Extract real b0
            _validate_nifti(input_dwi.img, self.logger, "Input DWI")
            try:
                img = nib.load(str(input_dwi.img))
            except Exception as e:
                size = input_dwi.img.stat().st_size
                raise ProcessingError(
                    f"Failed to read input DWI NIfTI: {input_dwi.img} (size={size} bytes). "
                    f"Original error: {e}"
                )
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
            freesurfer.mri_convert(in_file=t1w_path, out_file=t1w_mgz)
    
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
                return_mask=True,
                use_gpu=getattr(self.config, 'use_gpu', False)
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
                                           extra_args="-searchcost normmi -searchrx -180 180 -searchry -180 180 -searchrz -180 180")
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
                                            extra_args="-searchcost normmi -searchrx -180 180 -searchry -180 180 -searchrz -180 180")
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
            try:
                # Prepare arguments
                gpu_ids = self.config.gpu_ids
                
                self.logger.info(f"Launching Synb0 estimation in a separate process to manage GPU memory...")
                
                # Launch process
                p = multiprocessing.Process(
                    target=_run_synb0_worker,
                    kwargs={
                        'in_file': b0_in_mni,
                        't1_file': t1w_norm_atlas, 
                        'out_file': syn_b0_path,
                        'gpu_ids': gpu_ids
                    }
                )
                p.start()
                p.join()
                
                if p.exitcode != 0:
                    raise ProcessingError(f"Synb0 estimation subprocess failed with exit code {p.exitcode}")
                
            except Exception as e:
                 raise ProcessingError(f"Synb0 estimation execution failed: {e}")
                 
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
