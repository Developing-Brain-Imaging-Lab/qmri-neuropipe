"""
Registration module for generic coregistration.

Supports coregistration of any two images using ANTs, FSL, or FreeSurfer.
"""

from pathlib import Path
from typing import Optional, Literal, Tuple, Dict, Any
import logging
import nibabel as nib

from ...core import BaseProcessingStep, ValidationError, ProcessingError
from ...core.run import run_cmd
from ...core.types import ImageLike, DWIFile, ImageFile
from ...interfaces import ants, fsl, freesurfer, c3d, mrtrix
from ...io.bids import build_bids_name


class NonlinearRegistrationStep(BaseProcessingStep):
    """
    Nonlinear registration (SyN) to a template.
    output: warped image and warp field.
    """
    
    def __init__(self, config, logger=None, provenance=None, template: Optional[Path] = None):
        super().__init__(config, logger, provenance)
        self.template = template # If None, must be in config or passed in run

    def run(self, first_arg, output_dir: Path, template: Optional[Path]=None, **kwargs) -> Any:
        context, input_image = self.unpack_input(first_arg)
        if not input_image:
             raise ValidationError("No input image for nonlinear registration.")

        target = template or self.template or self.config.get("template")
        if not target:
             # Default to MNI?
             # For now require explicit template
             self.logger.warning("No template specified for NonlinearRegistration. Skipping.")
             return context if context else input_image
             
        target = Path(target)
        if not target.exists():
              raise ValidationError(f"Template not found: {target}")

        output_dir = self.get_step_output_dir(output_dir)
        entities = input_image.entities.copy() if hasattr(input_image, 'entities') else {}
        
        # New name
        # MNI or template space?
        # Usually 'space-MNI152NLin2009cAsym' etc.
        # Let's assume generic 'space-template' or just update desc.
        # "anat_proc" usually produces '..._space-MNI..._desc-preproc_T1w.nii.gz'
        
        # We'll rely on config to enforce naming if strict, else generic:
        new_desc = "std" # standardized space
        if 'space' not in entities:
             entities['space'] = 'Standard' 
        
        output_img = output_dir / build_bids_name({**entities, "desc": "norm"})
        output_transform = output_dir / build_bids_name({**entities, "desc": "norm", "suffix": "transform"})
        
        if output_img.exists() and not kwargs.get('force', False):
             self.logger.info(f"Skipping nonlinear registration (exists): {output_img}")
        else:
             in_p = self._extract_path(input_image)
             ants.registration(
                 fixed_file=target,
                 moving_file=in_p,
                 out_prefix=output_transform,
                 transform_type="SyN" # Nonlinear
             )
             
             # ANTs outputs:
             # prefixWarped.nii.gz -> output_img
             # prefix1Warp.nii.gz -> forward warp
             # prefix0GenericAffine.mat -> affine
             
             import shutil
             warped = output_transform.with_suffix("").parent / (output_transform.name + "Warped.nii.gz")
             if warped.exists():
                 shutil.copy(warped, output_img)
             else:
                 self.logger.warning("ANTs SyN completed but warped image not found?")

        result = ImageFile(img=output_img, entities=entities)
        
        if context:
             context["current_image"] = result
             context["template_transform"] = output_transform # Store prefix for applying later?
             return context
        return result


class CoregistrationStep(BaseProcessingStep):
    """
    Coregistration step (Moving <-> Fixed).
    
    Methods:
    - 'ants': Uses ANTs registration (Rigid).
    - 'fsl': Uses FSL FLIRT.
    - 'freesurfer': Uses FreeSurfer bbregister (for B0/T1w scenarios mostly).
    """
    
    def __init__(
        self,
        config,
        logger: Optional[logging.Logger] = None,
        provenance = None,
        method: Literal['ants', 'fsl', 'freesurfer'] = 'ants'
    ):
        super().__init__(config, logger, provenance)
        self.method = method
        self.logger.info(f"Initialized CoregistrationStep with method: {method}")

    def validate_inputs(self, first_arg, **kwargs) -> None:
        pass

    def validate_outputs(self, result) -> None:
        pass

    def run(self, first_arg, output_dir: Path, target: Optional[Path]=None, options: Optional[Dict[str, Any]]=None, **kwargs) -> Any:
        
        # Unpack input
        context, input_image = self.unpack_input(first_arg)
        
        if input_image is None:
             raise ValidationError("No input image provided for coregistration.")

        if hasattr(input_image, 'img'):
            in_path = input_image.img
            # Check if this is DWIFile to preserve type later
            is_dwi = isinstance(input_image, DWIFile)
            entities = input_image.entities.copy() if hasattr(input_image, 'entities') else {}
        else:
            in_path = self._extract_path(input_image)
            is_dwi = False
            entities = {}

        if not in_path or not in_path.exists():
             raise ValidationError(f"Input image not found: {in_path}")

        # Try to infer target from context if not provided
        if not target and context:
             # Heuristic: look for 't1w_file' or 'structural_reference'
             t1w = context.get('t1w_files', [])
             if t1w:
                  target = t1w[0].img
             # Add other lookups here if needed

        if not target:
             raise ProcessingError("CoregistrationStep requires a target image (reference).")

        output_dir = self.get_step_output_dir(output_dir)
        
        options = options or {}
        apply_method = options.get('apply_method', 'native').lower() # 'native' or 'mrtrix'
        
        # Suffix handling
        # Standardize stem cleanup
        input_stem = in_path.name.replace(".nii.gz", "").replace(".nii", "")
        
        new_desc = "coreg"
        # BIDS name generic approach often appends entities to a base or rebuilds.
        # "input_image.entities" is usually safer if available.
        # But if we just want to ensure NO double ext:
        # build_bids_name handles it if we pass clean entities.
        # If build_bids_name uses suffix, it appends ".extension" (usually).
        
        # Let's trust build_bids_name but ensure previous ".nii" isn't lingering in some entity value?
        # Usually entities are clean.
        
        output_img = output_dir / build_bids_name({**entities, "desc": new_desc})
        
        # For transform filename:
        transform_name_full = build_bids_name({**entities, "desc": "coreg", "suffix": "transform"})
        
        output_transform = output_dir / transform_name_full
        if output_transform.suffix: # if it has extension
             output_transform = output_transform.with_suffix("").with_suffix("") # handle .nii.gz
             # If .nii only, one with_suffix is enough.
             # Safe remove:
             while output_transform.suffix:
                  output_transform = output_transform.with_suffix("")
        
        # Skip if exists?
        should_run = True
        if output_img.exists() and not kwargs.get('force', False):
             # Check timestamps: if input is newer than output, we MUST re-run
             in_mtime = in_path.stat().st_mtime
             out_mtime = output_img.stat().st_mtime
             
             # Check dimensions consistency (especially for Outlier Removal re-runs)
             dims_consistent = True
             try:
                 in_shape = nib.load(in_path).shape
                 out_shape = nib.load(output_img).shape
                 # For 4D DWI, check 4th dim. For 3D, check all?
                 # Main issue is num volumes.
                 if len(in_shape) == 4 and len(out_shape) == 4:
                     if in_shape[3] != out_shape[3]:
                         dims_consistent = False
                         self.logger.info(f"Dimension mismatch (In: {in_shape[3]}, Out: {out_shape[3]}). Re-running.")
             except Exception as e:
                 # If read fails, assume incompatible/bad output
                 self.logger.warning(f"Could not check dimensions: {e}. Re-running.")
                 dims_consistent = False

             if not dims_consistent or in_mtime > out_mtime:
                 if in_mtime > out_mtime:
                      self.logger.info(f"Input ({in_path.name}) is newer than output. Re-running coregistration.")
                      self.logger.info(f"Debug: Input mtime={in_mtime}, Output mtime={out_mtime}, Diff={in_mtime-out_mtime:.2f}s")
                 else:
                      self.logger.info("Dimension mismatch (or read error). Re-running coregistration.")
                 should_run = True
             else:
                 self.logger.info(f"Skipping coregistration (output exists and is up-to-date): {output_img}")
                 self.logger.debug(f"Debug: Input mtime={in_mtime}, Output mtime={out_mtime}, In Shape={in_shape}, Out Shape={out_shape}")
                 should_run = False
                 
        # Get nthreads from kwargs or config
        nthreads = kwargs.get('nthreads', self.config.n_cpus)
        
        mrtrix_rotated_bvecs = None # Track if MRTrix handled bvecs
        
        if should_run:
            self.logger.info(f"Running {self.method} coregistration with {nthreads} threads...")
            self.logger.info(f"Application method: {apply_method}")

            # --- Pre-Registration: Ensure Moving Image is 3D ---
            # If input is 4D DWI, we need a 3D reference (b0 or mean b0) for:
            # 1. Defining the native grid for resampling target (if requested)
            # 2. Being the 'moving' image for registration
            moving_for_reg = in_path
            
            if is_dwi:
                 # Extract b0 or mean b0
                 b0_path = output_dir / "temp_b0_ref.nii.gz"
                 # Ensure we don't re-extract if it exists (though run_cmd might handle checks, explicit check is cheap)
                 if not b0_path.exists():
                     run_cmd(f"fslroi {in_path} {b0_path} 0 1", label="extract_b0_ref")
                 moving_for_reg = b0_path

            # --- Output Resolution Handling ---
            # Check for output_resolution option ('anatomical' [default] or 'dwi'/'native')
            # 'native' means we want the result in DWI space.
            # So we must DOWNsample the Target (T1w) to the DWI grid BEFORE registration.
            out_res = options.get('output_resolution', 'anatomical').lower()
            if out_res in ['dwi', 'native']:
                self.logger.info(f"Output resolution set to '{out_res}'. Resampling target (structural) to input (diffusion) grid...")
                
                # Define path for resampled target
                # Avoid double extension if using stem that might have dots
                safe_stem = in_path.name.replace(".nii.gz", "").replace(".nii", "")
                resampled_target = output_dir / f"target_resampled_to_dwi_{safe_stem}.nii.gz"
                
                try:
                    interpolator = options.get("interpolation", "linear").lower() # Normalize case
                    
                    if self.method == 'ants':
                         # ANTsPy expects: linear, nearestNeighbor, bspline...
                         if interpolator == 'nearest': interpolator = 'nearestNeighbor'
                         elif interpolator == 'cubic': interpolator = 'bspline'
                         
                         # Use moving_for_reg (3D) as reference
                         ants.resample_to_image(target, moving_for_reg, resampled_target, interpolator=interpolator, nthreads=nthreads)
                         
                    elif self.method == 'fsl':
                        # FLIRT: trilinear, nearestneighbour, sinc, spline
                        fsl_interp = interpolator
                        if interpolator == 'linear': fsl_interp = 'trilinear'
                        elif interpolator == 'nearest': fsl_interp = 'nearestneighbour'
                        elif interpolator == 'bspline': fsl_interp = 'spline'
                        elif interpolator == 'cubic': fsl_interp = 'spline'
                        
                        fsl.resample_to_image(target, moving_for_reg, resampled_target, interpolator=fsl_interp)
                    else:
                        self.logger.warning(f"Resampling not implemented for method '{self.method}'. Using original target.")
                        resampled_target = target

                    if resampled_target.exists():
                        target = resampled_target
                        self.logger.info(f"Target updated to resampled image: {target}")
                        
                except Exception as e:
                     self.logger.error(f"Failed to resample target: {e}.")
                     # If user explicitly asked for native, we should not fail silently
                     raise ProcessingError(f"Failed to resample target to native resolution: {e}")

            try:
                if self.method == 'ants':
                    # ANTs registration
                    transform_type = options.get("transform_type", "Rigid")
                    interpolator = options.get("interpolation", "linear")
                    
                    # moving_for_reg is already prepared above
                    
                    warped, prefix = ants.registration(
                        fixed_file=target, 
                        moving_file=moving_for_reg, 
                        out_prefix=output_transform, 
                        transform_type=transform_type,
                        interpolator=interpolator,
                        nthreads=nthreads,
                        **{k:v for k,v in options.items() if k not in ['transform_type', 'interpolation', 'apply_method']}
                    )

                    # Now we have the transform. We must apply it to the FULL 4D image.
                    if is_dwi:
                         # Application
                         if apply_method == 'mrtrix':
                             # Identify ANTs Transform (prefix0GenericAffine.mat)
                             # ants.registration returns 'prefix', which is a LIST of transforms usually
                             # e.g. [ ...0GenericAffine.mat ]
                             ants_transform = None
                             for t in prefix:
                                 if str(t).endswith("0GenericAffine.mat"):
                                     ants_transform = Path(t)
                                     break
                             
                             if not ants_transform:
                                 # Fallback guess
                                 ants_transform = Path(str(output_transform) + "0GenericAffine.mat")
                                 
                             if not ants_transform.exists():
                                 raise ProcessingError(f"Could not find ANTs transform for MRTrix application: {ants_transform}")

                             self.logger.info("Applying ANTs transform using MRTrix (with gradient reorientation)...")
                             
                             # MRTrix needs the transform in its format or FSL/ITK?
                             # mrtransform -linear using ITK/ANTs needs checking.
                             # `mrtransform` supports FSL via -linear (import) but ANTs?
                             # Usually we convert ANTs to FSL first if using mrtransform's fsl support,
                             # OR we use `transformconvert itk_import`.
                             # My updated apply_mrtrix_transform supports 'flirt' type (via transformconvert flirt_import).
                             # If ANTs produces ITK format, we might need 'itk_import'.
                             # For now, let's convert ANTs -> FSL, then use MRTrix via FSL mode (safer if transformconvert handles it easily).
                             # Or better: `transformconvert` has `itk_import` operation.
                             
                             # Let's try converting ANTs -> FSL mat using c3d (we already have it) -> then MRTrix.
                             # This is robust because we know how to handle FSL mat in mrtrix wrapper.
                             
                             fsl_mat = output_dir / build_bids_name({**entities, "desc": new_desc}, suffix="fsl_affine", extension=".mat")
                             # We need moving_for_reg as ref for conversion
                             c3d.ants2fsl(target, moving_for_reg, ants_transform, fsl_mat)
                             
                             # Now apply using MRTrix
                             _, mrtrix_rotated_bvecs, _ = mrtrix.apply_mrtrix_transform(
                                 dwi_file=input_image,
                                 out_dwi=output_img,
                                 transform_file=fsl_mat,
                                 transform_type='flirt',
                                 ref_image=target,
                                 interp=interpolator,
                                 nthreads=nthreads,
                                 force=True
                             )
                             
                         else:
                             # Native ANTs Apply
                             self.logger.info(f"Applying transform to full 4D DWI (native/ANTs, interpolation={interpolator})...")
                             ants.apply_transforms(
                                 fixed_file=target,
                                 moving_file=in_path,
                                 out_file=output_img,
                                 transforms=prefix, # Now prefix IS the list of transforms
                                 interpolator=interpolator, 
                                 imagetype=3, # 3 = Time Series (4D)
                                 nthreads=nthreads
                             )
                    else:
                        # Standard 3D copy
                        import shutil
                        if warped and Path(warped).exists():
                             shutil.copy(warped, output_img)
                        else:
                             self.logger.warning("ANTs registration reported success but warped file not found?")
                
                elif self.method == 'fsl':
                    # FSL FLIRT
                    output_mat = output_transform.with_suffix(".mat")
                    dof = options.get("dof", 6)
                    cost = options.get("cost", "normmi")
                    extra_args = options.get("extra_args", "")
                    
                    # --- BBR Handling ---
                    if cost == 'bbr':
                        self.logger.info("BBR cost function requested. Ensuring WM segmentation is available...")
                        try:
                            from ..anat.segmentation import generate_wm_segmentation
                            wm_seg_method = options.get('wm_seg_method', 'fast')
                            
                            # Generate/Get WM segmentation of the target (structural)
                            # We use output_dir for the segmentation artifacts
                            wm_seg = generate_wm_segmentation(
                                target, 
                                output_dir, 
                                method=wm_seg_method, 
                                nthreads=nthreads
                            )
                            
                            # Pass to FSL FLIRT
                            # FLIRT expects -wmseg <path>
                            options['wmseg'] = wm_seg
                            self.logger.info(f"Using WM segmentation for BBR: {wm_seg}")
                            
                        except ImportError:
                            self.logger.error("Could not import segmentation module. Skipping BBR specific setup, which might fail.")
                        except Exception as e:
                            self.logger.error(f"Failed to generate WM segmentation for BBR: {e}. BBR might fail.")
                            raise ProcessingError(f"BBR segmentation failed: {e}")

                    if is_dwi and not moving_for_reg:
                         # Fallback if somehow not set (unlikely)
                         moving_for_reg = in_path

                    
                    # Collect extra options (exclude known args)
                    known_args = ['dof', 'cost', 'extra_args', 'output_resolution', 'interpolation', 'enabled', 'reference_image', 'method', 'wm_seg_method', 'apply_method']
                    fsl_extra_opts = {k: v for k, v in options.items() if k not in known_args}
                    
                    # Calculate transform
                    fsl.flirt(in_file=moving_for_reg, ref_file=target, out_file=output_img, omat=output_mat, dof=dof, cost=cost, extra_args=extra_args, extra_opts=fsl_extra_opts)
                    
                    if is_dwi:
                        if apply_method == 'mrtrix':
                             self.logger.info("Applying FLIRT transform using MRTrix (with gradient reorientation)...")
                             # use output_mat directly
                             interp = options.get("interpolation", "cubic")
                             if interp == 'linear': interp = 'linear' # matching mrtrix names? cubic is default. mrtransform accepts linear/cubic/sinc/nearest.
                             # Check mapping: flirt 'trilinear' -> mrtrix 'linear'?
                             
                             _, mrtrix_rotated_bvecs, _ = mrtrix.apply_mrtrix_transform(
                                 dwi_file=input_image,
                                 out_dwi=output_img,
                                 transform_file=output_mat,
                                 transform_type='flirt',
                                 ref_image=target,
                                 interp=interp,
                                 nthreads=nthreads,
                                 force=True
                             )
                        else:
                            # FLIRT Apply
                            # FLIRT produced a 3D output (resampled B0).
                            # We need to apply the matrix to the full 4D DWI.
                            self.logger.info("Applying FLIRT transform to full 4D DWI...")
                            # flirt -applyxfm -init <mat> -in <4D> -ref <target> -out <4D_out>
                            
                            apply_cmd = f"flirt -applyxfm -init {output_mat} -in {in_path} -ref {target} -out {output_img}"
                            # Note: We overwrite the 3D output from the first flirt call with the 4D output
                            run_cmd(apply_cmd, label="flirt_apply_4d")
                    
                elif self.method == 'freesurfer':
                    # FreeSurfer BBRegister
                    output_dat = output_transform.with_suffix(".dat")
                    freesurfer.bbregister(
                        in_file=in_path, 
                        target_file=target, 
                        out_reg_file=output_dat, 
                        contrast_type="t2"
                    )
                    
                    # NOTE: bbregister does NOT produce a resampled image by default. 
                    # For now we duplicate input to output to allow pipeline continuation,
                    # but log a clear warning.
                    import shutil
                    shutil.copy(in_path, output_img) 
                    self.logger.warning("Freesurfer bbregister only calculates transform. Image not resampled.")

                else:
                     raise ValueError(f"Unknown coregistration method: {self.method}")

            except Exception as e:
                 raise ProcessingError(f"Coregistration failed: {e}", step_name="coregistration") from e

            # Rotation (Run only if coreg ran)
            if is_dwi:
                 # Standard Rigid/Affine registration of dMRI requires bvec rotation.
                 rotated_bvecs = input_image.bvec # Default
                 
                 # If MRTrix was used, it returned the rotated bvecs (embedded in MIF, then exported)
                 # We should use that.
                 if apply_method == 'mrtrix' and mrtrix_rotated_bvecs:
                     self.logger.info("Using b-vectors rotated by MRTrix.")
                     rotated_bvecs = mrtrix_rotated_bvecs
                     
                 else:
                     # Manual Rotation Fallback
                     if self.method == 'fsl':
                         # Rotate bvecs for FSL
                         mat_file = output_transform.with_suffix(".mat")
                         if hasattr(input_image, 'bvec') and input_image.bvec and input_image.bvec.exists() and mat_file.exists():
                             new_bvec_path = output_dir / build_bids_name({**entities, "desc": new_desc}, suffix="bvec", extension=".bvec")
                             try:
                                 self.logger.info("Rotating b-vectors...")
                                 fsl.rotate_bvecs(input_image.bvec, mat_file, new_bvec_path)
                                 rotated_bvecs = new_bvec_path
                             except Exception as e:
                                 self.logger.warning(f"Failed to rotate b-vectors: {e}")
                     
                     elif self.method == 'ants':
                         # ANTs rotation logic
                         # Need to recover or find affine mat.
                         # In 'should_run' block, we have local variables but might be tricky.
                         # Re-find best affine candidate
                         ants_affine = None
                         
                         # Check locals for 'prefix' (transform list)
                         transform_list = locals().get('prefix', [])
                         for t in transform_list:
                             if str(t).endswith(".mat"):
                                 ants_affine = Path(t)
                                 break
                         
                         if not ants_affine:
                              potential_mat = Path(str(output_transform) + "0GenericAffine.mat")
                              if potential_mat.exists():
                                   ants_affine = potential_mat
                         
                         if ants_affine and ants_affine.exists():
                             fsl_mat = output_dir / build_bids_name({**entities, "desc": new_desc}, suffix="fsl_affine", extension=".mat")
                             try:
                                 # C3d ants2fsl needs refs. 
                                 # 'moving_for_reg' is defined in this block above.
                                 ref_moving = locals().get('moving_for_reg')
                                 if not ref_moving:
                                      ref_moving = output_dir / "temp_b0_ref.nii.gz" if is_dwi else in_path
                                 
                                 if ref_moving.exists():
                                     if not fsl_mat.exists():
                                         self.logger.info("Converting ANTs transform to FSL format for bvec rotation...")
                                         c3d.ants2fsl(target, ref_moving, ants_affine, fsl_mat)
                                     
                                     if fsl_mat.exists() and hasattr(input_image, 'bvec') and input_image.bvec and input_image.bvec.exists():
                                         new_bvec_path = output_dir / build_bids_name({**entities, "desc": new_desc}, suffix="bvec", extension=".bvec")
                                         self.logger.info("Rotating b-vectors (ANTs -> FSL)...")
                                         fsl.rotate_bvecs(input_image.bvec, fsl_mat, new_bvec_path)
                                         rotated_bvecs = new_bvec_path

                             except Exception as e:
                                 self.logger.warning(f"Error during ANTs bvec rotation: {e}")

        # --- Result Construction ---
        if is_dwi:
             # Check if we have a rotated bvec from this run
             final_bvec = locals().get('rotated_bvecs')
             
             import shutil
             
             # If not (skipped or failed rotation), look for existing file or use original
             if not final_bvec:
                 bvec_cand = output_img.parent / (output_img.name.split('.')[0] + ".bvec")
                 # Check strict suffix replacement to avoid double extensions if name has dots?
                 # safest is output_img.with_suffix("").with_suffix(".bvec") if .nii.gz 
                 # But we handled double extension logic for output_img construction.
                 # Let's use string replace of extension.
                 base_name = output_img.name
                 for ext in ['.nii.gz', '.nii']:
                     if base_name.endswith(ext):
                         base_name = base_name[:-len(ext)]
                         break
                 
                 bvec_cand = output_dir / (base_name + ".bvec")
                 
                 if bvec_cand.exists():
                     final_bvec = bvec_cand
                 elif input_image.bvec and input_image.bvec.exists():
                     # Copy original bvec if no rotation happened (e.g. rigid with no rotation? Unlikely for coreg)
                     # But if we did coreg, we SHOULD have rotated. 
                     # If we are here, maybe rotation failed or method didn't support it (e.g. freesurfer without manual fallback).
                     # Copying original is better than nothing, but we warn?
                     # Actually generic ANTs/FSL logic above tries to rotate.
                     shutil.copy(input_image.bvec, bvec_cand)
                     final_bvec = bvec_cand
             
             # Ensure BVAL is present (Copy)
             final_bval = None
             if input_image.bval and input_image.bval.exists():
                 base_name = output_img.name
                 for ext in ['.nii.gz', '.nii']:
                     if base_name.endswith(ext):
                         base_name = base_name[:-len(ext)]
                         break
                 bval_path = output_dir / (base_name + ".bval")
                 
                 if not bval_path.exists():
                     shutil.copy(input_image.bval, bval_path)
                 final_bval = bval_path
             
             # Ensure JSON is present (Copy)
             final_json = None
             if input_image.json and input_image.json.exists():
                 base_name = output_img.name
                 for ext in ['.nii.gz', '.nii']:
                     if base_name.endswith(ext):
                         base_name = base_name[:-len(ext)]
                         break
                 json_path = output_dir / (base_name + ".json")
                 
                 if not json_path.exists():
                     shutil.copy(input_image.json, json_path)
                 final_json = json_path

             
             result = DWIFile(img=output_img, bvec=final_bvec, bval=final_bval, entities=entities, json=final_json)
        else:
             result = ImageFile(img=output_img, entities=entities)

        if not output_img.exists():
             raise ProcessingError(f"Coregistration step finished but output not found: {output_img}")

        if context is not None:
             context["current_image"] = result
             
             # Native Reference for GNL (if resampled)
             out_res_chk = options.get('output_resolution', 'anatomical').lower()
             if out_res_chk == 'anatomical' and is_dwi:
                  context['native_dwi_for_gnl'] = input_image
                  
             return context
             
        return result
