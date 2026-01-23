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
from ...core.utils import check_nifti_integrity


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

    def run(self, first_arg, output_dir: Path, target: Optional[Path]=None, options: Optional[Dict[str, Any]]=None, target_modality: str = "T1w", **kwargs) -> Any:
        
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

        target_path = self._extract_path(target)
        if not target_path.exists():
            raise ProcessingError(f"Coregistration target (reference) image not found: {target_path}")

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
        
        # Define standard transform paths for both calculation and application
        output_mat = output_transform.with_suffix(".mat")
        mrtrix_transform = output_dir / "transform_mrtrix.txt"
        reg_lta = output_dir / "bbregister.lta" # for FreeSurfer

        # Determine nthreads
        nthreads = kwargs.get('nthreads', self.config.n_cpus)

        # --- PRE-REGISTRATION: Extract Reference for Calculation/Application ---
        # We extract this even if should_run is False, as it may be needed for mask transform conversion
        moving_for_reg = in_path
        if is_dwi:
            # Modality-specific reference extraction
            if target_modality == "T1w":
                self.logger.info("Target is T1w: Extracting and averaging non-b0 volumes for coregistration reference...")
                avg_dwi_path = output_dir / "temp_avg_dwi_ref.nii.gz"
                if not avg_dwi_path.exists() or kwargs.get('force', False):
                    mrtrix.dwiextract(input_image, avg_dwi_path, no_bzero=True, nthreads=nthreads, force=True)
                    mrtrix.mrmath(avg_dwi_path, "mean", avg_dwi_path, axis=3, nthreads=nthreads, force=True)
                moving_for_reg = avg_dwi_path
            else:
                self.logger.info(f"Target is {target_modality}: Extracting and averaging b0 volumes for coregistration reference...")
                avg_b0_path = output_dir / "temp_avg_b0_ref.nii.gz"
                if not avg_b0_path.exists() or kwargs.get('force', False):
                    try:
                        mrtrix.dwiextract(input_image, avg_b0_path, bzero=True, nthreads=nthreads, force=True)
                        mrtrix.mrmath(avg_b0_path, "mean", avg_b0_path, axis=3, nthreads=nthreads, force=True)
                    except Exception as e:
                        self.logger.warning(f"MRtrix extraction failed: {e}. Falling back to first volume.")
                        run_cmd(f"fslroi {in_path} {avg_b0_path} 0 1", label="extract_first_vol")
                moving_for_reg = avg_b0_path

        # Skip main coregistration if output exists and is valid
        should_run = True
        if output_img.exists() and not kwargs.get('force', False):
             # 0. Check Integrity
             if not check_nifti_integrity(output_img):
                  self.logger.warning(f"Output file corrupted: {output_img}. Re-running.")
                  should_run = True
             else:
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
                     if len(in_shape) != len(out_shape):
                         dims_consistent = False
                         self.logger.info(f"Dimension mismatch (Rank: {len(in_shape)} vs {len(out_shape)}). Re-running.")
                     elif len(in_shape) == 4 and len(out_shape) == 4:
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

            # --- Registration Options Processing ---
            dof = options.get("dof", 6)
            cost = options.get("cost", "normmi")
            
            # Known args to exclude from extra_opts
            known_args = ['dof', 'cost', 'extra_args', 'output_resolution', 'interpolation', 'enabled', 'reference_image', 'method', 'wm_seg_method', 'apply_method']
            fsl_opts = {k: v for k, v in options.items() if k not in known_args}

            # Setup BBR if requested
            if cost == 'bbr':
                try:
                    from ..anat.segmentation import generate_wm_segmentation
                    wm_seg = generate_wm_segmentation(target, output_dir, method=options.get('wm_seg_method', 'fast'), nthreads=nthreads)
                    fsl_opts['wmseg'] = wm_seg
                except Exception as e:
                    self.logger.warning(f"BBR setup failed: {e}. Falling back to default cost function.")
                    cost = "normmi"

            try:
                # --- Logic Branch: Application Method ---
                if apply_method == 'mrtrix':
                    # MRTrix-based Coregistration (handles 4D and gradients correctly)
                    res_val = options.get('output_resolution', 'anatomical').lower()
                    self.logger.info(f"Executing Coregistration via MRtrix (Resolution: {res_val})...")
                    
                    # 1. Calculate Registration
                    transform_file = None
                    transform_type = 'flirt' 
                    
                    if self.method == 'freesurfer':
                         subject_id = context.get('subject') if context else None
                         if not subject_id:
                              raise ProcessingError("FreeSurfer coregistration requires 'subject' in context.")
                         
                         # Define Output Files
                         # reg_lta and output_mat already defined

                         # Determine SUBJECTS_DIR and Subject ID from context logic
                         subjects_dir = None
                         if context and 'freesurfer_dir' in context:
                             # context['freesurfer_dir'] points to the specific subject folder (e.g. sub-01)
                             fs_rec_path = Path(context['freesurfer_dir'])
                             subjects_dir = fs_rec_path.parent
                             # IMPORTANT: Use the folder name as the ID, as FS expects the ID to exist in SUBJECTS_DIR
                             subject_id = fs_rec_path.name
                         elif not context.get('freesurfer_dir'):
                             # Fallback: Construct paths from BIDS structure
                             self.logger.warning("Freesurfer directory not found in context. Attempting to reconstruct from BIDS config.")
                             
                             bids_dir = self.config.get('bids_dir')
                             if bids_dir:
                                 subjects_dir = Path(bids_dir) / 'derivatives' / 'freesurfer'
                                 
                                 # Reconstruct Subject ID (FS format: sub-XX_ses-YY)
                                 # We need sub/ses from context or input_image entities
                                 sub = context.get('subject') if context else None
                                 ses = context.get('session') if context else None
                                 
                                 if not sub and hasattr(input_image, 'entities'):
                                      sub = input_image.entities.get('sub')
                                      if not ses: ses = input_image.entities.get('ses')
                                 
                                 if sub:
                                     candidate_id = f"sub-{sub}"
                                     if ses: candidate_id += f"_ses-{ses}"
                                     
                                     if (subjects_dir / candidate_id).exists():
                                          subject_id = candidate_id
                                          self.logger.info(f"Found FreeSurfer subject directory: {subjects_dir / subject_id}")
                                     else:
                                          self.logger.warning(f"Constructed FreeSurfer path does not exist: {subjects_dir / candidate_id}")
                                 else:
                                      self.logger.warning("Could not determine subject/session for FreeSurfer fallback.")
                             else:
                                  self.logger.warning("No 'bids_dir' in config to reconstruct FreeSurfer paths.")
                         
                         freesurfer.bbregister(
                             in_file=moving_for_reg,
                             target_file=subject_id,
                             out_reg_file=reg_lta,
                             contrast_type='t2',
                             fsl_mat_out=transform_file,
                             subjects_dir=subjects_dir
                         )
                         
                    elif self.method == 'ants':
                         transform_file = output_dir / "coreg_dwi_to_anat_0GenericAffine.mat"
                         out_prefix = output_dir / "coreg_dwi_to_anat_"
                         
                         ants.registration(
                             fixed_file=target,
                             moving_file=moving_for_reg,
                             out_prefix=out_prefix,
                             transform_type='Rigid',
                             nthreads=nthreads
                         )
                         
                         if not transform_file.exists():
                              transform_file = Path(str(out_prefix) + "0GenericAffine.mat")
                         
                         if not transform_file.exists():
                              raise ProcessingError(f"ANTs registration failed to produce transform: {transform_file}")
                          
                         # Convert ANTs -> FSL for consistency
                         fsl_mat = output_dir / "coreg_dwi_to_anat_fsl.mat"
                         c3d.ants2fsl(target, moving_for_reg, transform_file, fsl_mat)
                         transform_file = fsl_mat
                         
                    else:
                         transform_file = output_dir / "coreg_dwi_to_anat.mat"
                         temp_reg_out = output_dir / "temp_flirt_calc.nii.gz"
                         fsl.flirt(
                             in_file=moving_for_reg,
                             ref_file=target,
                             out_file=temp_reg_out,
                             omat=transform_file,
                             dof=dof,
                             cost=cost,
                             extra_opts=fsl_opts
                         )
                         if temp_reg_out.exists(): temp_reg_out.unlink()

                    # 2. Apply via MRtrix
                    temp_mif_in = output_dir / "temp_input.mif"
                    
                    # Ensure Gradients are embedded in the MIF
                    conv_kwargs = {'in_file': in_path, 'out_file': temp_mif_in, 'nthreads': nthreads, 'force': True}
                    if is_dwi:
                         bvec_in = getattr(input_image, 'bvec', None)
                         bval_in = getattr(input_image, 'bval', None)
                         
                         if not bvec_in or not bval_in:
                             # Try sidecars based on filename
                             candidate_bvec = in_path.with_suffix("").with_suffix(".bvec")
                             # Handle double suffix .nii.gz -> .nii -> .bvec if needed, but .with_suffix(".bvec") on .nii.gz yields .nii.bvec usually? 
                             # Path("foo.nii.gz").with_suffix("") is "foo.nii". .with_suffix(".bvec") is "foo.bvec".
                             # But in_path might be .nii.gz.
                             
                             # Reliable way:
                             base_path = str(in_path).split(".nii")[0]
                             candidate_bvec = Path(base_path + ".bvec")
                             candidate_bval = Path(base_path + ".bval")

                             if candidate_bvec.exists() and candidate_bval.exists():
                                 bvec_in = candidate_bvec
                                 bval_in = candidate_bval
                         
                         if bvec_in and bval_in:
                             conv_kwargs['in_bvec'] = bvec_in
                             conv_kwargs['in_bval'] = bval_in
                             
                    mrtrix.mrconvert(**conv_kwargs)
                    
                    mrtrix_transform = output_dir / "transform_mrtrix.txt"
                    mrtrix.transformconvert(
                        in_transform=transform_file,
                        out_mrtrix_transform=mrtrix_transform,
                        operation="flirt_import",
                        ref_image=target, 
                        in_image=moving_for_reg, 
                        force=True
                    )
                    
                    # Apply transform with STRIDES logic
                    temp_mif_out = output_dir / "temp_output.mif"
                    mrtrix_interp = options.get("interpolation", "linear").lower()
                    # Map standard interp names to mrtrix
                    if mrtrix_interp == 'linear': mrtrix_interp = 'linear'
                    elif mrtrix_interp == 'nearest': mrtrix_interp = 'nearest'
                    elif mrtrix_interp == 'sinc': mrtrix_interp = 'sinc'
                    elif mrtrix_interp == 'cubic': mrtrix_interp = 'cubic'

                    mt_kwargs = {
                        'in_file': temp_mif_in,
                        'out_file': temp_mif_out,
                        'linear_transform': mrtrix_transform,
                        'strides': target,
                        'interp': mrtrix_interp,
                        'nthreads': nthreads,
                        'force': True
                    }
                    
                    # If anatomical resolution requested, use target as template for regridding
                    if options.get('output_resolution', 'anatomical').lower() == 'anatomical':
                        mt_kwargs['template'] = target

                    mrtrix.mrtransform(**mt_kwargs)
                    
                    # Export to NIfTI
                    out_bvec = output_img.with_suffix("").with_suffix(".bvec")
                    out_bval = output_img.with_suffix("").with_suffix(".bval")
                    
                    mrtrix.mrconvert(
                        in_file=temp_mif_out,
                        out_file=output_img,
                        export_grad_fsl=(out_bvec, out_bval),
                        nthreads=nthreads,
                        force=True
                    )
                    
                    if temp_mif_in.exists(): temp_mif_in.unlink()
                    if temp_mif_out.exists(): temp_mif_out.unlink()
                    
                    mrtrix_rotated_bvecs = out_bvec

                else:
                    # --- STANDARD LOGIC ---
                    out_res = options.get('output_resolution', 'anatomical').lower()
                    if out_res in ['dwi', 'native']:
                        self.logger.info(f"Output resolution set to '{out_res}'. Resampling target (structural) to input (diffusion) grid...")
                        resampled_target = output_dir / f"target_resampled.nii.gz"
                        
                        interp = options.get("interpolation", "linear").lower()
                        if self.method == 'ants':
                             if interp == 'nearest': interp = 'nearestNeighbor'
                             elif interp == 'cubic': interp = 'bspline'
                             ants.resample_to_image(target, moving_for_reg, resampled_target, interpolator=interp, nthreads=nthreads)
                        elif self.method == 'fsl':
                            fsl_interp = interp
                            if interp == 'linear': fsl_interp = 'trilinear'
                            elif interp == 'nearest': fsl_interp = 'nearestneighbour'
                            fsl.resample_to_image(target, moving_for_reg, resampled_target, interpolator=fsl_interp)
                        else:
                            resampled_target = target
    
                        if resampled_target.exists():
                            target = resampled_target

                    if self.method == 'ants':
                        transform_type = options.get("transform_type", "Rigid")
                        warped, prefix = ants.registration(
                            fixed_file=target, 
                            moving_file=moving_for_reg, 
                            out_prefix=output_transform, 
                            transform_type=transform_type,
                            nthreads=nthreads,
                            **{k:v for k,v in options.items() if k not in ['transform_type', 'interpolation', 'apply_method']}
                        )
    
                        if is_dwi:
                             ants.apply_transforms(
                                 fixed_file=target,
                                 moving_file=in_path,
                                 out_file=output_img,
                                 transforms=prefix, 
                                 interpolator=options.get("interpolation", "linear"), 
                                 imagetype=3,
                                 nthreads=nthreads
                             )
                        elif warped and Path(warped).exists():
                              import shutil
                              shutil.copy(warped, output_img)

                    elif self.method == 'fsl':
                        output_mat = output_transform.with_suffix(".mat")
                        
                        # Ensure we clean up before running to force execution
                        if output_img.exists(): output_img.unlink()
                        if output_mat.exists(): output_mat.unlink()
                        
                        self.logger.info(f"DEBUG: Calling fsl.flirt with in={moving_for_reg}, ref={target}, out={output_img}, cost={cost}, dof={dof}")
                        fsl.flirt(
                            in_file=moving_for_reg, 
                            ref_file=target, 
                            out_file=output_img, 
                            omat=output_mat, 
                            dof=dof, 
                            cost=cost, 
                            extra_opts=fsl_opts
                        )
                        
                        if is_dwi:
                            self.logger.info(f"Applying 4D transform to full DWI series using FSL (interp={options.get('interpolation', 'trilinear')})...")
                            fsl.apply_xfm_4d(
                                in_file=in_path, 
                                ref_file=target, 
                                out_file=output_img, 
                                mat=output_mat,
                                interp=options.get("interpolation", "trilinear")
                            )
                    
                    elif self.method == 'freesurfer':
                        output_dat = output_transform.with_suffix(".dat")
                        freesurfer.bbregister(in_file=in_path, target_file=target, out_reg_file=output_dat, contrast_type="t2")
                        import shutil
                        shutil.copy(in_path, output_img) 
                        self.logger.warning("Freesurfer bbregister only calculates transform. Image not resampled.")

                    else:
                         raise ValueError(f"Unknown coregistration method: {self.method}")

            except Exception as e:
                 raise ProcessingError(f"Coregistration failed: {e}", step_name="coregistration") from e

            # --- Rotation Logic ---
            if is_dwi:
                 rotated_bvecs = input_image.bvec
                 
                 if apply_method == 'mrtrix' and mrtrix_rotated_bvecs:
                     self.logger.info("Using b-vectors rotated by MRTrix.")
                     rotated_bvecs = mrtrix_rotated_bvecs
                 else:
                     if self.method == 'fsl':
                         mat_file = output_transform.with_suffix(".mat")
                         if hasattr(input_image, 'bvec') and input_image.bvec and mat_file.exists():
                             new_bvec_path = output_dir / build_bids_name({**entities, "desc": new_desc}, suffix="bvec", extension=".bvec")
                             try:
                                 fsl.rotate_bvecs(input_image.bvec, mat_file, new_bvec_path)
                                 rotated_bvecs = new_bvec_path
                             except Exception: pass
                     
                     elif self.method == 'ants':
                         # Search for ANTs affine if consistent
                         ants_affine = None
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
                                 ref_moving = locals().get('moving_for_reg')
                                 if not ref_moving: ref_moving = output_dir / "temp_b0_ref.nii.gz" if is_dwi else in_path
                                 
                                 if ref_moving.exists():
                                     if not fsl_mat.exists():
                                         c3d.ants2fsl(target, ref_moving, ants_affine, fsl_mat)
                                     
                                     if fsl_mat.exists() and hasattr(input_image, 'bvec') and input_image.bvec and input_image.bvec.exists():
                                         new_bvec_path = output_dir / build_bids_name({**entities, "desc": new_desc}, suffix="bvec", extension=".bvec")
                                         fsl.rotate_bvecs(input_image.bvec, fsl_mat, new_bvec_path)
                                         rotated_bvecs = new_bvec_path

                             except Exception as e:
                                 self.logger.warning(f"Error during ANTs bvec rotation: {e}")

        # --- Result Construction ---
        if is_dwi:
             final_bvec = locals().get('rotated_bvecs')
             import shutil
             
             if not final_bvec:
                 base_name = output_img.name
                 for ext in ['.nii.gz', '.nii']:
                     if base_name.endswith(ext):
                         base_name = base_name[:-len(ext)]
                         break
                 bvec_cand = output_dir / (base_name + ".bvec")
                 if bvec_cand.exists(): final_bvec = bvec_cand
                 elif input_image.bvec and input_image.bvec.exists():
                      shutil.copy(input_image.bvec, bvec_cand)
                      final_bvec = bvec_cand
             
             final_bval = None
             if input_image.bval and input_image.bval.exists():
                 base_name = output_img.name
                 for ext in ['.nii.gz', '.nii']:
                     if base_name.endswith(ext):
                         base_name = base_name[:-len(ext)]
                         break
                 bval_path = output_dir / (base_name + ".bval")
                 if not bval_path.exists(): shutil.copy(input_image.bval, bval_path)
                 final_bval = bval_path
             
             final_json = None
             if input_image.json and input_image.json.exists():
                 base_name = output_img.name
                 for ext in ['.nii.gz', '.nii']:
                     if base_name.endswith(ext):
                         base_name = base_name[:-len(ext)]
                         break
                 json_path = output_dir / (base_name + ".json")
                 if not json_path.exists(): shutil.copy(input_image.json, json_path)
                 final_json = json_path

             result = DWIFile(img=output_img, bvec=final_bvec, bval=final_bval, entities=entities, json=final_json)
        else:
             result = ImageFile(img=output_img, entities=entities)

        if not output_img.exists():
             raise ProcessingError(f"Coregistration step finished but output not found: {output_img}")
             
        if not check_nifti_integrity(output_img):
             raise ProcessingError(f"Coregistration step finished but output is corrupt/truncated: {output_img}")

        # Dimension Verification
        try:
             chk_img = nib.load(output_img)
             self.logger.info(f"Coregistration output dimensions: {chk_img.shape}")
             if is_dwi and len(chk_img.shape) < 4:
                  self.logger.warning(f"CRITICAL: Coregistration produced a 3D output for a DWI series. DTI/DKI fitting will fail.")
        except Exception as e:
             self.logger.warning(f"Could not verify output dimensions: {e}")

        # --- Mask Handling (Fix for Dimension Mismatch) ---
        if context is not None and context.get("current_mask"):
            mask_input = context["current_mask"]
            mask_in_path = self._extract_path(mask_input)
            
            mask_entities = mask_input.entities.copy() if hasattr(mask_input, 'entities') else {}
            mask_entities['desc'] = new_desc
            mask_out_path_str = str(output_dir / build_bids_name({**mask_entities, "suffix": "mask"}))
            
            if not mask_out_path_str.endswith(".nii.gz") and not mask_out_path_str.endswith(".nii"):
                mask_out_path = Path(mask_out_path_str + ".nii.gz")
            else:
                mask_out_path = Path(mask_out_path_str)

            # Optimization: If output resolution is anatomical, try to find/use the structural mask 
            # instead of resampling the DWI mask.
            is_anatomical = options.get('output_resolution', 'anatomical').lower() == 'anatomical'
            struct_mask = context.get('structural_mask')
            
            if is_anatomical and not struct_mask:
                # Try to find structural mask near target
                target_img_path = Path(target_path)
                parent = target_img_path.parent
                # Pattern match based on target name
                t_stem = target_img_path.name.split(".nii")[0]
                # Common patterns: sub-01_T1w_mask.nii.gz, sub-01_desc-brain_mask.nii.gz
                potential_masks = [
                    parent / (t_stem + "_mask.nii.gz"),
                    parent / (t_stem.replace("_T1w", "_desc-brain_mask")), # sub-01_desc-brain_mask
                    parent / (t_stem.replace("_T1w", "_desc-brain_mask.nii.gz")),
                    parent / (t_stem + ".mask.nii.gz")
                ]
                for pm in potential_masks:
                    if pm.exists():
                        struct_mask = pm
                        self.logger.info(f"Automatically identified structural mask for anatomical space: {pm.name}")
                        break

            if is_anatomical and struct_mask:
                self.logger.info(f"Using structural mask for anatomical space: {Path(struct_mask).name}")
                import shutil
                shutil.copy(struct_mask, mask_out_path)
                mask_should_run = False
            else:
                # Fallback to resampling
                # Heuristic: If image is anatomical (usually >100 slices) and mask is native (usually ~60), trigger resampling
                mask_should_run = should_run
                if not mask_out_path.exists():
                    mask_should_run = True
                elif mask_out_path.exists():
                    try:
                        m_img = nib.load(mask_out_path)
                        # Use explicit target image for comparison if available, or the output image
                        ref_shape = chk_img.shape[:3] if 'chk_img' in locals() else nib.load(output_img).shape[:3]
                        
                        if m_img.shape != ref_shape:
                            self.logger.info(f"Existing mask {mask_out_path.name} shape {m_img.shape} mismatches image {ref_shape}. Re-resampling.")
                            mask_should_run = True
                    except Exception as e:
                        self.logger.warning(f"Could not verify mask dimensions: {e}")
                        mask_should_run = True

            if mask_should_run:
                self.logger.info(f"Applying coregistration transform to mask: {mask_in_path.name}")
                try:
                    # Determine which transform to use
                    # Priority 1: MRTrix transform if applying via MRTrix
                    if apply_method == 'mrtrix':
                        if not mrtrix_transform.exists() and output_mat.exists():
                             self.logger.info("Converting existing FSL transform to MRTrix for mask application...")
                             mrtrix.transformconvert(output_mat, mrtrix_transform, operation="flirt_import", ref_image=target, in_image=moving_for_reg, force=True)
                        
                        if mrtrix_transform.exists():
                            mrtrix.mrtransform(
                                in_file=mask_in_path,
                                out_file=mask_out_path,
                                linear_transform=mrtrix_transform,
                                strides=target,
                                interp='nearest',
                                nthreads=nthreads,
                                force=True
                            )
                        else:
                             self.logger.warning(f"MRTrix transform not found. Falling back to FSL for mask.")
                             if output_mat.exists():
                                 fsl.flirt(in_file=mask_in_path, ref_file=target, out_file=mask_out_path, extra_opts={"applyxfm": True, "init": output_mat, "interp": "nearestneighbour"})
                             
                    elif output_mat.exists():
                        # Default FSL application
                        fsl.flirt(
                            in_file=mask_in_path,
                            ref_file=target,
                            out_file=mask_out_path,
                            extra_opts={
                                "applyxfm": True,
                                "init": output_mat,
                                "interp": "nearestneighbour"
                            }
                        )
                    else:
                        self.logger.warning("Could not identify transform to apply to mask. Mask might be misaligned.")
                except Exception as e:
                    self.logger.warning(f"Failed to apply coregistration to mask: {e}")
            
            if mask_out_path.exists():
                context["current_mask"] = ImageFile(img=mask_out_path, entities=mask_entities)

        if context is not None:
             context["current_image"] = result
             
             # Native Reference for GNL (if resampled)
             out_res_chk = options.get('output_resolution', 'anatomical').lower()
             if out_res_chk == 'anatomical' and is_dwi:
                  context['native_dwi_for_gnl'] = input_image
                  
             return context
             
        return result
