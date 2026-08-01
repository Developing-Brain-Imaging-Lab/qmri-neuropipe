"""
Eddy Current correction module for diffusion MRI data.

This module provides eddy correction methods that work across dMRI data:
- dMRI: MP-PCA (Marchenko-Pastur PCA)

All eddy steps inherit from BaseProcessingStep for automatic
validation, logging, and provenance tracking.

Classes:
    EddyCorrectionStep: Main eddy current class with multiple methods
    eddy_current_correction: Convenience function for quick eddy current correction without full pipeline setup.
"""

from pathlib import Path
from typing import Optional, Literal, Tuple
import numpy as np
import nibabel as nib
import logging

from ...core import BaseProcessingStep, ValidationError, ProcessingError
from ...core.types import ImageFile, DWIFile, ImageLike
from ...interfaces import fsl
from ...io.bids import build_bids_name
from ..common.spatial_transforms import write_transform_chain_to_sidecar
from .b0_reference import select_optimal_b0


class EddyCorrectionStep(BaseProcessingStep):
    """
    General eddy current correction step for dMRI.
    
    Supports multiple eddy correction methods:
    - 'eddy-correct': FSL eddy-correct 
    - 'eddy': FSL eddy
    - 'two-pass': Combined eddy-correct and eddy correction for large motions
    
    Used by:
    - dMRI
    
    Attributes:
        method: Eddy correction method to use
        
    Example:
        >>> # For dMRI
        >>> corrected = EddyCorrectionStep(config, method='eddy')
        >>> corrected = eddy_current_correction(dwi_file, output_dir)
    """
    
    def __init__(
        self,
        config,
        logger: Optional[logging.Logger] = None,
        provenance = None,
        method: Literal['eddy-correct', 'eddy', 'two-pass'] = 'eddy',
        mask_dilation: int = 3, # Default 3 passes for eddy to ensure coverage
    ):
        """
        Initialize eddy current correction step.
        
        Args:
            config: Pipeline configuration
            method: Denoising method to use
            logger: Optional logger instance
            provenance: Optional provenance tracker
        
        Raises:
            ProcessingError: If required dependencies are missing
        """
        super().__init__(config, logger, provenance)
        
        self.method = method
        self.mask_dilation = mask_dilation
        
        self.logger.info(f"Initialized eddy current correction with method: {method}")

    # _unpack_input in base class handles context vs direct ImageLike

    def _validate_image(self, image: ImageLike) -> None:
        """Your existing validate_inputs logic, moved here."""
        if not image.img.exists():
            raise ValidationError(f"Input image not found: {image.img}")
        
        # Check it's a valid NIfTI file
        try:
            img = nib.load(str(image.img))
            # Eddy/Eddy-correct requires 4D
            if len(img.shape) != 4:
                raise ValidationError(
                    f"Eddy current correction requires 4D data, got shape {img.shape}"
                )
        except Exception as e:
            raise ValidationError(f"Invalid NIfTI file: {e}")

        if not image.img.suffix.endswith(".nii") and not "".join(image.img.suffixes).endswith(".nii.gz"):
            raise ValidationError(f"Input must be NIfTI, got {image.img}")
        
        self.logger.debug(f"Input validation passed for {image.img}")

    def _validate_result_image(self, result: ImageLike) -> None:
        """Your existing validate_outputs logic, moved here."""
        if not result.img.exists():
            raise ProcessingError(f"Eddy corrected image not created: {result.img}")
        
        try:
            img = nib.load(str(result.img))
            data = img.get_fdata()

            if np.any(np.isnan(data)):
                raise ProcessingError("Eddy current corrected image contains NaN values")
            if np.any(np.isinf(data)):
                raise ProcessingError("Eddy current corrected image contains Inf values")
            if np.any(data < 0):
                self.logger.warning("Eddy current corrected image contains negative values")

            self.logger.debug(
                f"Output validation passed. "
                f"Data range: [{data.min():.2f}, {data.max():.2f}]"
            )
        except Exception as e:
            raise ProcessingError(f"Output validation failed: {e}")

    def validate_inputs(self, first_arg, output_dir: Path, **kwargs) -> None:
        """
        Validate denoising inputs.
        """
        context, image = self.unpack_input(first_arg)
        if image is None:
             raise ValidationError("Input image is None (or not found in context)")
        self._validate_image(image)

    def validate_outputs(self, result) -> None:
        """
        Validate denoising outputs.
        
        Args:
            result: ImageLike object of denoised image
        
        Raises:
            ProcessingError: If output validation fails
        """

        if isinstance(result, dict):
            image = result.get("current_image")
            if image is None:
                raise ProcessingError(
                    "DenoisingStep expected 'current_image' in context result"
                )
        else:
            image = result

        self._validate_result_image(image)
    

    def run(self, first_arg, output_dir: Path, mask: Optional[Path]=None, **kwargs) -> Tuple[Path, Optional[Path]]:
        """
        Run eddy-current correction on input image.
        """
        context, input_img = self.unpack_input(first_arg)
        if input_img is None:
             raise ProcessingError("No input image provided")

        output_dir = self.get_step_output_dir(output_dir)
        output_img = output_dir / build_bids_name({**input_img.entities, "desc": "eddycorrected"})
        
        # Check if output exists (User request: skip if exists)
        # We must also ensure sidecars exist if they are expected (bvecs)
        out_base = output_img.with_suffix("").with_suffix("")
        out_bvec = out_base.with_suffix(".bvec")
        
        outputs_exist = self.check_output_validity(output_img)
        if input_img.bvec and not self.check_output_validity(out_bvec, min_size=5):
            outputs_exist = False
            
        if outputs_exist and not kwargs.get('force', False):
             self.logger.info(f"Skipping Eddy correction (Output exists: {output_img.name})")

             result_img = DWIFile(
                entities=input_img.entities,
                img=output_img,
                json=input_img.json,
                bval=input_img.bval,
                bvec=out_bvec if out_bvec.exists() else input_img.bvec, # Fallback to input bvec if rotated missing?
                Delta=getattr(input_img, "Delta", None),
                delta=getattr(input_img, "delta", None),
             )
             
             if context is not None:
                context["current_image"] = result_img
                
                # Ensure mask is available for downstream steps (e.g., QC)
                if "current_mask" not in context or not context["current_mask"]:
                     # Look for expected mask
                     possible_mask = output_dir / "eddy_mask.nii.gz"
                     
                     if not possible_mask.exists():
                         # If mask doesn't exist (legacy run), generate it!
                         self.logger.info("Output exists but mask missing. Generating temporary BET mask for QC...")
                         # We use the corrected output for the mask
                         fsl.bet(in_file=output_img, out_file=possible_mask, frac=0.1)
                         
                         # Dilate/Binarize for consistency
                         fsl.maths(possible_mask, possible_mask, args="-dilM -dilM -dilM -fillh -bin -fillh")
                      
                     if possible_mask.exists():
                         context["current_mask"] = ImageFile(img=possible_mask, entities=dict(result_img.entities, suffix="mask"))
                         self.logger.debug(f"Recovered/Generated mask: {possible_mask}")

                # Optional: maintain preprocessed list
                if isinstance(result_img, DWIFile):
                    pre_list = context.setdefault("preprocessed_dwis", [])
                    if result_img not in pre_list:
                        pre_list.append(result_img)
                return context
             else:
                return result_img

        # Run denoising based on method
        self.logger.info(f"Running {self.method} eddy current correction...")
        topup_base = kwargs.pop("topup_base", None)
        acqp = kwargs.pop("acqp", None)
        index = kwargs.pop("index", None)
        extra_opts = kwargs.pop("extra_opts", {})
        extra_opts = dict(extra_opts or {})

        # Merge with config options
        # Retrieve 'dmri.preprocessing.eddy' config if available
        dmri_cfg = self.config.get('dmri', {}).get('preprocessing', {}).get('eddy', {})
        # Assuming extra options might be top level keys in 'eddy' config or nested in 'options'?
        # Let's assume users might put other flags in the 'eddy' dict that are not 'enabled' or 'method'.
        # Or specifically look for an 'options' key.
        config_opts = dmri_cfg.get('options', {})
        if config_opts:
             extra_opts.update(config_opts)

        # FSL eddy supports an explicit zero-based reference scan.  Select it
        # with the shared TORTOISE-like b0 selector when requested, while
        # preserving eddy's historical behavior by default.
        motion_cfg = self.config.get('dmri', {}).get('preprocessing', {}).get(
            'motion_correction', {}
        ) or {}
        reference_cfg = (
            motion_cfg.get('reference_selection')
            or dmri_cfg.get('reference_selection')
            or {}
        )
        b0_selection = None
        if self.method in {'eddy', 'two-pass'} and reference_cfg.get('enabled', False):
            b0_selection = select_optimal_b0(
                input_img,
                output_dir / 'b0_reference',
                threshold=float(reference_cfg.get('b0_threshold', 50.0)),
                local_radius=int(reference_cfg.get('local_radius', 3)),
                force=bool(kwargs.get('force', False)),
            )
            extra_opts.setdefault('ref_scan_no', b0_selection.index)
            self.logger.info(
                "Selected volume %d as FSL eddy motion reference (paired with %d, score %.4f)",
                b0_selection.index,
                b0_selection.paired_index,
                b0_selection.score,
            )
            if context is not None:
                context['b0_reference_selection'] = b0_selection
                context['motion_reference'] = b0_selection.pair_average_image

        if context is not None:
            # Retrieve topup_base from map if available
            topup_map = context.get("topup_map", {})
            # Try exact match or string match
            if input_img.img in topup_map:
                topup_base = topup_map[input_img.img]
            elif str(input_img.img) in topup_map:
                topup_base = topup_map[str(input_img.img)]
            else:
                # Fallback
                topup_base = context.get("topup_base", topup_base)

            acqp = context.get("acqp", acqp)
            index = context.get("index", index)
            
            # Ensure mask exists for eddy and subsequent QC
            if not mask:
                mask = context.get("current_mask")
                if mask and hasattr(mask, 'img'):
                     mask = mask.img
                
                if not mask:
                     self.logger.info("No mask provided for eddy. Generating temporary robust BET mask...")
                     mask_path = output_dir / "eddy_mask.nii.gz"
                     temp_ref = output_dir / f"temp_eddy_ref_{input_img.img.name}"
                     
                     bet_input = input_img.img
                     # Check if 4D
                     try:
                         hdr = nib.load(str(input_img.img)).header
                         if len(hdr.get_data_shape()) > 3 and hdr.get_data_shape()[3] > 1:
                             # Calculate mean image over series for robust mask reference
                             from ...core.run import run_cmd
                             run_cmd(f"fslmaths {input_img.img} -Tmean {temp_ref}", label="calculate_mean_ref_for_eddy_mask")
                             bet_input = temp_ref
                     except Exception as e:
                         self.logger.warning(f"Failed to check image dimensions or calculate mean: {e}")

                     # Use fsl.bet
                     fsl.bet(in_file=bet_input, out_file=mask_path, frac=0.1)
                     
                     # Cleanup temp ref
                     if bet_input == temp_ref and temp_ref.exists():
                         try: temp_ref.unlink()
                         except: pass
                     
                     # Dilate and Binarize (Improve coverage)
                     # Use configured mask_dilation (default 3)
                     dilation = kwargs.get('mask_dilation', self.mask_dilation)
                     fsl_args = ""
                     if dilation > 0:
                         fsl_args += (" -dilM" * dilation)
                     
                     fsl_args += " -fillh -bin -fillh"
                     fsl.maths(mask_path, mask_path, args=fsl_args)
                     
                     mask = mask_path
                     
                     # Persist for QC
                     context["current_mask"] = ImageFile(img=mask, entities=dict(input_img.entities, suffix="mask"))
        
        # Fetch nthreads
        nthreads = kwargs.get('nthreads', self.config.n_cpus)
        
        # --- Memory Optimization: Int16 Conversion ---
        # Check config or extra_opts for 'datatype' or 'convert_to_int16'
        # eddy extra_opts doesn't usually allow 'datatype', so we look for custom keys or config
        
        target_datatype = kwargs.get('datatype') or extra_opts.get('datatype')
        # If user put datatype in extra_opts, remove it so it's not passed to fsl.eddy (which rejects it)
        if 'datatype' in extra_opts:
             del extra_opts['datatype']
             
        # Check for optimize_memory in extra_opts (user flexibility)
        if extra_opts.get('optimize_memory'):
             target_datatype = 'int16'
             del extra_opts['optimize_memory']
             
        # Also check explicit config flag if we add one later, e.g. 'optimize_memory'
        if not target_datatype and self.config.get('dmri', {}).get('preprocessing', {}).get('eddy', {}).get('optimize_memory', False):
             target_datatype = 'int16'

        current_input_img = input_img 
        if target_datatype == 'int16':
             self.logger.info("Reducing memory usage: Converting input to int16 before running Eddy...")
             from ...interfaces.mrtrix import mrconvert
             
             temp_int16 = output_dir / f"temp_int16_{input_img.img.name}"
             
             try:
                 mrconvert(
                     in_file=input_img,
                     out_file=temp_int16,
                     datatype="int16",
                     nthreads=nthreads
                 )
                 
                 if temp_int16.exists():
                     # Create a temporary DWIFile pointing to this int16 image
                     # preserving gradients
                     current_input_img = DWIFile(
                         entities=input_img.entities,
                         img=temp_int16,
                         json=input_img.json,
                         bval=input_img.bval,
                         bvec=input_img.bvec,
                         Delta=getattr(input_img, "Delta", None),
                         delta=getattr(input_img, "delta", None),
                     )
                 else:
                     self.logger.warning("Int16 conversion failed (output missing). Using original float input.")
                     
             except Exception as e:
                 self.logger.warning(f"Int16 conversion failed: {e}. Using original float input.")

        # Resolve GPU settings once — shared by 'eddy' and 'two-pass'
        cuda_enabled = self.config.use_gpu
        cuda_device = 0
        if self.config.gpu_ids is not None:
            cuda_enabled = True
            gpus = self.config.gpu_ids
            if isinstance(gpus, int):
                gpus = [gpus]
            cuda_device = gpus[0]

        # Subset of kwargs that fsl.eddy accepts as dedicated params
        _EDDY_PASSTHROUGH = ('force', 'json_file', 'external_field')

        try:
            if self.method == 'eddy-correct':
                ecc = fsl.eddy_correct(in_file=input_img,
                                       out_file=output_img
                                       )
            elif self.method == 'eddy':
                ecc = fsl.eddy(
                    in_file=current_input_img,
                    out_file=output_img,
                    mask=mask,
                    topup_base=topup_base,
                    acqp=acqp,
                    index=index,
                    extra_opts=extra_opts,
                    nthreads=nthreads,
                    cuda=cuda_enabled,
                    cuda_device=cuda_device,
                    **kwargs,
                )
            elif self.method == 'two-pass':
                # --- Pass 1: FSL eddy (model-based, GPU-aware) ---
                pass1_out = output_dir / build_bids_name(
                    {**input_img.entities, "desc": "eddypass1"}
                )
                self.logger.info("Two-pass eddy: running Pass 1 (fsl.eddy)...")
                pass1_result = fsl.eddy(
                    in_file=current_input_img,
                    out_file=pass1_out,
                    mask=mask,
                    topup_base=topup_base,
                    acqp=acqp,
                    index=index,
                    extra_opts=dict(extra_opts),   # copy — eddy mutates the dict in-place
                    nthreads=nthreads,
                    cuda=cuda_enabled,
                    cuda_device=cuda_device,
                    **{k: v for k, v in kwargs.items() if k in _EDDY_PASSTHROUGH},
                )

                # --- Pass 2: FSL eddy_correct (affine refinement) ---
                self.logger.info("Two-pass eddy: running Pass 2 (fsl.eddy_correct)...")
                ecc = fsl.eddy_correct(
                    in_file=pass1_result,
                    out_file=output_img,
                )

            else:
                raise ValueError(f"Unknown eddy current correction method: {self.method}")
            
        except Exception as e:
            raise ProcessingError(
                f"Eddy current correction failed with method {self.method}",
                step_name="eddy-current-correction",
                details=str(e)
            )
        
        # Save denoised image
        self.logger.info(f"Eddy current corrected image saved to: {output_img}")
               
        # ecc is a DWIFile
        result_img = ecc

        
        # ---- Return shape depends on input shape ----
        if context is not None:
            context["current_image"] = result_img
            spatial_transform = {
                "type": "motion_correction",
                "method": self.method,
                "usable_for_gnl_mapping": False,
                "notes": "Voxelwise motion/distortion correction applied; no explicit transform chain is currently serialized.",
            }
            context["spatial_transform"] = spatial_transform
            write_transform_chain_to_sidecar(getattr(result_img, "json", None), [spatial_transform])
            setattr(result_img, "spatial_transform", spatial_transform)
            # Ensure mask is updated in return context if we generated it
            if mask and "current_mask" not in context:
                context["current_mask"] = ImageFile(img=Path(mask), entities=dict(result_img.entities, suffix="mask"))

            # If this is DWI, you might want to maintain a list of preprocessed DWIs:
            if isinstance(result_img, DWIFile):
                pre_list = context.setdefault("preprocessed_dwis", [])
                if result_img not in pre_list:
                    pre_list.append(result_img)

            return context
        else:
            # Standalone behavior unchanged
            return result_img
        
       
    def _log_provenance(self, first_arg, output_dir: Path, result, **kwargs) -> None:
        if not self.provenance:
            return

        # Case 1: context dict
        if isinstance(first_arg, dict):
            context = first_arg
            in_dwis = context.get("dwi_files", [])
            out_dwis = result.get("dwi_files", result.get("preprocessed_dwis", []))
            inputs = {
                "subject": context.get("subject"),
                "session": context.get("session"),
                "input_dwis": [str(d.img) for d in in_dwis],
            }
            outputs = {
                "output_dwis": [str(d.img) for d in out_dwis],
            }

        # Case 2: ImageLike object
        else:
            image = first_arg
            inputs = {"image": str(image.img)}
            outputs = {"eddy_current_corrected_image": str(result.img)}

        parameters = {
            "method": self.method,
            "mask_dilation": self.mask_dilation,
            **kwargs,
        }

        self.provenance.log_step(
            step_name="eddy_current_correction",
            inputs=inputs,
            outputs=outputs,
            parameters=parameters,
            duration=(
                (self.end_time - self.start_time).total_seconds()
                if self.end_time and self.start_time
                else None
            ),
            commands=getattr(self, "last_commands", []),
        )
    


__version__ = '2.0.0'
__all__ = [
    'EddyCorrectionStep'
]
