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
from typing import Optional, Literal, Tuple, Union
import numpy as np
import nibabel as nib
import logging

from ...core import BaseProcessingStep, ValidationError, ProcessingError
from ...core.types import ImageFile, DWIFile, ImageLike
from ...interfaces import fsl
from ...io.bids import build_bids_name


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
        

        
        self.logger.info(f"Initialized eddy current correction with method: {method}")
    




        # --- Helper to get an image from either context or direct ImageLike ---
    
    # _extract_image logic replaced by self.unpack_input in base class

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
        
        outputs_exist = output_img.exists()
        if input_img.bvec and not out_bvec.exists():
            outputs_exist = False
            
        if outputs_exist:
             self.logger.info(f"Skipping Eddy correction (Output exists: {output_img.name})")

             result_img = DWIFile(
                entities=input_img.entities,
                img=output_img,
                json=input_img.json,
                bval=input_img.bval,
                bvec=out_bvec if out_bvec.exists() else input_img.bvec # Fallback to input bvec if rotated missing? 
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
                         fsl.bet(in_file=output_img, out_file=possible_mask)
                     
                     if possible_mask.exists():
                         context["current_mask"] = possible_mask
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
        if extra_opts is None:
            extra_opts = {}

        # Merge with config options
        # Retrieve 'dmri.preprocessing.eddy' config if available
        dmri_cfg = self.config.get('dmri', {}).get('preprocessing', {}).get('eddy', {})
        # Assuming extra options might be top level keys in 'eddy' config or nested in 'options'?
        # Let's assume users might put other flags in the 'eddy' dict that are not 'enabled' or 'method'.
        # Or specifically look for an 'options' key.
        config_opts = dmri_cfg.get('options', {})
        if config_opts:
             extra_opts.update(config_opts)

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
                     self.logger.info("No mask provided for eddy. Generating temporary BET mask...")
                     mask_path = output_dir / "eddy_mask.nii.gz"
                     # Use fsl.bet
                     # Note: fsl.bet expects ImageLike or Path
                     fsl.bet(in_file=input_img, out_file=mask_path)
                     mask = mask_path
                     
                     # Persist for QC
                     context["current_mask"] = mask
        
        # Fetch nthreads
        nthreads = kwargs.get('nthreads', self.config.n_cpus)
        
        try:
            if self.method == 'eddy-correct':
                ecc = fsl.eddy_correct(in_file=input_img, 
                                       out_file=output_img
                                       )
            elif self.method == 'eddy':
                # Determine GPU settings
                cuda_enabled = self.config.use_gpu
                cuda_device = 0
                if self.config.gpu_ids is not None:
                     cuda_enabled = True
                     gpus = self.config.gpu_ids
                     if isinstance(gpus, int):
                         gpus = [gpus]
                     cuda_device = gpus[0]
                
                ecc = fsl.eddy(
                    in_file=input_img,
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
                raise NotImplementedError("Two-pass eddy correction is not yet implemented.")

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
            # Ensure mask is updated in return context if we generated it
            if mask and "current_mask" not in context:
                context["current_mask"] = mask

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
        )
    


__version__ = '2.0.0'
__all__ = [
    'EddyCorrectionStep'
]
