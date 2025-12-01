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
        
        # Check dependencies
        self._check_dependencies()
        
        self.logger.info(f"Initialized eddy current correction with method: {method}")
    
    def _check_dependencies(self) -> None:
        """Check that required dependencies are available."""
        if self.method == 'eddy':
            # Assume MRtrix3 is installed and in PATH
            pass

        # if self.method == 'mppca' and not DIPY_AVAILABLE:
        #     raise ProcessingError(
        #         "DIPY not available. Install with: pip install dipy"
        #     )
        
        # if self.method == 'nlmeans' and not NLMEANS_AVAILABLE:
        #     raise ProcessingError(
        #         "DIPY nlmeans not available. Install with: pip install dipy"
        #     )

        # --- Helper to get an image from either context or direct ImageLike ---
    
    def _extract_image(self, first_arg) -> ImageLike:
        """
        Support both:
        - pipeline mode: first_arg is context dict
        - standalone mode: first_arg is an ImageLike
        """
        if isinstance(first_arg, dict):
            ctx = first_arg
            img = ctx.get("current_image")
            if img is None:
                raise ValidationError(
                    "DenoisingStep expects context['current_image'] to be set"
                )
            return img
        else:
            # Old behavior: first_arg is already an ImageLike
            return first_arg

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
        
        Args:
            input_image: Image Object
            output_dir: Output directory
            **kwargs: Additional arguments
        
        Raises:
            ValidationError: If inputs are invalid
        """

        """
        first_arg can be:
        - context dict (pipeline mode)
        - ImageLike (standalone denoise_image convenience)
        """
        image = self._extract_image(first_arg)
        self._validate_image(image)

    def validate_outputs(self, result) -> None:
        """
        Validate denoising outputs.
        
        Args:
            result: ImageLike object of denoised image
        
        Raises:
            ProcessingError: If output validation fails
        """
        """
        result can be:
        - context dict (pipeline mode) with context['current_image']
        - ImageLike (standalone mode)
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
        
        Args:
            input_img: ImageLike object to input NIfTI file
            output_dir: Path to output NIfTI file for denoised image
            mask: Optional brain mask (speeds up processing)
            **kwargs: Method-specific parameters
        
        Returns:
            Path to eddy-corrected image
            Path to eddy-corrected bvecs
        
        Raises:
            ProcessingError: If eddy correction fails
        """

        """
        Pipeline mode:
            first_arg: dict with 'current_image' (ImageLike)
            returns:   updated dict with 'current_image' replaced by eddy corrected ImageLike,
                       and optionally appends to 'preprocessed_dwis'
        
        Standalone mode (denoise_image helper):
            first_arg: ImageLike
            returns:   ImageLike
        """
        is_context = isinstance(first_arg, dict)
        if is_context:
            context = dict(first_arg)  # shallow copy to avoid in-place surprises
            input = self._extract_image(context)
        else:
            context = None
            input = first_arg


        output_dir.mkdir(parents=True, exist_ok=True)
        output_img = output_dir / build_bids_name({**input.entities, "desc": "eddycorrected"})
        
        # Run denoising based on method
        self.logger.info(f"Running {self.method} eddy current correction...")
        
        try:
            if self.method == 'eddy-correct':
                ecc = fsl.eddy_correct(in_dwi=input,
                                       out=output_img)
            elif self.method == 'eddy':
                ecc = fsl.eddy(in_dwi=input,
                               out=output_img,
                               mask=mask,
                               **kwargs)         
            elif self.method == 'two-pass':
                #Need to implement
                pass

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
               
        #Result image is a DWIFile
        result_img = DWIFile(entities=input.entities,
                             img=ecc,
                             json=input.json,
                             bval=ecc.bval,
                             bvec=ecc.bvec)

        
        # ---- Return shape depends on input shape ----
        if is_context:
            context["current_image"] = result_img

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
            step_name="denoising",
            inputs=inputs,
            outputs=outputs,
            parameters=parameters,
            duration=(
                (self.end_time - self.start_time).total_seconds()
                if self.end_time and self.start_time
                else None
            ),
        )
    

# Convenience function for quick denoising
def eddy_current_correction(input: ImageLike, output: ImageLike, method: str = 'mrtrix', mask: Optional[Path] = None, **kwargs) -> Path:
    """
    Convenience function for quick denoising without full pipeline setup.
    
    Args:
        input: Input image path
        output: Output image path
        method: Denoising method ('mrtrix', 'ants', 'mppca', 'nlmeans', etc.)
        mask: Optional mask path
        **kwargs: Method-specific parameters
    
    Returns:
        Path to denoised image
    
    Example:
        >>> from qmri_neuropipe.processing.common.denoising import denoise_image
        >>> 
        >>> denoised = denoise_image(
        ...     input=Path('dwi.nii.gz'),
        ...     output=Path('denoised.nii.gz'),
        ...     method='mppca'
        ... )
    """
    # Create minimal config
    from qmri_neuropipe.core import PipelineConfig
    
    config = PipelineConfig(
        bids_dir=Path('/tmp'),  # Dummy value
        output_dir=output.img.parent
    )
    
    # # Run denoising
    # denoiser = DenoisingStep(config, method=method)
    
    # # Temporarily disable validation for standalone use
    # denoiser.validate_inputs = lambda *args, **kwargs: None
    
    # result = denoiser.run(
    #     input,
    #     output.img.parent,
    #     mask=mask,
    #     **kwargs
    # )
    
    # return result


# Module version and metadata
__version__ = '2.0.0'
__all__ = [
    'DenoisingStep',
    'denoise_image'
]
