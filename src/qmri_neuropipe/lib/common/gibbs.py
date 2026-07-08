"""
Gibbs ringing correction module for neuroimaging data.

This module provides denoising methods that work across multiple modalities:
- dMRI: MP-PCA (Marchenko-Pastur PCA)
- fMRI: Non-local means, wavelets
- Anatomical: Non-local means, anisotropic diffusion

All gibbs unringing steps inherit from BaseProcessingStep for automatic
validation, logging, and provenance tracking.

Classes:
    GibbsUnringingStep: Main gibbs correction class with multiple methods

"""

from pathlib import Path
from typing import Optional, Literal, Tuple, Dict, Any
import numpy as np
import nibabel as nib
import logging

from ...core import BaseProcessingStep, ValidationError, ProcessingError
from ...core.types import ImageFile, DWIFile, ImageLike
from ...interfaces import dipy, ants, mrtrix
from ...io.bids import build_bids_name
from .json_metadata import copy_json_with_metadata

# Try to import optional dependencies
try:
    from dipy.denoise.gibbs import gibbs_removal
    DIPY_AVAILABLE = True
except ImportError:
    DIPY_AVAILABLE = False


class GibbsUnringingStep(BaseProcessingStep):
    """
    General denoising step that works for multiple modalities.
    
    Supports multiple unringing methods:
    - 'dipy': DIPY Implementation
    - 'mrtrix': MRtrix3 Implementation 
    
    Used by:
    - dMRI: Typically uses MP-PCA
    - fMRI: Typically uses non-local means or wavelets
    - Anatomical: Typically uses non-local means
    
    Attributes:
        method: Unringing method to use
        
    Example:
        >>> # For dMRI
        >>> unringing = GibbsUnringingStep(config, method='mrtrix')
        >>> corrected = gibss_unringing(dwi_file, output_dir)
        >>> 
        >>> # For fMRI
        >>> unringing = GibbsUnringingStep(config, method='dipy')
        >>> corrected = gibss_unringing(bold_file, output_dir)
    """
    
    def __init__(
        self,
        config,
        logger: Optional[logging.Logger] = None,
        provenance = None,
        method: Literal['dipy', 'mrtrix'] = 'dipy',
    ):
        """
        Initialize gibbs unringing step.
        
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
        
        self.logger.info(f"Initialized gibbs unringing with method: {method}")
    
    def _check_dependencies(self) -> None:
        """Check that required dependencies are available."""
        if self.method == 'dipy' and not DIPY_AVAILABLE:
            raise ProcessingError(
                "DIPY not available. Install with: pip install dipy"
            )
        
    # _extract_image logic replaced by self.unpack_input in base class

    def _validate_image(self, image: ImageLike) -> None:
        """Your existing validate_inputs logic, moved here."""
        if not image.img.exists():
            raise ValidationError(f"Input image not found: {image.img}")
        
        # Check it's a valid NIfTI file
        try:
            img = nib.load(str(image.img))
            
            # Check dimensions
            if len(img.shape) < 3:
                raise ValidationError(
                    f"Input must be at least 3D, got shape {img.shape}"
                )
            
        except Exception as e:
            raise ValidationError(f"Invalid NIfTI file: {e}")


        if not image.img.suffix.endswith(".nii") and not "".join(image.img.suffixes).endswith(".nii.gz"):
            raise ValidationError(f"Input must be NIfTI, got {image.img}")
        
        self.logger.debug(f"Input validation passed for {image.img}")

    def _validate_result_image(self, result: ImageLike) -> None:
        """Your existing validate_outputs logic, moved here."""
        if not result.img.exists():
            raise ProcessingError(f"Gibbs corrected image not created: {result.img}")
        
        # Check output is valid and has reasonable values
        try:
            img = nib.load(str(result.img))
            data = img.get_fdata()
            
            # Check for NaN or Inf
            if np.any(np.isnan(data)):
                raise ProcessingError("Gibbs corrected image contains NaN values")
            if np.any(np.isinf(data)):
                raise ProcessingError("Gibbs corrected image contains Inf values")
            
            # Check data range is reasonable (non-negative for MRI)
            if np.any(data < 0):
                self.logger.warning("Gibbs corrected image contains negative values")
            
            self.logger.debug(
                f"Output validation passed. "
                f"Data range: [{data.min():.2f}, {data.max():.2f}]"
            )
            
        except Exception as e:
            raise ProcessingError(f"Output validation failed: {e}")
    
    def validate_inputs(self, first_arg, **kwargs) -> None:
        """
        Validate gibbs ringing inputs.
        """
        context, image = self.unpack_input(first_arg)
        if image is None:
             raise ValidationError("Input image is None (or not found in context)")
        self._validate_image(image)

    def validate_outputs(self, result: ImageLike) -> None:
        """
        Validate denoising outputs.
        
        Args:
            result: ImageLike object of gibbs ringing corrected image
        
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
                    "GibbsUnringingStep expected 'current_image' in context result"
                )
        else:
            image = result

        self._validate_result_image(image)


    def run(self, first_arg, output_dir: Path, mask: Optional[Path]=None, **kwargs) -> Path:
        """
        Run gibbs unringing on input image.
        """
        context, input_img = self.unpack_input(first_arg)
        if input_img is None:
             raise ProcessingError("No input image provided")
        
        output_dir = self.get_step_output_dir(output_dir)
        
        # Append to desc (e.g. SPGRdenoised -> SPGRdenoisedGibbs)
        old_desc = input_img.entities.get('desc', '')
        new_desc = f"{old_desc}Gibbs" if old_desc else "Gibbs"
        
        output_img = output_dir / build_bids_name({**input_img.entities, "desc": new_desc})
        output_json = output_img.with_suffix("").with_suffix(".json")
               
        # Check if output exists and is valid
        should_skip = False
        if output_img.exists() and not kwargs.get('force', False):
             # Check timestamps
             in_mtime = input_img.img.stat().st_mtime
             out_mtime = output_img.stat().st_mtime
             
             if in_mtime > out_mtime:
                 self.logger.info(f"Gibbs input ({input_img.img.name}) is newer than output. Re-running.")
             else:
                 self.logger.info(f"Skipping {self.method} gibbs unringing (Output exists and up-to-date: {output_img.name})")
                 should_skip = True

        if should_skip:
             copy_json_with_metadata(getattr(input_img, "json", None), output_json)
             result_json = output_json if output_json.exists() else getattr(input_img, "json", None)
             if isinstance(input_img, DWIFile):
                 result_img = DWIFile(
                    entities=input_img.entities,
                    img=output_img,
                    json=result_json,
                    bval=input_img.bval,
                    bvec=input_img.bvec,
                    Delta=getattr(input_img, "Delta", None),
                    delta=getattr(input_img, "delta", None),
                 )
             else:
                 result_img = ImageFile(entities=input_img.entities, img=output_img, json=result_json)
                 
             if context is not None:
                context["current_image"] = result_img
                if isinstance(result_img, DWIFile):
                    pre_list = context.setdefault("preprocessed_dwis", [])
                    if result_img not in pre_list:
                         pre_list.append(result_img)
                return context
             else:
                return result_img

        # Run denoising based on method
        self.logger.info(f"Running {self.method} gibbs unringing...")
        
        # Get nthreads from kwargs or config
        nthreads = kwargs.get('nthreads', self.config.n_cpus)
        
        try:
            if self.method == 'dipy':
                corrected = dipy.gibbs_unring(in_file=input_img.img, 
                                                          out_file=output_img,
                                                          nthreads=nthreads)
            elif self.method == 'mrtrix':
                corrected = mrtrix.mrdegibbs(in_file=input_img.img, 
                                             out_file=output_img, 
                                             nthreads=nthreads,
                                             force=bool(kwargs.get('force', False)))
            else:
                raise ValueError(f"Unknown denoising method: {self.method}")
        
        except Exception as e:
            raise ProcessingError(
                f"Gibbs Unringing failed with method {self.method}",
                step_name="gibbs_unringing",
                details=str(e)
            )
                
        self.logger.info(f"Gibbs corrected image saved to: {output_img}")
        copy_json_with_metadata(getattr(input_img, "json", None), output_json)
        result_json = output_json if output_json.exists() else getattr(input_img, "json", None)
        
        if isinstance(input_img, DWIFile):
            result_img = DWIFile(entities=input_img.entities,
                                 img=corrected,
                                 json=result_json,
                                 bval=input_img.bval,
                                 bvec=input_img.bvec,
                                 Delta=getattr(input_img, "Delta", None),
                                 delta=getattr(input_img, "delta", None))
        else:
            result_img = ImageFile(entities=input_img.entities,
                                   img=corrected,
                                   json=result_json)
        
        # ---- Return shape depends on input shape ----
        if context is not None:
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
            outputs = {"gibbs_corrected_image": str(result.img)}

        parameters = {
            "method": self.method,
            **kwargs,
        }

        self.provenance.log_step(
            step_name="gibbs-correction",
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
    


# Convenience function for quick denoising
def gibbs_unringing(input_img: Path, output_img: Path, method: str = 'dipy', mask: Optional[Path] = None, **kwargs) -> Path:
    """
    Convenience function for quick gibbs unringing without full pipeline setup.
    
    Args:
        input_img: Input image path
        output_img: Output image path
        method: Gibbs Unringing method ('dipy', 'mrtrix', etc.)
        mask: Optional mask path
        **kwargs: Method-specific parameters
    
    Returns:
        Path to denoised image
    
    Example:
        >>> from qmri_neuropipe.processing.common.gibbs import gibbs_unringing
        >>> 
        >>> corrected = gibbs_unringing(
        ...     input_img=Path('dwi.nii.gz'),
        ...     output_img=Path('denoised.nii.gz'),
        ...     method='dipy'
        ... )
    """
    # Create minimal config
    from qmri_neuropipe.core import PipelineConfig
    
    config = PipelineConfig(
        bids_dir=Path('/tmp'),  # Dummy value
        output_dir=output_img.parent
    )
    
    # Run denoising
    unringer = GibbsUnringingStep(config, method=method)
    
    # Temporarily disable validation for standalone use
    unringer.validate_inputs = lambda *args, **kwargs: None
    
    result = unringer.run(
        input_img,
        output_img,
        mask=mask,
        **kwargs
    )
    
    return result


# Module version and metadata
__version__ = '2.0.0'
__all__ = [
    'GibbsUnringingStep',
    'gibbs_unringing'
]
