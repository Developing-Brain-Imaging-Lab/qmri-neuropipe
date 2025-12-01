"""
Denoising module for neuroimaging data.

This module provides denoising methods that work across multiple modalities:
- dMRI: MP-PCA (Marchenko-Pastur PCA)
- fMRI: Non-local means, wavelets
- Anatomical: Non-local means, anisotropic diffusion

All denoising steps inherit from BaseProcessingStep for automatic
validation, logging, and provenance tracking.

Classes:
    DenoisingStep: Main denoising class with multiple methods
    denoise_image: Convenience function for quick denoising without full pipeline setup.
"""

from pathlib import Path
from typing import Optional, Literal, Tuple, Union
import numpy as np
import nibabel as nib
import logging

from ...core import BaseProcessingStep, ValidationError, ProcessingError
from ...core.types import ImageFile, DWIFile, ImageLike
from ...interfaces import dipy, ants, mrtrix
from ...io.bids import build_bids_name



class DenoisingStep(BaseProcessingStep):
    """
    General denoising step that works for multiple modalities.
    
    Supports multiple denoising methods:
    - 'mrtrix': MRtrix3 dwidenoise
    - 'ants': ANTs DenoiseImage
    - 'mppca': Marchenko-Pastur PCA (best for dMRI)
    - 'nlmeans': Non-local means (works for all modalities)
    - 'wavelets': Wavelet-based denoising (experimental)
    - 'gaussian': Simple Gaussian smoothing (fallback)
    
    Used by:
    - dMRI: Typically uses MP-PCA
    - fMRI: Typically uses non-local means or wavelets
    - Anatomical: Typically uses non-local means
    
    Attributes:
        method: Denoising method to use
        patch_radius: Size of patch for local methods
        block_radius: Size of block for non-local methods
        
    Example:
        >>> # For dMRI
        >>> denoising = DenoisingStep(config, method='mppca')
        >>> denoised = denoising(dwi_file, output_dir)
        >>> 
        >>> # For fMRI
        >>> denoising = DenoisingStep(config, method='nlmeans')
        >>> denoised = denoising(bold_file, output_dir)
    """
    
    def __init__(
        self,
        config,
        logger: Optional[logging.Logger] = None,
        provenance = None,
        method: Literal['mrtrix', 'ants', 'mppca', 'nlmeans', 'wavelets', 'gaussian'] = 'mrtrix',
        patch_radius: int = 2,
        block_radius: int = 5
    ):
        """
        Initialize denoising step.
        
        Args:
            config: Pipeline configuration
            method: Denoising method to use
            patch_radius: Patch size for local methods (default: 2)
            block_radius: Block size for non-local methods (default: 5)
            logger: Optional logger instance
            provenance: Optional provenance tracker
        
        Raises:
            ProcessingError: If required dependencies are missing
        """
        super().__init__(config, logger, provenance)
        
        self.method = method
        self.patch_radius = patch_radius
        self.block_radius = block_radius
        
        # Check dependencies
        self._check_dependencies()
        
        self.logger.info(f"Initialized denoising with method: {method}")
    
    def _check_dependencies(self) -> None:
        """Check that required dependencies are available."""
        if self.method == 'mrtrix':
            # Assume MRtrix3 is installed and in PATH
            pass

        if self.method == 'ants':
            # Assume ANTs is installed and in PATH
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
            if len(img.shape) < 3:
                raise ValidationError(
                    f"Input must be at least 3D, got shape {img.shape}"
                )
            # MP-PCA requires 4D
            if self.method == 'mppca' and len(img.shape) != 4:
                raise ValidationError(
                    f"MP-PCA requires 4D data, got shape {img.shape}"
                )
        except Exception as e:
            raise ValidationError(f"Invalid NIfTI file: {e}")

        if not image.img.suffix.endswith(".nii") and not "".join(image.img.suffixes).endswith(".nii.gz"):
            raise ValidationError(f"Input must be NIfTI, got {image.img}")
        
        self.logger.debug(f"Input validation passed for {image.img}")

    def _validate_result_image(self, result: ImageLike) -> None:
        """Your existing validate_outputs logic, moved here."""
        if not result.img.exists():
            raise ProcessingError(f"Denoised image not created: {result.img}")
        
        try:
            img = nib.load(str(result.img))
            data = img.get_fdata()

            if np.any(np.isnan(data)):
                raise ProcessingError("Denoised image contains NaN values")
            if np.any(np.isinf(data)):
                raise ProcessingError("Denoised image contains Inf values")
            if np.any(data < 0):
                self.logger.warning("Denoised image contains negative values")

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
    

    def run(self, first_arg, output_dir: Path, mask: Optional[Path]=None, noise_map: Optional[Path]=None, **kwargs) -> Tuple[Path, Optional[Path]]:
        """
        Run denoising on input image.
        
        Args:
            input_img: ImageLike object to input NIfTI file
            output_dir: Path to output NIfTI file for denoised image
            mask: Optional brain mask (speeds up processing)
            **kwargs: Method-specific parameters
        
        Returns:
            Path to denoised image
            Path to noise map: (Optional)
        
        Raises:
            ProcessingError: If denoising fails
        """

        """
        Pipeline mode:
            first_arg: dict with 'current_image' (ImageLike)
            returns:   updated dict with 'current_image' replaced by denoised ImageLike,
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
        output_img = output_dir / build_bids_name({**input.entities, "desc": "denoised"})
        noise_map  = output_dir / build_bids_name({**input.entities, "desc": "NoiseMap"})
        
        # Run denoising based on method
        self.logger.info(f"Running {self.method} denoising...")
        
        try:
            if self.method == 'mrtrix':
                denoised, noise = mrtrix.dwidenoise(in_img=input.img,
                                                    out=output_img, 
                                                    mask=mask,
                                                    noise_map=noise_map,
                                                    nthreads=kwargs.get('nthreads', 2),
                                                    force=bool(kwargs.get('force', False)))
            elif self.method == 'ants':
                denoised, noise = ants.denoise_image(in_img=input.img,
                                                     out=output_img,
                                                     noise_model=kwargs.get('noise_model', 'Rician'),
                                                     mask=mask,
                                                     noise_map=noise_map,
                                                     nthreads=kwargs.get('nthreads', 2))
            elif self.method == 'mppca':
                denoised, noise = dipy.mppca(in_img=input.img, 
                                             out=output_img, 
                                             mask=mask,
                                             noise_map=noise_map,
                                             patch_radius=self.patch_radius,
                                             block_radius=self.block_radius, 
                                             **kwargs)
            elif self.method == 'nlmeans':
                denoised = dipy.nlmeans(in_img=input.img, 
                                        out=output_img, 
                                        mask=mask, 
                                        **kwargs)
            elif self.method == 'wavelets':
                denoised = self._run_wavelets(
                    nib.load(str(input.img)).get_fdata(),
                    mask=nib.load(str(mask)).get_fdata() if mask else None,
                    **kwargs
                )
            elif self.method == 'gaussian':
                denoised = self._run_gaussian(
                    nib.load(str(input.img)).get_fdata(),
                    mask=nib.load(str(mask)).get_fdata() if mask else None,
                    **kwargs
                )
            else:
                raise ValueError(f"Unknown denoising method: {self.method}")
            
        except Exception as e:
            raise ProcessingError(
                f"Denoising failed with method {self.method}",
                step_name="denoising",
                details=str(e)
            )
        
        # Save denoised image
        self.logger.info(f"Denoised image saved to: {output_img}")
        
        # Also save noise map if MP-PCA
        if noise_map is not None:
            self.logger.debug(f"Noise map saved to: {noise_map}")

        if isinstance(input, DWIFile):
            result_img = DWIFile(entities=input.entities,
                                 img=denoised,
                                 json=input.json,
                                 bval=input.bval,
                                 bvec=input.bvec)
        else:
            result_img = ImageFile(entities=input.entities,
                                   img=denoised,
                                   json=input.json)
        
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
        
    def _run_wavelets(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Run wavelet-based denoising.
        
        Uses soft thresholding in wavelet domain.
        
        Args:
            data: 3D or 4D array
            mask: Optional binary mask
            **kwargs: Additional parameters
        
        Returns:
            Denoised data
        """
        try:
            import pywt
        except ImportError:
            raise ProcessingError(
                "PyWavelets not available. Install with: pip install PyWavelets"
            )
        
        self.logger.debug("Running wavelet denoising")
        
        # Get wavelet type and threshold
        wavelet = kwargs.get('wavelet', 'db4')
        threshold_method = kwargs.get('threshold_method', 'BayesShrink')
        
        def denoise_3d(volume):
            """Denoise a 3D volume using wavelets."""
            # Perform 3D wavelet decomposition
            coeffs = pywt.wavedecn(volume, wavelet, level=3)
            
            # Threshold detail coefficients
            coeffs_thresh = [coeffs[0]]  # Keep approximation
            for detail in coeffs[1:]:
                detail_thresh = {}
                for key, value in detail.items():
                    if threshold_method == 'BayesShrink':
                        sigma = np.median(np.abs(value)) / 0.6745
                        thresh = sigma * np.sqrt(2 * np.log(value.size))
                    else:
                        thresh = kwargs.get('threshold', 1.0)
                    
                    detail_thresh[key] = pywt.threshold(
                        value, thresh, mode='soft'
                    )
                coeffs_thresh.append(detail_thresh)
            
            # Reconstruct
            denoised = pywt.waverecn(coeffs_thresh, wavelet)
            
            return denoised
        
        # Process data
        if len(data.shape) == 4:
            denoised_arr = np.zeros_like(data)
            for vol in range(data.shape[3]):
                denoised_arr[..., vol] = denoise_3d(data[..., vol])
        else:
            denoised_arr = denoise_3d(data)
        
        return denoised_arr
    
    def _run_gaussian(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Run simple Gaussian smoothing.
        
        Fallback method that doesn't require special dependencies.
        
        Args:
            data: 3D or 4D array
            mask: Optional binary mask
            **kwargs: Additional parameters
        
        Returns:
            Smoothed data
        """
        from scipy.ndimage import gaussian_filter
        
        sigma = kwargs.get('sigma', 1.0)
        self.logger.debug(f"Running Gaussian smoothing with sigma={sigma}")
        
        if len(data.shape) == 4:
            # Don't smooth in time dimension
            smoothed = np.zeros_like(data)
            for vol in range(data.shape[3]):
                smoothed[..., vol] = gaussian_filter(data[..., vol], sigma=sigma)
        else:
            smoothed = gaussian_filter(data, sigma=sigma)
        
        return smoothed
    
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
            outputs = {"denoised_image": str(result.img)}

        parameters = {
            "method": self.method,
            "patch_radius": self.patch_radius,
            "block_radius": self.block_radius,
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
def denoise_image(input: ImageLike, output: ImageLike, method: str = 'mrtrix', mask: Optional[Path] = None, **kwargs) -> Path:
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
    
    # Run denoising
    denoiser = DenoisingStep(config, method=method)
    
    # Temporarily disable validation for standalone use
    denoiser.validate_inputs = lambda *args, **kwargs: None
    
    result = denoiser.run(
        input,
        output.img.parent,
        mask=mask,
        **kwargs
    )
    
    return result


# Module version and metadata
__version__ = '2.0.0'
__all__ = [
    'DenoisingStep',
    'denoise_image'
]
