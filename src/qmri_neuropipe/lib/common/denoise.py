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
from ...interfaces import dipy, ants, mrtrix, fsl
from ...io.bids import build_bids_name
from ...core.run import run_cmd
from ...core.utils import get_nifti_stem


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
        method: Literal['mrtrix', 'ants', 'mppca', 'patch2self', 'nlmeans', 'wavelets', 'gaussian'] = 'mrtrix',
        patch_radius: int = 2,
        block_radius: int = 5,
        mask_dilation: int = 2,
        pca_method: str = 'eig',
        model: str = 'ridge',
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
        self.mask_dilation = mask_dilation
        self.pca_method = pca_method    
        self.model = model
        self.logger.info(f"Initialized denoising with method: {method}")
    

    # _unpack_input in base class handles context vs direct ImageLike

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
            # MP-PCA and Patch2Self require 4D
            if self.method in ['mppca', 'patch2self'] and len(img.shape) != 4:
                raise ValidationError(
                    f"{self.method} requires 4D data, got shape {img.shape}"
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
        """
        context, input_img = self.unpack_input(first_arg)
        if input_img is None:
             raise ProcessingError("No input image provided")

        output_dir = self.get_step_output_dir(output_dir)
        
        # Append to desc to preserve modality info (e.g. SPGRreor -> SPGRreordenoised)
        old_desc = input_img.entities.get('desc', '')
        # Avoid duplication if rerun? 
        # denoise shouldn't be in old_desc theoretically if we are running it now
        new_desc = f"{old_desc}denoised" if old_desc else "denoised"
        
        output_img_path = output_dir / build_bids_name({**input_img.entities, "desc": new_desc})
        
        # Noise map usually shares basename
        noise_map_desc = f"{new_desc}NoiseMap"
        noise_map_path  = output_dir / build_bids_name({**input_img.entities, "desc": noise_map_desc})
        
        # Check if output exists and is valid
        should_skip = False
        if output_img_path.exists() and not kwargs.get('force', False):
             # Check timestamps
             in_mtime = input_img.img.stat().st_mtime
             out_mtime = output_img_path.stat().st_mtime
             
             if in_mtime > out_mtime:
                 self.logger.info(f"Denoising input ({input_img.img.name}) is newer than output. Re-running.")
             else:
                 self.logger.info(f"Skipping {self.method} denoising (Output exists and up-to-date: {output_img_path.name})")
                 should_skip = True
        
        if should_skip:
             # Reconstruct result object
             if isinstance(input_img, DWIFile):
                 result_img = DWIFile(
                    entities=input_img.entities,
                    img=output_img_path,
                    json=input_img.json,
                    bval=input_img.bval,
                    bvec=input_img.bvec
                 )
             else:
                 result_img = ImageFile(entities=input_img.entities, img=output_img_path, json=input_img.json)
                 
             if context is not None:
                context["current_image"] = result_img
                if isinstance(result_img, DWIFile):
                    pre_list = context.setdefault("preprocessed_dwis", [])
                    if result_img not in pre_list:
                        pre_list.append(result_img)
                return context
             else:
                return result_img
        
        # Standardized threading resolution
        nthreads = kwargs.get('nthreads') or getattr(self, 'nthreads', None) or self.config.get('n_cpus', 1)
        
        # Optimization: Generate temporary mask if not provided to speed up denoising
        # Primarily for MRTrix/MP-PCA which benefits significantly from masking, but also valid for others (NLMeans, ANTs)
        # User requested: "Is this mask also created for other modalities... And does this work across implemented denoising methods..."
        # We enable this for all methods.
        if mask is None:
             try:
                 self.logger.info("No mask provided. Generating temporary dilated mask via FSL BET to accelerate denoising...")
                 
                 # Robust stem extraction
                 stem = get_nifti_stem(input_img.img)

                 temp_ref = output_dir / f"temp_denoise_ref_{stem}.nii.gz"
                 temp_brain = output_dir / f"temp_denoise_brain_{stem}.nii.gz"
                 temp_mask_dil = output_dir / f"temp_denoise_mask_dilated_{stem}.nii.gz"
                 
                 # 1. Prepare Reference (3D)
                 # Determine if 4D
                 is_4d = False
                 try:
                     # Check extraction logic or use nibabel
                     hdr = nib.load(str(input_img.img)).header
                     if len(hdr.get_data_shape()) > 3 and hdr.get_data_shape()[3] > 1:
                         is_4d = True
                 except:
                     pass # Assume 3D or fail later
                 
                 bet_input = input_img.img
                 if is_4d:
                      # Calculate mean image over series for robust mask reference
                      # Previously used first volume: fslroi {input_img.img} {temp_ref} 0 1
                      run_cmd(f"fslmaths {input_img.img} -Tmean {temp_ref}", label="calculate_mean_ref_for_mask")
                      bet_input = temp_ref
                 
                 # 2. Run FSL BET
                 # Returns (brain, mask)
                 # Force overwrite if temp files exist from previous failed run
                 if temp_brain.exists():
                     import shutil
                     if temp_brain.exists(): temp_brain.unlink()
                     # BET creates separate mask file, we need to know its name to clean it?
                     # Wrapper handles it, but let's just let BET overwrite or fail?
                     # FSL BET usually overwrites.
                 
                 _, temp_mask = fsl.bet(bet_input, temp_brain, frac=0.3, mask=True) # frac=0.3 is often safe for dMRI b0
                 
                 if temp_mask and temp_mask.exists():
                     # 3. Dilate
                     # Ensures whole coverage
                     # Use configured mask_dilation (default 2)
                     dilation = kwargs.get('mask_dilation', self.mask_dilation)
                     if dilation > 0:
                         mrtrix.maskfilter(temp_mask, temp_mask_dil, filter_type='dilate', npass=dilation, nthreads=nthreads, force=True)
                         if temp_mask_dil.exists():
                             mask = temp_mask_dil
                     else:
                         # If dilation is 0, just use the raw mask
                         mask = temp_mask
                     
                     if mask and mask.exists():
                         self.logger.info(f"Using temporary mask (dilation={dilation}): {mask}")
             except Exception as e:
                 self.logger.warning(f"Failed to generate temporary mask: {e}. Proceeding without mask.")

        # Run denoising based on method
        self.logger.info(f"Running {self.method} denoising...")
        
        # Prepare filtered kwargs for backends that don't accept 'force'
        call_kwargs = kwargs.copy()
        if 'force' in call_kwargs:
            del call_kwargs['force']
        if 'nthreads' in call_kwargs: # nthreads often passed explicitly
             del call_kwargs['nthreads']

        from threadpoolctl import threadpool_limits

        try:
            if self.method == 'mrtrix':
                # Force=True because we are in the execution block (so we decided to run)
                # This handles cases where auxiliary outputs (NoiseMap) exist but main output didn't.
                denoised, noise = mrtrix.dwidenoise(in_file=input_img.img,
                                                    out_file=output_img_path, 
                                                    mask=mask,
                                                    noise_map=noise_map_path,
                                                    nthreads=nthreads,
                                                    force=True)
            elif self.method == 'ants':
                denoised, noise = ants.denoise_image(in_file=input_img.img,
                                                     out_file=output_img_path,
                                                     noise_map=noise_map_path,
                                                     nthreads=nthreads) # ants.denoise_image does not accept kwargs
            else:
                # Python-based methods (mppca, patch2self, nlmeans, wavelets, gaussian)
                # Limit threads for BLAS/OpenMP operations
                with threadpool_limits(limits=nthreads):
                    if self.method == 'mppca':
                        # Fetch pca_method from kwargs OR use instance default (from config/init)
                        pca_method = call_kwargs.get('pca_method', self.pca_method)
                        
                        denoised, noise = dipy.mppca(in_file=input_img.img, 
                                                     out_file=output_img_path, 
                                                     mask=mask,
                                                     noise_map=noise_map_path,
                                                     patch_radius=self.patch_radius,
                                                     block_radius=self.block_radius,
                                                     pca_method=pca_method,
                                                     nthreads=nthreads,
                                                     **call_kwargs)
                    elif self.method == 'patch2self':
                        model = call_kwargs.get('model', self.model)
                        # Ensure input is DWI and has bval
                        if not isinstance(input_img, DWIFile) or not input_img.bval:
                            raise ProcessingError("Patch2Self requires DWI data with bval file")
                            
                        # Warn about unused patch_radius
                        if self.patch_radius != 2: # Default is 2 in Config/Init, but DIPY v3 ignores it.
                             self.logger.warning("patch_radius is ignored by Patch2Self (auto-determined).")

                        denoised = dipy.patch2self(in_file=input_img.img,
                                                   out_file=output_img_path,
                                                   bval_file=input_img.bval,
                                                   # patch_radius=self.patch_radius, # Unused in v3
                                                   model=model,
                                                   nthreads=nthreads,
                                                   **call_kwargs)
                        noise = None # Patch2Self wrapper doesn't return noise map
                    elif self.method == 'nlmeans':
                        denoised = dipy.nlmeans(in_file=input_img.img, 
                                                out_file=output_img_path, 
                                                mask=mask, 
                                                nthreads=nthreads,
                                                **call_kwargs)
                    elif self.method == 'wavelets':
                        denoised = self._run_wavelets(
                            nib.load(str(input_img.img)).get_fdata(),
                            mask=nib.load(str(mask)).get_fdata() if mask else None,
                            **call_kwargs
                        )
                    elif self.method == 'gaussian':
                        denoised = self._run_gaussian(
                            nib.load(str(input_img.img)).get_fdata(),
                            mask=nib.load(str(mask)).get_fdata() if mask else None,
                            **call_kwargs
                        )
                    else:
                        raise ValueError(f"Unknown denoising method: {self.method}")
            
        except Exception as e:
            raise ProcessingError(
                f"Denoising failed with method {self.method}",
                step_name="denoising",
                details=str(e)
            )
        
        # Check if denoised is an array (from internal python methods) and save it
        if isinstance(denoised, np.ndarray):
             self.logger.info(f"Saving {self.method} result to {output_img_path}")
             original = nib.load(str(input_img.img))
             # Ensure header matches data shape if needed, strictly we just want affine/header
             # If headers are stricter about dimensions (e.g. 4D vs 3D), Nifti1Image might handle or we clean header
             new_img = nib.Nifti1Image(denoised, original.affine, original.header)
             nib.save(new_img, str(output_img_path))
             denoised = output_img_path

        # Save denoised image log
        self.logger.info(f"Denoised image saved to: {output_img_path}")
        
        # Also save noise map if MP-PCA
        if noise_map is not None:
             # Logic fix: noise_map arg was meant to be input path or boolean? 
             # In run arg it is Optional[Path]. If passed, we use it.
             # but check if we actually created it.
             self.logger.debug(f"Noise map saved to: {noise_map_path}")

        if isinstance(input_img, DWIFile):
            result_img = DWIFile(entities=input_img.entities,
                                 img=denoised,
                                 json=input_img.json,
                                 bval=input_img.bval,
                                 bvec=input_img.bvec)
        else:
            result_img = ImageFile(entities=input_img.entities,
                                   img=denoised,
                                   json=input_img.json)
        
        # ---- Return shape depends on input shape ----
        if context is not None:
            context["current_image"] = result_img

            # If this is DWI, you might want to maintain a list of preprocessed DWIs:
            if isinstance(result_img, DWIFile):
                pre_list = context.setdefault("preprocessed_dwis", [])
                if result_img not in pre_list:
                    pre_list.append(result_img)
            
            # Save mask to context if we generated/used one (for reuse in Bias Correction)
            if mask:
                 context["temp_denoise_mask"] = mask
                 context["current_mask"] = mask

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
            "mask_dilation": self.mask_dilation,
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
    

# Module version and metadata
__version__ = '2.0.0'
__all__ = [
    'DenoisingStep'
]
