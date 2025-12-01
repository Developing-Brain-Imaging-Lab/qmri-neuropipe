
from pathlib import Path
from typing import Optional, Literal, Tuple, Dict, Any
import numpy as np
import nibabel as nib

# Try to import optional dependencies
try:
    from dipy.denoise.localpca import mppca as dipy_mppca
    from dipy.denoise.patch2self import patch2self as p2s
    from dipy.denoise.nlmeans import nlmeans as dipy_nlmeans
    from dipy.denoise.noise_estimate import estimate_sigma
    from dipy.denoise.pca_noise_estimate import pca_noise_estimate
    DIPY_AVAILABLE = True
except ImportError:
    DIPY_AVAILABLE = False


def patch2self(in_img: Path, out: Path, patch_radius: int = 1, model: str = "ridge"):
    """
        Run Patch2Self denoising.
        
        Best for dMRI data with multiple volumes.
        
        Args:
            data: 4D array (x, y, z, volumes)
            mask: Optional 3D binary mask
            **kwargs: Additional parameters
        
        Returns:
            Tuple of (denoised_data, sigma_map)
        """
    
    img = nib.load(in_img)

    if out.exists():
        return out

    den = p2s(img.get_fdata(), patch_radius=patch_radius, b0_threshold=50, model=model)
    nib.Nifti1Image(den, img.affine, img.header).to_filename(out)
    return out

def mppca(in_img: Path, out: Path, mask: Optional[Path]=None, noise_map: Optional[Path]=None, patch_radius: int=2, **kwargs)-> Tuple[Path, Optional[Path]]:
        """
        Run Marchenko-Pastur PCA denoising.
        
        Best for dMRI data with multiple volumes.
        
        Args:
            in_img: 4D array (x, y, z, volumes)
            mask: Optional 3D binary mask
            **kwargs: Additional parameters
        
        Returns:
            Tuple of (denoised_data, sigma_map)
        """
        # logger.debug(
        #     f"Running MP-PCA with patch_radius={self.patch_radius}"
        # )

        if out.exists() and (not noise_map or noise_map.exists()):
            return out, (noise_map if noise_map else None)
        
        # Get patch radius from kwargs or use default
        patch_radius = kwargs.get('patch_radius', patch_radius)

        img = nib.load(in_img)
        data = img.get_fdata()

        if mask is not None:
            mask_img = nib.load(mask)
            mask = mask_img.get_fdata().astype(bool)
        else:
            mask = None
        
        # Run MP-PCA
        denoised_arr, sigma = dipy_mppca(data, mask=mask, patch_radius=patch_radius, return_sigma=True)
        
        # Calculate noise reduction
        if mask is not None:
            original_std = np.std(data[mask])
            denoised_std = np.std(denoised_arr[mask])
        else:
            original_std = np.std(data)
            denoised_std = np.std(denoised_arr)
        
        noise_reduction = (1 - denoised_std / original_std) * 100

        # logger.info(
        #     f"MP-PCA reduced noise by {noise_reduction:.1f}% "
        #     f"(std: {original_std:.2f} → {denoised_std:.2f})"
        # )

        denoised_img = nib.Nifti1Image(denoised_arr, img.affine, img.header)
        nib.save(denoised_img, str(out))

        if noise_map is not None:
            sigma_img = nib.Nifti1Image(sigma, img.affine, img.header)
            nib.save(sigma_img, noise_map)
            
        return out, noise_map

def nlmeans(in_img: str, out: str, mask: Optional[str]=None, patch_radius: int=1, block_radius: int=5, **kwargs) -> np.ndarray:
        """
        Run non-local means denoising.
        
        Works for any modality (dMRI, fMRI, anatomical).
        
        Args:
            data: 3D or 4D array
            mask: Optional binary mask
            **kwargs: Additional parameters
        
        Returns:
            Denoised data
        """
        # logger.debug(
        #     f"Running non-local means with "
        #     f"patch_radius={self.patch_radius}, "
        #     f"block_radius={self.block_radius}"
        # )

        if out.exists():
            return out
        
        # Estimate noise level
        sigma = estimate_sigma(data, N=0)
        # logger.debug(f"Estimated noise level (sigma): {sigma:.4f}")
        
        # Get parameters
        patch_radius = kwargs.get('patch_radius', patch_radius)
        block_radius = kwargs.get('block_radius', block_radius)

        img = nib.load(in_img)
        data = img.get_fdata()

        if mask is not None:
            mask_img = nib.load(mask)
            mask = mask_img.get_fdata().astype(bool)
        else:
            mask = None
        
        # Run NLMeans
        if len(data.shape) == 4:
            # Process each volume separately for 4D data
            denoised_arr = np.zeros_like(data)
            for vol in range(data.shape[3]):
                denoised_arr[..., vol] = dipy_nlmeans(data[..., vol], sigma=sigma, mask=mask, patch_radius=patch_radius,block_radius=block_radius)
            #logger.debug(f"Processed {data.shape[3]} volumes")
        else:
            # 3D data
            denoised_arr = dipy_nlmeans(data, sigma=sigma, mask=mask, patch_radius=patch_radius, block_radius=block_radius)
        
        # Calculate noise reduction
        if mask is not None:
            original_std = np.std(data[mask])
            denoised_std = np.std(denoised_arr[mask])
        else:
            original_std = np.std(data)
            denoised_std = np.std(denoised_arr)
        
        noise_reduction = (1 - denoised_std / original_std) * 100
       #logger.info(f"NLMeans reduced noise by {noise_reduction:.1f}%")

        denoised_img = nib.Nifti1Image(denoised_arr, img.affine, img.header)
        nib.save(denoised_img, str(out))
        
        return out

def gibbs_ringing_correction(in_img: str, out: str, nthreads: int = 2):
    import nibabel as nib
    from dipy.denoise.gibbs import gibbs_removal

    if out.exists():
        return out

    img = nib.load(in_img)
    corrected = gibbs_removal(img.get_fdata(), num_processes=nthreads)
    nib.Nifti1Image(corrected, img.affine, img.header).to_filename(out)

    return out