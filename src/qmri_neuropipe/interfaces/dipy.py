
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


def patch2self(in_file: Path, out_file: Path, patch_radius: int = 1, model: str = "ridge"):
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
    
    img = nib.load(in_file)

    if out_file.exists():
        return out_file

    den = p2s(img.get_fdata(), patch_radius=patch_radius, b0_threshold=50, model=model)
    nib.Nifti1Image(den, img.affine, img.header).to_filename(out_file)
    return out_file

def mppca(in_file: Path, out_file: Path, mask: Optional[Path]=None, noise_map: Optional[Path]=None, patch_radius: int=2, **kwargs)-> Tuple[Path, Optional[Path]]:
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

        if out_file.exists() and (not noise_map or noise_map.exists()):
            return out_file, (noise_map if noise_map else None)
        
        # Get patch radius from kwargs or use default
        patch_radius = kwargs.get('patch_radius', patch_radius)

        img = nib.load(in_file)
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
        nib.save(denoised_img, str(out_file))

        if noise_map is not None:
            sigma_img = nib.Nifti1Image(sigma, img.affine, img.header)
            nib.save(sigma_img, noise_map)
            
        return out_file, noise_map

def nlmeans(in_file: str, out_file: str, mask: Optional[str]=None, patch_radius: int=1, block_radius: int=5, **kwargs) -> np.ndarray:
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

        if out_file.exists():
            return out_file
        
        # Estimate noise level
        sigma = estimate_sigma(data, N=0)
        # logger.debug(f"Estimated noise level (sigma): {sigma:.4f}")
        
        # Get parameters
        patch_radius = kwargs.get('patch_radius', patch_radius)
        block_radius = kwargs.get('block_radius', block_radius)

        img = nib.load(in_file)
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
        nib.save(denoised_img, str(out_file))
        
        return out_file

def gibbs_ringing_correction(in_file: str, out_file: str, nthreads: int = 2):
    import nibabel as nib
    from dipy.denoise.gibbs import gibbs_removal

    if out_file.exists():
        return out_file

    img = nib.load(in_file  )
    corrected = gibbs_removal(img.get_fdata(), num_processes=nthreads)
    nib.Nifti1Image(corrected, img.affine, img.header).to_filename(out_file)

    return out_file


def pca_noise_estimate(in_file: str, out_file: str, nthreads: int = 2):
    import nibabel as nib
    from dipy.denoise.pca_noise_estimate import pca_noise_estimate

    if out_file.exists():
        return out_file

    img = nib.load(in_file)
    corrected = pca_noise_estimate(img.get_fdata(), num_processes=nthreads)
    nib.Nifti1Image(corrected, img.affine, img.header).to_filename(out_file)

    return out_file

def estimate_sigma(in_file: str, out_file: str, nthreads: int = 2):
    import nibabel as nib
    from dipy.denoise.noise_estimate import estimate_sigma

    if out_file.exists():
        return out_file

    img = nib.load(in_file)
    corrected = estimate_sigma(img.get_fdata(), num_processes=nthreads)
    nib.Nifti1Image(corrected, img.affine, img.header).to_filename(out_file)

    return out_file

    return out


def synb0_estimation(in_file: Path, t1_file: Path, out_file: Path, b0_mask_path: Optional[Path] = None, t1_mask_path: Optional[Path] = None) -> Path:
    """
    Estimate synthetic b0 with reversed phase encoding using DIPY's Deep Learning Synb0.
    
    Args:
        b0_path: Path to b0 image
        t1_path: Path to T1w image
        out_path: Path to save synthetic b0
        b0_mask_path: Optional path to b0 brain mask
        t1_mask_path: Optional path to T1 brain mask
        
    Returns:
        Path to the generated synthetic b0 file.
    """
    import gc
    import nibabel as nib
    import numpy as np
    
    # Import inside function to avoid heavy TF import if not used
    try:
        from dipy.nn.tf.synb0 import Synb0
    except ImportError:
        # Fallback or error if module structure is different than expected
        # User specified dipy.nn.tf.synb0
        try:
             import dipy.nn.synb0 as synb0_module
             synb0 = synb0_module.synb0
        except ImportError:
             raise ImportError("Could not import dipy.nn.tf.synb0. Ensure DIPY and TensorFlow are installed correctly.")

    if out_file.exists():
        return out_file

    # Load images
    b0_img = nib.load(str(in_file))
    t1_img = nib.load(str(t1_file))
    
    b0_data = b0_img.get_fdata()
    t1_data = t1_img.get_fdata()
    
    # Run prediction
    # Assuming synb0 signature: synb0(b0, t1, b0_mask=None, t1_mask=None) -> synthetic_b0_data
    # Note: We might need to handle 3D/4D shapes. b0 should be 3D.
    if b0_data.ndim == 4:
        b0_data = b0_data[..., 0]

    SyNb0       = Synb0(False)
    rev_b0_data = SyNb0.predict(b0_data, t1_data)

    # Release GPU memory
    del SyNb0
    gc.collect()

    # Save output
    nib.Nifti1Image(rev_b0_data, b0_img.affine, b0_img.header).to_filename(out_file)
    
    return out_file

