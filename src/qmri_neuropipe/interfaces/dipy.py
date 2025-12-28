from pathlib import Path
import os, json
from typing import Optional, Literal, Tuple, Dict, Any, Union
import numpy as np
import nibabel as nib
from ..core.types import ImageLike, DWIFile
from ..core.utils import extract_image_path, ensure_dir

# Try to import optional dependencies
# Moved to local scope to optimize import time

def patch2self(in_file: Path, out_file: Path, bval_file: Path, patch_radius: int = 1, model: str = "ridge", nthreads: int = 1):
    """
        Run Patch2Self denoising.
        
        Best for dMRI data with multiple volumes.
        
        Args:
            in_file: Input 4D NIfTI file
            out_file: Output path
            bval_file: Path to bval file
            patch_radius: Patch radius (default 1)
            model: Regression model (default 'ridge')
            nthreads: Number of threads
        
        Returns:
            Output file path
        """
    try:
        from dipy.denoise.patch2self import patch2self as p2s
        from dipy.io.gradients import read_bvals_bvecs
    except ImportError:
         raise ImportError("DIPY is required for patch2self but not installed.")
    
    img = nib.load(in_file)

    if out_file.exists():
        return out_file
    
    # Load bvals
    # read_bvals_bvecs returns (bvals, bvecs)
    bvals, _ = read_bvals_bvecs(str(bval_file), None)

    os.environ['OMP_NUM_THREADS'] = str(nthreads)
    # patch2self signature: data, bvals, ...
    den = p2s(img.get_fdata(), bvals, patch_radius=patch_radius, b0_threshold=50, model=model)
    nib.Nifti1Image(den, img.affine, img.header).to_filename(out_file)
    return out_file

def mppca(in_file: Path, out_file: Path, mask: Optional[Path]=None, noise_map: Optional[Path]=None, patch_radius: int=2, pca_method: str="eig", nthreads: int = 1, **kwargs)-> Tuple[Path, Optional[Path]]:
        """
        Run Marchenko-Pastur PCA denoising.
        
        Best for dMRI data with multiple volumes.
        
        Args:
            in_img: 4D array (x, y, z, volumes)
            mask: Optional 3D binary mask
            pca_method: Method for PCA ('eig' or 'svd')
            **kwargs: Additional parameters
        
        Returns:
            Tuple of (denoised_data, sigma_map)
        """
        try:
             from dipy.denoise.localpca import mppca as dipy_mppca
             from dipy.denoise.pca_noise_estimate import pca_noise_estimate
             from dipy.denoise.denspeed import determine_num_threads
        except ImportError:
             raise ImportError("DIPY is required for mppca but not installed.")
             
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
        os.environ['OMP_NUM_THREADS'] = str(nthreads)
        denoised_arr, sigma = dipy_mppca(data, mask=mask, patch_radius=patch_radius, pca_method=pca_method, return_sigma=True)
        
        # Calculate noise reduction
        if mask is not None:
            original_std = np.std(data[mask])
            denoised_std = np.std(denoised_arr[mask])
        else:
            original_std = np.std(data)
            denoised_std = np.std(denoised_arr)
        
        noise_reduction = (1 - denoised_std / original_std) * 100



        denoised_img = nib.Nifti1Image(denoised_arr, img.affine, img.header)
        nib.save(denoised_img, str(out_file))

        if noise_map is not None:
            sigma_img = nib.Nifti1Image(sigma, img.affine, img.header)
            nib.save(sigma_img, noise_map)
            
        return out_file, noise_map

def nlmeans(in_file: Path, out_file: Path, mask: Optional[Path]=None, sigma: float=None, patch_radius: int=1, block_radius: int=5, nthreads: int = 1, **kwargs):
    """
    Run Non-Local Means denoising.
    """
    try:
        from dipy.denoise.nlmeans import nlmeans as dipy_nlmeans
        from dipy.denoise.noise_estimate import estimate_sigma
        from dipy.denoise.denspeed import determine_num_threads
    except ImportError:
         raise ImportError("DIPY is required for nlmeans but not installed.")
         
    # Optimize threads
    n_jobs = determine_num_threads(nthreads)
         
    in_path = extract_image_path(in_file)
    out_p = ensure_dir(out_file)
    
    if out_p.exists():
        return out_p

    img = nib.load(in_path)
    data = img.get_fdata()
    
    if sigma is None:
        sigma = estimate_sigma(data, N=0)

    den = dipy_nlmeans(data, sigma=sigma, mask=None, patch_radius=patch_radius, block_radius=block_radius, rician=True, num_threads=n_jobs)
    nib.Nifti1Image(den, img.affine, img.header).to_filename(out_p)
    return out_p

def gibbs_unring(in_file: Path, out_file: Path, nthreads: int = 1, **kwargs):
    """
    Run Gibbs unringing.
    """
    try:
        from dipy.denoise.gibbs import gibbs_removal
    except ImportError:
         raise ImportError("DIPY is required for gibbs_unring but not installed.")
         
    in_path = extract_image_path(in_file)
    out_p = ensure_dir(out_file)
    
    if out_p.exists():
        return out_p
        
    img = nib.load(in_path)
    data = img.get_fdata()

    # Optimize threads
    try:
         from dipy.denoise.denspeed import determine_num_threads
         n_jobs = determine_num_threads(nthreads)
    except ImportError:
         n_jobs = nthreads # Fallback

    den = gibbs_removal(data, num_processes=n_jobs) # slice_axis? n_points? defaults usually ok for full volume
    nib.Nifti1Image(den, img.affine, img.header).to_filename(out_p)
    return out_p




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


def _dti_fit_worker(data_chunk, gtab, fit_method, kwargs):
    """
    Worker function for parallel DTI fitting.
    """
    import dipy.reconst.dti as dipy_dti
    # Re-instantiate model to avoid pickling complex objects or issues with shared state
    # Ensure return_leverages is handled if passed in kwargs
    if 'return_leverages' not in kwargs and (fit_method in ['WLLS', 'OLS', 'NLLS']):
         # Default to True as per main function to match behavior
         kwargs['return_leverages'] = True
         
    if fit_method == 'RESTORE':
        # sigma logic
        sigma = kwargs.pop('sigma', None)
        model = dipy_dti.TensorModel(gtab, fit_method='RESTORE', sigma=sigma, **kwargs)
    else:
        model = dipy_dti.TensorModel(gtab, fit_method=fit_method, **kwargs)
        
    fit = model.fit(data_chunk)
    return fit.model_params

def fit_dti(
    in_file: Union[Path, ImageLike],
    out_dir: Path,
    bval_file: Optional[Path] = None,
    bvec_file: Optional[Path] = None,
    mask_file: Optional[Path] = None,
    fit_method: str = "WLLS",
    metrics: list[str] = ["fa", "md", "ad", "rd", "color_fa", "evals", "evecs"],
    nthreads: int = 1
) -> Dict[str, Path]:
    """
    Fit Diffusion Tensor Imaging (DTI) model using DIPY.
    
    Parameters
    ----------
    in_file : Path or ImageLike
        Input DWI NIfTI file or ImageLike object.
    out_dir : Path
        Output directory.
    bval_file : Path, optional
        Path to bval file.
    bvec_file : Path, optional
        Path to bvec file.
    mask_file : Path, optional
        Path to brain mask.
    fit_method : str
        Method: "WLLS" (Weighted Linear Least Squares), "OLS" (Ordinary), "NLLS" (Non-Linear), "RESTORE".
    metrics : list
        Metrics to calculate: fa, md, ad, rd, color_fa.
    nthreads : int
        Number of CPUs to use (not currently fully utilized by standard DIPY fit, but reserved for future parallelization).
    """
    import numpy as np
    import multiprocessing
    import nibabel as nib
    import dipy.reconst.dti as dipy_dti
    from dipy.core.gradients import gradient_table
    from dipy.io.gradients import read_bvals_bvecs
    from qmri_neuropipe.io.bids import build_bids_name, get_entities_from_path

    # Read data
    in_path = extract_image_path(in_file)
    img = nib.load(str(in_path))
    data = img.get_fdata()
    
    if bval_file is None or bvec_file is None:
        if isinstance(in_file, DWIFile):
            bval_file = bval_file or in_file.bval
            bvec_file = bvec_file or in_file.bvec
            
    if not bval_file or not bvec_file:
         raise ValueError("Gradient files (bval/bvec) are required but not provided or found in input DWIFile.")
         
    bvals, bvecs = read_bvals_bvecs(str(bval_file), str(bvec_file))
    gtab = gradient_table(bvals, bvecs=bvecs)

    # Read mask
    if mask_file and mask_file.exists():
        mask = nib.load(str(mask_file)).get_fdata().astype(bool)
    else:
        # Create full mask if none provided
        mask = np.ones(data.shape[:3], dtype=bool)

    # Initialize Model for metadata or serial fallback
    if fit_method == 'RESTORE':
        dti_model = dipy_dti.TensorModel(gtab, fit_method='RESTORE', sigma=None)
    else:
        # return_leverages=True to avoid KeyError in serial fit or internally
        dti_model = dipy_dti.TensorModel(gtab, fit_method=fit_method, return_leverages=True)

    # Fit
    try:
        if nthreads > 1:
            # Parallel Fit
            # 1. Flatten data within mask
            data_flat = data[mask]
            
            # 2. Chunk data
            n_samples = data_flat.shape[0]
            if n_samples > 0:
                chunk_size = int(np.ceil(n_samples / nthreads))
                chunks = [data_flat[i:i + chunk_size] for i in range(0, n_samples, chunk_size)]
                
                # 3. Prepare args
                # Pass necessary kwargs to worker
                worker_kwargs = {}
                if fit_method != 'RESTORE':
                    worker_kwargs['return_leverages'] = True
                    
                args_list = [(chunk, gtab, fit_method, worker_kwargs) for chunk in chunks]
                
                # 4. Run in parallel
                with multiprocessing.Pool(processes=nthreads) as pool:
                    results = pool.starmap(_dti_fit_worker, args_list)
                    
                # 5. Reassemble
                all_params = np.concatenate(results, axis=0)
                
                # 6. Map back to volume
                # Get shape of params from first result (or model definition). DTI is usually 12 params (quadric form + S0 etc)
                # But actually dipy returns a shape like (..., 6) or (..., 12).
                # model_params shape matches data shape except last dim.
                # If input was (N, D), output is (N, P).
                n_params = all_params.shape[-1]
                vol_params = np.zeros(mask.shape + (n_params,), dtype=all_params.dtype)
                vol_params[mask] = all_params
                
                dti_fit = dipy_dti.TensorFit(dti_model, vol_params)
            else:
                # Empty mask?
                 dti_fit = dti_model.fit(data, mask=mask)
        else:
            # Serial Fit
            dti_fit = dti_model.fit(data, mask=mask)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"DTI fitting failed (method={fit_method}): {e}") from e
    
    # Save Outputs
    output_files = {}
    ent_base = get_entities_from_path(in_path)
    if 'desc' in ent_base: del ent_base['desc']
    ent_base['model'] = 'DTI'
    
    sidecar = {
        "ModelName": "Diffusion Tensor Imaging",
        "FittingSoftware": "DIPY",
        "InputData": in_path.name,
        "FittingMethod": fit_method,
        "Metrics": metrics
    }
    
    for metric in metrics:
        metric_suffix = metric.upper()
        if metric == 'color_fa': metric_suffix = 'DECFA'
        
        # Build path
        out_name = build_bids_name({**ent_base, 'suffix': metric_suffix})
        out_path = out_dir / out_name
        
        if metric == 'fa':
            nib.save(nib.Nifti1Image(dti_fit.fa, img.affine), str(out_path))
        elif metric == 'md':
            nib.save(nib.Nifti1Image(dti_fit.md, img.affine), str(out_path))
        elif metric == 'ad':
            nib.save(nib.Nifti1Image(dti_fit.ad, img.affine), str(out_path))
        elif metric == 'rd':
            nib.save(nib.Nifti1Image(dti_fit.rd, img.affine), str(out_path))
        elif metric == 'color_fa':
            nib.save(nib.Nifti1Image(dti_fit.color_fa, img.affine), str(out_path))
        elif metric == 'evals':
             nib.save(nib.Nifti1Image(dti_fit.evals, img.affine), str(out_path))
        elif metric == 'evecs':
             nib.save(nib.Nifti1Image(dti_fit.evecs, img.affine), str(out_path))
             
        output_files[metric] = out_path
        
        # Save sidecar
        import json
        sidecar_path = str(out_path).replace('.nii.gz', '.json')
        with open(sidecar_path, 'w') as f:
             json.dump(sidecar, f, indent=4)
            
    return output_files

def _dki_fit_worker(data_chunk, gtab, kwargs):
    """
    Worker function for parallel DKI fitting.
    """
    import dipy.reconst.dki as dipy_dki
    
    # Instantiate model
    model = dipy_dki.DiffusionKurtosisModel(gtab, **kwargs)
    
    fit = model.fit(data_chunk)
    return fit.model_params
 
def fit_dki(
    in_file: Union[Path, ImageLike],
    out_dir: Path,
    bval_file: Optional[Path] = None,
    bvec_file: Optional[Path] = None,
    mask_file: Optional[Path] = None,
    metrics: list[str] = ["mk", "ak", "rk", "fa", "md"],
    nthreads: int = 1,
    **kwargs
) -> Dict[str, Path]:
    """
    Fit Diffusion Kurtosis Imaging (DKI) model.
    """
    from dipy.core.gradients import gradient_table
    from dipy.io.gradients import read_bvals_bvecs
    import dipy.reconst.dki as dipy_dki
    from qmri_neuropipe.io.bids import build_bids_name, get_entities_from_path

    in_path = extract_image_path(in_file)
    img = nib.load(str(in_path))
    data = img.get_fdata()
    
    if bval_file is None or bvec_file is None:
        if isinstance(in_file, DWIFile):
            bval_file = bval_file or in_file.bval
            bvec_file = bvec_file or in_file.bvec
            
    if not bval_file or not bvec_file:
         raise ValueError("Gradient files (bval/bvec) are required but not provided or found in input DWIFile.")

    bvals, bvecs = read_bvals_bvecs(str(bval_file), str(bvec_file))
    gtab = gradient_table(bvals, bvecs=bvecs)

    if mask_file and mask_file.exists():
        mask = nib.load(str(mask_file)).get_fdata().astype(bool)
    else:
        mask = None
        
    dkimodel = dipy_dki.DiffusionKurtosisModel(gtab)
    
    # Fit
    try:
        if nthreads > 1:
            import multiprocessing
            # Parallel Fit
            # 1. Flatten data within mask
            if mask is None:
                 mask = np.ones(data.shape[:3], dtype=bool)
                 
            data_flat = data[mask]
            
            # 2. Chunk data
            n_samples = data_flat.shape[0]
            if n_samples > 0:
                chunk_size = int(np.ceil(n_samples / nthreads))
                chunks = [data_flat[i:i + chunk_size] for i in range(0, n_samples, chunk_size)]
                
                # 3. Prepare args
                # Pass clean kwargs. Serial version ignores kwargs, so we ignore them here too.
                # If we want to support params, we need to handle them explicitly.
                args_list = [(chunk, gtab, {}) for chunk in chunks]
                
                # 4. Run in parallel
                with multiprocessing.Pool(processes=nthreads) as pool:
                    results = pool.starmap(_dki_fit_worker, args_list)
                    
                # 5. Reassemble
                all_params = np.concatenate(results, axis=0)
                
                # 6. Map back to volume
                n_params = all_params.shape[-1]
                vol_params = np.zeros(mask.shape + (n_params,), dtype=all_params.dtype)
                vol_params[mask] = all_params
                
                dkifit = dipy_dki.DiffusionKurtosisFit(dkimodel, vol_params)
            else:
                 # Empty mask?
                 dkifit = dkimodel.fit(data, mask=mask)
        else:
            # Serial Fit
            dkifit = dkimodel.fit(data, mask=mask)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"DKI fitting failed: {e}") from e
    
    output_files = {}
    # Save Outputs
    ent_base = get_entities_from_path(in_path)
    if 'desc' in ent_base: del ent_base['desc']
    ent_base['model'] = 'DKI'
    
    sidecar = {
        "ModelName": "Diffusion Kurtosis Imaging",
        "FittingSoftware": "DIPY",
        "InputData": in_path.name,
        "FittingMethod": "WLLS", # Default for DIPY DKI usually
        "Metrics": metrics
    }

    for metric in metrics:
        metric_suffix = metric.upper()
        # Create output path
        out_name = build_bids_name({**ent_base, 'suffix': metric_suffix})
        out_path = out_dir / out_name
        
        if metric == 'mk':
            nib.save(nib.Nifti1Image(dkifit.mk(), img.affine), str(out_path))
        elif metric == 'ak':
            nib.save(nib.Nifti1Image(dkifit.ak(), img.affine), str(out_path))
        elif metric == 'rk':
            nib.save(nib.Nifti1Image(dkifit.rk(), img.affine), str(out_path))
        elif metric == 'fa':
            nib.save(nib.Nifti1Image(dkifit.fa, img.affine), str(out_path))
        elif metric == 'md':
            nib.save(nib.Nifti1Image(dkifit.md, img.affine), str(out_path))
            
        output_files[metric] = out_path
        
        # Save sidecar
        sidecar_path = str(out_path).replace('.nii.gz', '.json')
        with open(sidecar_path, 'w') as f:
             json.dump(sidecar, f, indent=4)
            
    return output_files

def fit_mapmri(
    in_file: Union[Path, ImageLike],
    out_dir: Path,
    bval_file: Optional[Path] = None,
    bvec_file: Optional[Path] = None,
    mask_file: Optional[Path] = None,
    laplacian: bool = True,
    positivity: bool = True,
    global_constraints: bool = False,
    metrics: list[str] = ["rtop", "rtap", "rtpp", "qiv", "msd"],
    nthreads: int = 1
) -> Dict[str, Path]:
    """
    Fit MAP-MRI model.
    """
    from dipy.core.gradients import gradient_table
    from dipy.io.gradients import read_bvals_bvecs
    import dipy.reconst.mapmri as mapmri
    from qmri_neuropipe.io.bids import build_bids_name, get_entities_from_path

    in_path = extract_image_path(in_file)
    img = nib.load(str(in_path))
    data = img.get_fdata()
    
    if bval_file is None or bvec_file is None:
        if isinstance(in_file, DWIFile):
            bval_file = bval_file or in_file.bval
            bvec_file = bvec_file or in_file.bvec
            
    if not bval_file or not bvec_file:
         raise ValueError("Gradient files (bval/bvec) are required but not provided or found in input DWIFile.")

    bvals, bvecs = read_bvals_bvecs(str(bval_file), str(bvec_file))
    gtab = gradient_table(bvals, bvecs=bvecs)

    if mask_file and mask_file.exists():
        mask = nib.load(str(mask_file)).get_fdata().astype(bool)
    else:
        mask = None
    
    map_model = mapmri.MapmriModel(
        gtab, 
        laplacian_regularization=laplacian,
        positivity_constraint=positivity,
        global_constraints=global_constraints
    )
    
    map_fit = map_model.fit(data, mask=mask)
    
    output_files = {}
    # Save Outputs
    ent_base = get_entities_from_path(in_path)
    if 'desc' in ent_base: del ent_base['desc']
    ent_base['model'] = 'MAPMRI'
    
    sidecar = {
        "ModelName": "Mean Apparent Propagator MRI",
        "FittingSoftware": "DIPY",
        "InputData": in_path.name,
        "FittingParameters": {
             "laplacian": laplacian,
             "positivity": positivity,
             "global_constraints": global_constraints
        },
        "Metrics": metrics
    }
    
    for metric in metrics:
        metric_suffix = metric.upper()
        # Create output path
        out_name = build_bids_name({**ent_base, 'suffix': metric_suffix})
        out_path = out_dir / out_name
        
        if metric == 'rtop':
             nib.save(nib.Nifti1Image(map_fit.rtop(), img.affine), str(out_path))
        elif metric == 'rtap':
             nib.save(nib.Nifti1Image(map_fit.rtap(), img.affine), str(out_path))
        elif metric == 'rtpp':
             nib.save(nib.Nifti1Image(map_fit.rtpp(), img.affine), str(out_path))
        elif metric == 'qiv':
             # Note: suffix was QIV previously?
             nib.save(nib.Nifti1Image(map_fit.qiv(), img.affine), str(out_path))
        elif metric == 'msd':
             nib.save(nib.Nifti1Image(map_fit.msd(), img.affine), str(out_path))
             
        output_files[metric] = out_path
        
        # Save sidecar
        sidecar_path = str(out_path).replace('.nii.gz', '.json')
        with open(sidecar_path, 'w') as f:
             json.dump(sidecar, f, indent=4)

             
    return output_files
