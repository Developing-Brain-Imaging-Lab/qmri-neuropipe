from pathlib import Path
import os, json
from typing import Optional, Literal, Tuple, Dict, Any, Union
import numpy as np
import nibabel as nib
from ..core.types import ImageLike, DWIFile
from ..core.utils import extract_image_path, ensure_dir
import multiprocessing
import warnings

# Suppress resource_tracker warnings (benign semaphore "leaks" at shutdown in some envs)
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")
warnings.filterwarnings("ignore", message=".*resource_tracker: There appear to be.*leaked semaphore.*", category=UserWarning)
# Suppress cvxpy solution inaccuracy warnings (common in MAPMRI on noisy data)
warnings.filterwarnings("ignore", message=".*Solution may be inaccurate.*", category=UserWarning)


# Try to import optional dependencies
# Moved to local scope to optimize import time

def _load_timings(Delta_file: Optional[Path], delta_file: Optional[Path]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Helper to load diffusion timing files."""
    import numpy as np
    big_delta = None
    small_delta = None
    if Delta_file and Delta_file.exists():
         big_delta = np.loadtxt(str(Delta_file))
    if delta_file and delta_file.exists():
         small_delta = np.loadtxt(str(delta_file))
    return big_delta, small_delta


def _affine_direction_basis(affine: np.ndarray) -> np.ndarray:
    """Return direction cosines that map voxel axes into world axes."""
    linear = np.asarray(affine, dtype=float)[:3, :3]
    scales = np.linalg.norm(linear, axis=0)
    scales[scales == 0] = 1.0
    return linear / scales


def _reorient_tensor_matrices_to_world(tensor_matrices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Map voxel-basis tensor matrices into the image world basis."""
    basis = _affine_direction_basis(affine)
    return np.einsum("ia,...ab,jb->...ij", basis, tensor_matrices, basis)


def _reorient_eigenvectors_to_world(evecs: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Map voxel-basis eigenvectors into the image world basis."""
    basis = _affine_direction_basis(affine)
    world = np.einsum("ij,...kj->...ki", basis, evecs)
    norms = np.linalg.norm(world, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return world / norms


def _tensor_matrices_to_fsl_components(tensor_matrices: np.ndarray) -> np.ndarray:
    """Serialize tensors as FSL-order components."""
    out = np.empty(tensor_matrices.shape[:-2] + (6,), dtype=np.float32)
    out[..., 0] = tensor_matrices[..., 0, 0]
    out[..., 1] = tensor_matrices[..., 0, 1]
    out[..., 2] = tensor_matrices[..., 0, 2]
    out[..., 3] = tensor_matrices[..., 1, 1]
    out[..., 4] = tensor_matrices[..., 1, 2]
    out[..., 5] = tensor_matrices[..., 2, 2]
    return out


def _tensor_matrices_to_mrtrix_components(tensor_matrices: np.ndarray) -> np.ndarray:
    """Serialize tensors as MRtrix-order components."""
    out = np.empty(tensor_matrices.shape[:-2] + (6,), dtype=np.float32)
    out[..., 0] = tensor_matrices[..., 0, 0]
    out[..., 1] = tensor_matrices[..., 1, 1]
    out[..., 2] = tensor_matrices[..., 2, 2]
    out[..., 3] = tensor_matrices[..., 0, 1]
    out[..., 4] = tensor_matrices[..., 0, 2]
    out[..., 5] = tensor_matrices[..., 1, 2]
    return out


def patch2self(in_file: Path, out_file: Path, bval_file: Path, patch_radius: Optional[int] = None, model: str = "ridge", nthreads: int = 1):
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
    den = p2s(img.get_fdata(), bvals, b0_threshold=50, model=model)
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
    if out_file.exists():
        return out_file

    import gc
    import nibabel as nib
    import numpy as np

    try:
        import tensorflow  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "DIPY Synb0 requires TensorFlow, but TensorFlow is not installed. "
            "Install qmri-neuropipe with the synb0 extra (`pip install .[synb0]`) "
            "or rebuild the container after installing `tensorflow>=2.15,<2.16`."
        ) from e
    
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


# --- Unified Parallelization Helpers ---
def _global_driver_wrapper(args):
    """
    Global wrapper to unpack arguments for parallel worker.
    args: (chunk_id, chunk_data, gtab, worker_func, kwargs)
    """
    import os
    # Limit internal threading for each worker to avoid oversubscription
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1" 
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    chunk_id, chunk_data, gtab, worker_func, kwargs = args
    
    # Try using threadpoolctl for reliable limiting of BLAS/OpenMP
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            return worker_func(chunk_id, chunk_data, gtab, kwargs)
    except ImportError:
        return worker_func(chunk_id, chunk_data, gtab, kwargs)

def _parallel_fit_driver(data, mask, gtab, worker_func, nthreads, worker_kwargs=None):
    """
    Unified driver for parallel DIPY model fitting.
    
    Args:
        data: 4D numpy array (volumetric data)
        mask: 3D boolean mask
        gtab: Gradient table
        worker_func: Function that takes (chunk_id, chunk_data, gtab, kwargs) and returns fitted parameters
        nthreads: Number of parallel processes
        worker_kwargs: Additional kwargs to pass to the worker
        
    Returns:
        vol_params: 4D array of fitted parameters wrapped in the original volume shape
    """
    import multiprocessing
    import numpy as np
    
    if worker_kwargs is None:
        worker_kwargs = {}

    print(f"  - Starting parallel fit with {nthreads} threads...")
    
    # 1. Flatten data within mask
    if mask is None:
        mask = np.ones(data.shape[:3], dtype=bool)
    
    # Verification: data MUST be at least 4D for diffusion fitting (X, Y, Z, Vol)
    if data.ndim < 4:
         raise RuntimeError(f"Parallel fit driver received {data.ndim}D data (shape: {data.shape}). "
                            f"Diffusion models require a 4D volume series. "
                            f"This often happens if the coregistration or resampling step produced a 3D instead of 4D output.")

    # Ensure mask is 3D (X, Y, Z) and boolean
    if mask.ndim == 4:
         print(f"  - WARNING: 4D mask detected (shape {mask.shape}). Using first volume.")
         mask = mask[..., 0]
    mask = mask.astype(bool)

    # Ensure mask and data dimensions match
    if data.shape[:3] != mask.shape[:3]:
         raise RuntimeError(f"Dimension mismatch between data {data.shape[:3]} and mask {mask.shape[:3]}. "
                            f"The data may have been resampled (e.g. to anatomical space) but the mask was not, "
                            f"possibly due to a cached output. Please check the coregistration/masking steps.")
        
    data_flat = data[mask]
    n_samples = data_flat.shape[0]
    
    if n_samples == 0:
        raise ValueError("No voxels found in the provided mask (mask is empty). Cannot perform fitting.") 

    # 2. Create Chunks
    # Use array_split to ensure we strictly partition the data
    chunks = np.array_split(data_flat, nthreads)
    
    # Prune empty chunks if any
    chunks = [c for c in chunks if c.shape[0] > 0]
    
    # 3. Prepare Args
    # (chunk_id, chunk_data, gtab, worker_func, worker_kwargs)
    chunk_args = []
    for i, c in enumerate(chunks):
        # We must pass the worker_func in the args now if we use a generic wrapper
        chunk_args.append((i, c, gtab, worker_func, worker_kwargs))

    # 4. Run Pool
    results = []
    
    # Use 'spawn' for safety against BLAS/Numba crashes
    try:
        ctx = multiprocessing.get_context('spawn')
    except ValueError:
        ctx = multiprocessing.get_context('fork')
        
    with ctx.Pool(processes=nthreads) as pool:
        # Use imap to guarantee order
        # Use the GLOBAL wrapper defined below/above
        iterator = pool.imap(_global_driver_wrapper, chunk_args)
        
        for i, res in enumerate(iterator):
            results.append(res)


    # 5. Reassemble
    if not results:
        raise RuntimeError("No results from parallel fitting.")
    
    # Concatenate along the first axis (samples)
    all_params = np.concatenate(results, axis=0)
    
    # 6. Map back to volume
    if all_params.ndim == 1:
        out_shape = mask.shape
    else:
        out_shape = mask.shape + all_params.shape[1:]
        
    vol_params = np.zeros(out_shape, dtype=all_params.dtype)
    vol_params[mask] = all_params
    
    return vol_params


def _serial_fit_driver(data, mask, gtab, worker_func, worker_kwargs=None):
    """Run a parallel-driver compatible worker in-process and reassemble output."""
    import numpy as np

    if worker_kwargs is None:
        worker_kwargs = {}

    if mask is None:
        mask = np.ones(data.shape[:3], dtype=bool)
    if data.ndim < 4:
        raise RuntimeError(f"Serial fit driver received {data.ndim}D data (shape: {data.shape}). "
                           f"Diffusion models require a 4D volume series.")
    if mask.ndim == 4:
        print(f"  - WARNING: 4D mask detected (shape {mask.shape}). Using first volume.")
        mask = mask[..., 0]
    mask = mask.astype(bool)
    if data.shape[:3] != mask.shape[:3]:
        raise RuntimeError(f"Dimension mismatch between data {data.shape[:3]} and mask {mask.shape[:3]}.")

    data_flat = data[mask]
    if data_flat.shape[0] == 0:
        raise ValueError("No voxels found in the provided mask (mask is empty). Cannot perform fitting.")

    all_params = worker_func(0, data_flat, gtab, worker_kwargs)
    if all_params.ndim == 1:
        out_shape = mask.shape
    else:
        out_shape = mask.shape + all_params.shape[1:]

    vol_params = np.zeros(out_shape, dtype=all_params.dtype)
    vol_params[mask] = all_params
    return vol_params


# --- Worker Functions ---


def _default_model_param_shape(model_name):
    param_sizes = {
        "TensorModel": (12,),
        "DiffusionKurtosisModel": (27,),
        "MeanDiffusionKurtosisModel": (2,),
        "FreeWaterTensorModel": (13,),
    }
    return param_sizes.get(model_name)


def _fit_model_params_with_voxel_fallback(model, data_chunk, chunk_id, model_name, *, reshape_4d=False):
    """Fit a chunk, falling back to per-voxel NaNs if isolated voxels fail."""
    import numpy as np
    model_class_name = model.__class__.__name__

    def _fit_params(voxels):
        fit_data = voxels
        if reshape_4d:
            fit_data = voxels.reshape(voxels.shape[0], 1, 1, voxels.shape[1])
        fit = model.fit(fit_data)
        params = np.asarray(fit.model_params)
        if params.shape[0] == voxels.shape[0]:
            return params.reshape(voxels.shape[0], -1)
        if voxels.shape[0] == 1:
            return params.reshape(1, -1)
        return params.squeeze()

    try:
        return _fit_params(data_chunk)
    except Exception:
        pass

    rows = []
    pending_failed = 0
    param_shape = None
    failed = 0

    for vox_data in data_chunk:
        try:
            if not np.all(np.isfinite(vox_data)) or np.all(vox_data <= 0):
                raise ValueError("invalid signal vector")

            params = np.asarray(_fit_params(vox_data[None, :]))
            if params.ndim > 1 and params.shape[0] == 1:
                params = params[0]
            params = np.squeeze(params)
            if param_shape is None:
                param_shape = params.shape
                rows.extend(np.full(param_shape, np.nan, dtype=np.float32) for _ in range(pending_failed))
                pending_failed = 0
            rows.append(params)
        except Exception:
            failed += 1
            if param_shape is None:
                pending_failed += 1
            else:
                rows.append(np.full(param_shape, np.nan, dtype=np.float32))

    if param_shape is None:
        param_shape = _default_model_param_shape(model_class_name)
        if param_shape is None:
            raise RuntimeError(f"{model_name} worker {chunk_id} could not fit any voxels in its chunk.")
        rows.extend(np.full(param_shape, np.nan, dtype=np.float32) for _ in range(pending_failed))

    if failed:
        print(f"  - WARNING: {model_name} worker {chunk_id} failed on {failed}/{data_chunk.shape[0]} voxels; writing NaN for those voxels.")

    return np.asarray(rows)


def _dti_worker(chunk_id, data_chunk, gtab, kwargs):
    import dipy.reconst.dti as dipy_dti
    fit_method = kwargs.get('fit_method', 'WLLS')
    
    # Filter kwargs
    fit_kwargs = kwargs.copy()
    fit_kwargs.pop('n_cpus', None)
    fit_kwargs.pop('nthreads', None)
    fit_kwargs.pop('smoothing_fwhm', None)
    fit_kwargs.pop('grad_nonlin', None)
    fit_kwargs.pop('sub_method', None)  # Fix for NLLS/WLLS not accepting sub_method
    
    # Handle Legacy defaults
    if fit_method != 'RESTORE' and 'return_leverages' not in fit_kwargs:
         fit_kwargs['return_leverages'] = True
         
    model = dipy_dti.TensorModel(gtab, **fit_kwargs)
        
    # Reshape to 4D to ensure DIPY handles it as a volume (avoiding 2D broadcasting issues)
    # data_chunk is (N, B)
    n_vox = data_chunk.shape[0]
    
    if data_chunk.ndim == 1:
        # Unexpected 1D data (e.g. from 4D mask on 4D data)
        # We assume n_vox is 1 and B is the whole length (or vice versa? usually B is the dim)
        # But if it's 1D, we can't reliably know. Better to raise descriptive error or wrap.
        raise ValueError(f"Worker received 1D data chunk (shape {data_chunk.shape}). Expected 2D (N, B). Check mask dimensions.")

    n_vols = data_chunk.shape[1]
    
    # Reshape to (N, 1, 1, B)
    data_4d = data_chunk.reshape(n_vox, 1, 1, n_vols)
    
    return _fit_model_params_with_voxel_fallback(
        model,
        data_chunk,
        chunk_id,
        "DTI",
        reshape_4d=True,
    )

def _dki_worker(chunk_id, data_chunk, gtab, kwargs):
    import dipy.reconst.dki as dipy_dki
    import dipy.reconst.msdki as dipy_msdki
    import warnings
    
    # Suppress RuntimeWarnings
    warnings.simplefilter("ignore", RuntimeWarning)
    
    # Check for MSDKI
    use_msdki = kwargs.pop('use_msdki', False)
    
    # Filter out kwargs that dipy models don't accept but might be passed by pipeline
    fit_kwargs = kwargs.copy()
    fit_kwargs.pop('n_cpus', None)
    fit_kwargs.pop('nthreads', None)
    fit_kwargs.pop('grad_nonlin', None) # GNL is handled by splitting, not passed to fit directly here
    fit_kwargs.pop('sub_method', None)
    fit_method = _normalize_dki_fit_method(fit_kwargs.pop('fit_method', 'WLLS'))
    if fit_kwargs.get('weights_method') is not None or fit_method == 'IRLS':
        fit_kwargs['return_S0_hat'] = True
        fit_kwargs.setdefault('min_signal', 1e-6)
        if fit_method == 'IRLS':
            fit_method = 'WLS'

    if use_msdki:
        model = dipy_msdki.MeanDiffusionKurtosisModel(gtab, fit_method=fit_method, **fit_kwargs)
    else:
        model = dipy_dki.DiffusionKurtosisModel(gtab, fit_method=fit_method, **fit_kwargs)
        
    # Reshape to 4D to ensure safe broadcasting
    n_vox = data_chunk.shape[0]
    n_vols = data_chunk.shape[1]
    data_4d = data_chunk.reshape(n_vox, 1, 1, n_vols)
    
    return _fit_model_params_with_voxel_fallback(
        model,
        data_chunk,
        chunk_id,
        "DKI",
        reshape_4d=True,
    )

def _mapmri_worker(chunk_id, data_chunk, gtab, kwargs):
    import dipy.reconst.mapmri as mapmri
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore", message=".*Solution may be inaccurate.*", category=UserWarning)
    
    fit_kwargs = kwargs.copy()
    fit_kwargs.pop('n_cpus', None)
    fit_kwargs.pop('nthreads', None)
    fit_kwargs.pop('smoothing_fwhm', None)
    fit_kwargs.pop('grad_nonlin', None)
    
    # Check if metrics requested
    metrics = fit_kwargs.pop('metrics', None)
    peak_npeaks = int(fit_kwargs.pop('peak_npeaks', 3))
    peak_relative_threshold = float(fit_kwargs.pop('peak_relative_threshold', 0.5))
    peak_min_separation_angle = float(fit_kwargs.pop('peak_min_separation_angle', 25.0))
    
    model = mapmri.MapmriModel(gtab, **fit_kwargs)
    
    if metrics:
        failed = 0
        rows = []
        for vox_data in data_chunk:
            try:
                if not np.all(np.isfinite(vox_data)) or np.all(vox_data <= 0):
                    raise ValueError("invalid MAPMRI signal vector")

                fit = model.fit(vox_data[None, :])
                row = []
                for m in metrics:
                    if m == 'rtop': val = fit.rtop()
                    elif m == 'rtap': val = fit.rtap()
                    elif m == 'rtpp': val = fit.rtpp()
                    elif m == 'qiv': val = fit.qiv()
                    elif m == 'msd': val = fit.msd()
                    elif m == 'ng': val = fit.ng()
                    elif m == 'peaks':
                        val = _mapmri_peaks_from_fit(
                            fit,
                            n_peaks=peak_npeaks,
                            relative_peak_threshold=peak_relative_threshold,
                            min_separation_angle=peak_min_separation_angle,
                        )[0]
                    elif m == 'ng_par': val = fit.ng_parallel()
                    elif m == 'ng_perp': val = fit.ng_perpendicular()
                    else: val = np.nan

                    val = np.asarray(val)
                    if val.size == 1:
                        row.append(val.item())
                    else:
                        row.append(val.reshape(-1))
            except Exception:
                failed += 1
                row = []
                for m in metrics:
                    if m == 'peaks':
                        row.append(np.full(peak_npeaks * 3, np.nan, dtype=np.float32))
                    else:
                        row.append(np.nan)

            if len(row) == 1:
                rows.append(row[0])
            else:
                rows.append(row)

        if failed:
            print(f"  - WARNING: MAPMRI worker {chunk_id} failed on {failed}/{data_chunk.shape[0]} voxels; writing NaN for those voxels.")
        return np.asarray(rows)

    fit = model.fit(data_chunk)
    
    if hasattr(fit, 'mapmri_params'):
        return fit.mapmri_params
    if hasattr(fit, 'mapmri_coeffs'):
        return fit.mapmri_coeffs
    if hasattr(fit, 'mapmri_coeff'):
        return fit.mapmri_coeff
    return fit.model_params


_MAPMRI_METRIC_ALIASES = {
    'rtop': 'rtop',
    'rtap': 'rtap',
    'rtpp': 'rtpp',
    'qiv': 'qiv',
    'msd': 'msd',
    'ng': 'ng',
    'peak': 'peaks',
    'peaks': 'peaks',
    'ng_par': 'ng_par',
    'ng_parallel': 'ng_par',
    'parng': 'ng_par',
    'ng_perp': 'ng_perp',
    'ng_perpendicular': 'ng_perp',
    'perng': 'ng_perp',
}


def _normalize_mapmri_metrics(metrics):
    normalized = []
    seen = set()
    for metric in metrics:
        canonical = _MAPMRI_METRIC_ALIASES.get(str(metric).strip().lower())
        if canonical and canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    return normalized


def _mapmri_metric_suffix(metric: str) -> str:
    canonical = _MAPMRI_METRIC_ALIASES.get(str(metric).strip().lower(), str(metric).strip().lower())
    if canonical == 'peaks':
        return 'PEAKS'
    if canonical == 'ng_par':
        return 'NG_PAR'
    if canonical == 'ng_perp':
        return 'NG_PERP'
    return canonical.upper()


def _mapmri_peaks_from_fit(
    fit,
    *,
    n_peaks: int = 3,
    relative_peak_threshold: float = 0.5,
    min_separation_angle: float = 25.0,
):
    import numpy as np
    from dipy.data import default_sphere
    from dipy.direction import peak_directions

    odf = np.asarray(fit.odf(default_sphere), dtype=np.float32)
    odf_flat = odf.reshape((-1, odf.shape[-1]))
    peaks = np.zeros((odf_flat.shape[0], n_peaks * 3), dtype=np.float32)

    for idx, voxel_odf in enumerate(odf_flat):
        directions, _, _ = peak_directions(
            voxel_odf,
            default_sphere,
            relative_peak_threshold=relative_peak_threshold,
            min_separation_angle=min_separation_angle,
            is_symmetric=True,
        )
        if directions is None or len(directions) == 0:
            continue
        directions = np.asarray(directions[:n_peaks], dtype=np.float32)
        peaks[idx, : directions.shape[0] * 3] = directions.reshape(-1)

    return peaks.reshape(odf.shape[:-1] + (n_peaks * 3,))

def _fwe_dti_worker(chunk_id, data_chunk, gtab, kwargs):
    import dipy.reconst.fwdti as fwdti
    fit_method = kwargs.get('fit_method', 'NLLS') # Default to NLLS for FWE
    
    # Filter out pipeline-specific kwargs
    fwe_kwargs = kwargs.copy()
    fwe_kwargs.pop('fit_method', None)
    fwe_kwargs.pop('n_cpus', None)
    fwe_kwargs.pop('nthreads', None)
    fwe_kwargs.pop('smoothing_fwhm', None)
    fwe_kwargs.pop('grad_nonlin', None)
    fwe_kwargs.pop('sub_method', None)
    
    model = fwdti.FreeWaterTensorModel(gtab, fit_method=fit_method, **fwe_kwargs)
    
    return _fit_model_params_with_voxel_fallback(
        model,
        data_chunk,
        chunk_id,
        "FWE-DTI",
    )

# --- Generic GNL Voxel-wise Driver ---

def _gnl_worker_func(chunk_id, chunk_data, _, kwargs):
    """
    Generic worker for voxel-wise fitting with varying gradient tables (GNL).
    """
    from dipy.core.gradients import gradient_table
    import numpy as np
    import warnings
    
    # Suppress RuntimeWarnings (e.g. overflow in exp) common in DKI fits on noisy data
    warnings.simplefilter("ignore", RuntimeWarning)
    # Suppress cvxpy warnings in GNL fits (MAPMRI)
    warnings.filterwarnings("ignore", message=".*Solution may be inaccurate.*", category=UserWarning)

    # Unpack kwargs
    gnl_chunk = kwargs['gnl_chunk'] # (N, ...)
    bvals = kwargs['bvals']
    bvecs = kwargs['bvecs'] # (N_gradients, 3)
    big_delta = kwargs.get('big_delta')
    small_delta = kwargs.get('small_delta')
    model_class = kwargs['model_class']
    # Copy model_kwargs so we can safely modify
    full_kwargs = kwargs.get('model_kwargs', {}).copy()
    metrics = full_kwargs.pop('metrics', None)
    peak_npeaks = int(full_kwargs.pop('peak_npeaks', 3))
    peak_relative_threshold = float(full_kwargs.pop('peak_relative_threshold', 0.5))
    peak_min_separation_angle = float(full_kwargs.pop('peak_min_separation_angle', 25.0))
    
    res_params = []
    pending_failed = 0
    param_shape = None
    failed = 0
    
    for i in range(chunk_data.shape[0]):
        vox_data = chunk_data[i]
        vox_gnl = gnl_chunk[i]
        
        # Reshape GNL tensor to 3x3 if needed
        # It handles flattened 9-element arrays or 3x3 matrices
        if vox_gnl.size == 9:
            rot_mat = vox_gnl.reshape(3, 3)
        elif vox_gnl.shape == (3,3):
            rot_mat = vox_gnl
        else:
            # Fallback or error assumption
            rot_mat = vox_gnl.reshape(3,3) 
            
        # Rotate bvecs: bvecs_new = bvecs @ R.T
        rot_bvecs = np.dot(bvecs, rot_mat.T)
        
        # Recalculate b-values and normalize b-vectors to satisfy Gtab requirements
        # Effective b-value scales with square of gradient amplitude change
        norms = np.linalg.norm(rot_bvecs, axis=1)
        new_bvals = bvals * (norms ** 2)
        
        # Normalize bvecs (handle zero norms safely, though unlikely for non-b0)
        safe_norms = norms.copy()
        safe_norms[safe_norms == 0] = 1.0
        new_bvecs = rot_bvecs / safe_norms[:, None]
        
        # Create new gradient table
        # Optimized: minimal check
        vox_gtab = gradient_table(new_bvals, bvecs=new_bvecs, big_delta=big_delta, small_delta=small_delta, b0_threshold=kwargs.get('b0_threshold', 50))
        
        # Instantiate model with compatibility guards for iterative DTI/DKI only.
        model_name = model_class.__name__
        supports_iterative_guards = model_name in {
            'TensorModel',
            'DiffusionKurtosisModel',
            'MeanDiffusionKurtosisModel',
        }
        if supports_iterative_guards and 'min_signal' not in full_kwargs:
             full_kwargs['min_signal'] = 1e-6
        if supports_iterative_guards and 'return_S0_hat' not in full_kwargs:
             full_kwargs['return_S0_hat'] = True

        model = model_class(vox_gtab, **full_kwargs)
        
        # Fit
        # Force 2D input (1, N_grads) to avoid indexing errors in some DIPY models (e.g. DKI iterative fit)
        # when processing single voxels.
        try:
            if model_name == 'MapmriModel' and (not np.all(np.isfinite(vox_data)) or np.all(vox_data <= 0)):
                raise ValueError("invalid MAPMRI signal vector")
            fit = model.fit(vox_data[None, :])
        except Exception:
            failed += 1
            if model_name != 'MapmriModel' and metrics:
                raise

            if metrics:
                failed_row = []
                for m in metrics:
                     if m == 'peaks':
                         failed_row.append(np.full(peak_npeaks * 3, np.nan, dtype=np.float32))
                     else:
                         failed_row.append(np.nan)

                if len(failed_row) == 1 and np.asarray(failed_row[0]).ndim > 0:
                    res_params.append(np.asarray(failed_row[0]))
                else:
                    res_params.append(failed_row)
            else:
                if param_shape is None:
                    pending_failed += 1
                else:
                    res_params.append(np.full(param_shape, np.nan, dtype=np.float32))
            continue
        
        # Check if Metrics requested
        # metrics extracted earlier

        
        if metrics:
            # We assume the fit object has methods corresponding to metric names
            m_res = []
            for m in metrics:
                 # Standardize lookup
                 if m == 'rtop' and hasattr(fit, 'rtop'): val = fit.rtop()
                 elif m == 'rtap' and hasattr(fit, 'rtap'): val = fit.rtap()
                 elif m == 'rtpp' and hasattr(fit, 'rtpp'): val = fit.rtpp()
                 elif m == 'qiv' and hasattr(fit, 'qiv'): val = fit.qiv()
                 elif m == 'msd' and hasattr(fit, 'msd'): val = fit.msd()
                 elif m == 'ng' and hasattr(fit, 'ng'): val = fit.ng()
                 elif m == 'peaks': val = _mapmri_peaks_from_fit(
                     fit,
                     n_peaks=peak_npeaks,
                     relative_peak_threshold=peak_relative_threshold,
                     min_separation_angle=peak_min_separation_angle,
                 )[0]
                 elif m == 'ng_par' and hasattr(fit, 'ng_parallel'): val = fit.ng_parallel()
                 elif m == 'ng_perp' and hasattr(fit, 'ng_perpendicular'): val = fit.ng_perpendicular()
                 elif hasattr(fit, m): val = getattr(fit, m) # generic property?
                 else: val = 0.0 # Nan?
                 
                 # Unwrap if array (since we passed batch of 1)
                 if isinstance(val, (np.ndarray, list)) and np.size(val) == 1:
                     val = np.array(val).item()
                 
                 m_res.append(val)
            if len(m_res) == 1 and np.asarray(m_res[0]).ndim > 0:
                res_params.append(np.asarray(m_res[0]))
            else:
                res_params.append(m_res)
        else:
            # Collect parameters
            # Most models stick params in model_params
            if hasattr(fit, 'mapmri_params'):
                 res = fit.mapmri_params
            elif hasattr(fit, 'mapmri_coeffs'):
                 res = fit.mapmri_coeffs
            elif hasattr(fit, 'mapmri_coeff'):
                 res = fit.mapmri_coeff
            elif hasattr(fit, 'model_params'):
                 res = fit.model_params
            else:
                 # Fallback
                 raise AttributeError(f"Fit object {type(fit)} has neither 'model_params' nor 'mapmri_params' nor 'mapmri_coeff'.")
            
            # Unwrap (1, P) -> (P,)
            if hasattr(res, 'ndim') and res.ndim > 1 and res.shape[0] == 1:
                res = res[0]

            res = np.asarray(res)
            if param_shape is None:
                param_shape = res.shape
                res_params.extend(np.full(param_shape, np.nan, dtype=np.float32) for _ in range(pending_failed))
                pending_failed = 0
            res_params.append(res)

    if pending_failed:
        param_shape = _default_model_param_shape(model_class.__name__)
        if param_shape is None:
            raise RuntimeError(f"{model_class.__name__} GNL worker {chunk_id} could not fit any voxels in its chunk.")
        res_params.extend(np.full(param_shape, np.nan, dtype=np.float32) for _ in range(pending_failed))
    if failed:
        print(f"  - WARNING: {model_class.__name__} GNL worker {chunk_id} failed on {failed}/{chunk_data.shape[0]} voxels; writing NaN for those voxels.")

    return np.array(res_params)

def _execute_gnl_fit(data, mask, gnl_map_path, bvals, bvecs, model_class, model_kwargs, nthreads=1, big_delta=None, small_delta=None):
    """
    Driver to execute GNL-corrected fitting.
    """
    import numpy as np
    import nibabel as nib
    import multiprocessing
    
    # 1. Load Map
    gnl_img = nib.load(str(gnl_map_path))
    gnl_data = gnl_img.get_fdata()

    if gnl_data.shape[:3] != data.shape[:3]:
        raise RuntimeError(f"GNL map dimensions {gnl_data.shape} do not match data {data.shape}")

    # 2. Flatten data
    if mask is None:
        mask = np.ones(data.shape[:3], dtype=bool)
        
    data_flat = data[mask]
    gnl_flat = gnl_data[mask]
    
    n_samples = data_flat.shape[0]
    if n_samples == 0:
        raise ValueError("No voxels in mask.")
        
    # 3. Parallelize
    if nthreads > 1:
        chunks_data = np.array_split(data_flat, nthreads)
        chunks_gnl = np.array_split(gnl_flat, nthreads)
        
        pool_args = []
        for i in range(nthreads):
            if chunks_data[i].size == 0: continue
            kw = {
                'gnl_chunk': chunks_gnl[i],
                'bvals': bvals,
                'bvecs': bvecs,
                'big_delta': big_delta,
                'small_delta': small_delta,
                'model_class': model_class,
                'model_kwargs': model_kwargs,
                'b0_threshold': model_kwargs.get('b0_threshold', 50)
            }
            # Use global generic wrapper
            pool_args.append((i, chunks_data[i], None, _gnl_worker_func, kw))
            
        try:
            ctx = multiprocessing.get_context('spawn')
        except ValueError:
            ctx = multiprocessing.get_context('fork')
            
        with ctx.Pool(processes=nthreads) as pool:
            res_list = pool.map(_global_driver_wrapper, pool_args)
            
        all_params = np.concatenate(res_list, axis=0) if res_list else np.array([])
        
    else:
        # Serial
        kw = {
            'gnl_chunk': gnl_flat,
            'bvals': bvals,
            'bvecs': bvecs,
            'big_delta': big_delta,
            'small_delta': small_delta,
            'model_class': model_class,
            'model_kwargs': model_kwargs,
            'b0_threshold': model_kwargs.get('b0_threshold', 50)
        }
        all_params = _gnl_worker_func(0, data_flat, None, kw)
        
    # 4. Map back to volume
    if all_params.size > 0:
        param_shape = all_params.shape[1:]
        out_shape = data.shape[:3] + param_shape
        vol_params = np.zeros(out_shape, dtype=all_params.dtype)
        vol_params[mask] = all_params
        return vol_params
    else:
        raise RuntimeError("GNL Fit resulted in empty parameters.")

def _resolve_iterative_params(fit_method, kwargs):
    """
    Resolve parameters for iterative fitting (IRLS).
    Ensures 'weights_method' is a callable and 'fit_type' is set.
    """
    if fit_method not in ['IRLS']:
        return

    # Extract optional cutoff (z-score threshold) - remove from kwargs as Model init doesn't take it
    cutoff = kwargs.pop('weights_cutoff', None)

    # 1. Resolve weights_method
    weights_method = kwargs.get('weights_method')
    
    try:
        import dipy.reconst.weights_method as wm
        from functools import partial
        
        # If using IRLS, weights_method is mandatory. Default if missing.
        if weights_method is None:
            kwargs['weights_method'] = wm.weights_method_wls_m_est
            
        elif isinstance(weights_method, str):
            w_str = weights_method.lower()
            if w_str in ['gm', 'geman-mcclure']:
                 kwargs['weights_method'] = partial(wm.weights_method_wls_m_est, m_est='gm')
            elif w_str == 'cauchy':
                 kwargs['weights_method'] = partial(wm.weights_method_wls_m_est, m_est='cauchy')
            elif w_str in ['wls_m_est', 'ols', 'default']:
                 kwargs['weights_method'] = wm.weights_method_wls_m_est
            elif w_str == 'nlls_m_est':
                 kwargs['weights_method'] = wm.weights_method_nlls_m_est
            else:
                 # Check if attribute exists
                 if hasattr(wm, weights_method):
                      kwargs['weights_method'] = getattr(wm, weights_method)
                 else:
                      print(f"  - WARNING: Unknown weights_method string '{weights_method}'. defaulting to wls_m_est.")
                      kwargs['weights_method'] = wm.weights_method_wls_m_est
        
        # Apply Custom Cutoff if provided
        if cutoff is not None and 'weights_method' in kwargs:
             # Wrap existing callable (partial or func) with new cutoff
             kwargs['weights_method'] = partial(kwargs['weights_method'], cutoff=float(cutoff))
             
    except ImportError:
        print("  - WARNING: Could not import dipy.reconst.weights_method to resolve string.")

    # 2. Resolve fit_type
    # iterative_fit_tensor requires fit_type ('WLS' or 'NLLS')
    if 'fit_type' not in kwargs:
         # Default to WLS as it aligns with default weights_method_wls_m_est
         kwargs['fit_type'] = 'WLS'


def _normalize_dki_fit_method(fit_method: str) -> str:
    """
    Map pipeline-friendly DKI fit labels to the subset accepted by DIPY's
    DiffusionKurtosisModel constructor.
    """
    normalized = str(fit_method or "WLLS").strip().upper()
    aliases = {
        "WLLS": "WLS",
        "UWLLS": "WLS",
        "WLS": "WLS",
        "OLS": "OLS",
        "ULLS": "OLS",
        "LS": "OLS",
        "LLS": "OLS",
        # DIPY's DKI constructor does not expose NLLS directly. If the caller
        # requests iterative robust fitting, weights_method drives that path.
        "NLLS": "WLS",
        "IRLS": "WLS",
        "CLS": "CLS",
        "CWLS": "CWLS",
    }
    return aliases.get(normalized, normalized)

def fit_dti(
    in_file: Union[Path, ImageLike],
    out_dir: Path,
    bval_file: Optional[Path] = None,
    bvec_file: Optional[Path] = None,
    mask_file: Optional[Path] = None,
    fit_method: str = "WLLS",
    metrics: list[str] = ["fa", "md", "ad", "rd", "color_fa", "evals", "evecs"],
    nthreads: int = 1,
    Delta_file: Optional[Path] = None,
    delta_file: Optional[Path] = None,
    **kwargs
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
    Delta_file: Path, optional
        Path to Big Delta timings.
    delta_file: Path, optional
        Path to small delta timings.
    """
    import numpy as np
    import multiprocessing
    import nibabel as nib
    import dipy.reconst.dti as dipy_dti
    from dipy.core.gradients import gradient_table
    from dipy.io.gradients import read_bvals_bvecs
    from qmri_neuropipe.io.bids import build_bids_name, get_entities_from_path

    # Resolve iterative parameters if needed
    _resolve_iterative_params(fit_method, kwargs)

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
    big_delta, small_delta = _load_timings(Delta_file, delta_file)
    gtab = gradient_table(bvals, bvecs=bvecs, big_delta=big_delta, small_delta=small_delta, b0_threshold=kwargs.get('b0_threshold', 50))
    if big_delta is not None and small_delta is not None:
         print(f"  - Using custom diffusion times (Delta/delta)")

    # Handle optional smoothing
    smoothing_fwhm = kwargs.pop('smoothing_fwhm', None)
    if smoothing_fwhm:
        import scipy.ndimage
        # FWHM = 2.355 * sigma
        sigma = float(smoothing_fwhm) / 2.3548200450309493
        print(f"  - Applying Gaussian smoothing (FWHM={smoothing_fwhm}mm, sigma={sigma:.2f})")
        # Apply smoothing spatially (axes 0,1,2), independent over volumes (axis 3)
        # Using 4D sigma with 0 on 4th dimension achieves this efficiently
        scipy.ndimage.gaussian_filter(data, sigma=[sigma, sigma, sigma, 0], output=data)

    # Read mask
    if mask_file and mask_file.exists():
        mask = nib.load(str(mask_file)).get_fdata().astype(bool)
    else:
        # Create full mask if none provided
        mask = np.ones(data.shape[:3], dtype=bool)

    # Initialize Model for metadata or serial fallback
    dti_kwargs = kwargs.copy()
    # Remove pipeline-only keys before creating DIPY model; these may be
    # forwarded into low-level fit methods and trigger unexpected kwarg errors.
    dti_kwargs.pop('grad_nonlin', None)
    dti_kwargs.pop('nthreads', None)
    dti_kwargs.pop('n_cpus', None)
    dti_kwargs.pop('smoothing_fwhm', None)
    dti_kwargs.pop('sub_method', None)
    
    if fit_method != 'RESTORE' and 'return_leverages' not in dti_kwargs:
         dti_kwargs['return_leverages'] = True

    dti_model = dipy_dti.TensorModel(gtab, fit_method=fit_method, **dti_kwargs)

    # Convert grad_nonlin path to Path if str
    if kwargs.get('grad_nonlin'):
         grad_nonlin = Path(kwargs['grad_nonlin'])
    else:
         grad_nonlin = None

    # Fit
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            
            if grad_nonlin:
                 # GNL Correction Voxel-wise Fit
                 
                 # Prepare model kwargs
                 model_kwargs = {
                     'fit_method': fit_method
                 }
                 if fit_method != 'RESTORE':
                     model_kwargs['return_leverages'] = True
                 else:
                     model_kwargs['sigma'] = None # or None
                     
                 vol_params = _execute_gnl_fit(
                    data=data,
                    mask=mask,
                    gnl_map_path=grad_nonlin,
                    bvals=bvals,
                    bvecs=bvecs,
                    model_class=dipy_dti.TensorModel,
                    model_kwargs=model_kwargs,
                    nthreads=nthreads,
                    big_delta=big_delta,
                    small_delta=small_delta
                 )
                 
                 dti_fit = dipy_dti.TensorFit(dti_model, vol_params)
    
            elif nthreads > 1:
                # Parallel Fit using Unified Driver
                worker_kwargs = kwargs.copy()
                worker_kwargs['fit_method'] = fit_method
                
                if fit_method != 'RESTORE' and 'return_leverages' not in worker_kwargs:
                     worker_kwargs['return_leverages'] = True
                     
                vol_params = _parallel_fit_driver(
                    data, 
                    mask, 
                    gtab, 
                    _dti_worker, 
                    nthreads, 
                    worker_kwargs=worker_kwargs
                )
                
                dti_fit = dipy_dti.TensorFit(dti_model, vol_params)
    
            else:
                 worker_kwargs = kwargs.copy()
                 worker_kwargs['fit_method'] = fit_method
                 if fit_method != 'RESTORE' and 'return_leverages' not in worker_kwargs:
                     worker_kwargs['return_leverages'] = True
                 vol_params = _serial_fit_driver(
                    data,
                    mask,
                    gtab,
                    _dti_worker,
                    worker_kwargs=worker_kwargs
                 )
                 dti_fit = dipy_dti.TensorFit(dti_model, vol_params)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"DTI fitting failed: {e}") from e

    
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

    def _write_sidecar(path: Path, output_metric: str, extras: Optional[Dict[str, Any]] = None) -> None:
        payload = dict(sidecar)
        payload["OutputMetric"] = output_metric
        if extras:
            payload.update(extras)
        sidecar_path = str(path).replace('.nii.gz', '.json')
        with open(sidecar_path, 'w') as f:
            json.dump(payload, f, indent=4)

    tensor_matrices = np.asarray(dti_fit.quadratic_form, dtype=np.float32)
    tensor_matrices_world = _reorient_tensor_matrices_to_world(tensor_matrices, img.affine).astype(np.float32)
    evecs_world = _reorient_eigenvectors_to_world(np.asarray(dti_fit.evecs, dtype=np.float32), img.affine).astype(np.float32)
    evals = np.asarray(dti_fit.evals, dtype=np.float32)
    metrics_norm = {metric.strip().lower() for metric in metrics}
    
    for metric in metrics:
        m_norm = metric.strip().lower()
        metric_suffix = m_norm.upper()
        if m_norm == 'color_fa': metric_suffix = 'DECFA'
        
        # Build path
        out_name = build_bids_name({**ent_base, 'suffix': metric_suffix})
        out_path = out_dir / out_name
        
        saved = False
        if m_norm == 'fa':
            nib.save(nib.Nifti1Image(dti_fit.fa, img.affine), str(out_path))
            saved = True
        elif m_norm == 'md':
            nib.save(nib.Nifti1Image(dti_fit.md, img.affine), str(out_path))
            saved = True
        elif m_norm == 'ad':
            nib.save(nib.Nifti1Image(dti_fit.ad, img.affine), str(out_path))
            saved = True
        elif m_norm == 'rd':
            nib.save(nib.Nifti1Image(dti_fit.rd, img.affine), str(out_path))
            saved = True
        elif m_norm == 'color_fa':
            nib.save(nib.Nifti1Image(dti_fit.color_fa, img.affine), str(out_path))
            saved = True
        elif m_norm == 'evals':
             nib.save(nib.Nifti1Image(evals, img.affine), str(out_path))
             saved = True
        elif m_norm == 'evecs':
             nib.save(nib.Nifti1Image(evecs_world, img.affine), str(out_path))
             saved = True
        elif m_norm == 'l1':
             nib.save(nib.Nifti1Image(evals[..., 0], img.affine), str(out_path))
             saved = True
        elif m_norm == 'l2':
             nib.save(nib.Nifti1Image(evals[..., 1], img.affine), str(out_path))
             saved = True
        elif m_norm == 'l3':
             nib.save(nib.Nifti1Image(evals[..., 2], img.affine), str(out_path))
             saved = True
        elif m_norm == 'v1':
             nib.save(nib.Nifti1Image(evecs_world[..., 0, :], img.affine), str(out_path))
             saved = True
        elif m_norm == 'v2':
             nib.save(nib.Nifti1Image(evecs_world[..., 1, :], img.affine), str(out_path))
             saved = True
        elif m_norm == 'v3':
             nib.save(nib.Nifti1Image(evecs_world[..., 2, :], img.affine), str(out_path))
             saved = True
             
        if saved:
             output_files[m_norm] = out_path
             _write_sidecar(
                 out_path,
                 metric_suffix,
                 extras={"VectorConvention": "world"} if m_norm in {'evecs', 'v1', 'v2', 'v3'} else None,
             )
        elif m_norm not in ['tensor', 'tensor_fsl', 'tensor_mrtrix']:
             # Tensors are handled separately below, don't warn for them
             print(f"Warning: Unknown or unhandled DTI metric requested: {metric}")
            
    # Handle explicit tensor outputs if requested
    tensor_fsl = _tensor_matrices_to_fsl_components(tensor_matrices)
    tensor_mrtrix = _tensor_matrices_to_mrtrix_components(tensor_matrices_world)

    if "tensor" in metrics_norm:
        out_path = out_dir / build_bids_name({**ent_base, 'suffix': 'tensor'})
        nib.save(nib.Nifti1Image(tensor_fsl, img.affine), str(out_path))
        output_files['tensor'] = out_path
        _write_sidecar(out_path, "tensor", extras={"TensorConvention": "FSL", "TensorBasis": "voxel"})

    if "tensor_fsl" in metrics_norm:
        out_path = out_dir / build_bids_name({**ent_base, 'suffix': 'tensorFSL'})
        nib.save(nib.Nifti1Image(tensor_fsl, img.affine), str(out_path))
        output_files['tensor_fsl'] = out_path
        _write_sidecar(out_path, "tensorFSL", extras={"TensorConvention": "FSL", "TensorBasis": "voxel"})

    if "tensor_mrtrix" in metrics_norm:
        out_path = out_dir / build_bids_name({**ent_base, 'suffix': 'tensorMRTRIX'})
        nib.save(nib.Nifti1Image(tensor_mrtrix, img.affine), str(out_path))
        output_files['tensor_mrtrix'] = out_path
        _write_sidecar(out_path, "tensorMRTRIX", extras={"TensorConvention": "MRtrix", "TensorBasis": "world"})

    return output_files


def fit_dki(
    in_file: Union[Path, ImageLike],
    out_dir: Path,
    bval_file: Optional[Path] = None,
    bvec_file: Optional[Path] = None,
    mask_file: Optional[Path] = None,
    fit_method: str = "WLLS",
    metrics: list[str] = ["mk", "ak", "rk", "fa", "md"],
    nthreads: int = 1,
    Delta_file: Optional[Path] = None,
    delta_file: Optional[Path] = None,
    **kwargs
) -> Dict[str, Path]:
    """
    Fit Diffusion Kurtosis Imaging (DKI) model.
    """
    from dipy.core.gradients import gradient_table
    from dipy.io.gradients import read_bvals_bvecs
    import dipy.reconst.dki as dipy_dki
    from qmri_neuropipe.io.bids import build_bids_name, get_entities_from_path

    # Resolve iterative parameters if needed
    _resolve_iterative_params(fit_method, kwargs)
    dki_fit_method = _normalize_dki_fit_method(fit_method)

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
    big_delta, small_delta = _load_timings(Delta_file, delta_file)
    gtab = gradient_table(bvals, bvecs=bvecs, big_delta=big_delta, small_delta=small_delta, b0_threshold=kwargs.get('b0_threshold', 50))
    if big_delta is not None and small_delta is not None:
         print(f"  - Using custom diffusion times (Delta/delta)")
    
    # Handle optional smoothing
    smoothing_fwhm = kwargs.pop('smoothing_fwhm', None)
    if smoothing_fwhm:
        import scipy.ndimage
        # FWHM = 2.355 * sigma
        sigma = float(smoothing_fwhm) / 2.3548200450309493
        print(f"  - Applying Gaussian smoothing (FWHM={smoothing_fwhm}mm, sigma={sigma:.2f})")
        # Apply smoothing spatially (axes 0,1,2), independent over volumes (axis 3)
        # Using 4D sigma with 0 on 4th dimension achieves this efficiently
        scipy.ndimage.gaussian_filter(data, sigma=[sigma, sigma, sigma, 0], output=data)

    if mask_file and mask_file.exists():
        mask = nib.load(str(mask_file)).get_fdata().astype(bool)
    else:
        mask = None
        
    # Check for MSDKI flag
    use_msdki = kwargs.pop('mean_signal', False)
    
    if use_msdki:
        import dipy.reconst.msdki as dipy_msdki
        ModelClass = dipy_msdki.MeanDiffusionKurtosisModel
        FitClass = dipy_msdki.MeanDiffusionKurtosisFit
    else:
        ModelClass = dipy_dki.DiffusionKurtosisModel
        FitClass = dipy_dki.DiffusionKurtosisFit
        
    # Build clean kwargs for direct DIPY model initialization
    dki_kwargs = kwargs.copy()
    dki_kwargs.pop('n_cpus', None)
    dki_kwargs.pop('nthreads', None)
    dki_kwargs.pop('smoothing_fwhm', None)
    dki_kwargs.pop('grad_nonlin', None)
    dki_kwargs.pop('sub_method', None)
    dki_kwargs['fit_method'] = dki_fit_method
    if dki_kwargs.get('weights_method') is not None or str(fit_method).strip().upper() == 'IRLS':
        dki_kwargs['return_S0_hat'] = True
        dki_kwargs.setdefault('min_signal', 1e-6)

    dkimodel = ModelClass(gtab, **dki_kwargs)
    
    # Fit
    try:
        # Check for GNL
        if kwargs.get('grad_nonlin'):
            grad_nonlin = Path(kwargs['grad_nonlin'])
            # Clean kwargs for GNL fit function if needed, but it passes model_kwargs to init
            # _execute_gnl_fit splits data then calls model_class(gtab, **model_kwargs).fit(data)
            
            # For GNL fit, we must pass the correct class.
            
            # We need to make sure 'grad_nonlin' key is passed? 
            # Actually, `_execute_gnl_fit` expects `model_kwargs` to be clean enough for initialization.
            # But we popped 'mean_signal' and 'smoothing_fwhm' so they are gone.
            # We added 'use_msdki'. 
            # _dki_worker handles 'use_msdki' but `_execute_gnl_fit` uses `voxel_fit` directly usually?
            # Wait, `_execute_gnl_fit` rotates bvecs and runs.
            # If `_execute_gnl_fit` instantiates `ModelClass`, does it handle `use_msdki`?
            # No, `ModelClass` (DIPY model) doesn't know `use_msdki`.
            # So `model_kwargs` passed to `_execute_gnl_fit` MUST NOT have `use_msdki` if passing `MeanDiffusionKurtosisModel` explicitly.
            
            gnl_kwargs = dki_kwargs.copy()
            gnl_kwargs.pop('grad_nonlin', None)
            gnl_kwargs = {k:v for k,v in gnl_kwargs.items() if k not in ['n_cpus', 'nthreads']}

            vol_params = _execute_gnl_fit(
                data=data,
                mask=mask,
                gnl_map_path=grad_nonlin,
                bvals=bvals,
                bvecs=bvecs,
                model_class=ModelClass,
                model_kwargs=gnl_kwargs, 
                nthreads=nthreads,
                big_delta=big_delta,
                small_delta=small_delta
            )
            dkifit = FitClass(dkimodel, vol_params)
            
        elif nthreads > 1:
            # Parallel Fit
            worker_kwargs = dki_kwargs.copy()
            worker_kwargs['use_msdki'] = use_msdki
            vol_params = _parallel_fit_driver(
                data, 
                mask, 
                gtab, 
                _dki_worker, 
                nthreads, 
                worker_kwargs=worker_kwargs
            )
            
            dkifit = FitClass(dkimodel, vol_params)

        else:
            # Serial Fit
            worker_kwargs = dki_kwargs.copy()
            worker_kwargs['use_msdki'] = use_msdki
            vol_params = _serial_fit_driver(
                data,
                mask,
                gtab,
                _dki_worker,
                worker_kwargs=worker_kwargs
            )
            dkifit = FitClass(dkimodel, vol_params)
            
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
        "FittingMethod": fit_method,
        "ResolvedFittingMethod": dki_fit_method,
        "Metrics": metrics
    }

    for metric in metrics:
        m_norm = metric.strip().lower()
        metric_suffix = m_norm.upper()
        
        # Create output path
        out_name = build_bids_name({**ent_base, 'suffix': metric_suffix})
        out_path = out_dir / out_name
        
        saved = False
        if m_norm == 'mk':
            nib.save(nib.Nifti1Image(dkifit.mk(), img.affine), str(out_path))
            saved = True
        elif m_norm == 'ak':
            nib.save(nib.Nifti1Image(dkifit.ak(), img.affine), str(out_path))
            saved = True
        elif m_norm == 'rk':
            nib.save(nib.Nifti1Image(dkifit.rk(), img.affine), str(out_path))
            saved = True
        elif m_norm == 'fa':
            nib.save(nib.Nifti1Image(dkifit.fa, img.affine), str(out_path))
            saved = True
        elif m_norm == 'md':
            nib.save(nib.Nifti1Image(dkifit.md, img.affine), str(out_path))
            saved = True
        elif m_norm == 'ad':
            nib.save(nib.Nifti1Image(dkifit.ad, img.affine), str(out_path))
            saved = True
        elif m_norm == 'rd':
            nib.save(nib.Nifti1Image(dkifit.rd, img.affine), str(out_path))
            saved = True
        elif m_norm == 'kfa':
            nib.save(nib.Nifti1Image(dkifit.kfa, img.affine), str(out_path))
            saved = True
            
        if saved:
             output_files[m_norm] = out_path
        else:
             print(f"Warning: Unknown DKI metric requested: {metric}")
        
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
    metrics: list[str] = ["rtop", "rtap", "rtpp", "qiv", "msd", "ng"],
    nthreads: int = 1,
    Delta_file: Optional[Path] = None,
    delta_file: Optional[Path] = None,
    **kwargs
) -> Dict[str, Path]:
    """
    Fit MAP-MRI model.
    """
    import numpy as np
    from dipy.core.gradients import gradient_table
    from dipy.io.gradients import read_bvals_bvecs
    import dipy.reconst.mapmri as mapmri
    from qmri_neuropipe.io.bids import build_bids_name, get_entities_from_path

    in_path = extract_image_path(in_file)
    img = nib.load(str(in_path))
    data = img.get_fdata()
    metrics = _normalize_mapmri_metrics(metrics)
    if not metrics:
        metrics = ["rtop", "rtap", "rtpp", "qiv", "msd", "ng"]
    scalar_metrics = [metric for metric in metrics if metric != 'peaks']
    peaks_requested = 'peaks' in metrics
    
    if bval_file is None or bvec_file is None:
        if isinstance(in_file, DWIFile):
            bval_file = bval_file or in_file.bval
            bvec_file = bvec_file or in_file.bvec
            
    if not bval_file or not bvec_file:
         raise ValueError("Gradient files (bval/bvec) are required but not provided or found in input DWIFile.")

    bvals, bvecs = read_bvals_bvecs(str(bval_file), str(bvec_file))
    big_delta, small_delta = _load_timings(Delta_file, delta_file)
    gtab = gradient_table(bvals, bvecs=bvecs, big_delta=big_delta, small_delta=small_delta, b0_threshold=kwargs.get('b0_threshold', 50))
    if big_delta is not None and small_delta is not None:
         print(f"  - Using custom diffusion times (Delta/delta)")

    # Handle optional smoothing
    smoothing_fwhm = kwargs.pop('smoothing_fwhm', None)
    if smoothing_fwhm:
        import scipy.ndimage
        sigma = float(smoothing_fwhm) / 2.3548200450309493
        print(f"  - Applying Gaussian smoothing (FWHM={smoothing_fwhm}mm, sigma={sigma:.2f})")
        scipy.ndimage.gaussian_filter(data, sigma=[sigma, sigma, sigma, 0], output=data)

    if mask_file and mask_file.exists():
        mask = nib.load(str(mask_file)).get_fdata().astype(bool)
    else:
        mask = None
    
    # Prepare MapmriModel kwargs
    map_kwargs = kwargs.copy()
    # Map explicit function args to model init args
    map_kwargs.setdefault('laplacian_regularization', laplacian)
    map_kwargs.setdefault('positivity_constraint', positivity)
    map_kwargs.setdefault('global_constraints', global_constraints)
    peak_npeaks = int(map_kwargs.pop('peak_npeaks', 3))
    peak_relative_threshold = float(map_kwargs.pop('peak_relative_threshold', 0.5))
    peak_min_separation_angle = float(map_kwargs.pop('peak_min_separation_angle', 25.0))
    
    # Extract GNL path and remove from kwargs passed to Model
    if 'grad_nonlin' in map_kwargs:
        grad_nonlin_path = map_kwargs.pop('grad_nonlin')
        if grad_nonlin_path:
             grad_nonlin = Path(grad_nonlin_path)
        else:
             grad_nonlin = None
    else:
        grad_nonlin = None

    final_data = None
    mapfit = None
    peak_data = None
    map_model_kwargs = dict(map_kwargs)

    try:
        if scalar_metrics:
            scalar_kwargs = dict(map_model_kwargs)
            scalar_kwargs['metrics'] = scalar_metrics

            if grad_nonlin:
                 print(f"  - applying Gradient Nonlinearity Correction (voxel-wise)...")
                 
                 final_data = _execute_gnl_fit(
                    data=data,
                    mask=mask,
                    gnl_map_path=grad_nonlin,
                    bvals=bvals,
                    bvecs=bvecs,
                    model_class=mapmri.MapmriModel,
                    model_kwargs=scalar_kwargs,
                    nthreads=nthreads,
                    big_delta=big_delta,
                    small_delta=small_delta
                 )
            elif nthreads > 1:
                 final_data = _parallel_fit_driver(
                    data,
                    mask,
                    gtab,
                    _mapmri_worker,
                    nthreads,
                    worker_kwargs=scalar_kwargs
                 )
            else:
                 final_data = _serial_fit_driver(
                    data,
                    mask,
                    gtab,
                    _mapmri_worker,
                    worker_kwargs=scalar_kwargs
                 )

        if peaks_requested:
            peak_kwargs = dict(map_model_kwargs)
            peak_kwargs['metrics'] = ['peaks']
            peak_kwargs['peak_npeaks'] = peak_npeaks
            peak_kwargs['peak_relative_threshold'] = peak_relative_threshold
            peak_kwargs['peak_min_separation_angle'] = peak_min_separation_angle

            if grad_nonlin:
                peak_data = _execute_gnl_fit(
                    data=data,
                    mask=mask,
                    gnl_map_path=grad_nonlin,
                    bvals=bvals,
                    bvecs=bvecs,
                    model_class=mapmri.MapmriModel,
                    model_kwargs=peak_kwargs,
                    nthreads=nthreads,
                    big_delta=big_delta,
                    small_delta=small_delta
                )
            elif nthreads > 1:
                peak_data = _parallel_fit_driver(
                    data,
                    mask,
                    gtab,
                    _mapmri_worker,
                    nthreads,
                    worker_kwargs=peak_kwargs
                )
            else:
                peak_data = _serial_fit_driver(
                    data,
                    mask,
                    gtab,
                    _mapmri_worker,
                    worker_kwargs=peak_kwargs
                )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"MAPMRI fitting failed: {e}") from e
    
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
             "global_constraints": global_constraints,
             "peak_npeaks": peak_npeaks,
             "peak_relative_threshold": peak_relative_threshold,
             "peak_min_separation_angle": peak_min_separation_angle,
        },
        "Metrics": metrics
    }
    
    # Handling Outputs
    for i, metric in enumerate(scalar_metrics):
        metric_lower = _MAPMRI_METRIC_ALIASES.get(metric.lower(), metric.lower())
        metric_suffix = _mapmri_metric_suffix(metric_lower)
        out_name = build_bids_name({**ent_base, 'suffix': metric_suffix})
        out_path = out_dir / out_name
        
        val = None
        if final_data is not None:
             # Data is in final_data (vol_params)
             if final_data.ndim == 4:
                  val = final_data[..., i]
             else:
                  # Should not happen with current driver but for safety
                  val = final_data
        else:
             # Using MapmriFit object methods (Serial path)
             if metric_lower == 'rtop': val = mapfit.rtop()
             elif metric_lower == 'rtap': val = mapfit.rtap()
             elif metric_lower == 'rtpp': val = mapfit.rtpp()
             elif metric_lower == 'qiv': val = mapfit.qiv()
             elif metric_lower == 'msd': val = mapfit.msd()
             elif metric_lower == 'ng': val = mapfit.ng()
             elif metric_lower == 'ng_par': val = mapfit.ng_parallel()
             elif metric_lower == 'ng_perp': val = mapfit.ng_perpendicular()
        
        if val is not None:
             nib.save(nib.Nifti1Image(val, img.affine), str(out_path))
             output_files[metric_lower] = out_path
             
             sidecar_path = str(out_path).replace('.nii.gz', '.json')
             with open(sidecar_path, 'w') as f:
                  json.dump(sidecar, f, indent=4)

    if peaks_requested and peak_data is not None:
         peak_suffix = _mapmri_metric_suffix('peaks')
         peak_name = build_bids_name({**ent_base, 'suffix': peak_suffix})
         peak_path = out_dir / peak_name
         nib.save(nib.Nifti1Image(np.asarray(peak_data, dtype=np.float32), img.affine), str(peak_path))
         output_files['peaks'] = peak_path

         peak_sidecar = dict(sidecar)
         peak_sidecar["OutputMetric"] = "peaks"
         peak_sidecar["PeakComponents"] = [
             "peak1_x", "peak1_y", "peak1_z",
             "peak2_x", "peak2_y", "peak2_z",
             "peak3_x", "peak3_y", "peak3_z",
         ][:peak_npeaks * 3]
         with open(str(peak_path).replace('.nii.gz', '.json'), 'w') as f:
              json.dump(peak_sidecar, f, indent=4)

    return output_files


def fit_fwe_dti(
    in_file: Union[Path, ImageLike],
    out_dir: Path,
    bval_file: Optional[Path] = None,
    bvec_file: Optional[Path] = None,
    mask_file: Optional[Path] = None,
    fit_method: str = "NLLS",
    metrics: list[str] = ["fa", "md", "ad", "rd", "f"],
    nthreads: int = 1,
    **kwargs
) -> Dict[str, Path]:
    """
    Fit Free-Water Elimination DTI (FWE-DTI) model using DIPY.
    """
    from dipy.core.gradients import gradient_table
    from dipy.io.gradients import read_bvals_bvecs
    import dipy.reconst.fwdti as fwdti
    from qmri_neuropipe.io.bids import build_bids_name, get_entities_from_path

    in_path = extract_image_path(in_file)
    img = nib.load(str(in_path))
    data = img.get_fdata()
    
    if bval_file is None or bvec_file is None:
        if isinstance(in_file, DWIFile):
            bval_file = bval_file or in_file.bval
            bvec_file = bvec_file or in_file.bvec
            
    if not bval_file or not bvec_file:
         raise ValueError("Gradient files (bval/bvec) are required.")

    bvals, bvecs = read_bvals_bvecs(str(bval_file), str(bvec_file))
    gtab = gradient_table(bvals, bvecs=bvecs)

    if mask_file and mask_file.exists():
        mask = nib.load(str(mask_file)).get_fdata().astype(bool)
    else:
        mask = None
        
    fwe_kwargs = kwargs.copy()
    fwe_kwargs.pop('n_cpus', None)
    fwe_kwargs.pop('nthreads', None)
    fwe_kwargs.pop('smoothing_fwhm', None)
    fwe_kwargs.pop('grad_nonlin', None)
    fwe_kwargs.pop('sub_method', None)
        
    fwe_model = fwdti.FreeWaterTensorModel(gtab, fit_method=fit_method, **fwe_kwargs)
    
    try:
        if kwargs.get('grad_nonlin'):
            grad_nonlin = Path(kwargs['grad_nonlin'])
            # Reuse cleaned kwargs for voxel-wise model construction
            model_kwargs = fwe_kwargs.copy()
            model_kwargs['fit_method'] = fit_method
            
            vol_params = _execute_gnl_fit(
                data=data,
                mask=mask,
                gnl_map_path=grad_nonlin,
                bvals=bvals,
                bvecs=bvecs,
                model_class=fwdti.FreeWaterTensorModel,
                model_kwargs=model_kwargs,
                nthreads=nthreads
            )
            fwe_fit = fwdti.FreeWaterTensorFit(fwe_model, vol_params)
            
        elif nthreads > 1:
            worker_kwargs = fwe_kwargs.copy()
            worker_kwargs['fit_method'] = fit_method
            vol_params = _parallel_fit_driver(
                data, 
                mask, 
                gtab, 
                _fwe_dti_worker, 
                nthreads, 
                worker_kwargs=worker_kwargs
            )
            fwe_fit = fwdti.FreeWaterTensorFit(fwe_model, vol_params)
        else:
            worker_kwargs = fwe_kwargs.copy()
            worker_kwargs['fit_method'] = fit_method
            vol_params = _serial_fit_driver(
                data,
                mask,
                gtab,
                _fwe_dti_worker,
                worker_kwargs=worker_kwargs
            )
            fwe_fit = fwdti.FreeWaterTensorFit(fwe_model, vol_params)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"FWE-DTI fitting failed: {e}") from e
        
    # Save Outputs
    output_files = {}
    ent_base = get_entities_from_path(in_path)
    if 'desc' in ent_base: del ent_base['desc']
    ent_base['model'] = 'FWDTI'
    
    sidecar = {
        "ModelName": "Free-Water Elimination DTI",
        "FittingSoftware": "DIPY",
        "InputData": in_path.name,
        "FittingMethod": fit_method,
        "Metrics": metrics
    }
    
    for metric in metrics:
        metric_suffix = metric.upper()
        if metric == 'f': metric_suffix = 'F' # Free Water Fraction
        
        out_name = build_bids_name({**ent_base, 'suffix': metric_suffix})
        out_path = out_dir / out_name
        
        # Extract metric
        val = None
        if metric == 'fa': val = fwe_fit.fa
        elif metric == 'md': val = fwe_fit.md
        elif metric == 'ad': val = fwe_fit.ad
        elif metric == 'rd': val = fwe_fit.rd
        elif metric == 'f': val = fwe_fit.f
        elif metric == 'prediction': val = fwe_fit.predict(gtab, S0=1.) # Optional?
        
        if val is not None:
             nib.save(nib.Nifti1Image(val, img.affine), str(out_path))
             output_files[metric] = out_path
             
             sidecar_path = str(out_path).replace('.nii.gz', '.json')
             with open(sidecar_path, 'w') as f:
                 json.dump(sidecar, f, indent=4)
                 
    return output_files
