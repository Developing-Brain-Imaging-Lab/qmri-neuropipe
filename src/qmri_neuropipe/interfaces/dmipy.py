
from pathlib import Path
from pathlib import Path
import os
import multiprocessing
import argparse
from threadpoolctl import threadpool_limits
from typing import Optional, Dict, Union
import nibabel as nib
import numpy as np

from ..core import ProcessingError
from ..core.utils import ensure_dir, extract_image_path
from ..core.types import ImageLike, DWIFile
from ..io.bids import build_bids_name, get_entities_from_path

def _fit_chunk(args):
    """
    Helper function to fit a chunk of data in a separate process.
    """
    chunk_id, data_chunk, model, scheme, keys_to_keep, solver, solver_kwargs = args
    
    print(f"[Worker {chunk_id}] Started chunk with {len(data_chunk)} voxels.")
    
    # Enforce single-threaded execution within the worker
    import os
    import sys
    import warnings
    import contextlib

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    try:
        # Fit the chunk
        # dmipy fit returns a MicrostructureFit object
        # CRITICAL: number_of_processors=1 ensures we don't spawn nested pools
        
        # Suppress dmipy print statements (optimizer setup) and warnings
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
             with warnings.catch_warnings():
                 warnings.simplefilter("ignore")
                 fit_obj = model.fit(
                    scheme, 
                    data_chunk, 
                    number_of_processors=1,
                    solver=solver,
                    **solver_kwargs
                )
        
        # Return only the requested parameters to minimize pickling/transfer overhead
        full_params = fit_obj.fitted_parameters
        if keys_to_keep:
            ret = {k: v for k, v in full_params.items() if k in keys_to_keep}
        else:
            ret = full_params
            
        print(f"[Worker {chunk_id}] Finished fitting.")
        return ret
        
    except Exception as e:
        print(f"[Worker {chunk_id}] Crash/Error: {e}")
        raise e

def fit_noddi(
    in_file: Union[Path, ImageLike],
    out_dir: Path,
    bval_file: Optional[Path] = None,
    bvec_file: Optional[Path] = None,
    mask_file: Optional[Path] = None,
    metrics: list[str] = ["odi", "ficvf", "fiso"],
    nthreads: int = 1,
    parallel_diffusivity: float = 1.7e-9,
    iso_diffusivity: float = 3.0e-9,
    distribution: str = "Watson",
    solver: str = "brute2fine",
    solver_kwargs: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Path]:
    """
    Fit NODDI model using Dmipy.
    
    nthreads : int
        Number of CPUs for parallel processing.
    """
    # --- Parallelization Config (MUST BE FIRST) ---
    # These configurations must run before any imports that might initialize libraries
    import os
    import multiprocessing
    from threadpoolctl import threadpool_limits

    # 1. Environment Variables
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
    os.environ["JOBLIB_START_METHOD"] = "fork"

    # 2. Numba Monkeypatch
    # Critical: Must run before dmipy imports numba
    try:
        import numba
        numba.set_num_threads(1)
        if hasattr(numba, 'config'):
            numba.config.THREADING_LAYER = 'workqueue'
        
        # Monkeypatch: Disable further changes to prevent external overrides
        numba.set_num_threads = lambda n: None
    except (ImportError, RuntimeError):
        pass

    # 3. Force Fork - REMOVED, using spawn context in pool instead
    # try:
    #     if multiprocessing.get_start_method() != 'fork':
    #          multiprocessing.set_start_method('fork', force=True)
    # except RuntimeError:
    #     pass
        
    # Imports for NODDI
    try:
        from dmipy.signal_models import cylinder_models, gaussian_models
        from dmipy.core.modeling_framework import MultiCompartmentModel
        from dmipy.distributions import distribute_models
        from dmipy.core import acquisition_scheme
    except ImportError:
        raise ProcessingError("Dmipy required but not installed.")

    in_path = extract_image_path(in_file)
    out_dir = ensure_dir(out_dir)
    
    # Extract gradients
    if bval_file is None or bvec_file is None:
        if isinstance(in_file, DWIFile):
             bval_file = bval_file or in_file.bval
             bvec_file = bvec_file or in_file.bvec
             
    if not bval_file or not bvec_file:
         raise ValueError("Gradient files (bval/bvec) are required for NODDI.")

    if solver_kwargs is None:
        solver_kwargs = {}
    
    # Load data
    img = nib.load(str(in_path))
    data = img.get_fdata()
    affine = img.affine
    
    bvals = np.loadtxt(bval_file)
    bvecs = np.loadtxt(bvec_file).T
    
    # Create Scheme (Dmipy handles various bval/bvec formats, usually expects SI units for computation?)
    # Dmipy usually works with b-values in s/mm^2 if diffusivities are in mm^2/s.
    # Standard: b=1000 s/mm^2. D ~ 0.001 mm^2/s.
    # qmri-neuropipe inputs are likely standard.
    
    # Warning: acquisition_scheme_from_bvalues_bvecs expects bvals in s/mm^2 and bvecs normalized.
    gtab = acquisition_scheme.acquisition_scheme_from_bvalues(bvals*1e6, bvecs)
    
    # Define Mask
    if mask_file and mask_file.exists():
        mask_data = nib.load(str(mask_file)).get_fdata().astype(bool)
    else:
        mask_data = None
        
    # --- Define NODDI Model ---
    # 1. Intra-cellular (Stick)
    # 2. Extra-cellular (Zeppelin)
    # 3. CSF (Ball)
    # 4. Distortion (Watson)
    
    # Models
    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    
    # Distributed Models
    print(f"Adding distributed models (Type: {distribution})...")
    if distribution.lower() == "watson":
         dispersed_bundle = distribute_models.SD1WatsonDistributed(models=[stick, zeppelin])
          
    elif distribution.lower() == "bingham":
         # Assuming SD2BinghamDistributed exists and works similarly
         try:
             dispersed_bundle = distribute_models.SD2BinghamDistributed(models=[stick, zeppelin])
         except AttributeError:
             raise ValueError("Bingham distribution (SD2BinghamDistributed) not found in dmipy.")

    else:
        raise ValueError(f"Unknown distribution: {distribution}. Supported: Watson, Bingham.")
    
    # Tortuosity Constraint
    dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp','C1Stick_1_lambda_par','partial_volume_0')
    dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
    dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', parallel_diffusivity)
    

    # Multi-compartment model
    noddi = MultiCompartmentModel(models=[ball, dispersed_bundle])
    noddi.set_fixed_parameter('G1Ball_1_lambda_iso', iso_diffusivity)
                                   
    print(f"Fitting NODDI (Dmipy) with {nthreads} CPUs (Custom Parallelization)...")
    
    # --- Custom Parallelization ---
    # 1. Prepare Data
    if mask_data is not None:
        valid_voxels = data[mask_data] # (N_valid, N_dwis)
    else:
        # Flatten all voxels
        valid_voxels = data.reshape(-1, data.shape[-1])
        
    n_voxels = valid_voxels.shape[0]
    print(f"Total voxels to fit: {n_voxels}")
    
    # 2. Split into chunks
    # Ensure at least one voxel per chunk
    n_chunks = nthreads
    chunks = np.array_split(valid_voxels, n_chunks)
    
    # Prepare arguments for each chunk
    # Define keys we actually need to save memory/time
    # Prepare arguments for each chunk
    # Define keys we actually need to save memory/time
    
    # DYNAMIC KEY DISCOVERY
    # Hardcoded naming (e.g. SD1WatsonDistributed_1...) is fragile as instance counters may vary.
    # We inspect noddi.parameter_names to find the correct keys.
    all_params = noddi.parameter_names
    param_map = {}
    
    # 1. Global Volume Fractions
    # MultiCompartmentModel(models=[ball, dispersed_bundle])
    # partial_volume_0 -> ball (iso)
    # partial_volume_1 -> bundle (intra + extra)
    param_map['f_iso'] = 'partial_volume_0'
    param_map['f_bundle'] = 'partial_volume_1'
    
    # 2. Distributed Model Parameters (ODI, Stick Fraction)
    # We look for keys that belong to the distributed model but are not the global volumes.
    # Pattern likely contains 'Watson' or 'Bingham' and 'odi' or 'partial_volume_0'.
    
    # Find ODI key
    odi_candidates = [p for p in all_params if 'odi' in p.lower() or 'kappa' in p.lower()] # Watson uses odi? Or kappa? Dmipy Watson yields 'odi' usually.
    # If Watson, we expect 'odi'. 
    if odi_candidates:
        # If multiple, take the longest one? Or strictly match distribution name?
        # Expect: SD1WatsonDistributed_1_SD1Watson_1_odi
        # Filter by distribution name if possible
        filtered = [p for p in odi_candidates if distribution.lower() in p.lower()]
        if filtered:
            param_map['odi'] = filtered[0]
        else:
            param_map['odi'] = odi_candidates[0] # Fallback
    else:
        print("WARNING: Could not find ODI parameter key. ODI map will be empty.")
        param_map['odi'] = None
        
    # Find Intra-Bundle Stick Fraction (pv0 of the bundle)
    # Key usually ends in 'partial_volume_0' but is NOT the global 'partial_volume_0'
    # Expect: SD1WatsonDistributed_1_partial_volume_0
    
    # Exclude global 'partial_volume_0'
    pv_candidates = [p for p in all_params if 'partial_volume_0' in p and p != 'partial_volume_0']
    if pv_candidates:
         # Again, filter by distribution name
         filtered = [p for p in pv_candidates if distribution.lower() in p.lower()]
         if filtered:
             param_map['pv_stick'] = filtered[0]
         else:
             param_map['pv_stick'] = pv_candidates[0]
    else:
         print("WARNING: Could not find Masked Stick Fraction key. vf_intra calculation may fail.")
         param_map['pv_stick'] = None

    keys_to_keep = [v for v in param_map.values() if v is not None]
    
    print(f"DEBUG: Identified NODDI keys: {param_map}")

    # (chunk_id, data_chunk, model, scheme, keys_to_keep, solver, solver_kwargs)
    chunk_args = [(i, c, noddi, gtab, keys_to_keep, solver, solver_kwargs) for i, c in enumerate(chunks) if c.shape[0] > 0]
    
    # 4. Run Pool
    results = []
    # Use 'spawn' for safety against BLAS/Numba crashes
    try:
         ctx = multiprocessing.get_context('spawn')
    except ValueError:
         # Fallback if spawn is not available (rare on modern python)
         ctx = multiprocessing.get_context('fork')
         
    print(f"Starting process pool (method: {ctx.get_start_method()})...")
    
    # We use explicit try/finally to ensure termination if needed
    pool = ctx.Pool(processes=nthreads)
    try:
        # CRITICAL FIX: Use pool.imap (ordered) instead of imap_unordered.
        # imap yields results in the same order as chunk_args.
        # Since we use np.array_split to create chunks in order, we MUST reassemble them in order.
        iterator = pool.imap(_fit_chunk, chunk_args)
        
        # Collect results with progress
        import time
        start_t = time.time()
        for i, res in enumerate(iterator):
            results.append(res)
            # Parent process logging
            if len(chunk_args) > 10 and (i + 1) % (len(chunk_args) // 10) == 0:
                 elapsed = time.time() - start_t
                 print(f"  - Collected {i + 1}/{len(chunk_args)} chunks ({elapsed:.1f}s)")
            elif len(chunk_args) <= 10:
                 print(f"  - Collected chunk {i + 1}/{len(chunk_args)}")

    finally:
        # Ensure we close the pool
        pool.close()
        # Wait for workers to exit
        print("Waiting for workers to exit (joining pool)...")
        pool.join()
        
    print("Workers finished. Reassembling results...")

    # 5. Reassemble Results
    # results is a list of dicts. We need to concatenate the arrays for each key.
    if not results:
         raise ProcessingError("Fitting failed produced no results.")
         
    keys = results[0].keys()
    merged_params = {}
    
    # Since we used pool.imap, 'results' is guaranteed to be in the same order as 'chunks'.
    # We can safely simple concatenate.
    for k in keys:
        # Concatenate this parameter across all chunks
        arrays = [res[k] for res in results]
        merged_params[k] = np.concatenate(arrays, axis=0)
        
    # 6. Map back to volume
    # We create a pseudo-fit-results dict to act like the original object's property
    fit_results = argparse.Namespace(fitted_parameters={}) 
    
    # Repopulate full volume maps
    # If we used a mask, we need to embed valid voxels back into 3D
    # If no mask, we just reshape
    vol_shape = data.shape[:-1]
    full_maps = {}
    for k, v in merged_params.items():
        # v is 1D array of length N_valid (or N_total)
        
        # Determine output shape: (X, Y, Z) or (X, Y, Z, M) if parameter is multidimensional?
        # NODDI params like odi, f_iso are scalars per voxel.
        # Check shape of v
        if v.ndim == 1:
            out_arr = np.zeros(vol_shape, dtype=v.dtype)
        else:
            out_arr = np.zeros(vol_shape + (v.shape[1],), dtype=v.dtype)
            
        if mask_data is not None:
             out_arr[mask_data] = v
        else:
             out_arr = v.reshape(out_arr.shape)
             
        full_maps[k] = out_arr
        
    # Assign to our mock object for compatibility with extraction code below
    fit_results.fitted_parameters = full_maps
    
    # --- Extract Metrics ---
    
    # Volume Fractions (Standard)
    f_iso = fit_results.fitted_parameters.get(param_map.get('f_iso'))
    f_intra = fit_results.fitted_parameters.get(param_map.get('f_bundle')) # Total non-iso fraction

    # Internal bundle fraction (Stick vs Zeppelin)
    # pv_stick key maps to the fraction of 'stick' within the bundle
    pv0 = None
    if param_map.get('pv_stick'):
        pv0 = fit_results.fitted_parameters.get(param_map['pv_stick'])
    
    odi_map = None
    if param_map.get('odi'):
        odi_map = fit_results.fitted_parameters.get(param_map['odi'])

    vf_intra = None
    vf_extra = None
    if pv0 is not None and f_intra is not None:
         vf_intra = pv0 * f_intra
         vf_extra = (1 - pv0) * f_intra

        
    # Save Outputs
    outputs = {}
    
    # Helper to save
    def save_map(name, array):
        if array is None: return
        out_p = out_dir / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(array, affine), out_p)
        outputs[name] = out_p
        
    save_map('odi', odi_map)
    save_map('fiso', f_iso) # also save fiso as it is useful
    save_map('vf_intra', vf_intra)  # Volume Fraction Intra
    save_map('vf_extra', vf_extra) # Volume Fraction Extra
    
    return outputs
