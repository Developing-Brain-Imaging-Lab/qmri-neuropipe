
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
    Models are instantiated LOCALLY to support chunk-specific fixed parameters (e.g., FISO map).
    """
    chunk_id, data_chunk, scheme, model_config, chunk_fixed_params, keys_to_keep, solver, solver_kwargs = args
    
    print(f"[Worker {chunk_id}] Started chunk with {len(data_chunk)} voxels.")
    
    # Enforce single-threaded execution within the worker
    import os
    import sys
    import warnings
    import contextlib
    from dmipy.signal_models import cylinder_models, gaussian_models
    from dmipy.distributions import distribute_models
    from dmipy.core.modeling_framework import MultiCompartmentModel, MultiCompartmentSphericalMeanModel

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    try:
        # --- Reconstruct Model Locally ---
        # Unpack config
        parallel_diffusivity = model_config.get('parallel_diffusivity', 1.7e-9)
        iso_diffusivity = model_config.get('iso_diffusivity', 3.0e-9)
        distribution = model_config.get('distribution', 'Watson')
        model_type = model_config.get('model_type', 'standard') # 'standard' or 'smt'
        
        # Core Models
        ball = gaussian_models.G1Ball()
        stick = cylinder_models.C1Stick()
        zeppelin = gaussian_models.G2Zeppelin()
        
        if model_type == 'smt':
            # SMT-NODDI: Spherical Mean of Stick + Zeppelin + Ball
            # Note: SMT doesn't use the explicit "Distributed" wrappers usually, as it models the mean signal directly.
            # We fit the microstructure parameters (fractions, intrinsic diffusivities).
            noddi = MultiCompartmentSphericalMeanModel(models=[stick, zeppelin, ball])
            
            # Fix Diffusivities
            noddi.set_fixed_parameter('C1Stick_1_lambda_par', parallel_diffusivity)
            noddi.set_fixed_parameter('G2Zeppelin_1_lambda_par', parallel_diffusivity)
            noddi.set_fixed_parameter('G1Ball_1_lambda_iso', iso_diffusivity)
            
            # SMT Tortuosity?
            # Ideally: lambda_perp_zepp = lambda_par * (1 - f_intra). 
            # But standard SMT-NODDI often just fixes lambda_perp if not linked?
            # Dmipy SMT supports tortuosity linking if we define it.
            # For now, let's duplicate the standard NODDI tortuosity if possible.
            noddi.set_tortuous_parameter('G2Zeppelin_1_lambda_perp','C1Stick_1_lambda_par','partial_volume_0')
            noddi.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
            
        else:
            # Standard NODDI
            if distribution.lower() == "watson":
                 dispersed_bundle = distribute_models.SD1WatsonDistributed(models=[stick, zeppelin])
            elif distribution.lower() == "bingham":
                 dispersed_bundle = distribute_models.SD2BinghamDistributed(models=[stick, zeppelin])
            else:
                 raise ValueError(f"Unknown distribution: {distribution}")
            
            # Tortuosity & Constraints
            dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp','C1Stick_1_lambda_par','partial_volume_0')
            dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
            dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', parallel_diffusivity)
            
            noddi = MultiCompartmentModel(models=[ball, dispersed_bundle])
            noddi.set_fixed_parameter('G1Ball_1_lambda_iso', iso_diffusivity)

        # --- Apply Chunk-Specific Fixed Parameters ---
        if chunk_fixed_params:
            for param_key, param_val in chunk_fixed_params.items():
                # param_val is an array matching data_chunk length
                if param_val is not None:
                    noddi.set_fixed_parameter(param_key, param_val)

        # --- Fit ---
        # Suppress dmipy print statements (optimizer setup) and warnings
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
             with warnings.catch_warnings():
                 warnings.simplefilter("ignore")
                 fit_obj = noddi.fit(
                    scheme, 
                    data_chunk, 
                    number_of_processors=1,
                    solver=solver,
                    **solver_kwargs
                )
        
        # Return only the requested parameters
        full_params = fit_obj.fitted_parameters
        if keys_to_keep:
            # Warning: SMT keys might differ from keys_to_keep logic. 
            # Ideally caller knows what to ask for, or we return everything if unsure.
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
    # New options
    model_type: str = "standard", # 'standard' or 'smt'
    fiso_file: Optional[Path] = None,
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
        from dmipy.core.modeling_framework import MultiCompartmentModel, MultiCompartmentSphericalMeanModel
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
        mask_img = nib.load(str(mask_file))
        mask_data = mask_img.get_fdata().astype(bool)
    else:
        mask_data = None
        
    # --- Load External FISO if provided ---
    fiso_data_flat = None
    if fiso_file:
        # Resolve path if it is a string
        fiso_path = Path(fiso_file)
        
        # Check if it's a glob pattern
        if not fiso_path.exists() and ('*' in str(fiso_file) or '?' in str(fiso_file)):
             print(f"Searching for FISO map with pattern: {fiso_file}")
             # Search in input directory
             parent_dir = Path(in_file).parent
             matches = list(parent_dir.glob(str(fiso_file)))
             if matches:
                 fiso_path = matches[0]
                 print(f"Found FISO map: {fiso_path}")
                 if len(matches) > 1:
                      print(f"WARNING: Multiple matches found for {fiso_file}. Using {fiso_path}.")
             else:
                 print(f"WARNING: No file found matching pattern {fiso_file} in {parent_dir}. FISO constraint will NOT be applied.")
                 fiso_path = None
                 
        if fiso_path and fiso_path.exists():
            print(f"Loading external FISO constraint from: {fiso_path}")
            fiso_img = nib.load(str(fiso_path))
            # Verify shape
            if fiso_img.shape != data.shape[:-1]:
                 raise ValueError(f"FISO map shape {fiso_img.shape} does not match DWI shape {data.shape[:-1]}")
            
            # Flatten based on mask
            if mask_data is not None:
                 fiso_data_flat = fiso_img.get_fdata()[mask_data]
            else:
                 fiso_data_flat = fiso_img.get_fdata().reshape(-1)
        elif fiso_file and not fiso_path:
             # pattern failed
             pass
        else:
             raise FileNotFoundError(f"FISO constraint file not found: {fiso_file}")
             
    # --- Prepare Data for Parallelization ---
    if mask_data is not None:
        valid_voxels = data[mask_data] # (N_valid, N_dwis)
    else:
        valid_voxels = data.reshape(-1, data.shape[-1])
        
    n_voxels = valid_voxels.shape[0]
    print(f"Total voxels to fit: {n_voxels}")
    
    # 2. Split into chunks
    n_chunks = nthreads
    chunks = np.array_split(valid_voxels, n_chunks)
    
    # Split FISO data if present
    fiso_chunks = [None]*n_chunks
    if fiso_data_flat is not None:
        fiso_chunks = np.array_split(fiso_data_flat, n_chunks)
    
    # --- Define Parameter Mapping (Dynamic) ---
    # We need to know parameter names WITHOUT instantiating the full model on all data yet.
    # We can instantiate a dummy model to inspect names.
    
    print(f"Inspecting model parameters for type: {model_type}...")
    
    # Dummy models
    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    
    if model_type == 'smt':
         dummy_model = MultiCompartmentSphericalMeanModel(models=[stick, zeppelin, ball])
    else:
         if distribution.lower() == "watson":
             dispersed_bundle = distribute_models.SD1WatsonDistributed(models=[stick, zeppelin])
         elif distribution.lower() == "bingham":
             # Try/Except for older dmipy versions if needed
             try:
                 dispersed_bundle = distribute_models.SD2BinghamDistributed(models=[stick, zeppelin])
             except AttributeError:
                  # Fallback or error
                  dispersed_bundle = distribute_models.SD1WatsonDistributed(models=[stick, zeppelin])
         else:
             # Default
             dispersed_bundle = distribute_models.SD1WatsonDistributed(models=[stick, zeppelin])
             
         dummy_model = MultiCompartmentModel(models=[ball, dispersed_bundle])

    all_params = dummy_model.parameter_names
    param_map = {}
    
    # 1. Global Volume Fractions
    # Usually: partial_volume_0 (ball), partial_volume_1 (bundle)
    # Check if 'partial_volume_0' exists
    if 'partial_volume_0' in all_params:
         param_map['f_iso'] = 'partial_volume_0'
    
    # For 'f_bundle', it's partial_volume_1 if it exists
    if 'partial_volume_1' in all_params:
         param_map['f_bundle'] = 'partial_volume_1'
    
    # 2. ODI / SMT specifics
    if model_type == 'standard':
        # ODI logic
        odi_candidates = [p for p in all_params if 'odi' in p.lower() or 'kappa' in p.lower()]
        if odi_candidates:
            filtered = [p for p in odi_candidates if distribution.lower() in p.lower()]
            param_map['odi'] = filtered[0] if filtered else odi_candidates[0]
        
        # Stick Fraction (intra-bundle)
        pv_candidates = [p for p in all_params if 'partial_volume_0' in p and p != 'partial_volume_0']
        if pv_candidates:
             filtered = [p for p in pv_candidates if distribution.lower() in p.lower()]
             param_map['pv_stick'] = filtered[0] if filtered else pv_candidates[0]
             
    else:
        # SMT Model
        # parameters are usually implicit.
        # We assume f_iso (ball), f_bundle (zeppelin+stick).
        # But wait, MultiCompartmentSphericalMeanModel(models=[stick, zeppelin, ball])
        # This usually has 3 partial volumes? partial_volume_0, _1, _2 ?
        # Or does it group?
        # Usually it respects input order: stick(0), zeppelin(1), ball(2).
        # Let's inspect in practice.
        # If we didn't use a bundler, we have 3 compartments.
        # We need to sum stick+zeppelin for f_bundle.
        pass

    print(f"Identified keys: {param_map}")
    print(f"All params: {all_params}") # Debug
    
    keys_to_keep = None # Retrieve everything to be safe given the complexity, or filter if confident.
    # For efficiency we should ideally filter, but with new SMT we might miss things.
    # Let's try to map dynamically AFTER results if we return all. 
    # But for transfer efficiency, let's keep all standard volume/diffusivity keys.
    # Actually, return all is safest for now.
    
    # --- Prepare Config for Workers ---
    model_config = {
        'model_type': model_type,
        'distribution': distribution,
        'parallel_diffusivity': parallel_diffusivity,
        'iso_diffusivity': iso_diffusivity
    }
    
    # chunk_args: (id, data, scheme, config, fixed_params, keys, solver, kwargs)
    chunk_args = []
    for i in range(n_chunks):
        if chunks[i].shape[0] == 0: continue
        
        # Fixed params for this chunk
        c_fixed = {}
        if fiso_chunks[i] is not None:
             # Map 'f_iso' key to this chunk
             # Usually 'partial_volume_0' if ball is first?
             # NOTE: In our standard NODDI (ball, bundle), ball is 0.
             # In SMT (stick, zeppelin, ball), ball is 2 (last).
             # We must get the key correct.
             
             # Standard: noddi = MultiCompartmentModel(models=[ball, dispersed_bundle]) -> ball is 0.
             # SMT: noddi = MultiCompartmentSphericalMeanModel(models=[stick, zeppelin, ball]) -> ball is 2?
             
             # Let's check param_map or all_params.
             # If SMT, 'G1Ball_1_partial_volume_0' ?? No, usually 'partial_volume_X'.
             # Dmipy normalizes volume fractions.
             
             # Strategy: Use the param name found in param_map['f_iso']/etc if we can trust it.
             # For SMT, we need to be careful.
             
             # For standard NODDI:
             if model_type == 'standard':
                  target_key = 'partial_volume_0' 
             else:
                  # For SMT with [stick, zeppelin, ball]
                  # It typically creates partial_volume_0, _1, _2 ?
                  # If we fix one, the others renormalize?
                  # Dmipy fixed parameter by array works on 'partial_volume_X'.
                  # We'll need to know which one is ball.
                  # Usually Dmipy assigns indices in order of models.
                  # stick=0, zeppelin=1, ball=2.
                  target_key = 'partial_volume_2' 
                  
             c_fixed[target_key] = fiso_chunks[i]

        chunk_args.append(
            (i, chunks[i], gtab, model_config, c_fixed, keys_to_keep, solver, solver_kwargs)
        )

    # 4. Run Pool
    print(f"Fitting NODDI (Type: {model_type}, FISO constraint: {fiso_file is not None})...")
    
    # We use explicit try/finally to ensure termination if needed
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
    # Handle SMT vs Standard keys
    
    f_iso = None
    f_intra = None
    pv0 = None
    odi_map = None
    
    # Extract based on map
    if param_map.get('f_iso'):
         f_iso = fit_results.fitted_parameters.get(param_map['f_iso'])
         
    if param_map.get('f_bundle'):
         # Standard NODDI bundle
         f_intra = fit_results.fitted_parameters.get(param_map['f_bundle'])
    
    if param_map.get('pv_stick'):
         pv0 = fit_results.fitted_parameters.get(param_map['pv_stick'])
         
    if param_map.get('odi'):
         odi_map = fit_results.fitted_parameters.get(param_map['odi'])
         
    # Logic for SMT Reconstruction of f_intra/f_extra
    if model_type == 'smt':
         # In SMT (Stick, Zeppelin, Ball), results are typically:
         # partial_volume_0 -> Stick (Intra)
         # partial_volume_1 -> Zeppelin (Extra)
         # partial_volume_2 -> Ball (Iso)
         
         # We try to extract these directly
         vf_intra = fit_results.fitted_parameters.get('partial_volume_0')
         vf_extra = fit_results.fitted_parameters.get('partial_volume_1')
         
         # Update f_iso if not already set correctly via param_map (which might fail for SMT)
         if f_iso is None:
             f_iso = fit_results.fitted_parameters.get('partial_volume_2')

    # Recalculate if standard (and not set by SMT)
    if vf_intra is None and pv0 is not None and f_intra is not None:
         vf_intra = pv0 * f_intra
         vf_extra = (1 - pv0) * f_intra
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
