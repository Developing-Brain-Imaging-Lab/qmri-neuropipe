import os
from pathlib import Path
from typing import Optional, Dict, Union
import nibabel as nib
import numpy as np
import warnings
import contextlib
import multiprocessing

from qmri_neuropipe.core import ProcessingError
from qmri_neuropipe.core.utils import ensure_dir, extract_image_path
from qmri_neuropipe.core.types import ImageLike, DWIFile
from qmri_neuropipe.io.bids import build_bids_name, get_entities_from_path

def _fit_microglia_chunk(args):
    chunk_id, data_chunk, scheme, model_config, solver, solver_kwargs = args
    print(f"[Worker {chunk_id}] Started Microglia chunk with {len(data_chunk)} voxels.")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from dmipy.signal_models import cylinder_models, gaussian_models, sphere_models
        from dmipy.distributions import distribute_models
        from dmipy.core.modeling_framework import MultiCompartmentModel

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    try:
        stick = cylinder_models.C1Stick()
        zeppelin = gaussian_models.G2Zeppelin()
        dispersed_bundle = distribute_models.SD1WatsonDistributed(models=[stick, zeppelin])
        
        # Link stick and zeppelin parallel diffusivity
        dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp','C1Stick_1_lambda_par','partial_volume_0')
        dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
        
        # Fix parallel diffusivity to 1.7e-9 or similar based on config
        d_par = float(model_config.get('parallel_diffusivity', 1.7e-9))
        dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', d_par)
        
        # Spheres
        small_sphere = sphere_models.S2SphereStejskalTannerApproximation(name="SmallSphere")
        large_sphere = sphere_models.S2SphereStejskalTannerApproximation(name="LargeSphere")
        
        # Fix diameters based on config
        d_small = float(model_config.get('small_diameter', 4e-6))
        d_large = float(model_config.get('large_diameter', 8e-6))
        
        small_sphere.set_fixed_parameter('SmallSphere_1_diameter', d_small)
        large_sphere.set_fixed_parameter('LargeSphere_1_diameter', d_large)
        
        # Free water ball
        ball = gaussian_models.G1Ball()
        d_iso = float(model_config.get('iso_diffusivity', 3.0e-9))
        ball.set_fixed_parameter('G1Ball_1_lambda_iso', d_iso)
        
        # Assemble 4-compartment model
        microglia_model = MultiCompartmentModel(
            models=[dispersed_bundle, small_sphere, large_sphere, ball]
        )
        
        # Fit
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
             with warnings.catch_warnings():
                 warnings.simplefilter("ignore")
                 fit_obj = microglia_model.fit(
                    scheme, 
                    data_chunk, 
                    number_of_processors=1,
                    solver=solver,
                    **solver_kwargs
                )
        
        print(f"[Worker {chunk_id}] Finished fitting.")
        return fit_obj.fitted_parameters
        
    except Exception as e:
        print(f"[Worker {chunk_id}] Crash/Error: {e}")
        raise e

def fit_microglia(
    in_file: Union[Path, ImageLike],
    out_dir: Path,
    bval_file: Optional[Path] = None,
    bvec_file: Optional[Path] = None,
    delta_file: Optional[Path] = None,
    Delta_file: Optional[Path] = None,
    mask_file: Optional[Path] = None,
    nthreads: int = 1,
    parallel_diffusivity: float = 1.7e-9,
    iso_diffusivity: float = 3.0e-9,
    small_diameter: float = 4e-6,
    large_diameter: float = 8e-6,
    solver: str = "brute2fine",
    solver_kwargs: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Path]:
    """
    Fit the 4-compartment Microglia model using Dmipy.
    """
    # 1. Environment Variables and Numba setup
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
    os.environ["JOBLIB_START_METHOD"] = "fork"

    try:
        import numba
        numba.set_num_threads(1)
        if hasattr(numba, 'config'):
            numba.config.THREADING_LAYER = 'workqueue'
        numba.set_num_threads = lambda n: None
    except (ImportError, RuntimeError):
        pass

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from dmipy.core import acquisition_scheme
    except ImportError:
        raise ProcessingError("Dmipy required but not installed.")

    in_path = extract_image_path(in_file)
    out_dir = ensure_dir(out_dir)
    
    if bval_file is None or bvec_file is None:
        if isinstance(in_file, DWIFile):
             bval_file = bval_file or in_file.bval
             bvec_file = bvec_file or in_file.bvec
             delta_file = delta_file or in_file.delta
             Delta_file = Delta_file or in_file.Delta
             
    if not bval_file or not bvec_file:
         raise ValueError("Gradient files (bval/bvec) are required.")

    if solver_kwargs is None:
        solver_kwargs = {}
    
    img = nib.load(str(in_path))
    data = img.get_fdata()
    affine = img.affine
    
    bvals = np.loadtxt(bval_file) * 1e6 # dmipy expects SI units
    bvecs = np.loadtxt(bvec_file).T
    
    # Check if we have explicitly provided delta/Delta for sphere modeling
    # Dmipy's Sphere models REQUIRE gradient duration (delta) and diffusion time (Delta)
    if delta_file and delta_file.exists() and Delta_file and Delta_file.exists():
        delta_arr = np.loadtxt(delta_file)
        Delta_arr = np.loadtxt(Delta_file)
        gtab = acquisition_scheme.acquisition_scheme_from_bvalues(
            bvalues=bvals, gradient_directions=bvecs,
            delta=delta_arr, Delta=Delta_arr
        )
    else:
        # Fallback to standard (not ideal for sphere models which depend strictly on diffusion time)
        print("WARNING: No delta/Delta files found. Sphere radius estimation may be inaccurate without explicit diffusion times.")
        # Dmipy usually defaults delta=0.015, Delta=0.03 if not provided
        gtab = acquisition_scheme.acquisition_scheme_from_bvalues(bvals, bvecs)

    if mask_file and mask_file.exists():
        mask_data = nib.load(str(mask_file)).get_fdata().astype(bool)
        valid_voxels = data[mask_data]
    else:
        mask_data = None
        valid_voxels = data.reshape(-1, data.shape[-1])
        
    n_voxels = valid_voxels.shape[0]
    n_chunks = nthreads
    chunks = np.array_split(valid_voxels, n_chunks)
    
    model_config = {
        'parallel_diffusivity': parallel_diffusivity,
        'iso_diffusivity': iso_diffusivity,
        'small_diameter': small_diameter,
        'large_diameter': large_diameter
    }
    
    chunk_args = []
    for i in range(n_chunks):
        if chunks[i].shape[0] == 0: continue
        chunk_args.append(
            (i, chunks[i], gtab, model_config, solver, solver_kwargs)
        )

    print(f"Fitting Microglia Model ({n_voxels} voxels)...")
    
    # Run Pool
    try:
         ctx = multiprocessing.get_context('spawn')
    except ValueError:
         ctx = multiprocessing.get_context('fork')
         
    pool = ctx.Pool(processes=nthreads)
    results = []
    try:
        iterator = pool.imap(_fit_microglia_chunk, chunk_args)
        import time
        start_t = time.time()
        for i, res in enumerate(iterator):
            results.append(res)
            if len(chunk_args) > 10 and (i + 1) % (len(chunk_args) // 10) == 0:
                 elapsed = time.time() - start_t
                 print(f"  - Collected {i + 1}/{len(chunk_args)} chunks ({elapsed:.1f}s)")
    finally:
        pool.close()
        pool.join()
        
    if not results:
         raise ProcessingError("Fitting failed produced no results.")
         
    keys = results[0].keys()
    merged_params = {}
    for k in keys:
        arrays = [res[k] for res in results]
        merged_params[k] = np.concatenate(arrays, axis=0)
        
    vol_shape = data.shape[:-1]
    full_maps = {}
    for k, v in merged_params.items():
        if v.ndim == 1:
            out_arr = np.zeros(vol_shape, dtype=v.dtype)
        else:
            out_arr = np.zeros(vol_shape + (v.shape[1],), dtype=v.dtype)
            
        if mask_data is not None:
             out_arr[mask_data] = v
        else:
             out_arr = v.reshape(out_arr.shape)
        full_maps[k] = out_arr
        
    # Standardize map names
    # dmipy name resolution:
    # partial_volume_0 = dispersed_bundle
    # partial_volume_1 = SmallSphere
    # partial_volume_2 = LargeSphere
    # partial_volume_3 = free water (implicitly 1 - others)
    # The actual keys depend on order. We will save all keys out.
    
    outputs = {}
    ent_base = get_entities_from_path(in_path)
    if 'desc' in ent_base: del ent_base['desc']
    ent_base['model'] = 'Microglia'
    
    import json
    sidecar = {
        "ModelName": "Microglia (4-Compartment)",
        "FittingSoftware": "Dmipy",
        "InputData": in_path.name
    }
    
    for k, array in full_maps.items():
        # Clean up suffix (e.g. 'SD1WatsonDistributed_1_partial_volume_0' -> 'pv_bundle')
        suffix = k.replace('SD1WatsonDistributed_1_', '')
        suffix = suffix.replace('SmallSphere_1_', 'small_')
        suffix = suffix.replace('LargeSphere_1_', 'large_')
        
        # Simplified naming mapping based on known Dmipy internal structure
        if k == 'partial_volume_0': suffix = 'f_bundle'
        elif k == 'partial_volume_1': suffix = 'f_small_sphere'
        elif k == 'partial_volume_2': suffix = 'f_large_sphere'
        elif 'kappa' in k: suffix = 'dispersion_kappa'
        elif 'mu' in k: suffix = 'mu' # main orientation vector (3D)
        
        out_name = build_bids_name({**ent_base, 'suffix': suffix})
        out_p = out_dir / out_name
        nib.save(nib.Nifti1Image(array, affine), out_p)
        outputs[suffix] = out_p
        
        with open(str(out_p).replace('.nii.gz', '.json'), 'w') as f:
             json.dump(sidecar, f, indent=4)
             
    return outputs
