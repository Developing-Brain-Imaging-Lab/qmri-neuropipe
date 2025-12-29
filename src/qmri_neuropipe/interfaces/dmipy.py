
from pathlib import Path
import os
from threadpoolctl import threadpool_limits
from typing import Optional, Dict, Union
import nibabel as nib
import numpy as np

from ..core import ProcessingError
from ..core.utils import ensure_dir, extract_image_path
from ..core.types import ImageLike, DWIFile
from ..io.bids import build_bids_name, get_entities_from_path

def fit_noddi(
    in_file: Union[Path, ImageLike],
    out_dir: Path,
    bval_file: Optional[Path] = None,
    bvec_file: Optional[Path] = None,
    mask_file: Optional[Path] = None,
    metrics: list[str] = ["odi", "ficvf", "fiso"],
    nthreads: int = 1,
    **kwargs
) -> Dict[str, Path]:
    """
    Fit NODDI model using Dmipy.
    
    nthreads : int
        Number of CPUs for parallel processing.
    """
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
    
    # Distributed Models (Watson)
    # Create Watson-dispersed stick and zeppelin
    dispersed_bundle = distribute_models.SD1WatsonDistributed(models=[stick, zeppelin])

    dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp','C1Stick_1_lambda_par','partial_volume_0')
    dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
    dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.7e-9)
    
    # Multi-compartment model
    noddi = MultiCompartmentModel(models=[ball, dispersed_bundle])
    noddi.set_fixed_parameter('G1Ball_1_lambda_iso', 3.0e-9)
                                   
    # Tortuosity Constraint: lambda_perp_zeppelin = lambda_par_stick * (1 - f_intra)
    # This implies 1 - f_stick_volume_fraction??
    # This is tricky in Dmipy via simple calls without explicit function.
    # For now, we will assume Independent NODDI (or flexible) which is often preferred anyway.
    # Just linking orientation/dispersion is a strong enough constraint for "NODDI-like".
    
    # --- FIT ---
    # Optimize parallelization by disabling inner-loop threading
    # This prevents thread oversubscription when using multiprocessing
    # --- FIT ---
    # Optimize parallelization by disabling inner-loop threading using threadpoolctl
    # This dynamically limits threads for BLAS/OpenMP libraries during the fit
    # Also attempt to limit Numba threads if used
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
    
    try:
        import numba
        numba.set_num_threads(1)
        if hasattr(numba, 'config'):
            numba.config.THREADING_LAYER = 'workqueue'        
    except (ImportError, RuntimeError):
        pass

    # Ensure Pathos pools are clear if used
    try:
        import pathos.multiprocessing as mp
        # resizing pool to 0 or restarting helps clear state
        mp.ProcessingPool().restart() 
    except ImportError:
        pass

    print(f"Fitting NODDI (Dmipy) with {nthreads} CPUs...")
    
    with threadpool_limits(limits=1, user_api='blas'):
        # fit returns a MicrostructureFit object
        fit_results = noddi.fit(gtab, data, mask=mask_data, number_of_processors=nthreads)
    
    # --- Extract Metrics ---
    # ODI
    odi_map = fit_results.fitted_parameters['SD1WatsonDistributed_1_SD1Watson_1_odi']
    
    # Volume Fractions
    # Dmipy returns partial_volume_0, partial_volume_1, partial_volume_2 corresponding to models input order
    # models=[ball, watson_stick, watson_zeppelin]
    f_iso = fit_results.fitted_parameters['partial_volume_0']
    f_intra = fit_results.fitted_parameters['partial_volume_1']
    f_extra = fit_results.fitted_parameters['partial_volume_2']
    
    # Calculate ICVF (Intra-cellular Volume Fraction relative to non-CSF)
    # standard ICVF = f_intra / (f_intra + f_extra)
    denom = f_intra + f_extra
    denom[denom == 0] = 1.0 # Avoid div/0
    icvf_map = f_intra / denom
    
    # Handle mask (zeros outside)
    if mask_data is not None:
         icvf_map[~mask_data] = 0
         
    # Save Outputs
    outputs = {}
    
    # Helper to save
    def save_map(name, array):
        out_p = out_dir / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(array, affine), out_p)
        outputs[name] = out_p
        
    save_map('odi', odi_map)
    save_map('icvf', icvf_map)
    save_map('fiso', f_iso) # also save fiso as it is useful
    save_map('f_intra', f_intra)
    save_map('f_extra', f_extra)
    
    return outputs
