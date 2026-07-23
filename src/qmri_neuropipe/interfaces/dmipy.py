
from pathlib import Path
from pathlib import Path
import os
import multiprocessing
import argparse
import contextlib
import json
from threadpoolctl import threadpool_limits
from typing import Optional, Dict, Union, Any
import nibabel as nib
import numpy as np
import warnings


from ..core import ProcessingError
from ..core.utils import ensure_dir, extract_image_path
from ..core.types import ImageLike, DWIFile
from ..io.bids import build_bids_name, get_entities_from_path
from .dmipy_backend import DmipyRuntime, acquisition_scheme_from_bvalues


def _reshape_gnl_tensor(vox_gnl):
    vox_gnl = np.asarray(vox_gnl)
    if vox_gnl.size == 9:
        return vox_gnl.reshape(3, 3)
    if vox_gnl.shape == (3, 3):
        return vox_gnl
    return vox_gnl.reshape(3, 3)


def _rotate_gradients_for_gnl(bvals, bvecs, vox_gnl):
    rot_mat = _reshape_gnl_tensor(vox_gnl)
    rot_bvecs = np.dot(bvecs, rot_mat.T)
    norms = np.linalg.norm(rot_bvecs, axis=1)
    new_bvals = bvals * (norms ** 2)
    safe_norms = norms.copy()
    safe_norms[safe_norms == 0] = 1.0
    new_bvecs = rot_bvecs / safe_norms[:, None]
    return new_bvals, new_bvecs


def _build_dmipy_scheme(bvals, bvecs, delta=None, Delta=None):
    return acquisition_scheme_from_bvalues(
        bvals,
        bvecs,
        delta=delta,
        Delta=Delta,
    )


def _load_sandi_gradients(bval_file, bvec_file):
    """Load and validate FSL-style gradients in dmipy SI units."""
    bvals = np.asarray(np.loadtxt(bval_file), dtype=float).reshape(-1)
    bvecs = np.asarray(np.loadtxt(bvec_file), dtype=float)
    if bvecs.ndim != 2:
        raise ValueError("b-vectors must be a two-dimensional 3 x N or N x 3 array.")
    if bvecs.shape == (3, bvals.size):
        bvecs = bvecs.T
    elif bvecs.shape != (bvals.size, 3):
        raise ValueError(
            f"b-vector shape {bvecs.shape} is incompatible with {bvals.size} "
            "b-values; expected 3 x N or N x 3."
        )
    if not np.all(np.isfinite(bvals)) or np.any(bvals < 0):
        raise ValueError("b-values must be finite and non-negative.")
    if not np.all(np.isfinite(bvecs)):
        raise ValueError("b-vectors must be finite.")
    norms = np.linalg.norm(bvecs, axis=1)
    weighted = bvals > 10.0
    if np.any(norms[weighted] <= 0):
        raise ValueError("Every diffusion-weighted volume must have a non-zero b-vector.")
    if np.any(np.abs(norms[weighted] - 1.0) > 0.1):
        raise ValueError("Diffusion-weighted b-vectors must have approximately unit norm.")
    nonzero = norms > 0
    bvecs[nonzero] /= norms[nonzero, None]
    return bvals * 1e6, bvecs


def _load_sandi_timing(delta_file, Delta_file, n_measurements):
    """Load required PGSE timing in seconds and validate physical ordering."""
    if not delta_file or not Delta_file:
        raise ProcessingError(
            "DMIPY SANDI sphere fitting requires both small-delta and big-Delta "
            "timing files, with values in seconds."
        )

    def load(path, label):
        values = np.asarray(np.loadtxt(Path(path)), dtype=float).reshape(-1)
        if values.size not in (1, n_measurements):
            raise ValueError(
                f"{label} timing must contain one value or {n_measurements} values; "
                f"found {values.size}."
            )
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"{label} timing must be finite and positive seconds.")
        return np.full(n_measurements, values[0]) if values.size == 1 else values

    delta = load(delta_file, "small-delta")
    Delta = load(Delta_file, "big-Delta")
    if np.any(Delta <= delta):
        raise ValueError("Every big-Delta value must be greater than small-delta.")
    return delta, Delta


def _initialize_param_storage(model, n_voxels: int) -> Dict[str, np.ndarray]:
    cardinality = getattr(model, "parameter_cardinality", {}) or {}
    storage: Dict[str, np.ndarray] = {}
    for key in getattr(model, "parameter_names", []):
        card = int(cardinality.get(key, 1) or 1)
        shape = (n_voxels,) if card == 1 else (n_voxels, card)
        storage[key] = np.full(shape, np.nan, dtype=np.float32)
    return storage


def _store_param_result(storage: Dict[str, np.ndarray], index: int, params: Optional[Dict[str, Any]]) -> None:
    if not params:
        return
    for key, dest in storage.items():
        value = params.get(key)
        if value is None:
            continue
        value_arr = np.asarray(value, dtype=np.float32)
        if value_arr.size == 0:
            continue
        flat = value_arr.reshape(-1)
        if dest.ndim == 1:
            dest[index] = flat[0]
        else:
            width = dest.shape[1]
            if flat.size >= width:
                dest[index, :] = flat[:width]


def _voxel_signal_is_valid(voxel: np.ndarray) -> bool:
    voxel = np.asarray(voxel)
    if voxel.ndim != 1 or voxel.size == 0:
        return False
    if not np.all(np.isfinite(voxel)):
        return False
    return bool(np.any(voxel > 0))


def _safe_rotate_gradients_for_gnl(bvals, bvecs, vox_gnl):
    try:
        if vox_gnl is None or not np.all(np.isfinite(vox_gnl)):
            return bvals, bvecs
        new_bvals, new_bvecs = _rotate_gradients_for_gnl(bvals, bvecs, vox_gnl)
        if new_bvals.shape != bvals.shape or new_bvecs.shape != bvecs.shape:
            return bvals, bvecs
        if not np.all(np.isfinite(new_bvals)) or not np.all(np.isfinite(new_bvecs)):
            return bvals, bvecs
        return new_bvals, new_bvecs
    except Exception:
        return bvals, bvecs


def _build_noddi_model(model_config, fixed_params: Optional[Dict[str, Any]] = None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from dmipy_fit.signal_models import cylinder_models, gaussian_models
        from dmipy_fit.distributions import distribute_models
        from dmipy_fit.core.modeling_framework import MultiCompartmentModel, MultiCompartmentSphericalMeanModel

    parallel_diffusivity = float(model_config.get('parallel_diffusivity', 1.7e-9))
    iso_diffusivity = float(model_config.get('iso_diffusivity', 3.0e-9))
    distribution = model_config.get('distribution', 'Watson')
    model_type = model_config.get('model_type', 'standard')

    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()

    if distribution.lower() == "bingham":
        dispersed_bundle = distribute_models.SD2BinghamDistributed(models=[stick, zeppelin])
    else:
        dispersed_bundle = distribute_models.SD1WatsonDistributed(models=[stick, zeppelin])

    dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp', 'C1Stick_1_lambda_par', 'partial_volume_0')
    dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
    dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', parallel_diffusivity)

    if model_type == 'smt':
        noddi = MultiCompartmentSphericalMeanModel(models=[dispersed_bundle, ball])
    else:
        noddi = MultiCompartmentModel(models=[dispersed_bundle, ball])

    noddi.set_fixed_parameter('G1Ball_1_lambda_iso', iso_diffusivity)
    for param_key, param_val in (fixed_params or {}).items():
        if param_val is not None:
            noddi.set_fixed_parameter(param_key, param_val)
    return noddi


def _build_sandi_model(model_config):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from dmipy_fit.signal_models import cylinder_models, gaussian_models, sphere_models
        from dmipy_fit.distributions.distribute_models import BundleModel
        #from dmipy_fit.core.modeling_framework import MultiCompartmentModel
        from dmipy_fit.core.modeling_framework import MultiCompartmentSphericalMeanModel
        

    soma_diffusivity = float(model_config.get('soma_diffusivity', 3.0e-9))
    if not np.isfinite(soma_diffusivity) or soma_diffusivity <= 0:
        raise ValueError("soma_diffusivity must be finite and positive.")

    stick = cylinder_models.C1Stick()
    extra_cellular = gaussian_models.G1Ball()
    soma = sphere_models.S4SphereGaussianPhaseApproximation(diffusion_constant=soma_diffusivity)

    bundle = BundleModel([stick, soma])

    sandi_model = MultiCompartmentSphericalMeanModel(models=[bundle, extra_cellular])
    sandi_model.set_parameter_optimization_bounds('BundleModel_1_S4SphereGaussianPhaseApproximation_1_diameter',[2e-6, 24e-6])
    sandi_model.set_parameter_optimization_bounds('G1Ball_1_lambda_iso',[1e-10, 3e-9]) #D_ec
    sandi_model.set_parameter_optimization_bounds('BundleModel_1_C1Stick_1_lambda_par',[1e-10, 3e-9]) #D_in
    sandi_model.set_parameter_optimization_bounds('BundleModel_1_partial_volume_0',[0.01, 0.99]) #f_in
    sandi_model.set_parameter_optimization_bounds('partial_volume_1',[0.01, 0.99]) #f_ec

    # dispersed_bundle = distribute_models.SD1WatsonDistributed(models=[stick, zeppelin])
    # dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp', 'C1Stick_1_lambda_par', 'partial_volume_0')
    # dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
    # dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', parallel_diffusivity)
    
    return sandi_model


def _sandi_fraction_maps(full_maps):
    """Return absolute SANDI compartment fractions from dmipy's nested model."""
    bundle_fraction = full_maps.get('partial_volume_0')
    stick_fraction_in_bundle = full_maps.get('BundleModel_1_partial_volume_0')
    extra_fraction = full_maps.get('partial_volume_1')

    if bundle_fraction is None or stick_fraction_in_bundle is None:
        return None, None, extra_fraction

    neurite_fraction = bundle_fraction * stick_fraction_in_bundle
    soma_fraction = bundle_fraction * (1.0 - stick_fraction_in_bundle)
    return soma_fraction, neurite_fraction, extra_fraction

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
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from dmipy_fit.signal_models import cylinder_models, gaussian_models
        from dmipy_fit.distributions import distribute_models
        from dmipy_fit.core.modeling_framework import MultiCompartmentModel, MultiCompartmentSphericalMeanModel

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    try:
        # --- Reconstruct Model Locally ---
        # Unpack config
        parallel_diffusivity = float(model_config.get('parallel_diffusivity', 1.7e-9))
        iso_diffusivity = float(model_config.get('iso_diffusivity', 3.0e-9))
        distribution = model_config.get('distribution', 'Watson')
        model_type = model_config.get('model_type', 'standard') # 'standard' or 'smt'
        
        # Core Models
        ball = gaussian_models.G1Ball()
        stick = cylinder_models.C1Stick()
        zeppelin = gaussian_models.G2Zeppelin()

        if distribution.lower() == "bingham":
            dispersed_bundle = distribute_models.SD2BinghamDistributed(models=[stick, zeppelin])
        else:
            dispersed_bundle = distribute_models.SD1WatsonDistributed(models=[stick, zeppelin])

        dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp','C1Stick_1_lambda_par','partial_volume_0')
        dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
        dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', parallel_diffusivity)

        if model_type == 'smt':
            # SMT-NODDI: Spherical Mean of Stick + Zeppelin + Ball
            noddi = MultiCompartmentSphericalMeanModel(models=[dispersed_bundle, ball])        
        else:
            # Standard NODDI            
            noddi = MultiCompartmentModel(models=[dispersed_bundle, ball])
            
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


def _fit_chunk_gnl(args):
    chunk_id, data_chunk, bvals, bvecs, model_config, chunk_fixed_params, solver, solver_kwargs, gnl_chunk = args

    print(f"[Worker {chunk_id}] Started GNL-aware chunk with {len(data_chunk)} voxels.")

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    try:
        n_voxels = data_chunk.shape[0]
        merged = None
        failed_voxels = 0
        fallback_voxels = 0
        first_error = None
        base_scheme = _build_dmipy_scheme(bvals, bvecs)

        for vox_idx in range(data_chunk.shape[0]):
            voxel_fixed = {}
            for param_key, param_vals in (chunk_fixed_params or {}).items():
                if param_vals is not None:
                    voxel_fixed[param_key] = np.asarray([param_vals[vox_idx]])

            model = _build_noddi_model(model_config, fixed_params=voxel_fixed)
            if merged is None:
                merged = _initialize_param_storage(model, n_voxels)

            voxel_signal = np.asarray(data_chunk[vox_idx], dtype=np.float32)
            if not _voxel_signal_is_valid(voxel_signal):
                failed_voxels += 1
                continue

            new_bvals, new_bvecs = _safe_rotate_gradients_for_gnl(bvals, bvecs, gnl_chunk[vox_idx])
            scheme = _build_dmipy_scheme(new_bvals, new_bvecs)

            try:
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit_obj = model.fit(
                            scheme,
                            voxel_signal[None, :],
                            number_of_processors=1,
                            solver=solver,
                            **solver_kwargs
                        )
                _store_param_result(merged, vox_idx, fit_obj.fitted_parameters)
            except Exception as exc:
                try:
                    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            fit_obj = model.fit(
                                base_scheme,
                                voxel_signal[None, :],
                                number_of_processors=1,
                                solver=solver,
                                **solver_kwargs
                            )
                    _store_param_result(merged, vox_idx, fit_obj.fitted_parameters)
                    fallback_voxels += 1
                except Exception:
                    failed_voxels += 1
                    if first_error is None:
                        first_error = exc

        if merged is None:
            model = _build_noddi_model(model_config, fixed_params={})
            merged = _initialize_param_storage(model, n_voxels)
            failed_voxels = n_voxels

        if failed_voxels:
            msg = f"[Worker {chunk_id}] GNL-aware NODDI skipped/failed {failed_voxels}/{n_voxels} voxels."
            if first_error is not None:
                msg += f" First error: {first_error}"
            print(msg)
        if fallback_voxels:
            print(f"[Worker {chunk_id}] GNL-aware NODDI fell back to the original scheme for {fallback_voxels}/{n_voxels} voxels.")
        print(f"[Worker {chunk_id}] Finished GNL-aware chunk.")
        return merged

    except Exception as e:
        print(f"[Worker {chunk_id}] GNL crash/error: {e}")
        raise e


def _fit_sandi_chunk(args):
    chunk_id, data_chunk, scheme, model_config, solver, solver_kwargs = args
    print(f"[Worker {chunk_id}] Started SANDI chunk with {len(data_chunk)} voxels.")

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    try:
        sandi = _build_sandi_model(model_config)
        valid = np.asarray([_voxel_signal_is_valid(v) for v in data_chunk])
        merged = _initialize_param_storage(sandi, len(data_chunk))
        if not np.any(valid):
            print(f"[Worker {chunk_id}] SANDI skipped {len(data_chunk)} invalid voxels.")
            return merged
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_obj = sandi.fit(
                    scheme,
                    data_chunk[valid],
                    number_of_processors=1,
                    solver=solver,
                    **solver_kwargs
                )
        for key, dest in merged.items():
            fitted = fit_obj.fitted_parameters.get(key)
            if fitted is not None:
                dest[valid] = np.asarray(fitted, dtype=np.float32).reshape(dest[valid].shape)
        invalid_count = int(np.count_nonzero(~valid))
        if invalid_count:
            print(f"[Worker {chunk_id}] SANDI skipped {invalid_count} invalid voxels.")
        print(f"[Worker {chunk_id}] Finished SANDI chunk.")
        return merged
    except Exception as e:
        print(f"[Worker {chunk_id}] SANDI crash/error: {e}")
        raise e


def _fit_sandi_chunk_gnl(args):
    chunk_id, data_chunk, bvals, bvecs, delta_arr, Delta_arr, model_config, solver, solver_kwargs, gnl_chunk = args
    print(f"[Worker {chunk_id}] Started GNL-aware SANDI chunk with {len(data_chunk)} voxels.")

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    try:
        n_voxels = data_chunk.shape[0]
        merged = None
        failed_voxels = 0
        fallback_voxels = 0
        first_error = None
        base_scheme = _build_dmipy_scheme(bvals, bvecs, delta=delta_arr, Delta=Delta_arr)

        for vox_idx in range(data_chunk.shape[0]):
            sandi = _build_sandi_model(model_config)
            if merged is None:
                merged = _initialize_param_storage(sandi, n_voxels)

            voxel_signal = np.asarray(data_chunk[vox_idx], dtype=np.float32)
            if not _voxel_signal_is_valid(voxel_signal):
                failed_voxels += 1
                continue

            new_bvals, new_bvecs = _safe_rotate_gradients_for_gnl(bvals, bvecs, gnl_chunk[vox_idx])
            scheme = _build_dmipy_scheme(new_bvals, new_bvecs, delta=delta_arr, Delta=Delta_arr)

            try:
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit_obj = sandi.fit(
                            scheme,
                            voxel_signal[None, :],
                            number_of_processors=1,
                            solver=solver,
                            **solver_kwargs
                        )
                _store_param_result(merged, vox_idx, fit_obj.fitted_parameters)
            except Exception as exc:
                try:
                    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            fit_obj = sandi.fit(
                                base_scheme,
                                voxel_signal[None, :],
                                number_of_processors=1,
                                solver=solver,
                                **solver_kwargs
                            )
                    _store_param_result(merged, vox_idx, fit_obj.fitted_parameters)
                    fallback_voxels += 1
                except Exception:
                    failed_voxels += 1
                    if first_error is None:
                        first_error = exc

        if merged is None:
            sandi = _build_sandi_model(model_config)
            merged = _initialize_param_storage(sandi, n_voxels)
            failed_voxels = n_voxels

        if failed_voxels:
            msg = f"[Worker {chunk_id}] GNL-aware SANDI skipped/failed {failed_voxels}/{n_voxels} voxels."
            if first_error is not None:
                msg += f" First error: {first_error}"
            print(msg)
        if fallback_voxels:
            print(f"[Worker {chunk_id}] GNL-aware SANDI fell back to the original scheme for {fallback_voxels}/{n_voxels} voxels.")
        print(f"[Worker {chunk_id}] Finished GNL-aware SANDI chunk.")
        return merged
    except Exception as e:
        print(f"[Worker {chunk_id}] GNL-aware SANDI crash/error: {e}")
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
    device: str = "auto",
    solver_kwargs: Optional[Dict] = None,
    # New options
    model_type: str = "standard", # 'standard' or 'smt'
    fiso_file: Optional[Path] = None,
    grad_nonlin: Optional[Path] = None,
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
    # Imports for NODDI
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from dmipy_fit.signal_models import cylinder_models, gaussian_models
            from dmipy_fit.core.modeling_framework import MultiCompartmentModel, MultiCompartmentSphericalMeanModel
            from dmipy_fit.distributions import distribute_models
    except ImportError as exc:
        raise ProcessingError(f"Dmipy could not be imported: {exc}") from exc

    in_path = extract_image_path(in_file)
    out_dir = ensure_dir(out_dir)
    
    # Extract gradients
    if bval_file is None or bvec_file is None:
        if isinstance(in_file, DWIFile):
             bval_file = bval_file or in_file.bval
             bvec_file = bvec_file or in_file.bvec
             
    if not bval_file or not bvec_file:
         raise ValueError("Gradient files (bval/bvec) are required for NODDI.")

    runtime = DmipyRuntime.resolve(solver=solver, device=device)
    if solver_kwargs is None:
        solver_kwargs = {}
    else:
        solver_kwargs = dict(solver_kwargs)
    
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
    gtab = _build_dmipy_scheme(bvals * 1e6, bvecs)
    
    # Define Mask
    if mask_file and mask_file.exists():
        mask_img = nib.load(str(mask_file))
        mask_data = mask_img.get_fdata().astype(bool)
    else:
        mask_data = None

    gnl_data_flat = None
    if grad_nonlin:
        gnl_img = nib.load(str(grad_nonlin))
        gnl_data = gnl_img.get_fdata()
        if gnl_data.shape[:3] != data.shape[:3]:
            raise ValueError(f"GNL map shape {gnl_data.shape} does not match DWI shape {data.shape}")
        if mask_data is not None:
            gnl_data_flat = gnl_data[mask_data]
        else:
            gnl_data_flat = gnl_data.reshape(-1, *gnl_data.shape[3:])
        
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
    # dmipy-fit's JAX solver vectorizes internally and must not be replicated
    # across the legacy multiprocessing pool.
    n_chunks = 1 if runtime.uses_jax else nthreads
    chunks = np.array_split(valid_voxels, n_chunks)
    
    # Split FISO data if present
    fiso_chunks = [None]*n_chunks
    if fiso_data_flat is not None:
        fiso_chunks = np.array_split(fiso_data_flat, n_chunks)
    gnl_chunks = [None] * n_chunks
    if gnl_data_flat is not None:
        gnl_chunks = np.array_split(gnl_data_flat, n_chunks)
    
    # --- Define Parameter Mapping (Dynamic) ---
    # We need to know parameter names WITHOUT instantiating the full model on all data yet.
    # We can instantiate a dummy model to inspect names.
    
    print(f"Inspecting model parameters for type: {model_type}...")
    
    # Dummy models
    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    
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
    
    if model_type == 'smt':
         dummy_model = MultiCompartmentSphericalMeanModel(models=[dispersed_bundle, ball])
    else:
         dummy_model = MultiCompartmentModel(models=[dispersed_bundle, ball])

    all_params = dummy_model.parameter_names
    param_map = {}
    
    # 1. Global Volume Fractions
    # Usually: partial_volume_0 (ball), partial_volume_1 (bundle)
    # Check if 'partial_volume_0' exists
    if 'partial_volume_1' in all_params:
         param_map['f_iso'] = 'partial_volume_1'
    
    # For 'f_bundle', it's partial_volume_1 if it exists
    if 'partial_volume_0' in all_params:
         param_map['f_bundle'] = 'partial_volume_0'

    # Stick Fraction (intra-bundle)
    pv_candidates = [p for p in all_params if 'partial_volume_0' in p and p != 'partial_volume_0']
    if pv_candidates:
            filtered = [p for p in pv_candidates if distribution.lower() in p.lower()]
            param_map['pv_stick'] = filtered[0] if filtered else pv_candidates[0]
    
    # 2. ODI / SMT specifics
    if model_type == 'standard':
        # ODI logic
        odi_candidates = [p for p in all_params if 'odi' in p.lower() or 'kappa' in p.lower()]
        if odi_candidates:
            filtered = [p for p in odi_candidates if distribution.lower() in p.lower()]
            param_map['odi'] = filtered[0] if filtered else odi_candidates[0]
        

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
             c_fixed['partial_volume_1'] = fiso_chunks[i]

        if gnl_chunks[i] is not None:
            chunk_args.append(
                (i, chunks[i], bvals * 1e6, bvecs, model_config, c_fixed, solver, solver_kwargs, gnl_chunks[i])
            )
        else:
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
    pool = ctx.Pool(processes=n_chunks)
    try:
        # CRITICAL FIX: Use pool.imap (ordered) instead of imap_unordered.
        # imap yields results in the same order as chunk_args.
        # Since we use np.array_split to create chunks in order, we MUST reassemble them in order.
        worker = _fit_chunk_gnl if gnl_data_flat is not None else _fit_chunk
        iterator = pool.imap(worker, chunk_args)
        
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
    vf_intra = None
    vf_extra = None
    
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

    # Recalculate if standard (and not set by SMT)
    if vf_intra is None and pv0 is not None and f_intra is not None:
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


def fit_sandi(
    in_file: Union[Path, ImageLike],
    out_dir: Path,
    bval_file: Optional[Path] = None,
    bvec_file: Optional[Path] = None,
    delta_file: Optional[Path] = None,
    Delta_file: Optional[Path] = None,
    mask_file: Optional[Path] = None,
    nthreads: int = 1,
    parallel_diffusivity: Optional[float] = None,
    iso_diffusivity: Optional[float] = None,
    solver: str = "brute2fine",
    device: str = "auto",
    solver_kwargs: Optional[Dict] = None,
    grad_nonlin: Optional[Path] = None,
    soma_diffusivity: Optional[float] = None,
    **kwargs
) -> Dict[str, Path]:
    """Fit a native dmipy SANDI model."""
    in_path = extract_image_path(in_file)
    out_dir = ensure_dir(out_dir)

    if isinstance(in_file, DWIFile):
        bval_file = bval_file or in_file.bval
        bvec_file = bvec_file or in_file.bvec
        delta_file = delta_file or in_file.delta
        Delta_file = Delta_file or in_file.Delta

    if not bval_file or not bvec_file:
        raise ValueError("Gradient files (bval/bvec) are required for SANDI.")

    runtime = DmipyRuntime.resolve(solver=solver, device=device)
    if solver_kwargs is None:
        solver_kwargs = {}
    else:
        solver_kwargs = dict(solver_kwargs)

    img = nib.load(str(in_path))
    data = img.get_fdata()
    affine = img.affine
    if data.ndim != 4:
        raise ValueError(f"SANDI input must be a 4D DWI image; found shape {data.shape}.")
    if not isinstance(nthreads, int) or nthreads < 1:
        raise ValueError("nthreads must be a positive integer.")

    # Backward compatibility: iso_diffusivity previously controlled the soma
    # diffusion constant despite its ambiguous name. parallel_diffusivity was
    # accepted but never used because D_in is a fitted SANDI parameter.
    if soma_diffusivity is None:
        soma_diffusivity = iso_diffusivity if iso_diffusivity is not None else 3.0e-9

    bvals_si, bvecs = _load_sandi_gradients(bval_file, bvec_file)
    if bvals_si.size != data.shape[-1]:
        raise ValueError(
            f"DWI has {data.shape[-1]} volumes but gradients contain {bvals_si.size} measurements."
        )
    delta_arr, Delta_arr = _load_sandi_timing(delta_file, Delta_file, bvals_si.size)
    gtab = _build_dmipy_scheme(
        bvals_si,
        bvecs,
        delta=delta_arr,
        Delta=Delta_arr,
    )

    if mask_file and Path(mask_file).exists():
        mask_data = nib.load(str(mask_file)).get_fdata().astype(bool)
        if mask_data.shape != data.shape[:3]:
            raise ValueError(
                f"Mask shape {mask_data.shape} does not match DWI shape {data.shape[:3]}."
            )
        valid_voxels = data[mask_data]
    else:
        mask_data = None
        valid_voxels = data.reshape(-1, data.shape[-1])

    gnl_data_flat = None
    if grad_nonlin:
        gnl_img = nib.load(str(grad_nonlin))
        gnl_data = gnl_img.get_fdata()
        if gnl_data.shape[:3] != data.shape[:3]:
            raise ValueError(f"GNL map shape {gnl_data.shape} does not match DWI shape {data.shape}")
        if mask_data is not None:
            gnl_data_flat = gnl_data[mask_data]
        else:
            gnl_data_flat = gnl_data.reshape(-1, *gnl_data.shape[3:])

    n_chunks = 1 if runtime.uses_jax else nthreads
    chunks = np.array_split(valid_voxels, n_chunks)
    gnl_chunks = np.array_split(gnl_data_flat, n_chunks) if gnl_data_flat is not None else [None] * n_chunks

    model_config = {
        'soma_diffusivity': soma_diffusivity,
    }

    chunk_args = []
    for i in range(n_chunks):
        if chunks[i].shape[0] == 0:
            continue
        if gnl_chunks[i] is not None:
            chunk_args.append((i, chunks[i], bvals_si, bvecs, delta_arr, Delta_arr, model_config, solver, solver_kwargs, gnl_chunks[i]))
        else:
            chunk_args.append((i, chunks[i], gtab, model_config, solver, solver_kwargs))

    print(f"Fitting native dmipy SANDI ({valid_voxels.shape[0]} voxels)...")
    try:
        ctx = multiprocessing.get_context('spawn')
    except ValueError:
        ctx = multiprocessing.get_context('fork')

    worker = _fit_sandi_chunk_gnl if gnl_data_flat is not None else _fit_sandi_chunk
    results = []
    pool = ctx.Pool(processes=n_chunks)
    try:
        for res in pool.imap(worker, chunk_args):
            results.append(res)
    finally:
        pool.close()
        pool.join()

    if not results:
        raise ProcessingError("SANDI fitting produced no results.")

    merged_params = {}
    for key in results[0].keys():
        merged_params[key] = np.concatenate([np.asarray(res[key]) for res in results], axis=0)

    vol_shape = data.shape[:-1]
    full_maps = {}
    for key, values in merged_params.items():
        if values.ndim == 1:
            out_arr = np.zeros(vol_shape, dtype=values.dtype)
        else:
            out_arr = np.zeros(vol_shape + (values.shape[1],), dtype=values.dtype)
        if mask_data is not None:
            out_arr[mask_data] = values
        else:
            out_arr = values.reshape(out_arr.shape)
        full_maps[key] = out_arr

    fsoma, fneurite, fextra = _sandi_fraction_maps(full_maps)
    diameter = full_maps.get('BundleModel_1_S4SphereGaussianPhaseApproximation_1_diameter')
    rsoma = 0.5 * diameter if diameter is not None else None
    d_in = full_maps.get('BundleModel_1_C1Stick_1_lambda_par')
    d_ec = full_maps.get('G1Ball_1_lambda_iso')

    outputs = {}

    entities = get_entities_from_path(in_path)
    entities.pop('desc', None)
    entities.pop('suffix', None)
    entities['model'] = 'SANDI'
    metadata = {
        "ModelName": "SANDI (Soma and Neurite Density Imaging)",
        **runtime.provenance(),
        "FittingMethod": "MultiCompartmentSphericalMeanModel",
        "SomaDiffusivity": float(soma_diffusivity),
        "SomaDiffusivityUnits": "m^2/s",
    }
    metric_units = {
        'fsoma': 'unitless', 'fneurite': 'unitless', 'fextra': 'unitless',
        'Rsoma': 'm', 'd_in': 'm^2/s', 'd_ec': 'm^2/s',
    }

    def save_map(name, array):
        if array is None:
            return
        out_p = out_dir / build_bids_name(entities, suffix=name)
        nib.save(nib.Nifti1Image(array.astype(np.float32), affine), out_p)
        sidecar = out_p.with_name(out_p.name[:-7] + '.json')
        with sidecar.open('w') as f:
            json.dump({**metadata, "Metric": name, "MetricUnits": metric_units[name]}, f, indent=2)
        outputs[name] = out_p

    save_map('fsoma', fsoma)
    save_map('fneurite', fneurite)
    save_map('fextra', fextra)
    save_map('Rsoma', rsoma)
    save_map('d_in', d_in)
    save_map('d_ec', d_ec)

    return outputs
