import os
from pathlib import Path
from typing import Optional, Dict, Union
import nibabel as nib
import numpy as np
import warnings
import multiprocessing

from qmri_neuropipe.core import ProcessingError
from qmri_neuropipe.core.utils import ensure_dir, extract_image_path
from qmri_neuropipe.core.types import ImageLike, DWIFile
from qmri_neuropipe.interfaces.dmipy import (
    _rotate_gradients_for_gnl,
    _build_dmipy_scheme,
    _initialize_param_storage,
    _store_param_result,
    _voxel_signal_is_valid,
    _safe_rotate_gradients_for_gnl,
)
from qmri_neuropipe.interfaces.dmipy_backend import (
    DmipyRuntime,
    collect_pool_results_with_heartbeat,
    dmipy_fit_output,
    install_dmipy_jax_postprocessing_workaround,
    jax_run_summary,
)
from qmri_neuropipe.interfaces.dmipy_derivatives import write_dmipy_derivatives
from qmri_neuropipe.interfaces.dmipy_models import (
    microglia as build_microglia_reference_model,
    microglia_output_alias,
    microglia_output_maps,
)


def _load_microglia_gradients(bval_file, bvec_file):
    """Load FSL-style gradients and convert b-values to dmipy SI units."""
    bvals_fsl = np.asarray(np.loadtxt(bval_file), dtype=float).reshape(-1)
    bvecs = np.asarray(np.loadtxt(bvec_file), dtype=float)
    if bvecs.ndim != 2:
        raise ValueError("b-vectors must be a two-dimensional 3 x N or N x 3 array.")
    if bvecs.shape == (3, bvals_fsl.size):
        bvecs = bvecs.T
    elif bvecs.shape != (bvals_fsl.size, 3):
        raise ValueError(
            f"b-vector shape {bvecs.shape} is incompatible with "
            f"{bvals_fsl.size} b-values; expected 3 x N or N x 3."
        )
    if not np.all(np.isfinite(bvals_fsl)) or np.any(bvals_fsl < 0):
        raise ValueError("b-values must be finite and non-negative.")
    if not np.all(np.isfinite(bvecs)):
        raise ValueError("b-vectors must be finite.")

    norms = np.linalg.norm(bvecs, axis=1)
    diffusion_weighted = bvals_fsl > 10.0
    if np.any(norms[diffusion_weighted] <= 0):
        raise ValueError("Every diffusion-weighted volume must have a non-zero b-vector.")
    if np.any(np.abs(norms[diffusion_weighted] - 1.0) > 0.1):
        raise ValueError("Diffusion-weighted b-vectors must have approximately unit norm.")
    nonzero = norms > 0
    bvecs[nonzero] /= norms[nonzero, None]
    return bvals_fsl * 1e6, bvecs


def _load_microglia_timing(delta_file, Delta_file, n_measurements):
    """Load required PGSE timing in seconds and validate its physical ordering."""
    if not delta_file or not Delta_file:
        raise ProcessingError(
            "Microglia sphere fitting requires both small-delta and big-Delta "
            "timing files, with values in seconds."
        )

    def load_timing(path, label):
        values = np.asarray(np.loadtxt(Path(path)), dtype=float).reshape(-1)
        if values.size not in (1, n_measurements):
            raise ValueError(
                f"{label} timing must contain either one value or "
                f"{n_measurements} values; found {values.size}."
            )
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"{label} timing must be finite and positive seconds.")
        return np.full(n_measurements, values[0]) if values.size == 1 else values

    delta = load_timing(delta_file, "small-delta")
    Delta = load_timing(Delta_file, "big-Delta")
    if np.any(Delta <= delta):
        raise ValueError("Every big-Delta value must be greater than small-delta.")
    return delta, Delta


def _microglia_metric_name(parameter_name):
    """Map dmipy parameter names to stable, interpretable derivative suffixes."""
    return microglia_output_alias(parameter_name)


def _microglia_metric_metadata(metric):
    metadata = {
        "f_bundle": ("Signal fraction of the dispersed stick/zeppelin bundle.", "unitless"),
        "f_small_sphere": ("Signal fraction of the small-sphere compartment.", "unitless"),
        "f_large_sphere": ("Signal fraction of the large-sphere compartment.", "unitless"),
        "f_iso": ("Signal fraction of the isotropic ball compartment.", "unitless"),
        "f_stick": (
            "Whole-voxel restricted stick signal fraction (paper f_IC).",
            "unitless",
        ),
        "f_extracellular": (
            "Whole-voxel hindered extracellular tensor signal fraction (paper f_EC).",
            "unitless",
        ),
        "f_tissue": ("Tissue signal fraction, calculated as 1 - f_iso (paper f_T).", "unitless"),
        "bundle_stick_fraction": (
            "Stick fraction within the dispersed bundle; not a whole-voxel fraction.",
            "unitless",
        ),
        "odi": ("Watson orientation dispersion index.", "unitless"),
        "watson_kappa": (
            "Watson concentration converted from ODI as 1/tan(pi*ODI/2).",
            "unitless",
        ),
        "mu": (
            "Watson mean orientation as polar and azimuthal spherical angles.",
            "radian",
        ),
        "bundle_radial_diffusivity": (
            "Radial diffusivity of the bundle zeppelin compartment.",
            "m^2/s",
        ),
        "small_sphere_diameter": ("Fitted small-sphere diameter.", "m"),
        "large_sphere_diameter": ("Fitted large-sphere diameter.", "m"),
        "small_sphere_radius": ("Fitted small-sphere radius (paper R_SS).", "m"),
        "large_sphere_radius": ("Fitted large-sphere radius (paper R_LS).", "m"),
    }
    description, units = metadata.get(metric, ("Fitted dmipy model parameter.", "unknown"))
    result = {"MetricDescription": description, "MetricUnits": units}
    if metric == "mu":
        result["MetricComponents"] = ["polar_angle", "azimuthal_angle"]
    return result


def _add_paper_microglia_maps(full_maps):
    """Add paper-facing fractions, radii, and Watson concentration in place."""
    full_maps.update(microglia_output_maps(full_maps))
    return full_maps


def _build_microglia_model(
    cylinder_models,
    gaussian_models,
    sphere_models,
    distribute_models,
    MultiCompartmentModel,
    model_config,
):
    return build_microglia_reference_model(
        parallel_diffusivity=float(
            model_config.get("parallel_diffusivity", 1.0e-9)
        ),
        iso_diffusivity=float(model_config.get("iso_diffusivity", 3.0e-9)),
        small_diameter=float(model_config.get("small_diameter", 8e-6)),
        large_diameter=float(model_config.get("large_diameter", 16e-6)),
        small_diameter_bounds=tuple(
            model_config.get("small_diameter_bounds", (5e-6, 11e-6))
        ),
        large_diameter_bounds=tuple(
            model_config.get("large_diameter_bounds", (12e-6, 18e-6))
        ),
        _components={
            "cylinder_models": cylinder_models,
            "gaussian_models": gaussian_models,
            "sphere_models": sphere_models,
            "distribute_models": distribute_models,
            "MultiCompartmentModel": MultiCompartmentModel,
        },
    )


def _fit_microglia_chunk(args):
    chunk_id, data_chunk, scheme, model_config, solver, solver_kwargs = args
    batch_size = solver_kwargs.get("batch_size") if solver == "jax" else None
    batch_note = f"; optimizer batch size {batch_size}" if batch_size else ""
    print(
        f"[Worker {chunk_id}] Received {len(data_chunk)} Microglia voxels"
        f"{batch_note}. Initializing dmipy fit...",
        flush=True,
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from dmipy_fit.signal_models import cylinder_models, gaussian_models, sphere_models
        from dmipy_fit.distributions import distribute_models
        from dmipy_fit.core.modeling_framework import MultiCompartmentModel

    worker_threads = (
        os.environ.get("QMRI_DMIPY_WORKER_THREADS", "1")
        if solver == "jax"
        else "1"
    )
    os.environ["OMP_NUM_THREADS"] = worker_threads
    os.environ["MKL_NUM_THREADS"] = worker_threads
    os.environ["OPENBLAS_NUM_THREADS"] = worker_threads
    if solver == "jax" and install_dmipy_jax_postprocessing_workaround():
        print(
            "Enabled fast NumPy conversion of dmipy JAX fitted fractions.",
            flush=True,
        )
    
    try:
        microglia_model = _build_microglia_model(
            cylinder_models,
            gaussian_models,
            sphere_models,
            distribute_models,
            MultiCompartmentModel,
            model_config,
        )
        
        # Fit
        with dmipy_fit_output(solver):
             with warnings.catch_warnings():
                 warnings.simplefilter("ignore")
                 fit_obj = microglia_model.fit(
                    scheme, 
                    data_chunk, 
                    number_of_processors=1,
                    solver=solver,
                    **solver_kwargs
                )
        
        print(f"[Worker {chunk_id}] Finished fitting.", flush=True)
        return fit_obj.fitted_parameters
        
    except Exception as e:
        print(f"[Worker {chunk_id}] Crash/Error: {e}")
        raise e


def _fit_microglia_chunk_gnl(args):
    chunk_id, data_chunk, bvals, bvecs, delta_arr, Delta_arr, model_config, solver, solver_kwargs, gnl_chunk = args
    print(
        f"[Worker {chunk_id}] Received {len(data_chunk)} GNL-aware Microglia voxels.",
        flush=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from dmipy_fit.signal_models import cylinder_models, gaussian_models, sphere_models
        from dmipy_fit.distributions import distribute_models
        from dmipy_fit.core.modeling_framework import MultiCompartmentModel

    worker_threads = (
        os.environ.get("QMRI_DMIPY_WORKER_THREADS", "1")
        if solver == "jax"
        else "1"
    )
    os.environ["OMP_NUM_THREADS"] = worker_threads
    os.environ["MKL_NUM_THREADS"] = worker_threads
    os.environ["OPENBLAS_NUM_THREADS"] = worker_threads
    if solver == "jax" and install_dmipy_jax_postprocessing_workaround():
        print(
            "Enabled fast NumPy conversion of dmipy JAX fitted fractions.",
            flush=True,
        )

    try:
        n_voxels = data_chunk.shape[0]
        if solver == "jax":
            from .dmipy_jax_gnl import fit_model_jax_gnl

            model = _build_microglia_model(
                cylinder_models,
                gaussian_models,
                sphere_models,
                distribute_models,
                MultiCompartmentModel,
                model_config,
            )
            scheme = _build_dmipy_scheme(
                bvals,
                bvecs,
                delta=delta_arr,
                Delta=Delta_arr,
            )
            fit_obj = fit_model_jax_gnl(
                model,
                scheme,
                data_chunk,
                gnl_chunk,
                solver_kwargs=solver_kwargs,
            )
            print(
                f"[Worker {chunk_id}] Finished voxel-parallel JAX GNL fit.",
                flush=True,
            )
            return {
                key: np.asarray(value)
                for key, value in fit_obj.fitted_parameters.items()
            }
        merged = None
        failed_voxels = 0
        fallback_voxels = 0
        first_error = None
        base_scheme = _build_dmipy_scheme(bvals, bvecs, delta=delta_arr, Delta=Delta_arr)

        for vox_idx in range(data_chunk.shape[0]):
            model = _build_microglia_model(
                cylinder_models,
                gaussian_models,
                sphere_models,
                distribute_models,
                MultiCompartmentModel,
                model_config,
            )
            if merged is None:
                merged = _initialize_param_storage(model, n_voxels)

            voxel_signal = np.asarray(data_chunk[vox_idx], dtype=np.float32)
            if not _voxel_signal_is_valid(voxel_signal):
                failed_voxels += 1
                continue

            new_bvals, new_bvecs = _safe_rotate_gradients_for_gnl(bvals, bvecs, gnl_chunk[vox_idx])
            scheme = _build_dmipy_scheme(new_bvals, new_bvecs, delta=delta_arr, Delta=Delta_arr)

            try:
                with dmipy_fit_output(solver):
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
                    with dmipy_fit_output(solver):
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
            model = _build_microglia_model(
                cylinder_models,
                gaussian_models,
                sphere_models,
                distribute_models,
                MultiCompartmentModel,
                model_config,
            )
            merged = _initialize_param_storage(model, n_voxels)
            failed_voxels = n_voxels

        if failed_voxels:
            msg = f"[Worker {chunk_id}] GNL-aware Microglia skipped/failed {failed_voxels}/{n_voxels} voxels."
            if first_error is not None:
                msg += f" First error: {first_error}"
            print(msg)
        if fallback_voxels:
            print(f"[Worker {chunk_id}] GNL-aware Microglia fell back to the original scheme for {fallback_voxels}/{n_voxels} voxels.")
        print(f"[Worker {chunk_id}] Finished GNL-aware Microglia chunk.")
        return merged
    except Exception as e:
        print(f"[Worker {chunk_id}] GNL-aware Microglia crash/error: {e}")
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
    parallel_diffusivity: float = 1.0e-9,
    iso_diffusivity: float = 3.0e-9,
    small_diameter: float = 8e-6,
    large_diameter: float = 16e-6,
    small_diameter_bounds=(5e-6, 11e-6),
    large_diameter_bounds=(12e-6, 18e-6),
    solver: str = "brute2fine",
    device: str = "auto",
    gpu_device: Optional[int] = None,
    jax_cache_dir: Optional[Path] = None,
    jax_log_compiles: bool = False,
    heartbeat_interval: float = 30.0,
    solver_kwargs: Optional[Dict] = None,
    Ns: int = 5,
    maxiter: int = 300,
    N_sphere_samples: int = 30,
    grad_nonlin: Optional[Path] = None,
    **kwargs
) -> Dict[str, Path]:
    """
    Fit the 4-compartment Microglia model using Dmipy.
    """
    if not isinstance(nthreads, int) or nthreads < 1:
        raise ValueError("nthreads must be a positive integer.")

    # 1. Environment Variables and Numba setup
    worker_threads = nthreads if str(solver).lower() == "jax" else 1
    os.environ["QMRI_DMIPY_WORKER_THREADS"] = str(worker_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(worker_threads)
    os.environ["MKL_NUM_THREADS"] = str(worker_threads)
    os.environ["OMP_NUM_THREADS"] = str(worker_threads)
    os.environ["NUMBA_NUM_THREADS"] = str(worker_threads)
    os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
    os.environ["JOBLIB_START_METHOD"] = "fork"

    try:
        import numba
        numba.set_num_threads(worker_threads)
        if hasattr(numba, 'config'):
            numba.config.THREADING_LAYER = 'workqueue'
        numba.set_num_threads = lambda n: None
    except (ImportError, RuntimeError):
        pass

    in_path = extract_image_path(in_file)
    out_dir = ensure_dir(out_dir)
    
    if isinstance(in_file, DWIFile):
         bval_file = bval_file or in_file.bval
         bvec_file = bvec_file or in_file.bvec
         delta_file = delta_file or in_file.delta
         Delta_file = Delta_file or in_file.Delta
             
    if not bval_file or not bvec_file:
         raise ValueError("Gradient files (bval/bvec) are required.")

    runtime = DmipyRuntime.resolve(
        solver=solver,
        device=device,
        gpu_device=gpu_device,
        jax_cache_dir=jax_cache_dir,
        jax_log_compiles=jax_log_compiles,
    )
    solver = runtime.solver
    if solver_kwargs is None:
        solver_kwargs = {}
    else:
        solver_kwargs = dict(solver_kwargs)
    solver_kwargs.setdefault('Ns', int(Ns))
    solver_kwargs.setdefault('maxiter', int(maxiter))
    solver_kwargs.setdefault('N_sphere_samples', int(N_sphere_samples))
    
    img = nib.load(str(in_path))
    data = img.get_fdata()
    affine = img.affine
    
    bvals, bvecs = _load_microglia_gradients(bval_file, bvec_file)
    if bvals.size != data.shape[-1]:
        raise ValueError(
            f"Found {bvals.size} gradient entries for a DWI with "
            f"{data.shape[-1]} volumes."
        )
    delta_arr, Delta_arr = _load_microglia_timing(
        delta_file, Delta_file, bvals.size
    )
    gtab = _build_dmipy_scheme(
        bvals,
        bvecs,
        delta=delta_arr,
        Delta=Delta_arr,
    )

    if mask_file and mask_file.exists():
        mask_data = nib.load(str(mask_file)).get_fdata().astype(bool)
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
        
    n_voxels = valid_voxels.shape[0]
    n_chunks = 1 if runtime.uses_jax else nthreads
    chunks = np.array_split(valid_voxels, n_chunks)
    gnl_chunks = np.array_split(gnl_data_flat, n_chunks) if gnl_data_flat is not None else [None] * n_chunks
    
    model_config = {
        'parallel_diffusivity': parallel_diffusivity,
        'iso_diffusivity': iso_diffusivity,
        'small_diameter': small_diameter,
        'large_diameter': large_diameter,
        'small_diameter_bounds': small_diameter_bounds,
        'large_diameter_bounds': large_diameter_bounds,
    }
    
    chunk_args = []
    for i in range(n_chunks):
        if chunks[i].shape[0] == 0: continue
        if gnl_chunks[i] is not None:
            chunk_args.append(
                (i, chunks[i], bvals, bvecs, delta_arr, Delta_arr, model_config, solver, solver_kwargs, gnl_chunks[i])
            )
        else:
            chunk_args.append(
                (i, chunks[i], gtab, model_config, solver, solver_kwargs)
            )

    print(f"Fitting Microglia Model ({n_voxels} voxels)...")
    for line in jax_run_summary(runtime, n_voxels, solver_kwargs):
        print(line, flush=True)
    
    # Run Pool
    try:
         ctx = multiprocessing.get_context('spawn')
    except ValueError:
         ctx = multiprocessing.get_context('fork')
         
    pool = ctx.Pool(processes=n_chunks)
    results = []
    try:
        worker = _fit_microglia_chunk_gnl if gnl_data_flat is not None else _fit_microglia_chunk
        if runtime.uses_jax:
            results = collect_pool_results_with_heartbeat(
                pool,
                worker,
                chunk_args,
                heartbeat_interval=heartbeat_interval,
                label="Microglia JAX fitting",
            )
        else:
            for i, res in enumerate(pool.imap(worker, chunk_args)):
                results.append(res)
                print(f"  - Collected chunk {i + 1}/{len(chunk_args)}")
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    else:
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

    # Paper-facing quantities derived from dmipy's nested bundle and diameter
    # parameterization. Raw fitted maps remain available alongside these maps.
    _add_paper_microglia_maps(full_maps)
        
    aliases = {key: _microglia_metric_name(key) for key in full_maps}
    sidecar = {
        "ModelName": "Microglia (4-Compartment)",
        "ModelReference": "https://doi.org/10.1126/sciadv.abq2923",
        "BValueInputUnits": "s/mm^2",
        "BValueFittingUnits": "s/m^2",
        "SmallDeltaSeconds": {
            "Minimum": float(np.min(delta_arr)),
            "Maximum": float(np.max(delta_arr)),
        },
        "BigDeltaSeconds": {
            "Minimum": float(np.min(Delta_arr)),
            "Maximum": float(np.max(Delta_arr)),
        },
        "SphereSignalModel": "S2SphereStejskalTannerApproximation",
        "SmallSphereDiameterBoundsMeters": list(map(float, small_diameter_bounds)),
        "LargeSphereDiameterBoundsMeters": list(map(float, large_diameter_bounds)),
        "ParallelDiffusivityMetersSquaredPerSecond": parallel_diffusivity,
        "IsotropicDiffusivityMetersSquaredPerSecond": iso_diffusivity,
        "Solver": solver,
        "SolverOptions": solver_kwargs,
        "WatsonMeanOrientationFitted": True,
    }
    written = write_dmipy_derivatives(
        out_dir,
        in_path,
        affine,
        full_maps,
        runtime,
        model_name="microglia",
        model_label="Microglia",
        output_aliases=aliases,
        base_metadata=sidecar,
        parameter_metadata={
            key: {
                "Metric": alias,
                **_microglia_metric_metadata(alias),
            }
            for key, alias in aliases.items()
        },
    )
    return {aliases[key]: path for key, path in written.items()}
