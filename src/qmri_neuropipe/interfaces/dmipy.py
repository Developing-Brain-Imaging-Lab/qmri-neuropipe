
from pathlib import Path
import os
import multiprocessing
import argparse
from typing import Optional, Dict, Union, Any
import nibabel as nib
import numpy as np
import warnings


from ..core import ProcessingError
from ..core.utils import ensure_dir, extract_image_path
from ..core.types import ImageLike, DWIFile
from .dmipy_backend import (
    DmipyRuntime,
    DmipyFitRequest,
    acquisition_scheme_from_bvalues,
    collect_pool_results_with_heartbeat,
    dmipy_fit_output,
    execute_dmipy_fit,
    install_dmipy_jax_postprocessing_workaround,
    jax_run_summary,
)
from .dmipy_derivatives import write_dmipy_derivatives


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


def _load_external_fiso(
    fiso_file: Path,
    dwi_img: nib.spatialimages.SpatialImage,
    mask_data: Optional[np.ndarray],
) -> np.ndarray:
    """Load and validate a voxelwise FISO map in the DWI grid."""
    fiso_img = nib.load(str(fiso_file))
    expected_shape = dwi_img.shape[:-1]
    if fiso_img.shape != expected_shape:
        raise ValueError(
            f"FISO map shape {fiso_img.shape} does not match DWI shape "
            f"{expected_shape}."
        )
    if not np.allclose(fiso_img.affine, dwi_img.affine, rtol=1e-5, atol=1e-5):
        raise ValueError(
            "FISO map affine does not match the DWI affine. Resample the FISO "
            "map into the DWI grid before fitting."
        )

    fiso_data = np.asarray(fiso_img.get_fdata(), dtype=float)
    selected = (
        fiso_data[mask_data]
        if mask_data is not None
        else fiso_data.reshape(-1)
    )
    if not np.all(np.isfinite(selected)):
        raise ValueError("FISO values inside the fitting mask must be finite.")
    if np.any((selected < 0.0) | (selected > 1.0)):
        minimum = float(np.min(selected))
        maximum = float(np.max(selected))
        raise ValueError(
            "FISO values inside the fitting mask must be within [0, 1]; "
            f"observed range [{minimum:.6g}, {maximum:.6g}]."
        )
    return selected


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
    from .dmipy_models import noddi_variant

    return noddi_variant(
        parallel_diffusivity=float(
            model_config.get("parallel_diffusivity", 1.7e-9)
        ),
        iso_diffusivity=float(model_config.get("iso_diffusivity", 3.0e-9)),
        distribution=model_config.get("distribution", "Watson"),
        model_type=model_config.get("model_type", "standard"),
        fixed_parameters=fixed_params,
    )


def _build_sandi_model(model_config):
    from .dmipy_models import sandi_spherical_mean

    return sandi_spherical_mean(
        soma_diffusivity=float(model_config.get("soma_diffusivity", 3.0e-9))
    )


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


def _validate_sandi_fit_results(merged_params):
    """Reject an all-failed fit before empty SANDI derivatives are written."""
    if not merged_params:
        raise ProcessingError("SANDI fitting returned no model parameters.")

    fitted_voxels = None
    for values in merged_params.values():
        array = np.asarray(values)
        if array.ndim == 0:
            continue
        finite = np.all(
            np.isfinite(array),
            axis=tuple(range(1, array.ndim)),
        )
        fitted_voxels = finite if fitted_voxels is None else fitted_voxels | finite

    if fitted_voxels is None or not np.any(fitted_voxels):
        raise ProcessingError(
            "SANDI fitting failed for every voxel; no derivative maps were "
            "written. Review the first worker error and the selected "
            "solver/model combination."
        )
    return fitted_voxels


_NODDI_OUTPUT_SPECS = {
    "odi": (
        "ODI",
        "Orientation dispersion index.",
    ),
    "fiso": (
        "FISO",
        "Isotropic free-water signal fraction.",
    ),
    "vf_intra": (
        "ICVF",
        "Absolute intracellular neurite signal fraction.",
    ),
    "vf_extra": (
        "EXVF",
        "Absolute extracellular tissue signal fraction.",
    ),
}


def _save_noddi_outputs(
    out_dir: Path,
    in_path: Path,
    affine: np.ndarray,
    metric_arrays: Dict[str, Optional[np.ndarray]],
    runtime: DmipyRuntime,
    *,
    model_type: str,
    distribution: str,
    parallel_diffusivity: float,
    iso_diffusivity: float,
    solver_kwargs: Dict[str, Any],
    fiso_constrained: bool,
) -> Dict[str, Path]:
    """Write NODDI maps and sidecars using the project BIDS derivative scheme."""
    aliases = {
        key: suffix
        for key, (suffix, _) in _NODDI_OUTPUT_SPECS.items()
    }
    parameter_metadata = {
        key: {
            "Metric": suffix,
            "MetricDescription": description,
            "MetricUnits": "unitless",
        }
        for key, (suffix, description) in _NODDI_OUTPUT_SPECS.items()
    }
    return write_dmipy_derivatives(
        out_dir,
        in_path,
        affine,
        metric_arrays,
        runtime,
        model_name="noddi",
        output_aliases=aliases,
        base_metadata={
            "ModelName": (
                "NODDI (Neurite Orientation Dispersion and Density Imaging)"
            ),
            "FittingMethod": "dmipy-fit multi-compartment optimization",
            "ModelType": model_type,
            "OrientationDistribution": distribution,
            "ParallelDiffusivity": float(parallel_diffusivity),
            "IsotropicDiffusivity": float(iso_diffusivity),
            "DiffusivityUnits": "m^2/s",
            "ExternalFISOConstraint": bool(fiso_constrained),
            "SolverOptions": dict(solver_kwargs),
        },
        parameter_metadata=parameter_metadata,
    )


def _fit_chunk(args):
    """
    Helper function to fit a chunk of data in a separate process.
    Models are instantiated LOCALLY to support chunk-specific fixed parameters (e.g., FISO map).
    """
    chunk_id, data_chunk, scheme, model_config, chunk_fixed_params, keys_to_keep, solver, solver_kwargs = args
    
    batch_size = solver_kwargs.get("batch_size") if solver == "jax" else None
    batch_note = f"; optimizer batch size {batch_size}" if batch_size else ""
    print(
        f"[Worker {chunk_id}] Received {len(data_chunk)} voxels{batch_note}. "
        "Initializing dmipy fit...",
        flush=True,
    )
    
    # Native workers are single-threaded; the sole JAX worker may use the
    # CPU-thread allowance requested by the caller for setup and compilation.
    import os
    import sys
    import warnings
    
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
        noddi = _build_noddi_model(
            model_config,
            fixed_params={
                key: value
                for key, value in (chunk_fixed_params or {}).items()
                if value is not None
            },
        )

        runtime = DmipyRuntime.resolve(solver=solver, device="auto")
        fit_obj = execute_dmipy_fit(
            DmipyFitRequest(
                model_name="noddi",
                model=noddi,
                acquisition_scheme=scheme,
                data=data_chunk,
                runtime=runtime,
                nthreads=1,
                solver_options=solver_kwargs,
                heartbeat_interval=None,
            )
        ).fitted
        
        # Return only the requested parameters
        full_params = fit_obj.fitted_parameters
        if keys_to_keep:
            # Warning: SMT keys might differ from keys_to_keep logic. 
            # Ideally caller knows what to ask for, or we return everything if unsure.
            ret = {k: v for k, v in full_params.items() if k in keys_to_keep}
        else:
            ret = full_params
            
        print(f"[Worker {chunk_id}] Finished fitting.", flush=True)
        return ret
        
    except Exception as e:
        print(f"[Worker {chunk_id}] Crash/Error: {e}")
        raise e


def _fit_chunk_gnl(args):
    chunk_id, data_chunk, bvals, bvecs, model_config, chunk_fixed_params, solver, solver_kwargs, gnl_chunk = args

    print(
        f"[Worker {chunk_id}] Received {len(data_chunk)} GNL-aware voxels.",
        flush=True,
    )

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
        if (
            solver == "jax"
            and not chunk_fixed_params
            and str(model_config.get("model_type", "standard")).lower() != "smt"
        ):
            model = _build_noddi_model(model_config, fixed_params={})
            scheme = _build_dmipy_scheme(bvals, bvecs)
            runtime = DmipyRuntime.resolve(solver=solver, device="auto")
            fit_obj = execute_dmipy_fit(
                DmipyFitRequest(
                    model_name="noddi",
                    model=model,
                    acquisition_scheme=scheme,
                    data=data_chunk,
                    runtime=runtime,
                    gradient_tensors=gnl_chunk,
                    nthreads=1,
                    solver_options=solver_kwargs,
                    heartbeat_interval=None,
                )
            ).fitted
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
    batch_size = solver_kwargs.get("batch_size") if solver == "jax" else None
    batch_note = f"; optimizer batch size {batch_size}" if batch_size else ""
    print(
        f"[Worker {chunk_id}] Received {len(data_chunk)} SANDI voxels"
        f"{batch_note}. Initializing dmipy fit...",
        flush=True,
    )

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
        sandi = _build_sandi_model(model_config)
        valid = np.asarray([_voxel_signal_is_valid(v) for v in data_chunk])
        merged = _initialize_param_storage(sandi, len(data_chunk))
        if not np.any(valid):
            print(f"[Worker {chunk_id}] SANDI skipped {len(data_chunk)} invalid voxels.")
            return merged
        if solver == "jax":
            from .dmipy_sandi_jax import fit_sandi_jax

            fitted = fit_sandi_jax(
                sandi,
                scheme,
                data_chunk,
                solver_kwargs=solver_kwargs,
            )
            print(
                f"[Worker {chunk_id}] Finished vectorized JAX SANDI chunk.",
                flush=True,
            )
            return fitted
        runtime = DmipyRuntime.resolve(solver=solver, device="auto")
        fit_obj = execute_dmipy_fit(
            DmipyFitRequest(
                model_name="sandi",
                model=sandi,
                acquisition_scheme=scheme,
                data=data_chunk[valid],
                runtime=runtime,
                nthreads=1,
                solver_options=solver_kwargs,
                heartbeat_interval=None,
            )
        ).fitted
        for key, dest in merged.items():
            fitted = fit_obj.fitted_parameters.get(key)
            if fitted is not None:
                dest[valid] = np.asarray(fitted, dtype=np.float32).reshape(dest[valid].shape)
        invalid_count = int(np.count_nonzero(~valid))
        if invalid_count:
            print(f"[Worker {chunk_id}] SANDI skipped {invalid_count} invalid voxels.")
        print(f"[Worker {chunk_id}] Finished SANDI chunk.", flush=True)
        return merged
    except Exception as e:
        print(f"[Worker {chunk_id}] SANDI crash/error: {e}")
        raise e


def _fit_sandi_chunk_gnl(args):
    chunk_id, data_chunk, bvals, bvecs, delta_arr, Delta_arr, model_config, solver, solver_kwargs, gnl_chunk = args
    print(
        f"[Worker {chunk_id}] Received {len(data_chunk)} GNL-aware SANDI voxels.",
        flush=True,
    )

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
        sandi = _build_sandi_model(model_config)
        if solver == "jax":
            from .dmipy_sandi_jax import fit_sandi_jax

            scheme = _build_dmipy_scheme(
                bvals,
                bvecs,
                delta=delta_arr,
                Delta=Delta_arr,
            )
            fitted = fit_sandi_jax(
                sandi,
                scheme,
                data_chunk,
                gradient_tensors=gnl_chunk,
                solver_kwargs=solver_kwargs,
            )
            print(
                f"[Worker {chunk_id}] Finished vectorized JAX GNL SANDI fit.",
                flush=True,
            )
            return fitted
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
                with dmipy_fit_output(solver):
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
                    with dmipy_fit_output(solver):
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
    gpu_device: Optional[int] = None,
    jax_cache_dir: Optional[Path] = None,
    jax_log_compiles: bool = False,
    heartbeat_interval: float = 30.0,
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
    if not isinstance(nthreads, int) or nthreads < 1:
        raise ValueError("nthreads must be a positive integer.")

    # --- Parallelization Config (MUST BE FIRST) ---
    # These configurations must run before any imports that might initialize libraries
    import os
    import multiprocessing

    # 1. Environment Variables
    worker_threads = nthreads if str(solver).lower() == "jax" else 1
    os.environ["QMRI_DMIPY_WORKER_THREADS"] = str(worker_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(worker_threads)
    os.environ["MKL_NUM_THREADS"] = str(worker_threads)
    os.environ["OMP_NUM_THREADS"] = str(worker_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(worker_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(worker_threads)
    os.environ["NUMBA_NUM_THREADS"] = str(worker_threads)
    os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
    os.environ["JOBLIB_START_METHOD"] = "fork"

    # Device visibility and JAX diagnostics must be configured before any
    # dmipy module has an opportunity to import JAX.
    runtime = DmipyRuntime.resolve(
        solver=solver,
        device=device,
        gpu_device=gpu_device,
        jax_cache_dir=jax_cache_dir,
        jax_log_compiles=jax_log_compiles,
    )
    solver = runtime.solver

    # 2. Numba Monkeypatch
    try:
        import numba
        numba.set_num_threads(worker_threads)
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
        
    try:
        dummy_model = _build_noddi_model(
            {
                "model_type": model_type,
                "distribution": distribution,
                "parallel_diffusivity": parallel_diffusivity,
                "iso_diffusivity": iso_diffusivity,
            }
        )
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
            fiso_data_flat = _load_external_fiso(fiso_path, img, mask_data)
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
    for line in jax_run_summary(runtime, n_voxels, solver_kwargs):
        print(line, flush=True)
    
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
    fiso_constrained = fiso_data_flat is not None
    print(
        f"Fitting NODDI (Type: {model_type}, "
        f"FISO constraint: {fiso_constrained})..."
    )
    
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
        worker = _fit_chunk_gnl if gnl_data_flat is not None else _fit_chunk
        if runtime.uses_jax:
            results = collect_pool_results_with_heartbeat(
                pool,
                worker,
                chunk_args,
                heartbeat_interval=heartbeat_interval,
                label="NODDI JAX fitting",
            )
        else:
            # Ordered collection preserves the spatial order of the chunks.
            for i, res in enumerate(pool.imap(worker, chunk_args)):
                results.append(res)
                print(f"  - Collected chunk {i + 1}/{len(chunk_args)}")
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
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

        
    return _save_noddi_outputs(
        out_dir,
        in_path,
        affine,
        {
            "odi": odi_map,
            "fiso": f_iso,
            "vf_intra": vf_intra,
            "vf_extra": vf_extra,
        },
        runtime,
        model_type=model_type,
        distribution=distribution,
        parallel_diffusivity=parallel_diffusivity,
        iso_diffusivity=iso_diffusivity,
        solver_kwargs=solver_kwargs,
        fiso_constrained=fiso_constrained,
    )


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
    gpu_device: Optional[int] = None,
    jax_cache_dir: Optional[Path] = None,
    jax_log_compiles: bool = False,
    heartbeat_interval: float = 30.0,
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

    worker_threads = nthreads if str(solver).lower() == "jax" else 1
    os.environ["QMRI_DMIPY_WORKER_THREADS"] = str(worker_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(worker_threads)
    os.environ["MKL_NUM_THREADS"] = str(worker_threads)
    os.environ["OMP_NUM_THREADS"] = str(worker_threads)
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
    for line in jax_run_summary(runtime, valid_voxels.shape[0], solver_kwargs):
        print(line, flush=True)
    try:
        ctx = multiprocessing.get_context('spawn')
    except ValueError:
        ctx = multiprocessing.get_context('fork')

    worker = _fit_sandi_chunk_gnl if gnl_data_flat is not None else _fit_sandi_chunk
    results = []
    pool = ctx.Pool(processes=n_chunks)
    try:
        if runtime.uses_jax:
            results = collect_pool_results_with_heartbeat(
                pool,
                worker,
                chunk_args,
                heartbeat_interval=heartbeat_interval,
                label="SANDI JAX fitting",
            )
        else:
            for res in pool.imap(worker, chunk_args):
                results.append(res)
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()

    if not results:
        raise ProcessingError("SANDI fitting produced no results.")

    merged_params = {}
    for key in results[0].keys():
        merged_params[key] = np.concatenate([np.asarray(res[key]) for res in results], axis=0)
    fitted_voxels = _validate_sandi_fit_results(merged_params)
    failed_voxels = int(np.count_nonzero(~fitted_voxels))
    if failed_voxels:
        print(
            f"SANDI produced valid parameters for "
            f"{len(fitted_voxels) - failed_voxels}/{len(fitted_voxels)} "
            "voxels; failed voxels will remain non-finite.",
            flush=True,
        )

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

    metric_units = {
        'fsoma': 'unitless', 'fneurite': 'unitless', 'fextra': 'unitless',
        'Rsoma': 'm', 'd_in': 'm^2/s', 'd_ec': 'm^2/s',
    }
    metric_maps = {
        'fsoma': fsoma,
        'fneurite': fneurite,
        'fextra': fextra,
        'Rsoma': rsoma,
        'd_in': d_in,
        'd_ec': d_ec,
    }
    return write_dmipy_derivatives(
        out_dir,
        in_path,
        affine,
        metric_maps,
        runtime,
        model_name="sandi",
        output_aliases={name: name for name in metric_maps},
        base_metadata={
            "ModelName": "SANDI (Soma and Neurite Density Imaging)",
            "FittingMethod": (
                "VectorizedJAXSphericalMean"
                if runtime.uses_jax
                else "MultiCompartmentSphericalMeanModel"
            ),
            "GradientNonlinearityCorrection": bool(grad_nonlin),
            "SomaDiffusivity": float(soma_diffusivity),
            "SomaDiffusivityUnits": "m^2/s",
        },
        parameter_metadata={
            name: {"Metric": name, "MetricUnits": units}
            for name, units in metric_units.items()
        },
    )
