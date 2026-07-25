"""Voxel-parallel gradient-nonlinearity fitting for dmipy-fit's JAX solver.

dmipy-fit 2.1 normally closes its JAX forward function over one acquisition
scheme.  A gradient-deviation tensor changes both the b-value and direction at
each voxel, so calling ``model.fit`` separately for every voxel is exact but
defeats JAX batching.  This module keeps the scheme as a JAX pytree argument
and vmaps the optimizer over data and voxel-specific schemes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import os
import warnings

import numpy as np


@dataclass(frozen=True)
class CorrectedSchemes:
    """Dense per-voxel acquisition arrays consumed by the JAX fitter."""

    arrays: Mapping[str, np.ndarray]
    fallback_count: int


def corrected_scheme_arrays(
    acquisition_scheme: Any,
    gradient_tensors: np.ndarray,
) -> CorrectedSchemes:
    """Build exact voxel-specific PGSE arrays from gradient tensors.

    The tensor convention matches the existing qmri-neuropipe correction:
    ``g_local = L @ g_nominal``. Invalid tensors fall back to the nominal
    scheme, matching the historical voxel-wise implementation.
    """
    tensors = np.asarray(gradient_tensors, dtype=float)
    if tensors.ndim == 2 and tensors.shape[1] == 9:
        tensors = tensors.reshape(-1, 3, 3)
    elif tensors.ndim == 3 and tensors.shape[1:] == (3, 3):
        pass
    else:
        raise ValueError(
            "Gradient-nonlinearity tensors must have shape (N, 9) or (N, 3, 3)."
        )

    bvalues = np.asarray(acquisition_scheme.bvalues, dtype=float)
    directions = np.asarray(acquisition_scheme.gradient_directions, dtype=float)
    valid = np.all(np.isfinite(tensors), axis=(1, 2))
    safe_tensors = tensors.copy()
    safe_tensors[~valid] = np.eye(3)

    # Existing convention: row-form bvec @ L.T == (L @ column-bvec).T.
    local_vectors = np.einsum("nij,mj->nmi", safe_tensors, directions)
    norms = np.linalg.norm(local_vectors, axis=-1)
    weighted = bvalues > float(getattr(acquisition_scheme, "b0_threshold", 0.0))
    valid &= np.all(np.isfinite(norms), axis=1)
    valid &= np.all(norms[:, weighted] > 0, axis=1)
    valid &= np.all(
        np.isfinite(bvalues[None, :] * norms**2),
        axis=1,
    )
    if not np.all(valid):
        safe_tensors[~valid] = np.eye(3)
        local_vectors = np.einsum("nij,mj->nmi", safe_tensors, directions)
        norms = np.linalg.norm(local_vectors, axis=-1)
    corrected_bvalues = bvalues[None, :] * norms**2
    corrected_directions = np.zeros_like(local_vectors)
    np.divide(
        local_vectors,
        norms[..., None],
        out=corrected_directions,
        where=norms[..., None] > 0,
    )

    arrays: dict[str, np.ndarray] = {
        "bvalues": corrected_bvalues,
        "gradient_directions": corrected_directions,
    }
    for name in ("delta", "Delta", "TE"):
        value = getattr(acquisition_scheme, name, None)
        if value is not None:
            value = np.asarray(value, dtype=float)
            arrays[name] = np.broadcast_to(
                value, (len(tensors), value.size)
            ).copy()

    if "delta" in arrays and "Delta" in arrays:
        tau = arrays["Delta"] - arrays["delta"] / 3.0
        arrays["tau"] = tau
        arrays["qvalues"] = np.sqrt(corrected_bvalues / tau) / (2.0 * np.pi)
        # Match dmipy_fit.core.constants without importing the optional
        # dependency in this otherwise NumPy-only preprocessing helper.
        gamma = 267.513e6
        arrays["gradient_strengths"] = (
            np.sqrt(corrected_bvalues / tau) / (gamma * arrays["delta"])
        )

    # TE-dependent factors in dmipy use tau_perp. For PGSE this is TE.
    if "TE" in arrays:
        arrays["tau_perp"] = arrays["TE"]

    return CorrectedSchemes(
        arrays={key: np.asarray(value) for key, value in arrays.items()},
        fallback_count=int(np.count_nonzero(~valid)),
    )


def _build_scheme_argument_forward(model: Any, acquisition_scheme: Any):
    """Return ``forward(params_si, scheme_dict)`` using dmipy's JAX registry."""
    import jax
    import jax.numpy as jnp
    from dmipy_fit.jax.multicompartment_jax import (
        _build_dispatch,
        _extract_params_jax,
    )

    # Passing None deliberately selects numerical Watson/Bingham integration.
    # Their faster SH implementation captures a nominal scheme and therefore
    # cannot represent voxel-varying gradient directions.
    dispatch = _build_dispatch(model, None)
    n_meas = acquisition_scheme.number_of_measurements
    is_multi = model.N_models > 1
    rho_list = (
        [float(value) for value in model.S0_responses]
        if hasattr(model, "S0_responses")
        else [1.0] * model.N_models
    )

    name_to_start: dict[str, int] = {}
    index = 0
    for name, cardinality in model.parameter_cardinality.items():
        name_to_start[name] = index
        index += cardinality
    eta_idx = name_to_start.get("eta")
    s0_idx = name_to_start.get("S0_global")
    has_eta = eta_idx is not None
    has_s0 = s0_idx is not None
    has_t2 = any(name.endswith("_T2") for name in model.parameter_cardinality)
    te = getattr(acquisition_scheme, "TE", None)
    single_te = te is None or len(np.unique(np.atleast_1d(te))) == 1
    has_s0_tissue = getattr(model, "S0_tissue_responses", None) is not None
    normalize_b0 = (
        single_te and not has_s0 and not has_eta and not has_t2 and not has_s0_tissue
    )
    b0_indices = jnp.asarray(np.where(acquisition_scheme.b0_mask)[0])

    def forward(params_scaled, scheme):
        signal = jnp.zeros(n_meas, dtype=params_scaled.dtype)
        for compartment_index, entry in enumerate(dispatch):
            parameters = _extract_params_jax(
                params_scaled, entry["param_slices"]
            )
            compartment = entry["jax_fn"](scheme, parameters)
            volume_fraction = (
                params_scaled[entry["vf_idx"]] if is_multi else jnp.array(1.0)
            )
            signal = signal + (
                jnp.asarray(rho_list[compartment_index])
                * volume_fraction
                * compartment
            )
        if normalize_b0:
            signal = signal / jnp.mean(signal[b0_indices])
        if has_s0:
            signal = signal * params_scaled[s0_idx]
        if has_eta:
            signal = jnp.sqrt(signal**2 + params_scaled[eta_idx] ** 2)
        return signal

    return jax.jit(forward)


def fit_model_jax_gnl(
    model: Any,
    acquisition_scheme: Any,
    data: np.ndarray,
    gradient_tensors: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Fit a dmipy multi-compartment model with voxel-specific GNL schemes.

    A nominal-scheme brute grid supplies robust starting points. The complete
    LBFGS-B refinement, including every loss and gradient evaluation, uses the
    voxel-corrected scheme on the selected JAX CPU/GPU device.
    """
    import jax
    import jax.numpy as jnp
    from jaxopt import LBFGSB
    from dmipy_fit.core.fitted_modeling_framework import (
        FittedMultiCompartmentModel,
    )
    from dmipy_fit.jax.brute_jax import build_jax_brute_fn
    from dmipy_fit.jax.optimizers_jax import (
        JaxOptimizer,
        nested_to_normalized_fractions_jax,
    )

    options = dict(solver_kwargs or {})
    batch_size = options.pop("batch_size", None)
    ns = int(options.pop("Ns", 5))
    sphere_samples = int(options.pop("N_sphere_samples", 30))
    maxiter = int(options.pop("maxiter", 300))
    if options:
        unknown = ", ".join(sorted(options))
        raise TypeError(f"Unsupported JAX GNL solver options: {unknown}.")
    if batch_size is not None and (
        not isinstance(batch_size, int) or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer.")

    data_array = np.atleast_2d(np.asarray(data))
    if model.__class__.__name__ == "MultiCompartmentSphericalMeanModel":
        raise NotImplementedError(
            "Voxel-specific GNL correction is not defined for dmipy spherical-"
            "mean fitting because the correction changes shell membership and "
            "directions per voxel. Use the full signal model instead."
        )
    if getattr(acquisition_scheme, "_G", None) is not None:
        raise NotImplementedError(
            "JAX GNL fitting currently supports PGSE schemes, not arbitrary "
            "gradient waveforms."
        )
    if any(
        compartment.__class__.__name__ == "X0GeneralizedKarger"
        and "C1Stick_1_lambda_par" in compartment.parameter_ranges
        for compartment in model.models
    ):
        raise NotImplementedError(
            "dmipy-fit 2.1 does not provide a valid JAX forward model for the "
            "oriented Kärger/NEXI parameterization. Use a native solver without "
            "GNL acceleration until dmipy fixes that forward model."
        )
    spatial_shape = data_array.shape[:-1]
    if mask is None:
        fit_mask = data_array[..., 0] > 0
    else:
        fit_mask = np.asarray(mask, dtype=bool) & (data_array[..., 0] > 0)
    mask_pos = np.where(fit_mask)
    data_masked = data_array[mask_pos]
    tensors = np.asarray(gradient_tensors)
    if tensors.shape[: len(spatial_shape)] == spatial_shape:
        tensors_masked = tensors[mask_pos]
    elif tensors.shape[0] == len(data_masked):
        tensors_masked = tensors
    else:
        raise ValueError(
            "GNL tensor spatial shape must match the DWI or contain one tensor "
            "for every fitted voxel."
        )

    b0_mask = np.asarray(acquisition_scheme.b0_mask)
    t2_active = (
        acquisition_scheme.TE is not None
        and any(
            enabled
            for name, enabled in model.parameter_optimization_flags.items()
            if name.endswith("_T2")
        )
    )
    if t2_active:
        s0_masked = np.ones_like(data_masked)
    elif model.S0_tissue_responses is not None:
        s0_masked = model.max_S0_response * np.ones(len(data_masked))
    elif acquisition_scheme.TE is None or len(np.unique(acquisition_scheme.TE)) == 1:
        s0_masked = np.mean(data_masked[:, b0_mask], axis=-1)
    else:
        s0_masked = np.ones_like(data_masked)
        for echo_time in acquisition_scheme.shell_TE:
            te_mask = acquisition_scheme.TE == echo_time
            te_b0_mask = b0_mask & te_mask
            s0_masked[:, te_mask] = np.mean(
                data_masked[:, te_b0_mask], axis=-1
            )[:, None]
    finite = (
        np.all(np.isfinite(data_masked), axis=1)
        & np.all(np.isfinite(s0_masked), axis=-1)
        & np.all(s0_masked > 0, axis=-1)
        if s0_masked.ndim > 1
        else (
            np.all(np.isfinite(data_masked), axis=1)
            & np.isfinite(s0_masked)
            & (s0_masked > 0)
        )
    )
    if not np.all(finite):
        warnings.warn(
            f"Skipping {np.count_nonzero(~finite)} invalid voxel(s) in JAX GNL fit.",
            RuntimeWarning,
            stacklevel=2,
        )
    valid_data = data_masked[finite]
    valid_tensors = tensors_masked[finite]
    if len(valid_data) == 0:
        raise ValueError("No finite, positive-S0 voxels are available for JAX GNL fitting.")
    valid_s0 = s0_masked[finite]
    data_norm = (
        valid_data / (valid_s0 if valid_s0.ndim > 1 else valid_s0[:, None])
    ).astype(np.float32)
    corrected = corrected_scheme_arrays(acquisition_scheme, valid_tensors)
    if corrected.fallback_count:
        warnings.warn(
            f"{corrected.fallback_count} invalid GNL tensor(s) fell back to the "
            "nominal acquisition scheme.",
            RuntimeWarning,
            stacklevel=2,
        )

    optimizer = JaxOptimizer(
        model,
        acquisition_scheme,
        maxiter=maxiter,
        Ns=ns,
        N_sphere_samples=sphere_samples,
    )
    dynamic_forward = _build_scheme_argument_forward(model, acquisition_scheme)
    from dmipy_fit.jax.jax_compat import scheme_to_jax

    nominal_scheme = scheme_to_jax(acquisition_scheme)
    nominal_forward = jax.jit(
        lambda parameters: dynamic_forward(parameters, nominal_scheme)
    )
    print(
        "Building nominal-scheme JAX brute initializer for GNL fit...",
        flush=True,
    )
    brute = build_jax_brute_fn(
        model,
        acquisition_scheme,
        nominal_forward,
        optimizer,
        Ns=ns,
        N_sphere_samples=sphere_samples,
    )
    x0_all = np.asarray(brute(jnp.asarray(data_norm, dtype=jnp.float32)))
    scales = jnp.asarray(optimizer._scales, dtype=jnp.float32)

    def forward_nested(params_nested, scheme):
        if optimizer._is_multi:
            n_non_vf = len(scales) - optimizer._N_models
            non_vf = params_nested[:n_non_vf] * scales[:n_non_vf]
            fractions = nested_to_normalized_fractions_jax(
                params_nested[n_non_vf:]
            )
            params_si = jnp.concatenate([non_vf, fractions])
        else:
            params_si = params_nested * scales
        return dynamic_forward(params_si, scheme)

    def loss(params_nested, datum, scheme):
        residual = forward_nested(params_nested, scheme) - datum
        return jnp.mean(residual * residual)

    lower = jnp.asarray(optimizer._lower, dtype=jnp.float32)
    upper = jnp.asarray(optimizer._upper, dtype=jnp.float32)
    bounds = (lower, upper)
    solver = LBFGSB(
        fun=loss,
        maxiter=maxiter,
        tol=1e-5,
        implicit_diff=True,
    )

    @jax.jit
    def fit_batch(x0_batch, data_batch, scheme_batch):
        results = jax.vmap(
            lambda x0, datum, scheme: solver.run(
                x0, bounds=bounds, datum=datum, scheme=scheme
            )
        )(x0_batch, data_batch, scheme_batch)
        return results.params

    n_voxels = len(valid_data)
    if batch_size is None:
        batch_size = min(
            n_voxels, int(os.environ.get("DMIPY_JAX_BATCH", "8192"))
        )
        if batch_size < 1:
            raise ValueError("DMIPY_JAX_BATCH must be a positive integer.")
    fitted_nested = np.empty_like(x0_all)

    def run_batches(active_batch_size):
        n_batches = -(-n_voxels // active_batch_size)
        try:
            from tqdm import tqdm

            starts = tqdm(
                range(0, n_voxels, active_batch_size),
                total=n_batches,
                desc="JAX GNL LBFGS-B vmap",
                unit="batch",
            )
        except ImportError:
            starts = range(0, n_voxels, active_batch_size)

        for start in starts:
            end = min(start + active_batch_size, n_voxels)
            count = end - start
            x0_batch = jnp.asarray(x0_all[start:end], dtype=jnp.float32)
            data_batch = jnp.asarray(data_norm[start:end], dtype=jnp.float32)
            scheme_batch = {
                key: jnp.asarray(value[start:end], dtype=jnp.float32)
                for key, value in corrected.arrays.items()
            }
            if count < active_batch_size:
                pad = active_batch_size - count
                x0_batch = jnp.concatenate(
                    [x0_batch, jnp.repeat(x0_batch[-1:], pad, axis=0)]
                )
                data_batch = jnp.concatenate(
                    [data_batch, jnp.repeat(data_batch[-1:], pad, axis=0)]
                )
                scheme_batch = {
                    key: jnp.concatenate(
                        [value, jnp.repeat(value[-1:], pad, axis=0)]
                    )
                    for key, value in scheme_batch.items()
                }
            result = np.asarray(fit_batch(x0_batch, data_batch, scheme_batch))
            fitted_nested[start:end] = result[:count]

    active_batch_size = batch_size
    while True:
        try:
            run_batches(active_batch_size)
            break
        except RuntimeError as exc:
            message = str(exc)
            out_of_memory = (
                "RESOURCE_EXHAUSTED" in message
                or "out of memory" in message.lower()
            )
            if not out_of_memory or active_batch_size <= 1:
                raise
            smaller_batch = max(1, active_batch_size // 2)
            warnings.warn(
                "JAX GNL fit hit GPU OOM at batch_size="
                f"{active_batch_size}; retrying at {smaller_batch}.",
                RuntimeWarning,
                stacklevel=2,
            )
            active_batch_size = smaller_batch

    fitted_normalized = np.asarray(
        [optimizer._unnest(row) for row in fitted_nested]
    )
    fitted_si_valid = fitted_normalized * np.asarray(
        model.scales_for_optimization
    )
    fitted_vector = np.zeros(
        spatial_shape + (len(model.bounds_for_optimization),), dtype=float
    )
    masked_vectors = np.zeros(
        (len(data_masked), fitted_vector.shape[-1]), dtype=float
    )
    masked_vectors[finite] = fitted_si_valid
    fitted_vector[mask_pos] = masked_vectors
    s0_shape = spatial_shape + (
        (data_array.shape[-1],) if s0_masked.ndim > 1 else ()
    )
    s0 = np.zeros(s0_shape, dtype=float)
    s0[mask_pos] = s0_masked

    model.scheme = acquisition_scheme
    model._fit_solver = "jax"
    model._was_fitted = True
    return FittedMultiCompartmentModel(
        model,
        s0,
        fit_mask,
        fitted_vector,
        fitted_multi_tissue_fractions_vector=None,
    )
