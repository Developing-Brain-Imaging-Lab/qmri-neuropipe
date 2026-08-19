"""Vectorized JAX fitting for the dedicated spherical-mean SANDI model.

dmipy-fit 2.3 can evaluate the individual Stick, Ball, and GPA Sphere
compartments with JAX, but its generic spherical-mean dispatcher cannot
traverse the nested ``BundleModel`` used by the historical SANDI adapter.
This module composes those validated primitives directly and keeps the
acquisition arrays dynamic so a different GNL-corrected scheme can be used for
every voxel in a single ``vmap``.
"""

from __future__ import annotations

from itertools import product
import os
from typing import Any, Mapping
import warnings

import numpy as np

from .dmipy_jax_gnl import corrected_scheme_arrays


_DIAMETER = (
    "BundleModel_1_S4SphereGaussianPhaseApproximation_1_diameter"
)
_D_IN = "BundleModel_1_C1Stick_1_lambda_par"
_D_EC = "G1Ball_1_lambda_iso"
_F_NEURITE_IN_TISSUE = "BundleModel_1_partial_volume_0"
_F_TISSUE = "partial_volume_0"
_F_EXTRA = "partial_volume_1"
_OPTIMIZED_PARAMETERS = (
    _DIAMETER,
    _D_IN,
    _D_EC,
    _F_NEURITE_IN_TISSUE,
    _F_TISSUE,
)


def _sandi_bounds_and_scales(model: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized bounds and SI scales in the specialized JAX order."""
    missing = [
        name
        for name in _OPTIMIZED_PARAMETERS
        if name not in model.parameter_ranges
        or name not in model.parameter_scales
    ]
    if missing:
        raise ValueError(
            "The dmipy SANDI model does not expose the expected parameter(s): "
            + ", ".join(missing)
        )
    bounds = np.asarray(
        [model.parameter_ranges[name] for name in _OPTIMIZED_PARAMETERS],
        dtype=np.float32,
    )
    scales = np.asarray(
        [model.parameter_scales[name] for name in _OPTIMIZED_PARAMETERS],
        dtype=np.float32,
    ).reshape(-1)
    if bounds.shape != (len(_OPTIMIZED_PARAMETERS), 2):
        raise ValueError("Every specialized SANDI parameter must have scalar bounds.")
    return bounds, scales


def _nominal_scheme_arrays(acquisition_scheme: Any, n_voxels: int):
    """Broadcast the nominal PGSE fields into the dynamic-scheme layout."""
    arrays = {}
    for name in ("bvalues", "gradient_strengths", "delta", "Delta"):
        value = getattr(acquisition_scheme, name, None)
        if value is None:
            raise ValueError(
                f"Vectorized JAX SANDI requires acquisition field {name!r}."
            )
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        arrays[name] = np.broadcast_to(
            value, (n_voxels, value.size)
        ).copy()
    return arrays


def _model_constants(model: Any) -> tuple[float, float, np.ndarray]:
    """Extract the fixed S4 Sphere constants from the nested dmipy model."""
    try:
        sphere = model.models[0].models[1]
        diffusivity = float(sphere.diffusion_constant)
        gyromagnetic_ratio = float(sphere.gyromagnetic_ratio)
        roots = np.asarray(
            sphere.SPHERE_TRASCENDENTAL_ROOTS, dtype=np.float32
        )
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError(
            "The specialized JAX SANDI fitter requires the historical "
            "BundleModel([C1Stick, S4Sphere]) + G1Ball model layout."
        ) from exc
    return diffusivity, gyromagnetic_ratio, roots


def _build_forward(model: Any):
    """Build ``forward(normalized_params, scheme)`` from dmipy JAX primitives."""
    import jax
    import jax.numpy as jnp
    from dmipy_fit.jax.signal_models_jax import (
        g1ball_signal,
        s4sphere_pgse_signal_jax,
    )

    _, scales_np = _sandi_bounds_and_scales(model)
    soma_diffusivity, gyromagnetic_ratio, roots_np = _model_constants(model)
    scales = jnp.asarray(scales_np, dtype=jnp.float32)
    roots = jnp.asarray(roots_np, dtype=jnp.float32)
    soma_diffusivity = jnp.asarray(soma_diffusivity, dtype=jnp.float32)
    gyromagnetic_ratio = jnp.asarray(
        gyromagnetic_ratio, dtype=jnp.float32
    )

    def forward(parameters_normalized, scheme):
        parameters = parameters_normalized * scales
        diameter, d_in, d_ec, f_neurite_in_tissue, f_tissue = parameters
        # dmipy-fit 2.3's public Stick spherical-mean primitive evaluates a
        # 0/0 expression on b=0 before jnp.where selects its unit-signal
        # branch. The value is correct, but autodiff then returns NaN for d_in.
        # A safe inactive argument preserves the identical analytical formula
        # while keeping its derivative finite.
        bl = scheme["bvalues"] * d_in
        weighted = bl > 1e-7
        safe_bl = jnp.where(weighted, bl, jnp.ones_like(bl))
        sqrt_bl = jnp.sqrt(safe_bl)
        stick_weighted = (
            jnp.sqrt(jnp.pi)
            * jax.scipy.special.erf(sqrt_bl)
            / (2.0 * sqrt_bl)
        )
        stick = jnp.where(weighted, stick_weighted, jnp.ones_like(bl))
        extra = g1ball_signal(scheme["bvalues"], d_ec)
        soma = jax.vmap(
            lambda strength, small_delta, big_delta: (
                s4sphere_pgse_signal_jax(
                    strength,
                    small_delta,
                    big_delta,
                    diameter,
                    soma_diffusivity,
                    roots,
                    gyromagnetic_ratio,
                )
            )
        )(
            scheme["gradient_strengths"],
            scheme["delta"],
            scheme["Delta"],
        )
        tissue = (
            f_neurite_in_tissue * stick
            + (1.0 - f_neurite_in_tissue) * soma
        )
        return f_tissue * tissue + (1.0 - f_tissue) * extra

    return jax.jit(forward)


def _parameter_grid(bounds: np.ndarray, ns: int) -> np.ndarray:
    """Construct the compact five-parameter initializer grid."""
    axes = [
        np.linspace(low, high, ns, dtype=np.float32)
        for low, high in bounds
    ]
    return np.asarray(list(product(*axes)), dtype=np.float32)


def _is_out_of_memory(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "resource_exhausted" in message or "out of memory" in message


def fit_sandi_jax(
    model: Any,
    acquisition_scheme: Any,
    data: np.ndarray,
    *,
    gradient_tensors: np.ndarray | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Fit spherical-mean SANDI to full measurements using batched JAX.

    The response remains the direction-independent SANDI spherical mean, but
    the objective is evaluated at every acquired measurement. With GNL, each
    measurement uses its voxel-specific corrected b-value and gradient
    strength; no incorrect assignment back to nominal shells is required.
    """
    import jax
    import jax.numpy as jnp
    from jaxopt import LBFGSB

    options = dict(solver_kwargs or {})
    batch_size = options.pop("batch_size", None)
    ns = int(options.pop("Ns", 5))
    # Accepted for compatibility with dmipy's generic JAX solver. SANDI has no
    # fitted orientation, so no spherical orientation samples are needed.
    options.pop("N_sphere_samples", None)
    maxiter = int(options.pop("maxiter", 300))
    tolerance = float(options.pop("tol", 1e-5))
    if options:
        unknown = ", ".join(sorted(options))
        raise TypeError(f"Unsupported JAX SANDI solver options: {unknown}.")
    if ns < 2:
        raise ValueError("Ns must be at least 2 for JAX SANDI fitting.")
    if maxiter < 1:
        raise ValueError("maxiter must be a positive integer.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tol must be finite and positive.")

    data_array = np.atleast_2d(np.asarray(data, dtype=np.float32))
    if data_array.ndim != 2:
        raise ValueError("Vectorized JAX SANDI expects voxels x measurements data.")
    n_voxels, n_measurements = data_array.shape
    if n_measurements != acquisition_scheme.number_of_measurements:
        raise ValueError(
            f"SANDI data has {n_measurements} measurements but the acquisition "
            f"scheme has {acquisition_scheme.number_of_measurements}."
        )
    if batch_size is None:
        batch_size = min(
            n_voxels, int(os.environ.get("DMIPY_JAX_BATCH", "8192"))
        )
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    b0_mask = np.asarray(acquisition_scheme.b0_mask, dtype=bool)
    if not np.any(b0_mask):
        raise ValueError("JAX SANDI fitting requires at least one b=0 measurement.")
    s0 = np.mean(data_array[:, b0_mask], axis=1)
    valid = (
        np.all(np.isfinite(data_array), axis=1)
        & np.isfinite(s0)
        & (s0 > 0)
    )
    if not np.any(valid):
        raise ValueError(
            "No finite, positive-S0 voxels are available for JAX SANDI fitting."
        )
    invalid_count = int(np.count_nonzero(~valid))
    if invalid_count:
        warnings.warn(
            f"Skipping {invalid_count} invalid voxel(s) in JAX SANDI fit.",
            RuntimeWarning,
            stacklevel=2,
        )
    valid_data = data_array[valid] / s0[valid, None]

    if gradient_tensors is None:
        scheme_arrays = _nominal_scheme_arrays(
            acquisition_scheme, len(valid_data)
        )
    else:
        tensors = np.asarray(gradient_tensors)
        if tensors.shape[0] != n_voxels:
            raise ValueError(
                "JAX SANDI requires one gradient tensor per input voxel."
            )
        corrected = corrected_scheme_arrays(
            acquisition_scheme, tensors[valid]
        )
        scheme_arrays = {
            name: corrected.arrays[name]
            for name in ("bvalues", "gradient_strengths", "delta", "Delta")
        }
        if corrected.fallback_count:
            warnings.warn(
                f"{corrected.fallback_count} invalid GNL tensor(s) fell back "
                "to the nominal acquisition scheme.",
                RuntimeWarning,
                stacklevel=2,
            )

    bounds_np, scales_np = _sandi_bounds_and_scales(model)
    lower = jnp.asarray(bounds_np[:, 0], dtype=jnp.float32)
    upper = jnp.asarray(bounds_np[:, 1], dtype=jnp.float32)
    bounds = (lower, upper)
    forward = _build_forward(model)

    nominal = {
        name: jnp.asarray(
            getattr(acquisition_scheme, name), dtype=jnp.float32
        )
        for name in ("bvalues", "gradient_strengths", "delta", "Delta")
    }
    grid_np = _parameter_grid(bounds_np, ns)
    grid_parameters = jnp.asarray(grid_np)
    margin = 0.05 * (bounds_np[:, 1] - bounds_np[:, 0])
    grid_initials = jnp.asarray(
        np.clip(
            grid_np,
            bounds_np[:, 0] + margin,
            bounds_np[:, 1] - margin,
        )
    )
    print(
        f"Building JAX SANDI initializer ({len(grid_np)} grid points)...",
        flush=True,
    )
    evaluate_grid_chunk = jax.jit(
        jax.vmap(lambda parameters: forward(parameters, nominal))
    )
    grid_chunks = []
    grid_chunk_size = min(256, len(grid_np))
    for start in range(0, len(grid_np), grid_chunk_size):
        end = min(start + grid_chunk_size, len(grid_np))
        chunk = grid_parameters[start:end]
        count = end - start
        if count < grid_chunk_size:
            chunk = jnp.concatenate(
                [
                    chunk,
                    jnp.repeat(chunk[-1:], grid_chunk_size - count, axis=0),
                ]
            )
        signals = jax.block_until_ready(evaluate_grid_chunk(chunk))
        grid_chunks.append(signals[:count])
    grid_signals = jnp.concatenate(grid_chunks, axis=0)
    grid_sq = jnp.sum(grid_signals * grid_signals, axis=1)

    def initialize(data_batch):
        data_sq = jnp.sum(data_batch * data_batch, axis=1)
        cross = data_batch @ grid_signals.T
        mse = (
            data_sq[:, None] + grid_sq[None, :] - 2.0 * cross
        ) / n_measurements
        return grid_initials[jnp.argmin(mse, axis=1)]

    def loss(parameters, datum, scheme):
        residual = forward(parameters, scheme) - datum
        return jnp.mean(residual * residual)

    optimizer = LBFGSB(
        fun=loss,
        maxiter=maxiter,
        tol=tolerance,
        implicit_diff=True,
    )

    @jax.jit
    def fit_batch(data_batch, scheme_batch):
        x0 = initialize(data_batch)
        results = jax.vmap(
            lambda initial, datum, scheme: optimizer.run(
                initial,
                bounds=bounds,
                datum=datum,
                scheme=scheme,
            )
        )(x0, data_batch, scheme_batch)
        return results.params

    n_valid = len(valid_data)
    fitted_normalized = np.empty(
        (n_valid, len(_OPTIMIZED_PARAMETERS)), dtype=np.float32
    )

    def run_batches(active_batch_size: int):
        starts = range(0, n_valid, active_batch_size)
        try:
            from tqdm import tqdm

            starts = tqdm(
                starts,
                total=-(-n_valid // active_batch_size),
                desc="JAX SANDI LBFGS-B vmap",
                unit="batch",
            )
        except ImportError:
            pass
        for start in starts:
            end = min(start + active_batch_size, n_valid)
            count = end - start
            data_batch = jnp.asarray(valid_data[start:end])
            scheme_batch = {
                name: jnp.asarray(values[start:end], dtype=jnp.float32)
                for name, values in scheme_arrays.items()
            }
            if count < active_batch_size:
                pad = active_batch_size - count
                data_batch = jnp.concatenate(
                    [data_batch, jnp.repeat(data_batch[-1:], pad, axis=0)]
                )
                scheme_batch = {
                    name: jnp.concatenate(
                        [value, jnp.repeat(value[-1:], pad, axis=0)]
                    )
                    for name, value in scheme_batch.items()
                }
            fitted = np.asarray(
                jax.block_until_ready(fit_batch(data_batch, scheme_batch))
            )
            fitted_normalized[start:end] = fitted[:count]

    active_batch_size = min(batch_size, n_valid)
    while True:
        try:
            run_batches(active_batch_size)
            break
        except RuntimeError as exc:
            if not _is_out_of_memory(exc) or active_batch_size <= 1:
                raise
            smaller_batch = max(1, active_batch_size // 2)
            warnings.warn(
                "JAX SANDI fit hit GPU OOM at batch_size="
                f"{active_batch_size}; retrying at {smaller_batch}.",
                RuntimeWarning,
                stacklevel=2,
            )
            active_batch_size = smaller_batch

    fitted_si = fitted_normalized * scales_np[None, :]
    complete = np.full(
        (n_voxels, len(_OPTIMIZED_PARAMETERS)), np.nan, dtype=np.float32
    )
    complete[valid] = fitted_si
    result = {
        name: complete[:, index]
        for index, name in enumerate(_OPTIMIZED_PARAMETERS)
    }
    result[_F_EXTRA] = 1.0 - result[_F_TISSUE]
    return result
