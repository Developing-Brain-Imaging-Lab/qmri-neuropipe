"""Generic fitting entry point for dmipy-fit 2.x reference models."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Mapping
import time

import nibabel as nib
import numpy as np

from ..core.types import DWIFile, ImageLike
from ..core.utils import ensure_dir, extract_image_path
from .dmipy_backend import (
    DmipyRuntime,
    acquisition_scheme_from_bvalues,
    build_reference_model,
    dmipy_fit_output,
    fit_model,
    get_model_spec,
    jax_run_summary,
    model_output_maps,
)
from .dmipy_derivatives import write_dmipy_derivatives


def _load_gradients(
    bval_file: Path,
    bvec_file: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load FSL gradients and convert b-values from s/mm² to s/m²."""
    bvalues = np.asarray(np.loadtxt(bval_file), dtype=float).reshape(-1)
    bvecs = np.asarray(np.loadtxt(bvec_file), dtype=float)
    if bvecs.ndim != 2:
        raise ValueError("b-vectors must be a two-dimensional 3 x N or N x 3 array.")
    if bvecs.shape == (3, bvalues.size):
        bvecs = bvecs.T
    elif bvecs.shape != (bvalues.size, 3):
        raise ValueError(
            f"b-vector shape {bvecs.shape} is incompatible with {bvalues.size} "
            "b-values; expected 3 x N or N x 3."
        )
    if not np.all(np.isfinite(bvalues)) or np.any(bvalues < 0):
        raise ValueError("b-values must be finite and non-negative.")
    if not np.all(np.isfinite(bvecs)):
        raise ValueError("b-vectors must be finite.")
    norms = np.linalg.norm(bvecs, axis=1)
    weighted = bvalues > 10.0
    if np.any(norms[weighted] <= 0):
        raise ValueError("Every diffusion-weighted volume must have a non-zero b-vector.")
    if np.any(np.abs(norms[weighted] - 1.0) > 0.1):
        raise ValueError("Diffusion-weighted b-vectors must have approximately unit norm.")
    nonzero = norms > 0
    bvecs[nonzero] /= norms[nonzero, None]
    return bvalues * 1e6, bvecs


def _load_measurement_file(
    path: Path,
    *,
    label: str,
    n_measurements: int,
) -> np.ndarray:
    """Load one value or one value per DWI volume, expressed in seconds."""
    values = np.asarray(np.loadtxt(path), dtype=float).reshape(-1)
    if values.size not in (1, n_measurements):
        raise ValueError(
            f"{label} must contain one value or {n_measurements} values; "
            f"found {values.size}."
        )
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{label} values must be finite, positive seconds.")
    if values.size == 1:
        values = np.full(n_measurements, values[0], dtype=float)
    return values


def _load_acquisition_values(
    model_name: str,
    n_measurements: int,
    *,
    delta_file: Path | None,
    Delta_file: Path | None,
    TE_file: Path | None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Validate registry requirements and load per-measurement timing arrays."""
    spec = get_model_spec(model_name)
    paths = {
        "delta": Path(delta_file) if delta_file is not None else None,
        "Delta": Path(Delta_file) if Delta_file is not None else None,
        "TE": Path(TE_file) if TE_file is not None else None,
    }
    missing = [name for name in spec.acquisition_requirements if paths[name] is None]
    if missing:
        options = {"delta": "--delta", "Delta": "--big-delta", "TE": "--te"}
        required = ", ".join(options[name] for name in missing)
        raise ValueError(f"dmipy model {model_name!r} requires {required}.")
    if (paths["delta"] is None) != (paths["Delta"] is None):
        raise ValueError("--delta and --big-delta must be provided together.")
    if (
        paths["delta"] is not None
        and paths["Delta"] is not None
        and paths["delta"].samefile(paths["Delta"])
    ):
        raise ValueError(
            "--delta and --big-delta must point to two distinct files. "
            "Use names such as small_delta.txt and big_delta.txt."
        )

    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, object] = {}
    labels = {
        "delta": "small-delta",
        "Delta": "big-Delta",
        "TE": "echo time",
    }
    for name, path in paths.items():
        if path is None:
            continue
        values = _load_measurement_file(
            path,
            label=labels[name],
            n_measurements=n_measurements,
        )
        arrays[name] = values
        metadata[f"{name}File"] = path.name
        metadata[f"{name}Seconds"] = {
            "Minimum": float(values.min()),
            "Maximum": float(values.max()),
        }
    if "delta" in arrays and np.any(arrays["Delta"] <= arrays["delta"]):
        raise ValueError("Every big-Delta value must be greater than small-delta.")
    return arrays, metadata


def fit_dmipy_reference(
    in_file: Path | ImageLike,
    out_dir: Path,
    *,
    model_name: str,
    bval_file: Path | None = None,
    bvec_file: Path | None = None,
    mask_file: Path | None = None,
    delta_file: Path | None = None,
    Delta_file: Path | None = None,
    TE_file: Path | None = None,
    solver: str = "brute2fine",
    device: str = "auto",
    gpu_device: int | None = None,
    jax_cache_dir: Path | None = None,
    jax_log_compiles: bool = False,
    heartbeat_interval: float = 30.0,
    nthreads: int = 1,
    solver_kwargs: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    """Fit any allow-listed dmipy reference model and write all parameters."""
    spec = get_model_spec(model_name)
    in_path = extract_image_path(in_file)
    out_dir = ensure_dir(out_dir)
    if isinstance(in_file, DWIFile):
        bval_file = bval_file or in_file.bval
        bvec_file = bvec_file or in_file.bvec
        delta_file = delta_file or in_file.delta
        Delta_file = Delta_file or in_file.Delta
    if bval_file is None or bvec_file is None:
        raise ValueError("Generic dmipy fitting requires --bval and --bvec.")
    if not isinstance(nthreads, int) or nthreads < 1:
        raise ValueError("nthreads must be a positive integer.")
    if heartbeat_interval < 1:
        raise ValueError("heartbeat_interval must be at least one second.")

    image = nib.load(str(in_path))
    data = image.get_fdata()
    if data.ndim != 4:
        raise ValueError(
            f"dmipy input must be a 4D DWI image; found shape {data.shape}."
        )
    bvalues, bvecs = _load_gradients(Path(bval_file), Path(bvec_file))
    if bvalues.size != data.shape[-1]:
        raise ValueError(
            f"DWI has {data.shape[-1]} volumes but gradients contain "
            f"{bvalues.size} measurements."
        )

    timing, timing_metadata = _load_acquisition_values(
        spec.name,
        bvalues.size,
        delta_file=delta_file,
        Delta_file=Delta_file,
        TE_file=TE_file,
    )
    scheme = acquisition_scheme_from_bvalues(
        bvalues,
        bvecs,
        delta=timing.get("delta"),
        Delta=timing.get("Delta"),
        TE=timing.get("TE"),
    )

    mask = None
    if mask_file is not None:
        mask = nib.load(str(mask_file)).get_fdata().astype(bool)
        if mask.shape != data.shape[:3]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match DWI shape {data.shape[:3]}."
            )
    voxel_count = int(mask.sum()) if mask is not None else int(np.prod(data.shape[:3]))
    options = dict(solver_kwargs or {})
    # Resolve GPU visibility before constructing a model, in case a future
    # reference-model module imports JAX while it is being initialized.
    runtime = DmipyRuntime.resolve(
        solver,
        device,
        gpu_device=gpu_device,
        jax_cache_dir=jax_cache_dir,
        jax_log_compiles=jax_log_compiles,
    )
    model = build_reference_model(spec.name)
    print(f"Fitting dmipy model {spec.name} ({voxel_count} voxels)...", flush=True)
    for line in jax_run_summary(runtime, voxel_count, options):
        print(line, flush=True)

    def run_fit():
        with dmipy_fit_output(runtime.solver):
            return fit_model(
                model,
                scheme,
                data,
                mask=mask,
                solver=runtime.solver,
                nthreads=nthreads,
                solver_kwargs=options,
                runtime=runtime,
            )

    if runtime.uses_jax:
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(run_fit)
            while True:
                try:
                    fitted, runtime = pending.result(timeout=heartbeat_interval)
                except FutureTimeout:
                    elapsed = (time.monotonic() - started) / 60.0
                    print(
                        f"dmipy {spec.name} JAX fit is still running "
                        f"(elapsed {elapsed:.1f} min).",
                        flush=True,
                    )
                else:
                    break
    else:
        fitted, runtime = run_fit()

    return write_dmipy_derivatives(
        out_dir,
        in_path,
        image.affine,
        model_output_maps(spec.name, fitted),
        runtime,
        model_name=spec.name,
        base_metadata={
            "FittingMethod": "dmipy-fit reference model",
            "SolverOptions": options,
            **timing_metadata,
        },
    )
