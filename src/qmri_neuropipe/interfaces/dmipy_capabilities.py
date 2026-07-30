"""Inspection and numerical smoke probes for registered dmipy models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .dmipy_backend import (
    MODEL_REGISTRY,
    DmipyFitRequest,
    DmipyRuntime,
    SUPPORTED_SOLVERS,
    acquisition_scheme_from_bvalues,
    build_reference_model,
    get_model_spec,
    execute_dmipy_fit,
)
from ..utils.serialization import json_ready


@dataclass(frozen=True)
class RecoveryCase:
    """One identifiable scalar used as an optimizer merge gate."""

    model_name: str
    parameter: str
    truth: float
    relative_tolerance: float = 0.02
    absolute_tolerance: float = 1e-12


RECOVERY_CASES: tuple[RecoveryCase, ...] = (
    RecoveryCase("ball", "G1Ball_1_lambda_iso", 1.2e-9),
    RecoveryCase("zeppelin", "G2Zeppelin_1_lambda_par", 1.8e-9),
)


def model_summary(name: str) -> dict[str, Any]:
    """Return registry-level capabilities without importing dmipy-fit."""
    spec = get_model_spec(name)
    return {
        "name": spec.name,
        "family": spec.family,
        "source": (
            "dmipy-fit"
            if spec.factory_module
            == "dmipy_fit.custom_optimizers.reference_models"
            else "qmri-neuropipe"
        ),
        "factory": f"{spec.factory_module}.{spec.factory_name}",
        "acquisition_requirements": list(spec.acquisition_requirements),
        "solver_interfaces": sorted(
            SUPPORTED_SOLVERS
            if spec.jax_supported
            else SUPPORTED_SOLVERS - {"jax"}
        ),
        "jax_supported": spec.jax_supported,
        "jax_gnl_supported": spec.jax_gnl_supported,
        "output_alias_count": len(spec.output_aliases),
        "references": list(spec.references),
    }


def registry_summary() -> list[dict[str, Any]]:
    """Return stable summaries for all allow-listed models."""
    return [model_summary(name) for name in sorted(MODEL_REGISTRY)]


def validation_acquisition_scheme():
    """Build a compact PGSE/multi-TE scheme usable by every registry model."""
    directions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2**-0.5, 2**-0.5, 0.0],
            [2**-0.5, 0.0, 2**-0.5],
            [0.0, 2**-0.5, 2**-0.5],
        ],
        dtype=float,
    )
    shell_bvalues = np.array(
        [0.0, 0.5e9, 1.0e9, 1.5e9, 2.0e9, 2.5e9, 3.0e9]
    )
    bvalues = np.tile(shell_bvalues, 2)
    directions = np.vstack([directions, directions])
    return acquisition_scheme_from_bvalues(
        bvalues,
        directions,
        delta=np.full(bvalues.size, 0.012),
        Delta=np.full(bvalues.size, 0.035),
        TE=np.r_[np.full(7, 0.070), np.full(7, 0.090)],
    )


def representative_parameters(model) -> dict[str, Any]:
    """Choose deterministic in-range parameters for simulation smoke tests."""
    parameters: dict[str, Any] = {}
    volume_fractions = [
        name
        for name in model.parameter_names
        if name.startswith("partial_volume_")
    ]
    initial_guesses = getattr(model, "x0_parameters", {}) or {}
    for name in model.parameter_names:
        cardinality = int(model.parameter_cardinality[name])
        parameter_type = model.parameter_types.get(name)
        initial = initial_guesses.get(name)
        if name in volume_fractions:
            value: Any = 1.0 / len(volume_fractions)
        elif name.endswith("_T2"):
            value = 0.080
        elif initial is not None and np.all(np.isfinite(np.asarray(initial))):
            value = initial
        elif parameter_type == "orientation":
            value = (
                np.array([np.pi / 2.0, 0.0])
                if cardinality == 2
                else np.zeros(cardinality)
            )
        else:
            ranges = np.asarray(model.parameter_ranges[name], dtype=float)
            scales = np.asarray(model.parameter_scales[name], dtype=float)
            value = np.mean(ranges, axis=-1) * scales
            if cardinality == 1:
                value = float(np.ravel(value)[0])
        parameters[name] = value
    return parameters


def model_details(name: str) -> dict[str, Any]:
    """Construct a model and report its resolved parameter contract."""
    model = build_reference_model(name)
    summary = model_summary(name)
    parameters = []
    for parameter_name in model.parameter_names:
        ranges = (
            np.asarray(model.parameter_ranges[parameter_name], dtype=float)
            * np.asarray(model.parameter_scales[parameter_name], dtype=float)
        )
        parameters.append(
            {
                "name": parameter_name,
                "cardinality": int(model.parameter_cardinality[parameter_name]),
                "type": model.parameter_types.get(parameter_name),
                "optimized": bool(
                    model.parameter_optimization_flags.get(parameter_name, True)
                ),
                "physical_range": json_ready(ranges),
                "output_alias": get_model_spec(name).output_aliases.get(
                    parameter_name
                ),
            }
        )
    return {
        **summary,
        "model_class": type(model).__name__,
        "parameter_count": len(parameters),
        "parameters": parameters,
    }


def probe_model(name: str) -> dict[str, Any]:
    """Construct and simulate one deterministic signal for a registry model."""
    result = {
        "name": get_model_spec(name).name,
        "construction": "failed",
        "simulation": "not-run",
        "error": None,
    }
    try:
        model = build_reference_model(name)
        result["construction"] = "passed"
        scheme = validation_acquisition_scheme()
        signal = np.asarray(
            model.simulate_signal(
                scheme,
                representative_parameters(model),
            )
        )
        if signal.shape != (scheme.number_of_measurements,):
            raise ValueError(
                f"simulation returned shape {signal.shape}; expected "
                f"({scheme.number_of_measurements},)."
            )
        if not np.all(np.isfinite(signal)):
            raise ValueError("simulation returned non-finite signal values.")
        result.update(
            {
                "simulation": "passed",
                "signal_minimum": float(signal.min()),
                "signal_maximum": float(signal.max()),
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def probe_registry() -> list[dict[str, Any]]:
    """Run the construction/simulation smoke probe for every model."""
    return [probe_model(name) for name in sorted(MODEL_REGISTRY)]


def parameter_recovery_probe(
    case: RecoveryCase,
    *,
    solver: str,
    device: str = "auto",
    n_voxels: int = 2,
    solver_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit noiseless synthetic data and quantify recovery of a known parameter.

    Non-target parameters are fixed to deterministic in-range values. This
    deliberately tests optimizer scaling, parameter conversion, batching, and
    result extraction without confounding the merge gate with multi-parameter
    identifiability.
    """
    if n_voxels < 1:
        raise ValueError("n_voxels must be a positive integer.")
    simulation_model = build_reference_model(case.model_name)
    truth = representative_parameters(simulation_model)
    if case.parameter not in truth:
        raise ValueError(
            f"Recovery parameter {case.parameter!r} is not exposed by "
            f"{case.model_name!r}."
        )
    truth[case.parameter] = float(case.truth)
    scheme = validation_acquisition_scheme()
    signal = np.asarray(
        simulation_model.simulate_signal(scheme, truth),
        dtype=float,
    )

    fitting_model = build_reference_model(case.model_name)
    for parameter, value in truth.items():
        if parameter != case.parameter:
            fitting_model.set_fixed_parameter(parameter, value)
    if fitting_model.parameter_names != [case.parameter]:
        raise ValueError(
            f"Recovery case for {case.model_name!r} must leave only "
            f"{case.parameter!r} optimized; found {fitting_model.parameter_names}."
        )

    options = {"Ns": 7, "maxiter": 100, **dict(solver_options or {})}
    if str(solver).lower() == "jax":
        options.setdefault("batch_size", n_voxels)
    runtime = DmipyRuntime.resolve(solver, device)
    execution = execute_dmipy_fit(
        DmipyFitRequest(
            model_name=case.model_name,
            model=fitting_model,
            acquisition_scheme=scheme,
            data=np.repeat(signal[None, :], n_voxels, axis=0),
            runtime=runtime,
            solver_options=options,
            heartbeat_interval=None,
        )
    )
    estimates = np.asarray(
        execution.fitted.fitted_parameters[case.parameter],
        dtype=float,
    ).reshape(-1)
    absolute_error = np.abs(estimates - case.truth)
    relative_error = absolute_error / max(abs(case.truth), np.finfo(float).eps)
    passed = bool(
        np.all(
            absolute_error
            <= case.absolute_tolerance
            + case.relative_tolerance * abs(case.truth)
        )
    )
    return {
        "model_name": case.model_name,
        "parameter": case.parameter,
        "solver": runtime.solver,
        "backend": runtime.backend,
        "truth": case.truth,
        "estimates": estimates.tolist(),
        "maximum_absolute_error": float(absolute_error.max()),
        "maximum_relative_error": float(relative_error.max()),
        "relative_tolerance": case.relative_tolerance,
        "absolute_tolerance": case.absolute_tolerance,
        "passed": passed,
    }
