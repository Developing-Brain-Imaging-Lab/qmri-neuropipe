"""Inspection and numerical smoke probes for registered dmipy models."""

from __future__ import annotations

from typing import Any

import numpy as np

from .dmipy_backend import (
    MODEL_REGISTRY,
    SUPPORTED_SOLVERS,
    acquisition_scheme_from_bvalues,
    build_reference_model,
    get_model_spec,
)


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return str(value)


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
        "solver_interfaces": sorted(SUPPORTED_SOLVERS),
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
                "physical_range": _json_ready(ranges),
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
