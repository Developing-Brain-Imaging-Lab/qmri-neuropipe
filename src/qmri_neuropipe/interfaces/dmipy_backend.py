"""Shared integration layer for the dmipy 2.x analytical fitting engine.

The PyPI ``dmipy`` project became a non-importable umbrella package in 2.x.
Pipeline code must import the analytical engine from ``dmipy_fit`` directly.
This module keeps version checks, solver validation, acquisition construction,
reference-model discovery, and fit keyword translation out of model-specific
interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, Mapping
import inspect
import os
import sys

import numpy as np

from ..core import ProcessingError


DMIPY_FIT_MIN_VERSION = (2, 1)
DMIPY_FIT_MAX_VERSION = (2, 2)
SUPPORTED_SOLVERS = frozenset({"brute2fine", "mix", "jax"})
SUPPORTED_DEVICES = frozenset({"auto", "cpu", "gpu"})


def _release_tuple(value: str) -> tuple[int, ...]:
    """Return the numeric release prefix without requiring packaging at import."""
    release = value.split("+", 1)[0].split("-", 1)[0]
    parts: list[int] = []
    for item in release.split("."):
        digits = "".join(ch for ch in item if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


@dataclass(frozen=True)
class DmipyRuntime:
    """Resolved dmipy-fit runtime and optional JAX device information."""

    version: str
    solver: str
    requested_device: str
    backend: str
    devices: tuple[str, ...] = ()

    @property
    def uses_jax(self) -> bool:
        return self.solver == "jax"

    @classmethod
    def resolve(
        cls,
        solver: str = "brute2fine",
        device: str = "auto",
        *,
        require_device: bool = False,
    ) -> "DmipyRuntime":
        solver = str(solver).lower()
        device = str(device).lower()
        if solver not in SUPPORTED_SOLVERS:
            choices = ", ".join(sorted(SUPPORTED_SOLVERS))
            raise ValueError(f"Unknown dmipy solver {solver!r}; choose one of: {choices}.")
        if device not in SUPPORTED_DEVICES:
            choices = ", ".join(sorted(SUPPORTED_DEVICES))
            raise ValueError(f"Unknown dmipy device {device!r}; choose one of: {choices}.")
        if solver != "jax" and device == "gpu":
            raise ValueError("device='gpu' requires solver='jax'.")

        try:
            installed = version("dmipy-fit")
        except PackageNotFoundError as exc:
            raise ProcessingError(
                "dmipy-fit 2.1 is required. Install qmri-neuropipe with its "
                "default dependencies, or use .[dmipy-jax]/.[dmipy-cuda12]."
            ) from exc

        release = _release_tuple(installed)
        if not (release >= DMIPY_FIT_MIN_VERSION and release < DMIPY_FIT_MAX_VERSION):
            raise ProcessingError(
                f"Unsupported dmipy-fit version {installed}; this release supports "
                "dmipy-fit>=2.1,<2.2."
            )

        if solver != "jax":
            return cls(installed, solver, device, "native-cpu")

        if device == "cpu":
            if "jax" in sys.modules:
                active = getattr(sys.modules["jax"], "default_backend", lambda: "unknown")()
                if active != "cpu":
                    raise ProcessingError(
                        "device='cpu' must be selected before JAX initializes. "
                        "Start a fresh process or set JAX_PLATFORMS=cpu."
                    )
            os.environ["JAX_PLATFORMS"] = "cpu"

        try:
            jax = import_module("jax")
            import_module("jaxopt")
        except ImportError as exc:
            raise ProcessingError(
                "solver='jax' requires the dmipy-jax or dmipy-cuda12 optional extra."
            ) from exc

        available = tuple(str(item) for item in jax.devices())
        platforms = {getattr(item, "platform", "") for item in jax.devices()}
        has_gpu = "gpu" in platforms or "cuda" in platforms
        if device == "gpu" and not has_gpu:
            message = "device='gpu' was requested but JAX reports no usable GPU."
            if require_device:
                raise ProcessingError(message)
            backend = "cpu"
        elif device == "cpu":
            backend = "cpu"
        else:
            backend = "gpu" if has_gpu else "cpu"
        return cls(installed, solver, device, backend, available)

    def provenance(self) -> dict[str, Any]:
        return {
            "FittingSoftware": "dmipy-fit",
            "FittingSoftwareVersion": self.version,
            "Solver": self.solver,
            "RequestedDevice": self.requested_device,
            "ExecutionBackend": self.backend,
            "JAXDevices": list(self.devices),
        }


@dataclass(frozen=True)
class ModelSpec:
    """Stable pipeline-facing description of a dmipy reference model."""

    name: str
    factory_name: str
    family: str
    acquisition_requirements: tuple[str, ...] = ()
    output_aliases: Mapping[str, str] = field(default_factory=dict)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "ball": ModelSpec("ball", "ball", "gaussian"),
    "zeppelin": ModelSpec("zeppelin", "zeppelin", "gaussian"),
    "temporal_zeppelin": ModelSpec(
        "temporal_zeppelin", "temporal_zeppelin", "time-dependent", ("diffusion_time",)
    ),
    "ball_and_stick": ModelSpec("ball_and_stick", "ball_and_stick", "white-matter"),
    "ball_and_zeppelin": ModelSpec(
        "ball_and_zeppelin", "ball_and_zeppelin", "white-matter"
    ),
    "free_water_elimination": ModelSpec(
        "free_water_elimination", "free_water_elimination", "white-matter"
    ),
    "ivim": ModelSpec("ivim", "ivim", "perfusion"),
    "noddi": ModelSpec(
        "noddi",
        "noddi",
        "orientation-dispersion",
        output_aliases={
            "SD1WatsonDistributed_1_SD1Watson_1_odi": "ODI",
            "partial_volume_1": "FISO",
        },
    ),
    "bingham_noddi": ModelSpec(
        "bingham_noddi", "bingham_noddi", "orientation-dispersion"
    ),
    "noddida": ModelSpec("noddida", "noddida", "orientation-dispersion"),
    "mcsmt": ModelSpec("mcsmt", "mcsmt", "spherical-mean"),
    "two_fascicle_noddi": ModelSpec(
        "two_fascicle_noddi", "two_fascicle_noddi", "multi-fascicle"
    ),
    "charmed": ModelSpec(
        "charmed", "charmed", "axon-diameter", ("delta", "Delta")
    ),
    "axcaliber": ModelSpec(
        "axcaliber", "axcaliber", "axon-diameter", ("delta", "Delta")
    ),
    "active_ax": ModelSpec(
        "active_ax", "active_ax", "axon-diameter", ("delta", "Delta")
    ),
    "verdict": ModelSpec("verdict", "verdict", "soma", ("delta", "Delta")),
    "sandi": ModelSpec("sandi", "sandi", "soma", ("delta", "Delta")),
    "impulsed": ModelSpec(
        "impulsed", "impulsed", "soma", ("composite_waveform",)
    ),
    "nexi": ModelSpec("nexi", "nexi", "exchange", ("diffusion_time",)),
    "karger_two_compartment": ModelSpec(
        "karger_two_compartment",
        "karger_two_compartment",
        "exchange",
        ("diffusion_time",),
    ),
    "fexi": ModelSpec("fexi", "fexi", "exchange", ("composite_waveform",)),
    "sandix": ModelSpec(
        "sandix", "sandix", "exchange", ("delta", "Delta", "diffusion_time")
    ),
    "exchange_impulsed": ModelSpec(
        "exchange_impulsed", "exchange_impulsed", "exchange", ("composite_waveform",)
    ),
    "temporal_zeppelin_model": ModelSpec(
        "temporal_zeppelin_model",
        "temporal_zeppelin_model",
        "time-dependent",
        ("diffusion_time",),
    ),
    "mte_ball_stick": ModelSpec(
        "mte_ball_stick", "mte_ball_stick", "relaxometry", ("TE",)
    ),
    "mte_noddi": ModelSpec(
        "mte_noddi", "mte_noddi", "relaxometry", ("TE",)
    ),
    "mte_sandi": ModelSpec(
        "mte_sandi", "mte_sandi", "relaxometry", ("delta", "Delta", "TE")
    ),
    "wmti": ModelSpec("wmti", "wmti", "white-matter"),
    "noddida_mte": ModelSpec(
        "noddida_mte", "noddida_mte", "relaxometry", ("TE",)
    ),
    "mte_impulsed": ModelSpec(
        "mte_impulsed", "mte_impulsed", "relaxometry", ("composite_waveform", "TE")
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    key = str(name).lower()
    try:
        return MODEL_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown dmipy model {name!r}. Available models: {available}.") from exc


def build_reference_model(name: str):
    """Build an allow-listed dmipy-fit reference model."""
    spec = get_model_spec(name)
    reference_models = import_module("dmipy_fit.custom_optimizers.reference_models")
    factory: Callable[[], Any] = getattr(reference_models, spec.factory_name)
    return factory()


def acquisition_scheme_from_bvalues(
    bvalues: np.ndarray,
    gradient_directions: np.ndarray,
    *,
    delta: np.ndarray | float | None = None,
    Delta: np.ndarray | float | None = None,
    TE: np.ndarray | float | None = None,
):
    """Construct a dmipy-fit scheme from validated SI-unit arrays."""
    acquisition = import_module("dmipy_fit.core.acquisition_scheme")
    kwargs: dict[str, Any] = {
        "bvalues": np.asarray(bvalues, dtype=float),
        "gradient_directions": np.asarray(gradient_directions, dtype=float),
    }
    if delta is not None:
        kwargs["delta"] = delta
    if Delta is not None:
        kwargs["Delta"] = Delta
    if TE is not None:
        kwargs["TE"] = TE
    return acquisition.acquisition_scheme_from_bvalues(**kwargs)


def fit_model(
    model,
    acquisition_scheme,
    data: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    solver: str = "brute2fine",
    device: str = "auto",
    nthreads: int = 1,
    solver_kwargs: Mapping[str, Any] | None = None,
):
    """Fit a model using the released dmipy-fit API and return runtime metadata."""
    runtime = DmipyRuntime.resolve(solver, device)
    options = dict(solver_kwargs or {})
    options["solver"] = runtime.solver
    if mask is not None:
        options["mask"] = mask
    if runtime.solver != "jax" and nthreads > 1:
        options.setdefault("use_parallel_processing", True)
        options.setdefault("number_of_processors", nthreads)
    elif runtime.solver == "jax":
        options.pop("use_parallel_processing", None)
        options.pop("number_of_processors", None)

    accepted = set(inspect.signature(model.fit).parameters)
    unknown = sorted(set(options) - accepted)
    if unknown:
        raise ValueError(
            "Unsupported dmipy-fit solver option(s): " + ", ".join(unknown)
        )
    return model.fit(acquisition_scheme, data, **options), runtime
