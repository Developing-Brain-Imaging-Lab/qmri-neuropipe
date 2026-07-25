"""Shared integration layer for the dmipy 2.x analytical fitting engine.

The PyPI ``dmipy`` project became a non-importable umbrella package in 2.x.
Pipeline code must import the analytical engine from ``dmipy_fit`` directly.
This module keeps version checks, solver validation, acquisition construction,
reference-model discovery, and fit keyword translation out of model-specific
interfaces.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Mapping
import inspect
import multiprocessing
import os
import sys
import time

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
    gpu_device: int | None = None
    jax_cache_dir: str | None = None
    jax_log_compiles: bool = False

    @property
    def uses_jax(self) -> bool:
        return self.solver == "jax"

    @classmethod
    def resolve(
        cls,
        solver: str = "brute2fine",
        device: str = "auto",
        *,
        gpu_device: int | None = None,
        jax_cache_dir: str | os.PathLike[str] | None = None,
        jax_log_compiles: bool = False,
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
        if gpu_device is not None:
            if solver != "jax":
                raise ValueError("gpu_device requires solver='jax'.")
            if device == "cpu":
                raise ValueError("gpu_device cannot be combined with device='cpu'.")
            if not isinstance(gpu_device, int) or gpu_device < 0:
                raise ValueError("gpu_device must be a non-negative integer.")
        if solver != "jax" and jax_cache_dir is not None:
            raise ValueError("jax_cache_dir requires solver='jax'.")
        if solver != "jax" and jax_log_compiles:
            raise ValueError("jax_log_compiles requires solver='jax'.")

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

        resolved_cache_dir = None
        if gpu_device is not None:
            requested_selector = str(gpu_device)
            configured_selector = os.environ.get("QMRI_DMIPY_GPU_SELECTOR")
            if "jax" in sys.modules:
                if configured_selector != requested_selector:
                    raise ProcessingError(
                        "gpu_device must be selected before JAX is imported. "
                        "Start a fresh process and set --gpu-device again."
                    )
            if configured_selector is None:
                scheduler_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
                if scheduler_visibility in {"", "-1"}:
                    raise ProcessingError(
                        "CUDA_VISIBLE_DEVICES hides all GPUs; cannot apply "
                        "--gpu-device."
                    )
                if scheduler_visibility is not None and scheduler_visibility != "all":
                    visible_tokens = [
                        token.strip()
                        for token in scheduler_visibility.split(",")
                        if token.strip()
                    ]
                    if gpu_device >= len(visible_tokens):
                        raise ProcessingError(
                            f"gpu_device={gpu_device} is outside the "
                            f"{len(visible_tokens)} CUDA device(s) exposed by "
                            "the scheduler or container."
                        )
                    selected_cuda_device = visible_tokens[gpu_device]
                else:
                    selected_cuda_device = requested_selector
                os.environ["CUDA_VISIBLE_DEVICES"] = selected_cuda_device
                os.environ["QMRI_DMIPY_GPU_SELECTOR"] = requested_selector
            elif configured_selector != requested_selector:
                raise ProcessingError(
                    "A different GPU was already selected in this process. "
                    "Start a fresh process to change --gpu-device."
                )
            # CUDA visibility now contains one device, exposed to JAX as index 0.
            os.environ["JAX_CUDA_VISIBLE_DEVICES"] = "0"
        if jax_cache_dir is not None:
            resolved_cache_dir = str(Path(jax_cache_dir).expanduser().resolve())
            os.environ["JAX_COMPILATION_CACHE_DIR"] = resolved_cache_dir
        elif os.environ.get("JAX_COMPILATION_CACHE_DIR"):
            resolved_cache_dir = os.environ["JAX_COMPILATION_CACHE_DIR"]
        if jax_log_compiles:
            os.environ["JAX_LOG_COMPILES"] = "1"
        resolved_log_compiles = os.environ.get("JAX_LOG_COMPILES", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

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

        jax_devices = (
            jax.local_devices() if hasattr(jax, "local_devices") else jax.devices()
        )
        available = tuple(str(item) for item in jax_devices)
        platforms = {getattr(item, "platform", "") for item in jax_devices}
        has_gpu = "gpu" in platforms or "cuda" in platforms
        if (device == "gpu" or gpu_device is not None) and not has_gpu:
            raise ProcessingError(
                "A GPU was explicitly requested but JAX reports no usable GPU."
            )
        elif device == "cpu":
            backend = "cpu"
        else:
            backend = "gpu" if has_gpu else "cpu"
        if gpu_device is not None:
            visible_gpus = [
                item
                for item in jax_devices
                if getattr(item, "platform", "") in {"gpu", "cuda"}
            ]
            if len(visible_gpus) != 1:
                raise ProcessingError(
                    "--gpu-device did not isolate exactly one JAX GPU; "
                    f"JAX reports {len(visible_gpus)} visible GPUs."
                )
        return cls(
            installed,
            solver,
            device,
            backend,
            available,
            gpu_device=gpu_device,
            jax_cache_dir=resolved_cache_dir,
            jax_log_compiles=resolved_log_compiles,
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "FittingSoftware": "dmipy-fit",
            "FittingSoftwareVersion": self.version,
            "Solver": self.solver,
            "RequestedDevice": self.requested_device,
            "ExecutionBackend": self.backend,
            "JAXDevices": list(self.devices),
            "GPUDeviceSelection": self.gpu_device,
            "JAXCompilationCacheDirectory": self.jax_cache_dir,
            "JAXCompileLogging": self.jax_log_compiles,
        }


@contextmanager
def dmipy_fit_output(solver: str):
    """Keep JAX progress visible while retaining quiet native fits."""
    if str(solver).lower() == "jax":
        yield
        return
    with open(os.devnull, "w") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


def _nested_to_normalized_fractions_numpy(nested: np.ndarray) -> np.ndarray:
    """Convert one nested fraction vector without a per-voxel JAX dispatch."""
    nested = np.asarray(nested)
    dtype = nested.dtype
    normalized = np.empty(nested.size + 1, dtype=dtype)
    remaining = dtype.type(1.0)
    for index, value in enumerate(nested):
        fraction = remaining * value
        normalized[index] = fraction
        remaining = remaining - fraction
    normalized[-1] = dtype.type(1.0) - normalized[:-1].sum(dtype=dtype)
    return normalized


def install_dmipy_jax_postprocessing_workaround() -> bool:
    """Replace dmipy 2.1's per-row JAX fraction conversion with NumPy.

    dmipy-fit 2.1 converts every fitted voxel from nested to normalized volume
    fractions in a Python loop. Its released implementation dispatches a tiny
    JAX operation for every row, which can take longer than the GPU fit for a
    whole-brain mask. Patch only implementations that still contain that JAX
    call; future dmipy releases with a native fix are left untouched.
    """
    module = import_module("dmipy_fit.jax.optimizers_jax")
    optimizer_class = module.JaxOptimizer
    current = optimizer_class._unnest_model
    if getattr(current, "_qmri_numpy_postprocessing", False):
        return False
    try:
        source = inspect.getsource(current)
    except (OSError, TypeError):
        return False
    if "nested_to_normalized_fractions_jax" not in source:
        return False

    def _unnest_model_numpy(self, x_model_nested):
        x_model_nested = np.asarray(x_model_nested)
        if self._is_multi:
            n_non_vf = len(self._scales) - self._N_models
            non_vf = x_model_nested[:n_non_vf]
            nested_vf = x_model_nested[n_non_vf:]
            normalized_vf = _nested_to_normalized_fractions_numpy(nested_vf)
            return np.concatenate([non_vf, normalized_vf])
        return x_model_nested

    _unnest_model_numpy._qmri_numpy_postprocessing = True
    optimizer_class._unnest_model = _unnest_model_numpy
    return True


def jax_run_summary(
    runtime: DmipyRuntime,
    n_voxels: int,
    solver_kwargs: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Return user-facing JAX execution details before a long fit starts."""
    if not runtime.uses_jax:
        return ()
    options = dict(solver_kwargs or {})
    lines = [
        f"JAX execution backend: {runtime.backend}",
        "JAX devices: " + (", ".join(runtime.devices) if runtime.devices else "none"),
    ]
    if runtime.backend == "cpu":
        lines.append(
            "WARNING: JAX is running on CPU; use --device gpu to require a GPU."
        )
    if runtime.gpu_device is not None:
        lines.append(f"Requested JAX CUDA device selector: {runtime.gpu_device}")
    batch_size = options.get("batch_size")
    if batch_size is not None:
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        expected = (int(n_voxels) + batch_size - 1) // batch_size
        lines.append(
            f"JAX optimizer batch size: {batch_size} voxels "
            f"({expected} expected batches for {n_voxels} voxels)"
        )
    else:
        lines.append("JAX optimizer batch size: automatic")
    if runtime.jax_cache_dir:
        lines.append(f"JAX persistent compilation cache: {runtime.jax_cache_dir}")
    else:
        lines.append("JAX persistent compilation cache: disabled")
    if runtime.jax_log_compiles:
        lines.append("JAX compilation logging: enabled")
    return tuple(lines)


def collect_pool_results_with_heartbeat(
    pool,
    worker: Callable[[Any], Any],
    chunk_args: list[Any],
    *,
    heartbeat_interval: float = 30.0,
    label: str = "JAX fitting",
) -> list[Any]:
    """Collect ordered worker results and report that long JAX work is alive."""
    if heartbeat_interval <= 0:
        raise ValueError("heartbeat_interval must be greater than zero.")
    pending = [pool.apply_async(worker, (args,)) for args in chunk_args]
    results = []
    started = time.monotonic()
    for index, result in enumerate(pending):
        while True:
            try:
                value = result.get(timeout=heartbeat_interval)
            except multiprocessing.TimeoutError:
                elapsed = time.monotonic() - started
                print(
                    f"{label} is still running "
                    f"(elapsed {elapsed / 60:.1f} min; worker "
                    f"{index + 1}/{len(pending)}).",
                    flush=True,
                )
            else:
                results.append(value)
                print(
                    f"Collected worker {index + 1}/{len(pending)} "
                    f"after {(time.monotonic() - started) / 60:.1f} min.",
                    flush=True,
                )
                break
    return results


@dataclass(frozen=True)
class ModelSpec:
    """Stable pipeline-facing description of a dmipy reference model."""

    name: str
    factory_name: str
    family: str
    acquisition_requirements: tuple[str, ...] = ()
    output_aliases: Mapping[str, str] = field(default_factory=dict)
    factory_module: str = "dmipy_fit.custom_optimizers.reference_models"
    output_adapter_name: str | None = None
    references: tuple[str, ...] = ()


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "ball": ModelSpec("ball", "ball", "gaussian"),
    "zeppelin": ModelSpec("zeppelin", "zeppelin", "gaussian"),
    "temporal_zeppelin": ModelSpec(
        "temporal_zeppelin", "temporal_zeppelin", "time-dependent", ("delta", "Delta")
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
    "impulsed": ModelSpec("impulsed", "impulsed", "soma", ("delta", "Delta")),
    "nexi": ModelSpec("nexi", "nexi", "exchange", ("delta", "Delta")),
    "karger_two_compartment": ModelSpec(
        "karger_two_compartment",
        "karger_two_compartment",
        "exchange",
        ("delta", "Delta"),
    ),
    "fexi": ModelSpec("fexi", "fexi", "exchange", ("delta", "Delta")),
    "sandix": ModelSpec(
        "sandix", "sandix", "exchange", ("delta", "Delta")
    ),
    "exchange_impulsed": ModelSpec(
        "exchange_impulsed", "exchange_impulsed", "exchange", ("delta", "Delta")
    ),
    "temporal_zeppelin_model": ModelSpec(
        "temporal_zeppelin_model",
        "temporal_zeppelin_model",
        "time-dependent",
        ("delta", "Delta"),
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
        "mte_impulsed", "mte_impulsed", "relaxometry", ("delta", "Delta", "TE")
    ),
    "microglia": ModelSpec(
        "microglia",
        "microglia",
        "glial-activation",
        ("delta", "Delta"),
        output_aliases={
            "SD1WatsonDistributed_1_G2Zeppelin_1_lambda_perp": "bundle_radial_diffusivity",
            "SD1WatsonDistributed_1_SD1Watson_1_mu": "mu",
            "SD1WatsonDistributed_1_SD1Watson_1_odi": "odi",
            "SD1WatsonDistributed_1_partial_volume_0": "bundle_stick_fraction",
            "S2SphereStejskalTannerApproximation_1_diameter": "small_sphere_diameter",
            "S2SphereStejskalTannerApproximation_2_diameter": "large_sphere_diameter",
            "partial_volume_0": "f_bundle",
            "partial_volume_1": "f_small_sphere",
            "partial_volume_2": "f_large_sphere",
            "partial_volume_3": "f_iso",
            "derived_f_stick": "f_stick",
            "derived_f_extracellular": "f_extracellular",
            "derived_f_tissue": "f_tissue",
            "derived_small_sphere_radius": "small_sphere_radius",
            "derived_large_sphere_radius": "large_sphere_radius",
            "derived_watson_kappa": "watson_kappa",
        },
        factory_module="qmri_neuropipe.interfaces.dmipy_models",
        output_adapter_name="microglia_output_maps",
        references=("https://doi.org/10.1126/sciadv.abq2923",),
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    key = str(name).lower()
    try:
        return MODEL_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown dmipy model {name!r}. Available models: {available}.") from exc


def build_reference_model(name: str, **factory_kwargs):
    """Build an allow-listed dmipy-fit reference model."""
    spec = get_model_spec(name)
    reference_models = import_module(spec.factory_module)
    factory: Callable[[], Any] = getattr(reference_models, spec.factory_name)
    return factory(**factory_kwargs)


def model_output_maps(name: str, fit_result: Any) -> Mapping[str, np.ndarray]:
    """Return raw or model-adapted fitted parameter maps."""
    spec = get_model_spec(name)
    parameter_maps = getattr(fit_result, "fitted_parameters", None)
    if not isinstance(parameter_maps, Mapping):
        raise TypeError(
            "dmipy fit result must expose fitted_parameters as a mapping."
        )
    if spec.output_adapter_name is None:
        return parameter_maps
    adapter_module = import_module(spec.factory_module)
    adapter = getattr(adapter_module, spec.output_adapter_name)
    return adapter(parameter_maps)


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
    gpu_device: int | None = None,
    jax_cache_dir: str | os.PathLike[str] | None = None,
    jax_log_compiles: bool = False,
    nthreads: int = 1,
    solver_kwargs: Mapping[str, Any] | None = None,
    runtime: DmipyRuntime | None = None,
):
    """Fit a model using the released dmipy-fit API and return runtime metadata."""
    if runtime is None:
        runtime = DmipyRuntime.resolve(
            solver,
            device,
            gpu_device=gpu_device,
            jax_cache_dir=jax_cache_dir,
            jax_log_compiles=jax_log_compiles,
        )
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
        install_dmipy_jax_postprocessing_workaround()

    accepted = set(inspect.signature(model.fit).parameters)
    unknown = sorted(set(options) - accepted)
    if unknown:
        raise ValueError(
            "Unsupported dmipy-fit solver option(s): " + ", ".join(unknown)
        )
    return model.fit(acquisition_scheme, data, **options), runtime
