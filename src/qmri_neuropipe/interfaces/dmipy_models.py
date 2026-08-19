"""Project-maintained model factories built from dmipy-fit 2.x components."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


MICROGLIA_MODEL_REFERENCE = "https://doi.org/10.1126/sciadv.abq2923"


def noddi_variant(
    *,
    parallel_diffusivity: float = 1.7e-9,
    iso_diffusivity: float = 3.0e-9,
    distribution: str = "Watson",
    model_type: str = "standard",
    fixed_parameters: Mapping[str, Any] | None = None,
):
    """Build the configurable NODDI variant used by the dedicated adapter."""
    from dmipy_fit.core.modeling_framework import (
        MultiCompartmentModel,
        MultiCompartmentSphericalMeanModel,
    )
    from dmipy_fit.distributions import distribute_models
    from dmipy_fit.signal_models import cylinder_models, gaussian_models

    distribution_key = str(distribution).lower()
    if distribution_key not in {"watson", "bingham"}:
        raise ValueError("NODDI distribution must be 'Watson' or 'Bingham'.")
    model_type_key = str(model_type).lower()
    if model_type_key not in {"standard", "smt"}:
        raise ValueError("NODDI model_type must be 'standard' or 'smt'.")

    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    if distribution_key == "bingham":
        bundle = distribute_models.SD2BinghamDistributed(
            models=[stick, zeppelin]
        )
    else:
        bundle = distribute_models.SD1WatsonDistributed(
            models=[stick, zeppelin]
        )
    bundle.set_tortuous_parameter(
        "G2Zeppelin_1_lambda_perp",
        "C1Stick_1_lambda_par",
        "partial_volume_0",
    )
    bundle.set_equal_parameter(
        "G2Zeppelin_1_lambda_par",
        "C1Stick_1_lambda_par",
    )
    bundle.set_fixed_parameter(
        "G2Zeppelin_1_lambda_par",
        float(parallel_diffusivity),
    )
    model_class = (
        MultiCompartmentSphericalMeanModel
        if model_type_key == "smt"
        else MultiCompartmentModel
    )
    model = model_class(models=[bundle, ball])
    model.set_fixed_parameter("G1Ball_1_lambda_iso", float(iso_diffusivity))
    for name, value in (fixed_parameters or {}).items():
        if value is not None:
            model.set_fixed_parameter(name, value)
    return model


def sandi_spherical_mean(
    *,
    soma_diffusivity: float = 3.0e-9,
):
    """Build the historical spherical-mean SANDI dedicated-command model."""
    from dmipy_fit.core.modeling_framework import (
        MultiCompartmentSphericalMeanModel,
    )
    from dmipy_fit.distributions.distribute_models import BundleModel
    from dmipy_fit.signal_models import (
        cylinder_models,
        gaussian_models,
        sphere_models,
    )

    soma_diffusivity = float(soma_diffusivity)
    if not np.isfinite(soma_diffusivity) or soma_diffusivity <= 0:
        raise ValueError("soma_diffusivity must be finite and positive.")
    stick = cylinder_models.C1Stick()
    soma = sphere_models.S4SphereGaussianPhaseApproximation(
        diffusion_constant=soma_diffusivity
    )
    model = MultiCompartmentSphericalMeanModel(
        models=[BundleModel([stick, soma]), gaussian_models.G1Ball()]
    )
    model.set_parameter_optimization_bounds(
        "BundleModel_1_S4SphereGaussianPhaseApproximation_1_diameter",
        [2e-6, 24e-6],
    )
    model.set_parameter_optimization_bounds(
        "G1Ball_1_lambda_iso", [1e-10, 3e-9]
    )
    model.set_parameter_optimization_bounds(
        "BundleModel_1_C1Stick_1_lambda_par", [1e-10, 3e-9]
    )
    model.set_parameter_optimization_bounds(
        "BundleModel_1_partial_volume_0", [0.01, 0.99]
    )
    model.set_parameter_optimization_bounds(
        "partial_volume_1", [0.01, 0.99]
    )
    return model


def microglia(
    *,
    parallel_diffusivity: float = 1.0e-9,
    iso_diffusivity: float = 3.0e-9,
    small_diameter: float = 8e-6,
    large_diameter: float = 16e-6,
    small_diameter_bounds: tuple[float, float] = (5e-6, 11e-6),
    large_diameter_bounds: tuple[float, float] = (12e-6, 18e-6),
    _components: Mapping[str, Any] | None = None,
):
    """Build the Garcia-Hernandez et al. glial activation model.

    The signal comprises a Watson-dispersed stick/zeppelin bundle, small and
    large restricted spheres, and free water. The bare dmipy-fit 2.3 sphere
    compartments are diffusion-only; relaxation factors are not composed into
    this model.
    """
    if _components is None:
        from dmipy_fit.core.modeling_framework import MultiCompartmentModel
        from dmipy_fit.distributions import distribute_models
        from dmipy_fit.signal_models import (
            cylinder_models,
            gaussian_models,
            sphere_models,
        )
    else:
        MultiCompartmentModel = _components["MultiCompartmentModel"]
        distribute_models = _components["distribute_models"]
        cylinder_models = _components["cylinder_models"]
        gaussian_models = _components["gaussian_models"]
        sphere_models = _components["sphere_models"]

    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    dispersed_bundle = distribute_models.SD1WatsonDistributed(
        models=[stick, zeppelin]
    )
    dispersed_bundle.set_equal_parameter(
        "G2Zeppelin_1_lambda_par",
        "C1Stick_1_lambda_par",
    )
    dispersed_bundle.set_fixed_parameter(
        "G2Zeppelin_1_lambda_par",
        float(parallel_diffusivity),
    )

    # Garcia-Hernandez et al. use Neuman's finite-pulse restricted-sphere
    # expression (their reference 46, equation 18).  dmipy's matching model is
    # the Gaussian phase approximation, not the q-only short-gradient-pulse S2
    # approximation.  The paper fixes diffusivity inside every restriction to
    # 1e-9 m^2/s, supplied here through parallel_diffusivity.
    small_sphere = sphere_models.S4SphereGaussianPhaseApproximation(
        diffusion_constant=float(parallel_diffusivity)
    )
    large_sphere = sphere_models.S4SphereGaussianPhaseApproximation(
        diffusion_constant=float(parallel_diffusivity)
    )
    ball = gaussian_models.G1Ball()
    model = MultiCompartmentModel(
        models=[dispersed_bundle, small_sphere, large_sphere, ball]
    )

    small_bounds = list(map(float, small_diameter_bounds))
    large_bounds = list(map(float, large_diameter_bounds))
    for label, bounds in (
        ("small-sphere", small_bounds),
        ("large-sphere", large_bounds),
    ):
        if len(bounds) != 2 or not 0 < bounds[0] < bounds[1]:
            raise ValueError(
                f"{label} diameter bounds must contain two increasing "
                "positive values."
            )
    if small_bounds[1] >= large_bounds[0]:
        raise ValueError("Small- and large-sphere diameter bounds must not overlap.")

    for label, initial, bounds in (
        ("microglia", float(small_diameter), small_bounds),
        ("astrocyte", float(large_diameter), large_bounds),
    ):
        if not bounds[0] <= initial <= bounds[1]:
            raise ValueError(
                f"Initial {label} diameter {initial:g} m is outside the "
                f"optimization bounds [{bounds[0]:g}, {bounds[1]:g}] m."
            )

    small_key = "S4SphereGaussianPhaseApproximation_1_diameter"
    large_key = "S4SphereGaussianPhaseApproximation_2_diameter"
    model.set_parameter_optimization_bounds(small_key, small_bounds)
    model.set_parameter_optimization_bounds(large_key, large_bounds)
    model.set_initial_guess_parameter(small_key, float(small_diameter))
    model.set_initial_guess_parameter(large_key, float(large_diameter))
    model.set_fixed_parameter("G1Ball_1_lambda_iso", float(iso_diffusivity))
    return model


def microglia_output_alias(parameter_name: str) -> str:
    """Map dmipy parameter names to paper-facing microglia metrics."""
    exact = {
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
    }
    if parameter_name in exact:
        return exact[parameter_name]
    if parameter_name.endswith("SD1Watson_1_mu"):
        return "mu"
    if parameter_name.endswith("SD1Watson_1_odi"):
        return "odi"
    if parameter_name.endswith("G2Zeppelin_1_lambda_perp"):
        return "bundle_radial_diffusivity"
    if parameter_name.endswith("SD1WatsonDistributed_1_partial_volume_0"):
        return "bundle_stick_fraction"
    if parameter_name.endswith("S4SphereGaussianPhaseApproximation_1_diameter"):
        return "small_sphere_diameter"
    if parameter_name.endswith("S4SphereGaussianPhaseApproximation_2_diameter"):
        return "large_sphere_diameter"
    return parameter_name.replace("SD1WatsonDistributed_1_", "")


def microglia_output_maps(
    parameter_maps: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Add the paper-facing fractions, radii, and Watson concentration."""
    maps = dict(parameter_maps)
    f_bundle = maps.get("partial_volume_0")
    bundle_stick = maps.get("SD1WatsonDistributed_1_partial_volume_0")
    f_iso = maps.get("partial_volume_3")
    small_diameter = maps.get(
        "S4SphereGaussianPhaseApproximation_1_diameter"
    )
    large_diameter = maps.get(
        "S4SphereGaussianPhaseApproximation_2_diameter"
    )
    odi = maps.get("SD1WatsonDistributed_1_SD1Watson_1_odi")
    if f_bundle is not None and bundle_stick is not None:
        maps["derived_f_stick"] = f_bundle * bundle_stick
        maps["derived_f_extracellular"] = f_bundle * (1.0 - bundle_stick)
    if f_iso is not None:
        maps["derived_f_tissue"] = 1.0 - f_iso
    if small_diameter is not None:
        maps["derived_small_sphere_radius"] = 0.5 * small_diameter
    if large_diameter is not None:
        maps["derived_large_sphere_radius"] = 0.5 * large_diameter
    if odi is not None:
        safe_odi = np.clip(
            odi,
            np.finfo(float).eps,
            1.0 - np.finfo(float).eps,
        )
        maps["derived_watson_kappa"] = 1.0 / np.tan(0.5 * np.pi * safe_odi)
    return maps
