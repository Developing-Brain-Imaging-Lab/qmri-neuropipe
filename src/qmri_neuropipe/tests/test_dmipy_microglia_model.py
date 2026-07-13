import numpy as np
import pytest

from qmri_neuropipe.interfaces.dmipy_microglia import _build_microglia_model


def _make_model(config=None):
    pytest.importorskip("dmipy")

    from dmipy.core.modeling_framework import MultiCompartmentModel
    from dmipy.distributions import distribute_models
    from dmipy.signal_models import cylinder_models, gaussian_models, sphere_models

    model = _build_microglia_model(
        cylinder_models,
        gaussian_models,
        sphere_models,
        distribute_models,
        MultiCompartmentModel,
        config or {},
    )
    return model


def test_microglia_model_matches_published_parameterization():
    model = _make_model()

    bundle = model.models[0]
    assert "SD1Watson_1_mu" not in bundle.parameter_names
    mu_link = next(link for link in bundle.parameter_links if link[1] == "mu")
    np.testing.assert_array_equal(mu_link[2].value, [0.0, 0.0])

    # Radial extracellular diffusivity remains independently fitted: no
    # tortuosity constraint links it to the stick fraction or diffusivity.
    assert "G2Zeppelin_1_lambda_perp" in bundle.parameter_names

    diameter_parameters = {
        "S2SphereStejskalTannerApproximation_1_diameter": (
            [5e-6, 11e-6],
            8e-6,
        ),
        "S2SphereStejskalTannerApproximation_2_diameter": (
            [12e-6, 18e-6],
            16e-6,
        ),
    }
    for parameter, values in diameter_parameters.items():
        expected_bounds, expected_initial = values
        assert parameter in model.parameter_names
        physical_bounds = (
            np.asarray(model.parameter_ranges[parameter])
            * model.parameter_scales[parameter]
        )
        np.testing.assert_allclose(physical_bounds, expected_bounds)
        assert model.x0_parameters[parameter] == pytest.approx(expected_initial)
        assert model.parameter_optimization_flags[parameter]


@pytest.mark.parametrize(
    ("config", "label"),
    [
        ({"small_diameter": 4e-6}, "microglia"),
        ({"large_diameter": 20e-6}, "astrocyte"),
    ],
)
def test_microglia_model_rejects_initial_diameter_outside_bounds(config, label):
    with pytest.raises(ValueError, match=f"Initial {label} diameter"):
        _make_model(config)
