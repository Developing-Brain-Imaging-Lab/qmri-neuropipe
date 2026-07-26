import numpy as np
import pytest

from qmri_neuropipe.core import ProcessingError
from qmri_neuropipe.interfaces.dmipy_microglia import (
    _add_paper_microglia_maps,
    _build_microglia_model,
    _load_microglia_gradients,
    _load_microglia_timing,
    _microglia_metric_name,
    _microglia_metric_metadata,
)


def _make_model(config=None):
    pytest.importorskip("dmipy_fit")

    from dmipy_fit.core.modeling_framework import MultiCompartmentModel
    from dmipy_fit.distributions import distribute_models
    from dmipy_fit.signal_models import cylinder_models, gaussian_models, sphere_models

    model = _build_microglia_model(
        cylinder_models,
        gaussian_models,
        sphere_models,
        distribute_models,
        MultiCompartmentModel,
        config or {},
    )
    return model


def test_microglia_model_fits_orientation_and_dispersion_independently():
    model = _make_model()

    bundle = model.models[0]
    assert "SD1Watson_1_mu" in bundle.parameter_names
    assert bundle.parameter_cardinality["SD1Watson_1_mu"] == 2
    assert "SD1Watson_1_odi" in bundle.parameter_names
    assert bundle.parameter_cardinality["SD1Watson_1_odi"] == 1
    assert not any(link[1] == "mu" for link in bundle.parameter_links)

    axial_link = next(
        link for link in bundle.parameter_links if link[1] == "lambda_par" and hasattr(link[2], "value")
    )
    assert axial_link[2].value == pytest.approx(1.0e-9)

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

    # dmipy-fit 2.1 adds optional sphere surface-relaxivity parameters. They
    # are not identifiable in this diffusion-only model and must remain fixed.
    assert not any(
        name.endswith("surface_relaxivity") for name in model.parameter_names
    )


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


def test_microglia_model_rejects_overlapping_sphere_bounds():
    with pytest.raises(ValueError, match="must not overlap"):
        _make_model(
            {
                "small_diameter_bounds": [5e-6, 13e-6],
                "large_diameter_bounds": [12e-6, 18e-6],
            }
        )


def test_gradient_loading_accepts_fsl_and_row_major_layouts(tmp_path):
    bval = tmp_path / "test.bval"
    fsl_bvec = tmp_path / "fsl.bvec"
    row_bvec = tmp_path / "row.bvec"
    np.savetxt(bval, [[0, 1000, 2000, 3000]])
    vectors = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1.02, 0], [0, 0, 1]], dtype=float
    )
    np.savetxt(fsl_bvec, vectors.T)
    np.savetxt(row_bvec, vectors)

    for path in (fsl_bvec, row_bvec):
        bvals, bvecs = _load_microglia_gradients(bval, path)
        np.testing.assert_array_equal(bvals, [0, 1e9, 2e9, 3e9])
        np.testing.assert_allclose(
            bvecs, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )


def test_timing_is_required_and_broadcast_from_seconds(tmp_path):
    with pytest.raises(ProcessingError, match="requires both"):
        _load_microglia_timing(None, None, 3)

    delta = tmp_path / "small_delta.txt"
    Delta = tmp_path / "big_delta.txt"
    np.savetxt(delta, [0.012])
    np.savetxt(Delta, [0.032])
    small, big = _load_microglia_timing(delta, Delta, 3)
    np.testing.assert_allclose(small, [0.012] * 3)
    np.testing.assert_allclose(big, [0.032] * 3)


def test_timing_rejects_nonphysical_order(tmp_path):
    delta = tmp_path / "small_delta.txt"
    Delta = tmp_path / "big_delta.txt"
    np.savetxt(delta, [0.040])
    np.savetxt(Delta, [0.030])
    with pytest.raises(ValueError, match="greater than"):
        _load_microglia_timing(delta, Delta, 1)


@pytest.mark.parametrize(
    ("parameter", "metric"),
    [
        ("partial_volume_0", "f_bundle"),
        ("partial_volume_3", "f_iso"),
        ("SD1WatsonDistributed_1_SD1Watson_1_mu", "mu"),
        ("SD1WatsonDistributed_1_SD1Watson_1_odi", "odi"),
        ("SD1WatsonDistributed_1_partial_volume_0", "bundle_stick_fraction"),
        ("S2SphereStejskalTannerApproximation_1_diameter", "small_sphere_diameter"),
        ("S2SphereStejskalTannerApproximation_2_diameter", "large_sphere_diameter"),
    ],
)
def test_metric_names_are_stable_and_interpretable(parameter, metric):
    assert _microglia_metric_name(parameter) == metric


def test_orientation_metadata_describes_two_radian_angles():
    metadata = _microglia_metric_metadata("mu")
    assert metadata["MetricUnits"] == "radian"
    assert metadata["MetricComponents"] == ["polar_angle", "azimuthal_angle"]


def test_paper_maps_are_derived_from_nested_dmipy_parameters():
    maps = {
        "partial_volume_0": np.array([0.6]),
        "SD1WatsonDistributed_1_partial_volume_0": np.array([0.25]),
        "partial_volume_3": np.array([0.1]),
        "S2SphereStejskalTannerApproximation_1_diameter": np.array([8e-6]),
        "S2SphereStejskalTannerApproximation_2_diameter": np.array([16e-6]),
        "SD1WatsonDistributed_1_SD1Watson_1_odi": np.array([0.5]),
    }
    _add_paper_microglia_maps(maps)
    assert maps["derived_f_stick"] == pytest.approx([0.15])
    assert maps["derived_f_extracellular"] == pytest.approx([0.45])
    assert maps["derived_f_tissue"] == pytest.approx([0.9])
    assert maps["derived_small_sphere_radius"] == pytest.approx([4e-6])
    assert maps["derived_large_sphere_radius"] == pytest.approx([8e-6])
    assert maps["derived_watson_kappa"] == pytest.approx([1.0])


def test_registered_microglia_model_simulates_finite_signal():
    pytest.importorskip("dmipy_fit")
    from qmri_neuropipe.interfaces.dmipy_backend import (
        acquisition_scheme_from_bvalues,
        build_reference_model,
        get_model_spec,
    )

    model = build_reference_model("microglia")
    bvalues = np.array([0.0, 1e9, 1e9, 1e9, 2e9, 2e9, 2e9])
    directions = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=float,
    )
    scheme = acquisition_scheme_from_bvalues(
        bvalues,
        directions,
        delta=np.full(bvalues.size, 0.012),
        Delta=np.full(bvalues.size, 0.035),
    )
    parameters = {
        "SD1WatsonDistributed_1_G2Zeppelin_1_lambda_perp": 0.5e-9,
        "SD1WatsonDistributed_1_SD1Watson_1_mu": np.array([np.pi / 2, 0]),
        "SD1WatsonDistributed_1_SD1Watson_1_odi": 0.3,
        "SD1WatsonDistributed_1_partial_volume_0": 0.6,
        "S2SphereStejskalTannerApproximation_1_diameter": 8e-6,
        "S2SphereStejskalTannerApproximation_2_diameter": 16e-6,
        "partial_volume_0": 0.4,
        "partial_volume_1": 0.1,
        "partial_volume_2": 0.1,
        "partial_volume_3": 0.4,
    }

    signal = model.simulate_signal(scheme, parameters)

    assert get_model_spec("microglia").acquisition_requirements == (
        "delta",
        "Delta",
    )
    assert signal[0] == pytest.approx(1.0)
    assert np.all(np.isfinite(signal))
    assert np.all((signal >= 0) & (signal <= 1.0))
