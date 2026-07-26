from types import SimpleNamespace

import numpy as np
import pytest

from qmri_neuropipe.interfaces.dmipy_jax_gnl import corrected_scheme_arrays
from qmri_neuropipe.interfaces.dmipy_jax_gnl import fit_model_jax_gnl


def _scheme():
    return SimpleNamespace(
        bvalues=np.array([0.0, 1.0e9, 2.0e9]),
        gradient_directions=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        delta=np.full(3, 0.01),
        Delta=np.full(3, 0.04),
        TE=None,
    )


def test_corrected_scheme_arrays_applies_voxel_tensor_and_recomputes_pgse():
    tensors = np.array(
        [
            np.eye(3),
            np.diag([2.0, 0.5, 1.0]),
        ]
    )

    result = corrected_scheme_arrays(_scheme(), tensors)

    assert result.fallback_count == 0
    np.testing.assert_allclose(
        result.arrays["bvalues"],
        [
            [0.0, 1.0e9, 2.0e9],
            [0.0, 4.0e9, 0.5e9],
        ],
    )
    np.testing.assert_allclose(
        result.arrays["gradient_directions"][1],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    np.testing.assert_allclose(
        result.arrays["tau"],
        result.arrays["Delta"] - result.arrays["delta"] / 3.0,
    )
    np.testing.assert_allclose(
        result.arrays["qvalues"],
        np.sqrt(result.arrays["bvalues"] / result.arrays["tau"])
        / (2.0 * np.pi),
    )


def test_corrected_scheme_arrays_matches_existing_row_vector_convention():
    tensor = np.array(
        [[1.0, 0.2, 0.0], [0.0, 0.9, 0.1], [0.0, 0.0, 1.1]]
    )
    scheme = _scheme()

    result = corrected_scheme_arrays(scheme, tensor[None, ...])

    transformed = scheme.gradient_directions @ tensor.T
    norms = np.linalg.norm(transformed, axis=1)
    expected_directions = np.zeros_like(transformed)
    np.divide(
        transformed,
        norms[:, None],
        out=expected_directions,
        where=norms[:, None] > 0,
    )
    np.testing.assert_allclose(
        result.arrays["bvalues"][0], scheme.bvalues * norms**2
    )
    np.testing.assert_allclose(
        result.arrays["gradient_directions"][0], expected_directions
    )


def test_corrected_scheme_arrays_falls_back_for_nonfinite_tensor():
    result = corrected_scheme_arrays(
        _scheme(), np.full((1, 3, 3), np.nan)
    )

    assert result.fallback_count == 1
    np.testing.assert_allclose(result.arrays["bvalues"][0], _scheme().bvalues)
    np.testing.assert_allclose(
        result.arrays["gradient_directions"][0],
        _scheme().gradient_directions,
    )


def test_corrected_scheme_arrays_falls_back_for_collapsed_weighted_gradients():
    result = corrected_scheme_arrays(_scheme(), np.zeros((1, 3, 3)))

    assert result.fallback_count == 1
    np.testing.assert_allclose(result.arrays["bvalues"][0], _scheme().bvalues)


def test_corrected_scheme_arrays_accepts_flat_tensor_layout():
    result = corrected_scheme_arrays(_scheme(), np.eye(3).reshape(1, 9))

    np.testing.assert_allclose(result.arrays["bvalues"][0], _scheme().bvalues)


@pytest.mark.parametrize("shape", [(2, 8), (2, 2, 2), (9,)])
def test_corrected_scheme_arrays_rejects_invalid_tensor_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        corrected_scheme_arrays(_scheme(), np.zeros(shape))


def test_fit_model_jax_gnl_runs_vectorized_optimizer_and_returns_maps():
    pytest.importorskip("jax")
    pytest.importorskip("jaxopt")
    acquisition = pytest.importorskip("dmipy_fit.core.acquisition_scheme")
    modeling = pytest.importorskip("dmipy_fit.core.modeling_framework")
    gaussian = pytest.importorskip("dmipy_fit.signal_models.gaussian_models")

    bvalues = np.array([0.0, 0.0, 0.5e9, 0.5e9, 1.0e9, 1.0e9])
    directions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    scheme = acquisition.acquisition_scheme_from_bvalues(
        bvalues, directions
    )
    model = modeling.MultiCompartmentModel(models=[gaussian.G1Ball()])
    true_diffusivity = 1.5e-9
    signal = np.exp(-bvalues * true_diffusivity)
    data = np.stack([1000.0 * signal, 900.0 * signal]).astype(np.float32)

    fitted = fit_model_jax_gnl(
        model,
        scheme,
        data,
        np.stack([np.eye(3), np.eye(3)]),
        solver_kwargs={
            "Ns": 3,
            "N_sphere_samples": 30,
            "maxiter": 20,
            "batch_size": 2,
        },
    )

    estimate = fitted.fitted_parameters["G1Ball_1_lambda_iso"]
    np.testing.assert_allclose(
        estimate,
        [true_diffusivity, true_diffusivity],
        rtol=2e-3,
    )
