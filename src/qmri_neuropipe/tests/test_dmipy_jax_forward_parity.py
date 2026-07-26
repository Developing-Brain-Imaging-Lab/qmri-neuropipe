import numpy as np
import pytest

pytest.importorskip("dmipy_fit")
pytest.importorskip("jax")
pytest.importorskip("jaxopt")

from dmipy_fit.jax.multicompartment_jax import build_mc_forward_fn
from dmipy_fit.jax.jax_compat import scheme_to_jax

from qmri_neuropipe.interfaces.dmipy_backend import build_reference_model
from qmri_neuropipe.interfaces.dmipy_capabilities import (
    RECOVERY_CASES,
    parameter_recovery_probe,
    representative_parameters,
    validation_acquisition_scheme,
)
from qmri_neuropipe.interfaces.dmipy_jax_gnl import (
    _build_scheme_argument_forward,
)


@pytest.mark.parametrize(
    "model_name",
    ["ball", "noddi", "microglia", "mte_noddi"],
)
def test_jax_forward_model_matches_native_signal(model_name):
    model = build_reference_model(model_name)
    scheme = validation_acquisition_scheme()
    parameters = representative_parameters(model)
    parameter_vector = model.parameters_to_parameter_vector(**parameters)

    native_signal = model.simulate_signal(scheme, parameters)
    jax_forward = build_mc_forward_fn(model, scheme)
    jax_signal = np.asarray(jax_forward(parameter_vector))

    np.testing.assert_allclose(
        jax_signal,
        native_signal,
        rtol=5e-3,
        atol=5e-4,
    )


@pytest.mark.parametrize("model_name", ["noddi", "microglia"])
def test_voxel_scheme_jax_forward_matches_nominal_jax_forward(model_name):
    model = build_reference_model(model_name)
    scheme = validation_acquisition_scheme()
    parameters = representative_parameters(model)
    parameter_vector = model.parameters_to_parameter_vector(**parameters)

    nominal = np.asarray(build_mc_forward_fn(model, scheme)(parameter_vector))
    voxel_scheme_forward = _build_scheme_argument_forward(model, scheme)
    dynamic = np.asarray(
        voxel_scheme_forward(parameter_vector, scheme_to_jax(scheme))
    )

    np.testing.assert_allclose(dynamic, nominal, rtol=5e-3, atol=5e-4)


@pytest.mark.parametrize("case", RECOVERY_CASES, ids=lambda case: case.model_name)
def test_jax_optimizer_recovers_identifiable_synthetic_parameter(case):
    result = parameter_recovery_probe(case, solver="jax", device="cpu")

    assert result["passed"], result
    assert result["backend"] == "cpu"
