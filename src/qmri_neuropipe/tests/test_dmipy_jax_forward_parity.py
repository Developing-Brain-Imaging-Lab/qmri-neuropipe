import numpy as np
import pytest

pytest.importorskip("dmipy_fit")
pytest.importorskip("jax")
pytest.importorskip("jaxopt")

from dmipy_fit.jax.multicompartment_jax import build_mc_forward_fn

from qmri_neuropipe.interfaces.dmipy_backend import build_reference_model
from qmri_neuropipe.interfaces.dmipy_capabilities import (
    representative_parameters,
    validation_acquisition_scheme,
)


@pytest.mark.parametrize(
    "model_name",
    ["ball", "noddi", "nexi", "microglia", "mte_noddi"],
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
