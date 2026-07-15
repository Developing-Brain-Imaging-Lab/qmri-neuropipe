import pytest

from qmri_neuropipe.interfaces.amico import (
    SANDI_AMICO_DEFAULTS,
    _configure_sandi_model,
)


class FakeSANDI:
    def set(self, **kwargs):
        self.model = kwargs

    def set_solver(self, **kwargs):
        self.solver = kwargs


def test_paper_defaults_are_converted_to_amico_units():
    model = FakeSANDI()
    metadata = _configure_sandi_model(model)

    assert model.model["d_is"] == pytest.approx(0.003)
    assert model.model["Rs"] == pytest.approx([1e-6, 3.8e-6, 6.6e-6, 9.4e-6, 12.2e-6, 15e-6])
    assert model.model["d_in"] == pytest.approx([0.00025, 0.0011666666667, 0.0020833333333, 0.003])
    assert model.model["d_isos"] == pytest.approx([0.00025, 0.0009375, 0.001625, 0.0023125, 0.003])
    assert model.solver == {"lambda1": 0.0, "lambda2": 7.5e-4}
    assert metadata["SomaRadii"] == list(SANDI_AMICO_DEFAULTS["soma_radii"])


def test_custom_dictionary_and_solver_are_supported():
    model = FakeSANDI()
    _configure_sandi_model(
        model,
        soma_diffusivity=2.5,
        soma_radii=[2, 8],
        neurite_diffusivities=[0.5, 2.0],
        extra_diffusivities=[0.4, 1.5],
        l1_regularization=0.1,
        l2_regularization=0.2,
    )
    assert model.model["d_is"] == pytest.approx(0.0025)
    assert model.model["Rs"] == pytest.approx([2e-6, 8e-6])
    assert model.solver == {"lambda1": 0.1, "lambda2": 0.2}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"soma_radii": []},
        {"soma_radii": [2, 1]},
        {"neurite_diffusivities": [0.0, 1.0]},
        {"l2_regularization": -1.0},
        {"soma_diffusivity": float("nan")},
    ],
)
def test_invalid_sandi_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        _configure_sandi_model(FakeSANDI(), **kwargs)
