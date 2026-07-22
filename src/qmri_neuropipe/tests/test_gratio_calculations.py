import numpy as np
import pytest

from qmri_neuropipe.lib.microstructure.gratio import (
    Calibration,
    compute_aggregate_gratio,
    compute_conduction_measures,
    compute_myelin_thickness,
)


def test_noddi_aggregate_gratio_identity_calibration():
    result = compute_aggregate_gratio(
        np.array([0.2]),
        np.array([0.5]),
        np.array([0.2]),
        calibration=Calibration(mode="identity"),
    )
    assert result.mvf[0] == pytest.approx(0.2)
    assert result.avf[0] == pytest.approx(0.4)
    assert result.fvf[0] == pytest.approx(0.6)
    assert result.gratio[0] == pytest.approx(np.sqrt(2 / 3))
    assert result.valid[0]


def test_linear_calibration_and_direct_avf():
    result = compute_aggregate_gratio(
        np.array([0.1]),
        np.array([0.3]),
        calibration=Calibration(mode="linear", slope=2, intercept=0.05),
        axonal_input_is_avf=True,
    )
    assert result.mvf[0] == pytest.approx(0.25)
    assert result.gratio[0] == pytest.approx(np.sqrt(0.3 / 0.55))


def test_invalid_values_are_nan_and_small_overshoot_is_clipped():
    result = compute_aggregate_gratio(
        np.array([1.0000001, 1.2, 0.0]),
        np.array([0.0, 0.1, 0.0]),
        calibration=Calibration(),
        axonal_input_is_avf=True,
    )
    assert result.valid.tolist() == [True, False, False]
    assert result.clipped.tolist() == [True, False, False]
    assert np.isnan(result.gratio[1:]).all()


def test_thickness_and_conduction_outputs():
    g = np.array([0.8])
    diameter = np.array([4.0])
    thickness = compute_myelin_thickness(diameter, g)
    assert thickness[0] == pytest.approx(0.5)

    outputs = compute_conduction_measures(
        g,
        diameter,
        rushton_coefficient=2.0,
        waxman_bennett_coefficient=3.0,
    )
    factor = np.sqrt(-np.log(0.8))
    assert outputs["ConductionFactor"][0] == pytest.approx(factor)
    assert outputs["RushtonCVIndex"][0] == pytest.approx(4 * factor)
    assert outputs["WaxmanBennettCVIndex"][0] == pytest.approx(5.0)
    assert outputs["RushtonCV"][0] == pytest.approx(8 * factor)
    assert outputs["WaxmanBennettCV"][0] == pytest.approx(15.0)


def test_no_diameter_emits_only_conduction_factor():
    outputs = compute_conduction_measures(np.array([0.75]))
    assert set(outputs) == {"ConductionFactor"}


def test_linear_calibration_requires_both_parameters():
    with pytest.raises(ValueError, match="slope and intercept"):
        Calibration(mode="linear", slope=2)


def test_invalid_noddi_source_fraction_cannot_hide_in_valid_avf_product():
    result = compute_aggregate_gratio(
        np.array([0.2, 0.2]),
        np.array([1.2, 0.5]),
        np.array([0.5, -0.2]),
    )
    assert result.valid.tolist() == [False, False]
