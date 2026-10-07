import sys
from types import SimpleNamespace

from qmri_neuropipe.interfaces import ants as ants_interface
from qmri_neuropipe.interfaces.ants import (
    _REGISTRATION_SCHEDULE_KEYS,
    _normalize_interpolator,
    _normalize_registration_schedule_kwargs,
)


def test_generic_interpolation_aliases_are_valid_ants_names():
    assert _normalize_interpolator("sinc") == "lanczosWindowedSinc"
    assert _normalize_interpolator("cubic") == "bSpline"
    assert _normalize_interpolator("nearest") == "nearestNeighbor"
    assert _normalize_interpolator("trilinear") == "linear"


def test_canonical_ants_interpolation_name_is_preserved():
    assert _normalize_interpolator("welchWindowedSinc") == "welchWindowedSinc"


def test_yaml_registration_schedules_are_converted_to_tuples():
    normalized = _normalize_registration_schedule_kwargs(
        {
            "aff_iterations": [3000, 2000, 1000, 200],
            "aff_shrink_factors": [8, 4, 2, 1],
            "aff_smoothing_sigmas": [4, 2, 1, 0],
            "aff_metric": "mattes",
        }
    )

    assert normalized == {
        "aff_iterations": (3000, 2000, 1000, 200),
        "aff_shrink_factors": (8, 4, 2, 1),
        "aff_smoothing_sigmas": (4, 2, 1, 0),
        "aff_metric": "mattes",
    }


def test_registration_does_not_forward_schedule_to_transform_application(
    tmp_path, monkeypatch
):
    registration_call = {}
    application_call = {}

    def fake_registration(**kwargs):
        registration_call.update(kwargs)
        return {"fwdtransforms": [tmp_path / "transform.mat"]}

    fake_ants = SimpleNamespace(
        image_read=lambda path: path,
        registration=fake_registration,
    )
    monkeypatch.setitem(sys.modules, "ants", fake_ants)
    monkeypatch.setattr(
        ants_interface,
        "apply_transforms",
        lambda **kwargs: application_call.update(kwargs),
    )

    fixed = tmp_path / "fixed.nii.gz"
    moving = tmp_path / "moving.nii.gz"
    fixed.touch()
    moving.touch()
    ants_interface.registration(
        fixed,
        moving,
        tmp_path / "registered_",
        aff_iterations=[3000, 2000, 1000, 200],
        aff_shrink_factors=[8, 4, 2, 1],
        aff_smoothing_sigmas=[4, 2, 1, 0],
    )

    assert registration_call["aff_iterations"] == (3000, 2000, 1000, 200)
    assert registration_call["aff_shrink_factors"] == (8, 4, 2, 1)
    assert registration_call["aff_smoothing_sigmas"] == (4, 2, 1, 0)
    assert not _REGISTRATION_SCHEDULE_KEYS & application_call.keys()
