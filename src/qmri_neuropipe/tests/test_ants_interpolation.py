import sys
from types import SimpleNamespace

import numpy as np

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


def test_registration_runs_rotation_search_and_rigidizes_initializer(
    tmp_path, monkeypatch
):
    initializer_call = {}
    registration_call = {}
    created_transform = {}

    class FakeImage:
        dimension = 3

    class FakeTransform:
        parameters = np.asarray(
            [2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 4.0, 5.0, 6.0]
        )
        fixed_parameters = np.asarray([10.0, 11.0, 12.0])

    def fake_initializer(**kwargs):
        initializer_call.update(kwargs)
        return kwargs["txfn"]

    def fake_create(**kwargs):
        created_transform.update(kwargs)
        return "rigid-transform"

    def fake_registration(**kwargs):
        registration_call.update(kwargs)
        return {"fwdtransforms": [tmp_path / "transform.mat"]}

    fake_ants = SimpleNamespace(
        image_read=lambda path: FakeImage(),
        affine_initializer=fake_initializer,
        read_transform=lambda path: FakeTransform(),
        create_ants_transform=fake_create,
        write_transform=lambda transform, path: None,
        registration=fake_registration,
    )
    monkeypatch.setitem(sys.modules, "ants", fake_ants)
    monkeypatch.setattr(ants_interface, "apply_transforms", lambda **kwargs: None)

    ants_interface.registration(
        tmp_path / "fixed.nii.gz",
        tmp_path / "moving.nii.gz",
        tmp_path / "registered_",
        transform_type="DenseRigid",
        initial_transform="Identity",
        rotation_search={
            "enabled": True,
            "search_factor": 10,
            "radian_fraction": 0.5,
            "local_search_iterations": 20,
        },
    )

    assert initializer_call["search_factor"] == 10
    assert initializer_call["radian_fraction"] == 0.5
    assert initializer_call["local_search_iterations"] == 20
    rotation = created_transform["matrix"]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-7)
    np.testing.assert_allclose(created_transform["translation"], [4, 5, 6])
    assert registration_call["initial_transform"].endswith(
        "RotationSearchRigid.mat"
    )
    assert "rotation_search" not in registration_call


def test_rotation_search_rejects_custom_initial_transform(tmp_path, monkeypatch):
    class FakeImage:
        dimension = 3

    fake_ants = SimpleNamespace(image_read=lambda path: FakeImage())
    monkeypatch.setitem(sys.modules, "ants", fake_ants)

    import pytest

    with pytest.raises(ValueError, match="custom initial_transform"):
        ants_interface.registration(
            tmp_path / "fixed.nii.gz",
            tmp_path / "moving.nii.gz",
            tmp_path / "registered_",
            initial_transform="custom.mat",
            rotation_search={"enabled": True},
        )
