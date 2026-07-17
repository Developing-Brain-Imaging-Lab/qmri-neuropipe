import numpy as np
import pytest

from qmri_neuropipe.interfaces.dmipy import _sandi_fraction_maps
from qmri_neuropipe.interfaces.dmipy import (
    _build_sandi_model,
    _load_sandi_gradients,
    _load_sandi_timing,
)
from qmri_neuropipe.core import ProcessingError


def test_sandi_fractions_are_derived_from_nested_dmipy_parameters():
    maps = {
        "partial_volume_0": np.array([0.8, 0.3]),
        "BundleModel_1_partial_volume_0": np.array([0.25, 0.6]),
        "partial_volume_1": np.array([0.2, 0.7]),
    }

    fsoma, fneurite, fextra = _sandi_fraction_maps(maps)

    assert fsoma == pytest.approx([0.6, 0.12])
    assert fneurite == pytest.approx([0.2, 0.18])
    assert fextra == pytest.approx([0.2, 0.7])
    assert fsoma + fneurite + fextra == pytest.approx([1.0, 1.0])


def test_sandi_fraction_derivation_requires_both_nested_fractions():
    extra = np.array([0.4])

    fsoma, fneurite, fextra = _sandi_fraction_maps(
        {"partial_volume_1": extra}
    )

    assert fsoma is None
    assert fneurite is None
    assert fextra is extra


def test_sandi_gradients_accept_both_fsl_orientations(tmp_path):
    bval = tmp_path / "dwi.bval"
    bvec = tmp_path / "dwi.bvec"
    np.savetxt(bval, [0, 1000])
    np.savetxt(bvec, [[0, 1], [0, 0], [0, 0]])

    bvals, bvecs = _load_sandi_gradients(bval, bvec)
    assert bvals == pytest.approx([0, 1e9])
    assert bvecs.shape == (2, 3)

    np.savetxt(bvec, [[0, 0, 0], [1, 0, 0]])
    _, transposed = _load_sandi_gradients(bval, bvec)
    assert transposed == pytest.approx(bvecs)


def test_sandi_timing_is_required_and_scalar_values_are_expanded(tmp_path):
    with pytest.raises(ProcessingError, match="requires both"):
        _load_sandi_timing(None, None, 2)

    delta = tmp_path / "small_delta.txt"
    Delta = tmp_path / "big_delta.txt"
    np.savetxt(delta, [0.012])
    np.savetxt(Delta, [0.032])
    small, big = _load_sandi_timing(delta, Delta, 2)
    assert small == pytest.approx([0.012, 0.012])
    assert big == pytest.approx([0.032, 0.032])


def test_sandi_timing_rejects_nonphysical_ordering(tmp_path):
    delta = tmp_path / "small_delta.txt"
    Delta = tmp_path / "big_delta.txt"
    np.savetxt(delta, [0.040])
    np.savetxt(Delta, [0.030])
    with pytest.raises(ValueError, match="greater than"):
        _load_sandi_timing(delta, Delta, 1)


def test_sandi_model_uses_named_soma_diffusivity():
    model = _build_sandi_model({"soma_diffusivity": 2.5e-9})
    sphere = model.models[0].models[1]
    assert sphere.diffusion_constant == pytest.approx(2.5e-9)
