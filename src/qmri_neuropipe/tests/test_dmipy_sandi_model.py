import numpy as np
import pytest

from qmri_neuropipe.core.types import DWIFile
from qmri_neuropipe.interfaces.dmipy import _sandi_fraction_maps
from qmri_neuropipe.interfaces.dmipy import (
    _build_sandi_model,
    _load_sandi_gradients,
    _load_sandi_timing,
    _validate_sandi_fit_results,
    fit_sandi,
)
from qmri_neuropipe.interfaces.dmipy_backend import DmipyRuntime
from qmri_neuropipe.interfaces.dmipy_sandi_jax import (
    _build_forward,
    _nominal_scheme_arrays,
    _parameter_grid,
    _sandi_bounds_and_scales,
    fit_sandi_jax,
)
from qmri_neuropipe.lib.dmri.fitting import SANDIFittingStep
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
    pytest.importorskip("dmipy_fit")
    model = _build_sandi_model({"soma_diffusivity": 2.5e-9})
    sphere = model.models[0].models[1]
    assert sphere.diffusion_constant == pytest.approx(2.5e-9)


def test_sandi_jax_continues_to_input_loading(tmp_path, monkeypatch):
    runtime = DmipyRuntime(
        version="2.1.0",
        solver="jax",
        requested_device="gpu",
        backend="gpu",
    )
    monkeypatch.setattr(
        DmipyRuntime,
        "resolve",
        classmethod(lambda cls, *args, **kwargs: runtime),
    )

    with pytest.raises(FileNotFoundError):
        fit_sandi(
            tmp_path / "not-loaded.nii.gz",
            tmp_path / "out",
            bval_file=tmp_path / "dwi.bval",
            bvec_file=tmp_path / "dwi.bvec",
            delta_file=tmp_path / "dwi.delta",
            Delta_file=tmp_path / "dwi.bigdelta",
            solver="jax",
            device="gpu",
        )


def test_sandi_all_failed_results_raise_instead_of_writing_zero_maps():
    with pytest.raises(ProcessingError, match="failed for every voxel"):
        _validate_sandi_fit_results(
            {
                "partial_volume_0": np.full(3, np.nan),
                "partial_volume_1": np.full(3, np.nan),
            }
        )


def test_sandi_result_validation_reports_partial_success():
    fitted = _validate_sandi_fit_results(
        {
            "partial_volume_0": np.array([0.2, np.nan, 0.4]),
            "orientation": np.array(
                [[0.1, 0.2], [np.nan, np.nan], [0.3, 0.4]]
            ),
        }
    )

    assert fitted.tolist() == [True, False, True]


def test_sandi_jax_parameter_layout_uses_dmipy_bounds_and_scales():
    class Model:
        parameter_ranges = {
            "BundleModel_1_S4SphereGaussianPhaseApproximation_1_diameter": (
                2.0,
                24.0,
            ),
            "BundleModel_1_C1Stick_1_lambda_par": (0.1, 3.0),
            "G1Ball_1_lambda_iso": (0.1, 3.0),
            "BundleModel_1_partial_volume_0": (0.01, 0.99),
            "partial_volume_0": (0.01, 0.99),
        }
        parameter_scales = {
            "BundleModel_1_S4SphereGaussianPhaseApproximation_1_diameter": 1e-6,
            "BundleModel_1_C1Stick_1_lambda_par": 1e-9,
            "G1Ball_1_lambda_iso": 1e-9,
            "BundleModel_1_partial_volume_0": 1.0,
            "partial_volume_0": 1.0,
        }

    bounds, scales = _sandi_bounds_and_scales(Model())
    grid = _parameter_grid(bounds, 2)

    assert bounds.shape == (5, 2)
    assert scales == pytest.approx([1e-6, 1e-9, 1e-9, 1.0, 1.0])
    assert grid.shape == (32, 5)
    assert grid[0] == pytest.approx(bounds[:, 0])
    assert grid[-1] == pytest.approx(bounds[:, 1])


def test_sandi_jax_nominal_scheme_is_broadcast_per_voxel():
    class Scheme:
        bvalues = np.array([0.0, 1e9])
        gradient_strengths = np.array([0.0, 0.04])
        delta = np.array([0.012, 0.012])
        Delta = np.array([0.032, 0.032])

    arrays = _nominal_scheme_arrays(Scheme(), 3)

    assert set(arrays) == {
        "bvalues",
        "gradient_strengths",
        "delta",
        "Delta",
    }
    assert all(value.shape == (3, 2) for value in arrays.values())
    assert arrays["bvalues"][2] == pytest.approx([0.0, 1e9])


def test_sandi_jax_forward_matches_native_compartments_and_has_finite_gradient():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("dmipy_fit")
    from qmri_neuropipe.interfaces.dmipy import _build_dmipy_scheme

    bvalues = np.array([0.0, 1e9, 2e9])
    directions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    small_delta = np.full(3, 0.012)
    big_delta = np.full(3, 0.032)
    scheme = _build_dmipy_scheme(
        bvalues,
        directions,
        delta=small_delta,
        Delta=big_delta,
    )
    model = _build_sandi_model({"soma_diffusivity": 3e-9})
    normalized = np.array([0.5, 0.55, 0.75, 0.35, 0.65], dtype=np.float32)
    _, scales = _sandi_bounds_and_scales(model)
    diameter, d_in, d_ec, f_neurite, f_tissue = normalized * scales

    bl = bvalues * d_in
    stick = np.ones_like(bl)
    weighted = bl > 1e-7
    stick[weighted] = (
        np.sqrt(np.pi)
        * pytest.importorskip("scipy").special.erf(np.sqrt(bl[weighted]))
        / (2.0 * np.sqrt(bl[weighted]))
    )
    sphere = model.models[0].models[1](
        scheme,
        diameter=float(diameter),
        surface_relaxivity=None,
    )
    extra = model.models[1](scheme, lambda_iso=float(d_ec))
    expected = f_tissue * (
        f_neurite * stick + (1.0 - f_neurite) * sphere
    ) + (1.0 - f_tissue) * extra

    dynamic_scheme = {
        name: jnp.asarray(getattr(scheme, name), dtype=jnp.float32)
        for name in ("bvalues", "gradient_strengths", "delta", "Delta")
    }
    forward = _build_forward(model)
    observed = np.asarray(
        forward(jnp.asarray(normalized), dynamic_scheme)
    )
    jacobian = np.asarray(
        jax.jacrev(forward)(jnp.asarray(normalized), dynamic_scheme)
    )

    assert observed == pytest.approx(expected, rel=2e-5, abs=2e-6)
    assert np.all(np.isfinite(jacobian))


def test_sandi_jax_vectorized_gnl_fit_handles_padded_batch():
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("jaxopt")
    pytest.importorskip("dmipy_fit")
    from qmri_neuropipe.interfaces.dmipy import _build_dmipy_scheme

    bvalues = np.array([0.0, 1e9, 2e9])
    scheme = _build_dmipy_scheme(
        bvalues,
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        delta=np.full(3, 0.012),
        Delta=np.full(3, 0.032),
    )
    model = _build_sandi_model({"soma_diffusivity": 3e-9})
    dynamic_scheme = {
        name: jnp.asarray(getattr(scheme, name), dtype=jnp.float32)
        for name in ("bvalues", "gradient_strengths", "delta", "Delta")
    }
    signal = np.asarray(
        _build_forward(model)(
            jnp.asarray([0.5, 0.55, 0.75, 0.35, 0.65]),
            dynamic_scheme,
        )
    )
    data = np.vstack([signal * 900.0, signal * 1000.0, signal * 1100.0])
    tensors = np.repeat(np.eye(3)[None, ...], 3, axis=0)

    fitted = fit_sandi_jax(
        model,
        scheme,
        data,
        gradient_tensors=tensors,
        solver_kwargs={
            "Ns": 2,
            "maxiter": 2,
            "batch_size": 2,
        },
    )

    assert all(np.all(np.isfinite(values)) for values in fitted.values())
    assert fitted["partial_volume_0"] + fitted[
        "partial_volume_1"
    ] == pytest.approx(np.ones(3))


def test_sandi_model_configuration_can_disable_available_gnl(
    tmp_path,
    monkeypatch,
):
    from qmri_neuropipe.interfaces import dmipy

    captured = {}

    def fake_fit_sandi(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(dmipy, "fit_sandi", fake_fit_sandi)
    dwi = DWIFile(
        entities={"sub": "01", "suffix": "dwi"},
        img=tmp_path / "sub-01_dwi.nii.gz",
    )
    step = SANDIFittingStep(
        config={},
        logger=None,
        provenance=None,
        method="dmipy",
        gradient_nonlinearity=False,
        solver="mix",
    )

    step.run(
        {
            "current_image": dwi,
            "gnl_map": tmp_path / "sub-01_desc-gnl_tensor.nii.gz",
        },
        tmp_path / "out",
    )

    assert captured["grad_nonlin"] is None
    assert captured["solver"] == "mix"
