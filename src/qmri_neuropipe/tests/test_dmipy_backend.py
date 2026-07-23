from types import SimpleNamespace

import numpy as np
import pytest

from qmri_neuropipe.interfaces import dmipy_backend


def test_model_registry_exposes_released_model_families():
    expected = {
        "noddi",
        "bingham_noddi",
        "noddida",
        "mcsmt",
        "charmed",
        "axcaliber",
        "active_ax",
        "verdict",
        "sandi",
        "impulsed",
        "nexi",
        "fexi",
        "sandix",
        "mte_noddi",
        "mte_sandi",
        "wmti",
    }
    assert expected <= set(dmipy_backend.MODEL_REGISTRY)


def test_model_registry_rejects_arbitrary_factory_names():
    with pytest.raises(ValueError, match="Unknown dmipy model"):
        dmipy_backend.get_model_spec("__import__('os').system('false')")


def test_runtime_rejects_incompatible_version(monkeypatch):
    monkeypatch.setattr(dmipy_backend, "version", lambda _: "2.2.0")
    with pytest.raises(Exception, match=r"supports dmipy-fit>=2.1,<2.2"):
        dmipy_backend.DmipyRuntime.resolve()


def test_native_runtime_does_not_import_jax(monkeypatch):
    monkeypatch.setattr(dmipy_backend, "version", lambda _: "2.1.0")

    def fail_import(name):
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(dmipy_backend, "import_module", fail_import)
    runtime = dmipy_backend.DmipyRuntime.resolve("mix", "cpu")
    assert runtime.backend == "native-cpu"
    assert runtime.provenance()["FittingSoftwareVersion"] == "2.1.0"


def test_jax_gpu_requirement_is_explicit(monkeypatch):
    monkeypatch.setattr(dmipy_backend, "version", lambda _: "2.1.0")
    fake_device = SimpleNamespace(platform="cpu")
    fake_jax = SimpleNamespace(devices=lambda: [fake_device])
    monkeypatch.setattr(dmipy_backend, "import_module", lambda _: fake_jax)

    with pytest.raises(Exception, match="no usable GPU"):
        dmipy_backend.DmipyRuntime.resolve("jax", "gpu", require_device=True)


def test_fit_model_translates_cpu_parallelism(monkeypatch):
    monkeypatch.setattr(dmipy_backend, "version", lambda _: "2.1.0")

    class FakeModel:
        def fit(
            self,
            scheme,
            data,
            mask=None,
            solver="brute2fine",
            use_parallel_processing=False,
            number_of_processors=None,
            Ns=5,
        ):
            return {
                "scheme": scheme,
                "shape": data.shape,
                "mask": mask,
                "solver": solver,
                "parallel": use_parallel_processing,
                "workers": number_of_processors,
                "Ns": Ns,
            }

    result, runtime = dmipy_backend.fit_model(
        FakeModel(),
        "scheme",
        np.ones((3, 4)),
        mask=np.ones(3, dtype=bool),
        solver="brute2fine",
        nthreads=3,
        solver_kwargs={"Ns": 7},
    )
    assert runtime.backend == "native-cpu"
    assert result["parallel"] is True
    assert result["workers"] == 3
    assert result["Ns"] == 7


def test_fit_model_rejects_unknown_solver_options(monkeypatch):
    monkeypatch.setattr(dmipy_backend, "version", lambda _: "2.1.0")

    class FakeModel:
        def fit(self, scheme, data, solver="brute2fine"):
            return None

    with pytest.raises(ValueError, match="Unsupported dmipy-fit solver option"):
        dmipy_backend.fit_model(
            FakeModel(),
            "scheme",
            np.ones((1, 2)),
            solver_kwargs={"not_an_option": True},
        )


def test_released_reference_factories_match_registry():
    pytest.importorskip("dmipy_fit")
    from dmipy_fit.custom_optimizers import reference_models

    missing = [
        spec.factory_name
        for spec in dmipy_backend.MODEL_REGISTRY.values()
        if not hasattr(reference_models, spec.factory_name)
    ]
    assert missing == []


def test_released_cpu_solver_recovers_synthetic_ball_diffusivity():
    pytest.importorskip("dmipy_fit")
    from dmipy_fit.custom_optimizers import reference_models

    rng = np.random.default_rng(4)
    bvalues = np.r_[0.0, np.full(6, 1e9), np.full(6, 2e9)]
    directions = rng.normal(size=(12, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    bvecs = np.vstack([np.zeros((1, 3)), directions])
    scheme = dmipy_backend.acquisition_scheme_from_bvalues(bvalues, bvecs)

    expected = 1.2e-9
    simulation_model = reference_models.ball()
    signal = simulation_model.simulate_signal(
        scheme, {"G1Ball_1_lambda_iso": expected}
    )
    fitted, runtime = dmipy_backend.fit_model(
        reference_models.ball(),
        scheme,
        signal[None, :],
        solver="brute2fine",
        solver_kwargs={"Ns": 5},
    )

    recovered = fitted.fitted_parameters["G1Ball_1_lambda_iso"][0]
    assert runtime.backend == "native-cpu"
    assert recovered == pytest.approx(expected, rel=1e-3)


def test_custom_noddi_keeps_reference_model_parameter_contract():
    pytest.importorskip("dmipy_fit")
    from dmipy_fit.custom_optimizers.reference_models import noddi
    from qmri_neuropipe.interfaces.dmipy import _build_noddi_model

    custom = _build_noddi_model(
        {
            "parallel_diffusivity": 1.7e-9,
            "iso_diffusivity": 3.0e-9,
            "distribution": "Watson",
            "model_type": "standard",
        }
    )
    reference = noddi()
    assert custom.parameter_names == reference.parameter_names
    assert custom.parameter_cardinality == reference.parameter_cardinality


def test_released_jax_cpu_solver_matches_native_synthetic_result():
    pytest.importorskip("dmipy_fit")
    pytest.importorskip("jaxopt")

    rng = np.random.default_rng(5)
    bvalues = np.r_[0.0, np.full(4, 1e9), np.full(4, 2e9)]
    directions = rng.normal(size=(8, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    bvecs = np.vstack([np.zeros((1, 3)), directions])
    scheme = dmipy_backend.acquisition_scheme_from_bvalues(bvalues, bvecs)
    expected = 1.2e-9
    signal = dmipy_backend.build_reference_model("ball").simulate_signal(
        scheme, {"G1Ball_1_lambda_iso": expected}
    )

    with pytest.warns(RuntimeWarning, match="running JAX on CPU"):
        fitted, runtime = dmipy_backend.fit_model(
            dmipy_backend.build_reference_model("ball"),
            scheme,
            signal[None, :],
            solver="jax",
            device="cpu",
            solver_kwargs={
                "Ns": 3,
                "N_sphere_samples": 6,
                "maxiter": 50,
                "batch_size": 1,
            },
        )

    recovered = fitted.fitted_parameters["G1Ball_1_lambda_iso"][0]
    assert runtime.backend == "cpu"
    assert recovered == pytest.approx(expected, rel=1e-3)
