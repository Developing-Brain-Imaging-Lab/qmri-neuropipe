import multiprocessing
import os
import sys
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
    with pytest.raises(Exception, match=r"supports dmipy-fit>=2.3,<2.4"):
        dmipy_backend.DmipyRuntime.resolve()


def test_native_runtime_does_not_import_jax(monkeypatch):
    monkeypatch.setattr(dmipy_backend, "version", lambda _: "2.3.0")

    def fail_import(name):
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(dmipy_backend, "import_module", fail_import)
    runtime = dmipy_backend.DmipyRuntime.resolve("mix", "cpu")
    assert runtime.backend == "native-cpu"
    assert runtime.provenance()["FittingSoftwareVersion"] == "2.3.0"


def test_jax_gpu_requirement_is_explicit(monkeypatch):
    monkeypatch.setattr(dmipy_backend, "version", lambda _: "2.3.0")
    fake_device = SimpleNamespace(platform="cpu")
    fake_jax = SimpleNamespace(devices=lambda: [fake_device])
    monkeypatch.setattr(dmipy_backend, "import_module", lambda _: fake_jax)

    with pytest.raises(Exception, match="no usable GPU"):
        dmipy_backend.DmipyRuntime.resolve("jax", "gpu")


def test_jax_runtime_configures_device_cache_and_compile_logging(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(dmipy_backend, "version", lambda _: "2.3.0")
    monkeypatch.delitem(sys.modules, "jax", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("JAX_CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("QMRI_DMIPY_GPU_SELECTOR", raising=False)
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("JAX_LOG_COMPILES", raising=False)
    fake_device = SimpleNamespace(platform="gpu")
    fake_jax = SimpleNamespace(devices=lambda: [fake_device])
    monkeypatch.setattr(
        dmipy_backend,
        "import_module",
        lambda name: fake_jax if name == "jax" else SimpleNamespace(),
    )

    runtime = dmipy_backend.DmipyRuntime.resolve(
        "jax",
        "gpu",
        gpu_device=2,
        jax_cache_dir=tmp_path / "jax-cache",
        jax_log_compiles=True,
    )

    assert runtime.backend == "gpu"
    assert runtime.gpu_device == 2
    assert runtime.jax_cache_dir == str((tmp_path / "jax-cache").resolve())
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    assert os.environ["JAX_CUDA_VISIBLE_DEVICES"] == "0"
    assert os.environ["QMRI_DMIPY_GPU_SELECTOR"] == "2"
    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == runtime.jax_cache_dir
    assert os.environ["JAX_LOG_COMPILES"] == "1"


def test_gpu_selector_respects_scheduler_visible_device_order(monkeypatch):
    monkeypatch.setattr(dmipy_backend, "version", lambda _: "2.3.0")
    monkeypatch.delitem(sys.modules, "jax", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    monkeypatch.delenv("QMRI_DMIPY_GPU_SELECTOR", raising=False)
    fake_device = SimpleNamespace(platform="gpu")
    fake_jax = SimpleNamespace(devices=lambda: [fake_device])
    monkeypatch.setattr(
        dmipy_backend,
        "import_module",
        lambda name: fake_jax if name == "jax" else SimpleNamespace(),
    )

    dmipy_backend.DmipyRuntime.resolve(
        "jax",
        "gpu",
        gpu_device=1,
    )

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"
    assert os.environ["JAX_CUDA_VISIBLE_DEVICES"] == "0"


def test_jax_run_summary_distinguishes_worker_input_from_optimizer_batches():
    runtime = dmipy_backend.DmipyRuntime(
        version="2.1.0",
        solver="jax",
        requested_device="gpu",
        backend="gpu",
        devices=("CudaDevice(id=0)",),
        gpu_device=0,
        jax_cache_dir="/private/cache",
    )

    summary = dmipy_backend.jax_run_summary(
        runtime,
        188_941,
        {"batch_size": 4_000},
    )

    assert "JAX execution backend: gpu" in summary
    assert any("48 expected batches" in line for line in summary)
    assert any("/private/cache" in line for line in summary)


def test_jax_fit_output_is_visible_but_native_output_is_quiet(capsys):
    with dmipy_backend.dmipy_fit_output("jax"):
        print("visible JAX setup")
    with dmipy_backend.dmipy_fit_output("brute2fine"):
        print("hidden native setup")

    captured = capsys.readouterr()
    assert "visible JAX setup" in captured.out
    assert "hidden native setup" not in captured.out


def test_dmipy_jax_postprocessing_workaround_uses_equivalent_numpy_fractions(
    monkeypatch,
):
    class FakeOptimizer:
        def _unnest_model(self, x_model_nested):
            nested_to_normalized_fractions_jax = None
            return nested_to_normalized_fractions_jax(x_model_nested)

    monkeypatch.setattr(
        dmipy_backend,
        "import_module",
        lambda _: SimpleNamespace(JaxOptimizer=FakeOptimizer),
    )

    assert dmipy_backend.install_dmipy_jax_postprocessing_workaround()
    optimizer = FakeOptimizer()
    optimizer._is_multi = True
    optimizer._scales = np.ones(4)
    optimizer._N_models = 3

    converted = optimizer._unnest_model(
        np.array([0.25, 0.6, 0.5], dtype=np.float32)
    )

    assert converted == pytest.approx([0.25, 0.6, 0.2, 0.2])
    assert not dmipy_backend.install_dmipy_jax_postprocessing_workaround()


def test_pool_collection_emits_heartbeat(capsys):
    class FakeResult:
        def __init__(self):
            self.calls = 0

        def get(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise multiprocessing.TimeoutError
            return {"result": 1}

    class FakePool:
        def apply_async(self, worker, args):
            return FakeResult()

    results = dmipy_backend.collect_pool_results_with_heartbeat(
        FakePool(),
        lambda value: value,
        [("chunk",)],
        heartbeat_interval=0.01,
        label="Test JAX fit",
    )

    assert results == [{"result": 1}]
    assert "Test JAX fit is still running" in capsys.readouterr().out


def test_fit_model_translates_cpu_parallelism(monkeypatch):
    monkeypatch.setattr(dmipy_backend, "version", lambda _: "2.3.0")

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
    monkeypatch.setattr(dmipy_backend, "version", lambda _: "2.3.0")

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


def test_shared_execution_routes_native_fit_and_records_request(monkeypatch):
    runtime = dmipy_backend.DmipyRuntime(
        version="2.1.0",
        solver="brute2fine",
        requested_device="auto",
        backend="native-cpu",
    )
    captured = {}

    def fake_fit(model, scheme, data, **kwargs):
        captured.update(
            model=model,
            scheme=scheme,
            shape=data.shape,
            kwargs=kwargs,
        )
        return "fitted", runtime

    monkeypatch.setattr(dmipy_backend, "fit_model", fake_fit)
    execution = dmipy_backend.execute_dmipy_fit(
        dmipy_backend.DmipyFitRequest(
            model_name="ball",
            model="model",
            acquisition_scheme="scheme",
            data=np.ones((2, 3)),
            runtime=runtime,
            nthreads=2,
            solver_options={"Ns": 4},
            heartbeat_interval=None,
        )
    )

    assert execution.fitted == "fitted"
    assert execution.voxel_count == 2
    assert not execution.used_gradient_nonlinearity
    assert captured["kwargs"]["nthreads"] == 2
    assert captured["kwargs"]["solver_kwargs"] == {"Ns": 4}


def test_shared_execution_routes_jax_gradient_tensors(monkeypatch):
    from qmri_neuropipe.interfaces import dmipy_jax_gnl

    runtime = dmipy_backend.DmipyRuntime(
        version="2.1.0",
        solver="jax",
        requested_device="gpu",
        backend="gpu",
    )
    captured = {}

    def fake_gnl(model, scheme, data, tensors, **kwargs):
        captured.update(
            model=model,
            scheme=scheme,
            shape=data.shape,
            tensor_shape=tensors.shape,
            kwargs=kwargs,
        )
        return "gnl-fitted"

    monkeypatch.setattr(dmipy_jax_gnl, "fit_model_jax_gnl", fake_gnl)
    execution = dmipy_backend.execute_dmipy_fit(
        dmipy_backend.DmipyFitRequest(
            model_name="noddi",
            model="model",
            acquisition_scheme="scheme",
            data=np.ones((2, 3)),
            gradient_tensors=np.tile(np.eye(3), (2, 1, 1)),
            runtime=runtime,
            heartbeat_interval=None,
        )
    )

    assert execution.fitted == "gnl-fitted"
    assert execution.used_gradient_nonlinearity
    assert captured["tensor_shape"] == (2, 3, 3)


def test_shared_execution_rejects_unvalidated_nexi_jax():
    runtime = dmipy_backend.DmipyRuntime(
        version="2.1.0",
        solver="jax",
        requested_device="gpu",
        backend="gpu",
    )

    with pytest.raises(ValueError, match="does not have a validated JAX"):
        dmipy_backend.execute_dmipy_fit(
            dmipy_backend.DmipyFitRequest(
                model_name="nexi",
                model="model",
                acquisition_scheme="scheme",
                data=np.ones((1, 3)),
                runtime=runtime,
                heartbeat_interval=None,
            )
        )


def test_released_reference_factories_match_registry():
    pytest.importorskip("dmipy_fit")
    from dmipy_fit.custom_optimizers import reference_models

    missing = [
        spec.factory_name
        for spec in dmipy_backend.MODEL_REGISTRY.values()
        if spec.factory_module
        == "dmipy_fit.custom_optimizers.reference_models"
        if not hasattr(reference_models, spec.factory_name)
    ]
    assert missing == []


def test_project_model_factories_match_registry():
    project_specs = [
        spec
        for spec in dmipy_backend.MODEL_REGISTRY.values()
        if spec.factory_module
        != "dmipy_fit.custom_optimizers.reference_models"
    ]

    assert {spec.name for spec in project_specs} == {"microglia"}
    for spec in project_specs:
        module = dmipy_backend.import_module(spec.factory_module)
        assert callable(getattr(module, spec.factory_name))
        assert callable(getattr(module, spec.output_adapter_name))


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


def test_released_mix_solver_recovers_synthetic_ball_diffusivity():
    pytest.importorskip("dmipy_fit")

    rng = np.random.default_rng(14)
    bvalues = np.r_[0.0, np.full(6, 1e9), np.full(6, 2e9)]
    directions = rng.normal(size=(12, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    bvecs = np.vstack([np.zeros((1, 3)), directions])
    scheme = dmipy_backend.acquisition_scheme_from_bvalues(bvalues, bvecs)
    expected = 1.2e-9
    signal = dmipy_backend.build_reference_model("ball").simulate_signal(
        scheme,
        {"G1Ball_1_lambda_iso": expected},
    )

    fitted, runtime = dmipy_backend.fit_model(
        dmipy_backend.build_reference_model("ball"),
        scheme,
        signal[None, :],
        solver="mix",
        solver_kwargs={"maxiter": 100},
    )

    recovered = fitted.fitted_parameters["G1Ball_1_lambda_iso"][0]
    assert runtime.solver == "mix"
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
