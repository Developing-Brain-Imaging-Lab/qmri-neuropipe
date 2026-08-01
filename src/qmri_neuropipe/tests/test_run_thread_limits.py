import importlib
from types import SimpleNamespace


def test_run_cmd_sets_hard_and_library_thread_limits(monkeypatch):
    run_module = importlib.import_module("qmri_neuropipe.core.run")
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_module.subprocess, "run", fake_run)

    run_module.run_cmd("true", n_threads=6)

    environment = captured["env"]
    assert environment["OMP_NUM_THREADS"] == "6"
    assert environment["OMP_THREAD_LIMIT"] == "6"
    assert environment["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] == "6"
    assert environment["MKL_NUM_THREADS"] == "6"
    assert environment["OPENBLAS_NUM_THREADS"] == "6"
