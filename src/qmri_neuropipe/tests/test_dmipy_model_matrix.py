import numpy as np
import pytest

pytest.importorskip("dmipy_fit")

from qmri_neuropipe.interfaces.dmipy_backend import MODEL_REGISTRY
from qmri_neuropipe.interfaces.dmipy_capabilities import (
    RECOVERY_CASES,
    parameter_recovery_probe,
    probe_model,
)


@pytest.mark.parametrize("model_name", sorted(MODEL_REGISTRY))
def test_registered_model_constructs_and_simulates(model_name):
    if model_name == "axcaliber" and not hasattr(np, "trapezoid"):
        pytest.skip("dmipy-fit AxCaliber requires the project minimum NumPy 2.x.")

    result = probe_model(model_name)

    assert result["construction"] == "passed", result["error"]
    assert result["simulation"] == "passed", result["error"]
    assert 0 <= result["signal_minimum"]
    assert result["signal_maximum"] <= 1.0 + 1e-6


@pytest.mark.parametrize("case", RECOVERY_CASES, ids=lambda case: case.model_name)
def test_native_optimizer_recovers_identifiable_synthetic_parameter(case):
    result = parameter_recovery_probe(case, solver="brute2fine")

    assert result["passed"], result
    assert len(result["estimates"]) == 2
