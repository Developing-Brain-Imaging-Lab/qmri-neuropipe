import numpy as np
import pytest

pytest.importorskip("dmipy_fit")

from qmri_neuropipe.interfaces.dmipy_backend import MODEL_REGISTRY
from qmri_neuropipe.interfaces.dmipy_capabilities import probe_model


@pytest.mark.parametrize("model_name", sorted(MODEL_REGISTRY))
def test_registered_model_constructs_and_simulates(model_name):
    if model_name == "axcaliber" and not hasattr(np, "trapezoid"):
        pytest.skip("dmipy-fit AxCaliber requires the project minimum NumPy 2.x.")

    result = probe_model(model_name)

    assert result["construction"] == "passed", result["error"]
    assert result["simulation"] == "passed", result["error"]
    assert 0 <= result["signal_minimum"]
    assert result["signal_maximum"] <= 1.0 + 1e-6
