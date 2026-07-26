import json
from types import SimpleNamespace

import numpy as np
from typer.testing import CliRunner

from qmri_neuropipe import tools
from qmri_neuropipe.interfaces import dmipy_capabilities
from qmri_neuropipe.interfaces.dmipy_backend import MODEL_REGISTRY


def test_registry_summary_covers_every_model_without_importing_dmipy():
    records = dmipy_capabilities.registry_summary()

    assert len(records) == len(MODEL_REGISTRY)
    assert [record["name"] for record in records] == sorted(MODEL_REGISTRY)
    microglia = next(record for record in records if record["name"] == "microglia")
    assert microglia["source"] == "qmri-neuropipe"
    assert microglia["acquisition_requirements"] == ["delta", "Delta"]
    assert microglia["solver_interfaces"] == ["brute2fine", "jax", "mix"]
    nexi = next(record for record in records if record["name"] == "nexi")
    assert nexi["solver_interfaces"] == ["brute2fine", "mix"]
    assert not nexi["jax_supported"]
    assert not nexi["jax_gnl_supported"]


def test_representative_parameters_use_physical_midpoints_and_equal_fractions():
    model = SimpleNamespace(
        parameter_names=["lambda", "mu", "partial_volume_0", "partial_volume_1"],
        parameter_cardinality={
            "lambda": 1,
            "mu": 2,
            "partial_volume_0": 1,
            "partial_volume_1": 1,
        },
        parameter_types={
            "lambda": "normal",
            "mu": "orientation",
            "partial_volume_0": "normal",
            "partial_volume_1": "normal",
        },
        parameter_ranges={
            "lambda": (1.0, 3.0),
            "mu": ((0.0, np.pi), (-np.pi, np.pi)),
            "partial_volume_0": (0.01, 0.99),
            "partial_volume_1": (0.01, 0.99),
        },
        parameter_scales={
            "lambda": 1e-9,
            "mu": 1.0,
            "partial_volume_0": 1.0,
            "partial_volume_1": 1.0,
        },
        x0_parameters={},
    )

    parameters = dmipy_capabilities.representative_parameters(model)

    assert parameters["lambda"] == 2e-9
    np.testing.assert_allclose(parameters["mu"], [np.pi / 2, 0.0])
    assert parameters["partial_volume_0"] == 0.5
    assert parameters["partial_volume_1"] == 0.5


def test_dmipy_models_cli_emits_machine_readable_registry():
    result = CliRunner().invoke(
        tools.app,
        ["dmipy-models", "--format", "json"],
    )

    assert result.exit_code == 0
    records = json.loads(result.stdout)
    assert {record["name"] for record in records} == set(MODEL_REGISTRY)
    assert all("solver_interfaces" in record for record in records)


def test_dmipy_models_cli_merges_probe_results(monkeypatch):
    monkeypatch.setattr(
        dmipy_capabilities,
        "probe_registry",
        lambda: [
            {
                "name": name,
                "construction": "passed",
                "simulation": "passed",
                "error": None,
            }
            for name in sorted(MODEL_REGISTRY)
        ],
    )

    result = CliRunner().invoke(
        tools.app,
        ["dmipy-models", "--format", "json", "--probe"],
    )

    assert result.exit_code == 0
    records = json.loads(result.stdout)
    assert all(record["probe"]["simulation"] == "passed" for record in records)


def test_dmipy_model_info_cli_reports_resolved_contract(monkeypatch):
    monkeypatch.setattr(
        dmipy_capabilities,
        "model_details",
        lambda name: {
            **dmipy_capabilities.model_summary(name),
            "model_class": "FakeModel",
            "parameter_count": 1,
            "parameters": [
                {
                    "name": "lambda",
                    "cardinality": 1,
                    "type": "normal",
                    "optimized": True,
                    "physical_range": [1e-9, 3e-9],
                    "output_alias": None,
                }
            ],
        },
    )

    result = CliRunner().invoke(
        tools.app,
        ["dmipy-model-info", "--model", "ball", "--format", "json"],
    )

    assert result.exit_code == 0
    details = json.loads(result.stdout)
    assert details["name"] == "ball"
    assert details["model_class"] == "FakeModel"
    assert details["parameters"][0]["name"] == "lambda"
