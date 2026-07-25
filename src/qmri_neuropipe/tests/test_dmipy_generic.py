import json
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
from typer.testing import CliRunner

from qmri_neuropipe import tools
from qmri_neuropipe.interfaces import dmipy_generic
from qmri_neuropipe.interfaces.dmipy_backend import (
    DmipyFitExecution,
    DmipyRuntime,
    MODEL_REGISTRY,
)


def _write_dwi_inputs(tmp_path):
    dwi = tmp_path / "sub-01_desc-preproc_dwi.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2, 3)), np.eye(4)), dwi)
    bval = tmp_path / "sub-01_dwi.bval"
    bvec = tmp_path / "sub-01_dwi.bvec"
    bval.write_text("0 1000 2000\n")
    bvec.write_text("0 1 0\n0 0 1\n0 0 0\n")
    return dwi, bval, bvec


def test_timing_dependent_registry_models_require_separate_delta_files():
    for spec in MODEL_REGISTRY.values():
        requirements = set(spec.acquisition_requirements)
        assert ("delta" in requirements) == ("Delta" in requirements)
        assert "diffusion_time" not in requirements
        assert "composite_waveform" not in requirements

    assert MODEL_REGISTRY["nexi"].acquisition_requirements == ("delta", "Delta")
    assert MODEL_REGISTRY["mte_impulsed"].acquisition_requirements == (
        "delta",
        "Delta",
        "TE",
    )


def test_nexi_rejects_missing_big_delta(tmp_path):
    delta = tmp_path / "small_delta.txt"
    delta.write_text("0.012\n")

    with pytest.raises(ValueError, match="--big-delta"):
        dmipy_generic._load_acquisition_values(
            "nexi",
            3,
            delta_file=delta,
            Delta_file=None,
            TE_file=None,
        )


def test_timing_files_broadcast_scalars_and_enforce_order(tmp_path):
    delta = tmp_path / "small_delta.txt"
    big_delta = tmp_path / "big_delta.txt"
    delta.write_text("0.012\n")
    big_delta.write_text("0.032\n")

    values, metadata = dmipy_generic._load_acquisition_values(
        "nexi",
        3,
        delta_file=delta,
        Delta_file=big_delta,
        TE_file=None,
    )

    assert values["delta"] == pytest.approx([0.012] * 3)
    assert values["Delta"] == pytest.approx([0.032] * 3)
    assert metadata["deltaFile"] == "small_delta.txt"
    assert metadata["DeltaSeconds"]["Maximum"] == pytest.approx(0.032)

    big_delta.write_text("0.010\n")
    with pytest.raises(ValueError, match="greater than small-delta"):
        dmipy_generic._load_acquisition_values(
            "nexi",
            3,
            delta_file=delta,
            Delta_file=big_delta,
            TE_file=None,
        )


def test_generic_nexi_fit_passes_separate_timings_and_writes_bids_outputs(
    tmp_path, monkeypatch
):
    dwi, bval, bvec = _write_dwi_inputs(tmp_path)
    delta = tmp_path / "small_delta.txt"
    big_delta = tmp_path / "big_delta.txt"
    delta.write_text("0.010 0.012 0.014\n")
    big_delta.write_text("0.030 0.032 0.034\n")
    captured = {}

    def fake_scheme(bvalues, directions, **kwargs):
        captured["bvalues"] = bvalues
        captured["directions"] = directions
        captured["scheme_kwargs"] = kwargs
        return "scheme"

    runtime = DmipyRuntime(
        version="2.1.0",
        solver="brute2fine",
        requested_device="auto",
        backend="native-cpu",
    )

    def fake_execute(request):
        captured["fit"] = request
        result = SimpleNamespace(
            fitted_parameters={
                "X0GeneralizedKarger_1_kappa": np.ones(
                    request.data.shape[:3]
                )
            }
        )
        return DmipyFitExecution(
            fitted=result,
            runtime=runtime,
            model_name=request.model_name,
            voxel_count=8,
            used_gradient_nonlinearity=False,
        )

    monkeypatch.setattr(dmipy_generic, "acquisition_scheme_from_bvalues", fake_scheme)
    monkeypatch.setattr(dmipy_generic, "build_reference_model", lambda name: name)
    monkeypatch.setattr(dmipy_generic, "execute_dmipy_fit", fake_execute)
    monkeypatch.setattr(
        dmipy_generic.DmipyRuntime,
        "resolve",
        classmethod(lambda cls, *args, **kwargs: runtime),
    )

    outputs = dmipy_generic.fit_dmipy_reference(
        dwi,
        tmp_path / "out",
        model_name="nexi",
        bval_file=bval,
        bvec_file=bvec,
        delta_file=delta,
        Delta_file=big_delta,
        solver_kwargs={"Ns": 3},
    )

    assert captured["bvalues"] == pytest.approx([0, 1e9, 2e9])
    assert captured["scheme_kwargs"]["delta"] == pytest.approx(
        [0.010, 0.012, 0.014]
    )
    assert captured["scheme_kwargs"]["Delta"] == pytest.approx(
        [0.030, 0.032, 0.034]
    )
    assert captured["scheme_kwargs"]["TE"] is None
    assert captured["fit"].model_name == "nexi"
    assert captured["fit"].solver_options == {"Ns": 3}
    parameter = "X0GeneralizedKarger_1_kappa"
    output = outputs[parameter]
    assert output.name == (
        "sub-01_model-NEXI_"
        "desc-X0GeneralizedKarger1Kappa_parameter.nii.gz"
    )
    sidecar = output.with_name(output.name[:-7] + ".json")
    metadata = json.loads(sidecar.read_text())
    assert metadata["Parameter"] == parameter
    assert metadata["AcquisitionRequirements"] == ["delta", "Delta"]
    assert metadata["deltaFile"] == "small_delta.txt"
    assert metadata["DeltaFile"] == "big_delta.txt"


def test_fit_dmipy_cli_exposes_independent_timing_options():
    result = CliRunner().invoke(tools.app, ["fit-dmipy", "--help"])

    assert result.exit_code == 0
    assert "--delta" in result.stdout
    assert "--big-delta" in result.stdout
    assert "--te" in result.stdout
    assert "--grad-nonlin" in result.stdout
    assert "--gpu-device" in result.stdout
    assert "--heartbeat-interval" in result.stdout


def test_microglia_registry_adapter_adds_paper_outputs():
    fitted = SimpleNamespace(
        fitted_parameters={
            "partial_volume_0": np.array([0.6]),
            "partial_volume_3": np.array([0.1]),
            "SD1WatsonDistributed_1_partial_volume_0": np.array([0.25]),
            "SD1WatsonDistributed_1_SD1Watson_1_odi": np.array([0.5]),
            "S2SphereStejskalTannerApproximation_1_diameter": np.array([8e-6]),
            "S2SphereStejskalTannerApproximation_2_diameter": np.array([16e-6]),
        }
    )

    maps = dmipy_generic.model_output_maps("microglia", fitted)

    assert MODEL_REGISTRY["microglia"].references == (
        "https://doi.org/10.1126/sciadv.abq2923",
    )
    assert maps["derived_f_stick"] == pytest.approx([0.15])
    assert maps["derived_f_extracellular"] == pytest.approx([0.45])
    assert maps["derived_f_tissue"] == pytest.approx([0.9])
    assert maps["derived_small_sphere_radius"] == pytest.approx([4e-6])
    assert maps["derived_large_sphere_radius"] == pytest.approx([8e-6])
