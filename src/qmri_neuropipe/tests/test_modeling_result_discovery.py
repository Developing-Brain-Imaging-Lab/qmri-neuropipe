from pathlib import Path
from types import SimpleNamespace

from qmri_neuropipe.lib.dmri.fitting import (
    MAPMRIFittingStep,
    NODDIFittingStep,
    collect_model_derivatives,
)
from qmri_neuropipe.workflows.pipelines.integrated_modeling_workflow import ModelingWorkflow


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"nifti")
    return path


def test_collect_model_derivatives_preserves_compound_metric_names(tmp_path):
    model_dir = tmp_path / "mapmri"
    ng_par = _touch(model_dir / "sub-01_model-MAPMRI_NG_PAR.nii.gz")
    ng_perp = _touch(model_dir / "sub-01_model-MAPMRI_NG_PERP.nii.gz")

    assert collect_model_derivatives(model_dir, "MAPMRI") == {
        "NG_PAR": ng_par,
        "NG_PERP": ng_perp,
    }


def test_cached_results_include_every_noddi_and_mapmri_derivative(tmp_path):
    noddi_dir = tmp_path / "NODDI"
    expected_noddi = {
        metric: _touch(noddi_dir / f"sub-01_model-NODDI_{metric}.nii.gz")
        for metric in ("ODI", "ICVF", "EXVF", "FISO")
    }
    mapmri_dir = tmp_path / "mapmri"
    expected_mapmri = {
        metric: _touch(mapmri_dir / f"sub-01_model-MAPMRI_{metric}.nii.gz")
        for metric in (
            "RTOP", "RTAP", "RTPP", "QIV", "MSD", "NG",
            "NG_PAR", "NG_PERP", "PEAKS",
        )
    }
    sandi_dir = tmp_path / "sandi"
    expected_sandi = {
        metric: _touch(sandi_dir / f"sub-01_model-SANDI_{metric}.nii.gz")
        for metric in ("fsoma", "fneurite", "fextra", "Rsoma", "d_in", "d_ec")
    }
    dmipy_dir = tmp_path / "dmipy-ball"
    dmipy_md = _touch(dmipy_dir / "sub-01_model-Ball_MD.nii.gz")

    context = {}
    workflow = object.__new__(ModelingWorkflow)
    dwi = SimpleNamespace(entities={"sub": "01"})
    workflow._populate_modeling_results_from_cache([dwi], tmp_path, context)

    assert context["modeling_results"]["NODDI"] == expected_noddi
    assert context["modeling_results"]["MAPMRI"] == expected_mapmri
    assert context["modeling_results"]["SANDI"] == expected_sandi
    assert context["modeling_results"]["dmipy:ball"] == {"MD": dmipy_md}


def test_pipeline_cache_requires_all_outputs_for_each_enabled_model(tmp_path):
    workflow = object.__new__(ModelingWorkflow)
    noddi = object.__new__(NODDIFittingStep)
    noddi.kwargs = {}
    mapmri = object.__new__(MAPMRIFittingStep)
    mapmri.kwargs = {"metrics": ["rtop", "ng_parallel"]}
    workflow.steps = [noddi, mapmri]
    dwi = SimpleNamespace(entities={"sub": "01"})

    for metric in ("ODI", "ICVF", "FISO"):
        _touch(tmp_path / "NODDI" / f"sub-01_model-NODDI_{metric}.nii.gz")
    _touch(tmp_path / "mapmri" / "sub-01_model-MAPMRI_RTOP.nii.gz")

    assert not workflow._check_all_outputs_exist([dwi], tmp_path)

    _touch(tmp_path / "mapmri" / "sub-01_model-MAPMRI_NG_PAR.nii.gz")
    assert workflow._check_all_outputs_exist([dwi], tmp_path)
