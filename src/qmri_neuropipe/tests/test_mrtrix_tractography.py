from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest


def _capture(monkeypatch):
    from qmri_neuropipe.interfaces import mrtrix

    calls = []
    monkeypatch.setattr(mrtrix, "_run_mrtrix", lambda parts, **kwargs: calls.append((parts, kwargs)))
    return mrtrix, calls


def test_tckgen_builds_act_roi_and_tracking_options(tmp_path, monkeypatch):
    mrtrix, calls = _capture(monkeypatch)
    output = tmp_path / "tracks.tck"
    mrtrix.tckgen(
        tmp_path / "fod.nii.gz", output, algorithm="iFOD2", select=1000,
        act=tmp_path / "5tt.nii.gz", seed_gmwmi=tmp_path / "gmwmi.nii.gz",
        include=[tmp_path / "include.nii.gz"], exclude=[tmp_path / "exclude.nii.gz"],
        options={"backtrack": True, "cutoff": 0.06}, nthreads=4, force=True,
    )
    cmd = calls[0][0]
    assert cmd[:3] == ["tckgen", str(tmp_path / "fod.nii.gz"), str(output)]
    assert ["-algorithm", "iFOD2"] == cmd[3:5]
    for flag in ("-act", "-seed_gmwmi", "-include", "-exclude", "-backtrack", "-cutoff"):
        assert flag in cmd


def test_sift2_and_connectome_accept_weights(tmp_path, monkeypatch):
    mrtrix, calls = _capture(monkeypatch)
    weights = mrtrix.tcksift2(
        tmp_path / "tracks.tck", tmp_path / "fod.nii.gz", tmp_path / "weights.txt"
    )
    mrtrix.tck2connectome(
        tmp_path / "tracks.tck", tmp_path / "nodes.nii.gz", tmp_path / "matrix.csv",
        weights=weights,
    )
    assert calls[0][0][0] == "tcksift2"
    assert "-tck_weights_in" in calls[1][0]
    assert "-symmetric" in calls[1][0]
    assert "-zero_diagonal" in calls[1][0]


def test_modeling_workflow_builds_act_tracking_tractseg_and_tractometry(tmp_path):
    import logging
    from qmri_neuropipe.core.config import PipelineConfig
    from qmri_neuropipe.workflows.pipelines.integrated_modeling_workflow import ModelingWorkflow

    config = PipelineConfig(
        bids_dir=tmp_path / "bids", output_dir=tmp_path / "out", n_cpus=2,
        config_data={"dmri": {"modeling": {"tractography": {
            "mrtrix": {"enabled": True, "algorithm": "iFOD2", "act": {"enabled": True}},
            "tractseg": {"enabled": True},
            "tract_specific": {"enabled": True, "bundles": ["CST_L"]},
        }}}},
    )
    workflow = ModelingWorkflow(config, logging.getLogger("tractography-test"), None)
    workflow.build_pipeline({})
    names = [step.__class__.__name__ for step in workflow.steps]
    assert names == [
        "CSDFittingStep", "MRtrixAnatomicalConstraintsStep", "MRtrixTractographyStep",
        "TractSegStep", "TractSpecificAnalysisStep",
    ]


def test_tensor_tracking_auto_enables_dti_not_csd(tmp_path):
    import logging
    from qmri_neuropipe.core.config import PipelineConfig
    from qmri_neuropipe.workflows.pipelines.integrated_modeling_workflow import ModelingWorkflow

    config = PipelineConfig(
        bids_dir=tmp_path / "bids", output_dir=tmp_path / "out",
        config_data={"dmri": {"modeling": {"tractography": {
            "mrtrix": {"enabled": True, "algorithm": "Tensor_Det"}
        }}}},
    )
    workflow = ModelingWorkflow(config, logging.getLogger("tensor-tracking-test"), None)
    workflow.build_pipeline({})
    names = [step.__class__.__name__ for step in workflow.steps]
    assert "DTIFittingStep" in names
    assert "CSDFittingStep" not in names


def test_tractography_step_requires_fod(tmp_path):
    import logging
    from qmri_neuropipe.core.config import PipelineConfig
    from qmri_neuropipe.core.exceptions import ValidationError
    from qmri_neuropipe.lib.dmri.tractography import MRtrixTractographyStep

    config = PipelineConfig(bids_dir=tmp_path, output_dir=tmp_path)
    step = MRtrixTractographyStep(config, logging.getLogger("test"), None)
    with pytest.raises(ValidationError, match="CSD FOD"):
        step.run({}, tmp_path)


def test_act_uses_selected_t1w_and_regrids_5tt_to_dwi(tmp_path, monkeypatch):
    import logging
    from qmri_neuropipe.core.config import PipelineConfig
    from qmri_neuropipe.lib.dmri.tractography import MRtrixAnatomicalConstraintsStep

    t1w = tmp_path / "sub-01_acq-MPnRAGE_desc-preproc_T1w.nii.gz"
    dwi = tmp_path / "sub-01_desc-preproc_dwi.nii.gz"
    nib.Nifti1Image(np.zeros((8, 8, 8)), np.eye(4)).to_filename(t1w)
    nib.Nifti1Image(np.zeros((4, 4, 4, 2)), np.eye(4)).to_filename(dwi)

    calls = {}

    def fake_five_tt_gen(algorithm, in_file, out_file, **kwargs):
        calls["anatomical"] = Path(in_file)
        nib.Nifti1Image(np.zeros((8, 8, 8, 5)), np.eye(4)).to_filename(out_file)

    def fake_mrtransform(in_file, out_file, **kwargs):
        calls["template"] = Path(kwargs["template"])
        nib.Nifti1Image(np.zeros((4, 4, 4, 5)), np.eye(4)).to_filename(out_file)

    monkeypatch.setattr("qmri_neuropipe.interfaces.mrtrix.five_tt_gen", fake_five_tt_gen)
    monkeypatch.setattr("qmri_neuropipe.interfaces.mrtrix.mrtransform", fake_mrtransform)
    monkeypatch.setattr("qmri_neuropipe.interfaces.mrtrix.five_tt_check", lambda *a, **k: None)
    monkeypatch.setattr("qmri_neuropipe.interfaces.mrtrix.five_tt_to_gmwmi", lambda *a, **k: None)

    config = PipelineConfig(bids_dir=tmp_path, output_dir=tmp_path)
    step = MRtrixAnatomicalConstraintsStep(config, logging.getLogger("test"), None)
    context = {
        "current_image": SimpleNamespace(img=dwi, entities={"sub": "01", "suffix": "dwi"}),
        "t1w_files": [SimpleNamespace(img=t1w)],
        "spatial_transform": {"type": "linear", "application_mode": "header"},
    }

    result = step.run(context, tmp_path / "modeling")

    assert calls == {"anatomical": t1w, "template": dwi}
    assert nib.load(result["tractography"]["act_5tt"]).shape == (4, 4, 4, 5)


def test_act_requires_alignment_before_regridding_anatomical_5tt(tmp_path, monkeypatch):
    import logging
    from qmri_neuropipe.core.config import PipelineConfig
    from qmri_neuropipe.core.exceptions import ValidationError
    from qmri_neuropipe.lib.dmri.tractography import MRtrixAnatomicalConstraintsStep

    t1w, dwi = tmp_path / "T1w.nii.gz", tmp_path / "dwi.nii.gz"
    nib.Nifti1Image(np.zeros((8, 8, 8)), np.eye(4)).to_filename(t1w)
    nib.Nifti1Image(np.zeros((4, 4, 4, 2)), np.eye(4)).to_filename(dwi)

    def fake_five_tt_gen(algorithm, in_file, out_file, **kwargs):
        nib.Nifti1Image(np.zeros((8, 8, 8, 5)), np.eye(4)).to_filename(out_file)

    monkeypatch.setattr("qmri_neuropipe.interfaces.mrtrix.five_tt_gen", fake_five_tt_gen)
    config = PipelineConfig(bids_dir=tmp_path, output_dir=tmp_path)
    step = MRtrixAnatomicalConstraintsStep(config, logging.getLogger("test"), None)
    context = {
        "current_image": SimpleNamespace(img=dwi, entities={"sub": "01", "suffix": "dwi"}),
        "t1w_files": [SimpleNamespace(img=t1w)],
    }

    with pytest.raises(ValidationError, match="no header-based anatomical-to-diffusion alignment"):
        step.run(context, tmp_path / "modeling")


def test_tractography_outputs_use_source_entities_and_sidecar(tmp_path, monkeypatch):
    import json
    import logging
    from qmri_neuropipe.core.config import PipelineConfig
    from qmri_neuropipe.lib.dmri.tractography import MRtrixTractographyStep

    fod = tmp_path / "fod.nii.gz"
    fod.touch()
    monkeypatch.setattr(
        "qmri_neuropipe.interfaces.mrtrix.tckgen",
        lambda in_file, out_file, **kwargs: Path(out_file),
    )
    config = PipelineConfig(bids_dir=tmp_path, output_dir=tmp_path)
    step = MRtrixTractographyStep(config, logging.getLogger("test"), None, select=123)
    context = {
        "current_image": SimpleNamespace(entities={"sub": "01", "ses": "02", "suffix": "dwi"}),
        "modeling_results": {"CSD": {"wmFOD": fod}},
    }
    result = step.run(context, tmp_path)
    tracks = result["tractography"]["whole_brain"]
    assert tracks.name == "sub-01_ses-02_space-dwi_desc-wholebrainiFOD2_tractography.tck"
    sidecar = tracks.with_suffix(".json")
    assert sidecar.exists()
    metadata = json.loads(sidecar.read_text())
    assert metadata["NumberOfStreamlinesRequested"] == 123
    assert metadata["TrackingAlgorithm"] == "iFOD2"
