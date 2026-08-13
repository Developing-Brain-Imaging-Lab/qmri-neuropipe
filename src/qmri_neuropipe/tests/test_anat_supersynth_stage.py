import logging
from pathlib import Path

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.core.types import ImageFile
from qmri_neuropipe.lib.anat.super_synth import SuperSynthStep
from qmri_neuropipe.workflows.pipelines.anat import AnatPreprocessingWorkflow


def _workflow(tmp_path: Path) -> AnatPreprocessingWorkflow:
    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "derivatives",
        skip_existing=False,
        config_data={"anat": {"super_synth": {"enabled": True}}},
    )
    return AnatPreprocessingWorkflow(config, logging.getLogger(__name__), None)


def _t1w(tmp_path: Path) -> ImageFile:
    path = tmp_path / "sub-01_T1w.nii.gz"
    path.touch()
    return ImageFile(
        entities={"sub": "01", "suffix": "T1w"},
        img=path,
    )


def test_supersynth_is_not_run_as_an_image_preprocessing_step(
    tmp_path: Path,
    monkeypatch,
):
    workflow = _workflow(tmp_path)
    super_synth = next(
        step for step in workflow.steps if isinstance(step, SuperSynthStep)
    )
    calls = []
    monkeypatch.setattr(
        super_synth,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    t1w = _t1w(tmp_path)

    context, _ = workflow._preprocess_t1w(
        tmp_path / "work",
        {"subject": "01", "t1w_files": [t1w]},
        None,
        None,
        tmp_path / "figures",
        [],
    )

    assert calls == []
    assert context["preprocessed_t1w"] is t1w


def test_dedicated_supersynth_stage_merges_returned_context(
    tmp_path: Path,
    monkeypatch,
):
    workflow = _workflow(tmp_path)
    super_synth = next(
        step for step in workflow.steps if isinstance(step, SuperSynthStep)
    )
    t1w = _t1w(tmp_path)
    volumes_csv = tmp_path / "sub-01_desc-supersynth_volumes.csv"
    volumes_csv.touch()

    def fake_run(context, **kwargs):
        result = dict(context)
        result["super_synth_dir"] = tmp_path / "super_synth"
        result["super_synth_volumes"] = {"Left-Caudate": 1234.5}
        result["super_synth_volumes_csv"] = volumes_csv
        return result

    monkeypatch.setattr(super_synth, "run", fake_run)
    context = {"subject": "01", "preprocessed_t1w": t1w}

    result, _ = workflow._run_super_synth(
        tmp_path / "work",
        context,
        None,
        None,
        [],
    )

    assert result is context
    assert result["super_synth_volumes"] == {"Left-Caudate": 1234.5}
    assert result["super_synth_volumes_csv"] == volumes_csv
