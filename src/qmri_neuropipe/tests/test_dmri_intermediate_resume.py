import logging
from pathlib import Path

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.workflows.pipelines.integrated_preprocessing_workflow import (
    PreprocessingWorkflow,
)


def test_recover_intermediates_restores_all_saved_steps_except_deleted_step(tmp_path: Path):
    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "derivatives",
        config_data={"save_intermediates": True},
    )
    workflow = PreprocessingWorkflow(config, logging.getLogger(__name__), None)
    work_dir = tmp_path / "work" / "sub-01" / "ses-01" / "dwi"
    output_dir = tmp_path / "derivatives" / "sub-01" / "ses-01" / "dwi"
    intermediate = output_dir / "intermediate"

    for step_name in ("denoising", "gibbs", "eddy", "bias"):
        step_dir = intermediate / step_name
        step_dir.mkdir(parents=True)
        (step_dir / f"{step_name}.nii.gz").write_bytes(b"valid-cache-placeholder")
    (intermediate / "resume-manifest.json").write_text("{}")
    # Simulate explicitly removing coregistration so it must run again.
    assert not (intermediate / "registration").exists()

    workflow.recover_intermediates(work_dir, output_dir)

    for step_name in ("denoising", "gibbs", "eddy", "bias"):
        assert (work_dir / step_name / f"{step_name}.nii.gz").exists()
    assert (work_dir / "resume-manifest.json").exists()
    assert not (work_dir / "registration").exists()


def test_recovery_does_not_overwrite_an_existing_work_cache(tmp_path: Path):
    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "derivatives",
        config_data={"save_intermediates": True},
    )
    workflow = PreprocessingWorkflow(config, logging.getLogger(__name__), None)
    work_dir = tmp_path / "work"
    output_dir = tmp_path / "out"
    (work_dir / "eddy").mkdir(parents=True)
    (work_dir / "eddy" / "marker.txt").write_text("work")
    (output_dir / "intermediate" / "eddy").mkdir(parents=True)
    (output_dir / "intermediate" / "eddy" / "marker.txt").write_text("saved")

    workflow.recover_intermediates(work_dir, output_dir)

    assert (work_dir / "eddy" / "marker.txt").read_text() == "work"
