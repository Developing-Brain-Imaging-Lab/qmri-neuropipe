import logging
from pathlib import Path

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.core.types import DWIFile
from qmri_neuropipe.io.bids import build_bids_name
from qmri_neuropipe.utils.data_io import DataIOManager


def _manager(tmp_path: Path) -> DataIOManager:
    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "out",
        work_dir=tmp_path / "work",
        config_data={"dmri": {}},
    )
    return DataIOManager(config, logging.getLogger(__name__))


def test_single_phase_encoding_derivative_drops_dir_and_run(tmp_path: Path):
    dwi = DWIFile(
        img=tmp_path / "sub-101_ses-post_dir-PA_run-1_dwi.nii.gz",
        entities={"sub": "101", "ses": "post", "dir": "PA", "run": "1", "suffix": "dwi"},
    )
    context = {"subject": "101", "session": "post", "preprocessed_dwis": [dwi]}

    ents = _manager(tmp_path).derivative_entities_for_dwis([dwi], context)[0]

    assert "dir" not in ents
    assert "run" not in ents
    assert build_bids_name(ents) == "sub-101_ses-post_desc-preproc_dwi.nii.gz"


def test_entity_cleanup_keeps_run_when_removal_would_collide(tmp_path: Path):
    dwis = [
        DWIFile(
            img=tmp_path / f"sub-101_ses-base_run-{run}_dwi.nii.gz",
            entities={"sub": "101", "ses": "base", "run": run, "suffix": "dwi"},
        )
        for run in ("1", "2")
    ]
    context = {"subject": "101", "session": "base", "preprocessed_dwis": dwis}

    cleaned = _manager(tmp_path).derivative_entities_for_dwis(dwis, context)

    assert [ents["run"] for ents in cleaned] == ["1", "2"]
    assert len({build_bids_name(ents) for ents in cleaned}) == 2
