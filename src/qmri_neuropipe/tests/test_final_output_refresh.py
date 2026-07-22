import logging
import os
from pathlib import Path

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.core.types import DWIFile
from qmri_neuropipe.utils.data_io import DataIOManager


def test_newer_workflow_output_replaces_stale_final_image_with_skip_existing(tmp_path: Path):
    output_dir = tmp_path / "derivatives"
    source = tmp_path / "work" / "registration" / "sub-01_desc-coreg_dwi.nii.gz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"new-coregistration")
    target = output_dir / "sub-01" / "dwi" / "sub-01_desc-preproc_dwi.nii.gz"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old-preprocessed")
    os.utime(target, (100, 100))
    os.utime(source, (200, 200))

    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=output_dir,
        config_data={"skip_existing": True},
    )
    dwi = DWIFile(
        img=source,
        entities={"sub": "01", "desc": "coreg", "suffix": "dwi"},
    )

    DataIOManager(config, logging.getLogger(__name__)).save_final_outputs(
        {"subject": "01", "preprocessed_dwis": [dwi], "preprocessed_masks": [None]},
        output_dir,
        skip_existing=True,
    )

    assert target.read_bytes() == b"new-coregistration"
