import logging
from pathlib import Path

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.workflows.pipelines.dmri import DMRIPipeline


def test_absolute_anatomical_pattern_preserves_entities_for_selection(tmp_path: Path):
    bids_dir = tmp_path / "rawdata"
    anat_dir = tmp_path / "derivatives" / "mpnrage-processed" / "sub-311140" / "ses-01" / "anat"
    anat_dir.mkdir(parents=True)
    t1w = anat_dir / "sub-311140_ses-01_acq-MPnRAGE_desc-preproc_T1w.nii.gz"
    t1w.write_bytes(b"placeholder")

    config = PipelineConfig(
        bids_dir=bids_dir,
        output_dir=tmp_path / "output",
        config_data={
            "anat": {
                "input": {
                    "t1w_search_pattern": str(
                        tmp_path
                        / "derivatives"
                        / "mpnrage-processed"
                        / "sub-{subject}"
                        / "ses-{session}"
                        / "anat"
                        / "*_acq-MPnRAGE_desc-preproc_T1w.nii.gz"
                    ),
                    "t1w_match": {
                        "entities": {"acq": "MPnRAGE", "desc": "preproc"}
                    },
                }
            }
        },
    )
    pipeline = object.__new__(DMRIPipeline)
    pipeline.config = config
    pipeline.logger = logging.getLogger(__name__)

    matches = pipeline._find_anat_files("311140", "01", "T1w")

    assert [image.img for image in matches] == [t1w]
    assert matches[0].entities["acq"] == "MPnRAGE"
    assert matches[0].entities["desc"] == "preproc"
