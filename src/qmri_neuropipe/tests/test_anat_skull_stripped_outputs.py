import logging
from pathlib import Path

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.core.types import ImageFile
from qmri_neuropipe.io.bids import build_bids_name
from qmri_neuropipe.workflows.pipelines.anat import AnatPreprocessingWorkflow


def _workflow(config_data: dict) -> AnatPreprocessingWorkflow:
    config = PipelineConfig(
        bids_dir=Path("/tmp/bids"),
        output_dir=Path("/tmp/derivatives"),
        config_data=config_data,
    )
    return AnatPreprocessingWorkflow(config, logging.getLogger(__name__), None)


def test_normalization_skull_stripped_output_can_be_enabled_independently():
    workflow = _workflow(
        {
            "anat": {
                "preprocessing": {
                    "brain_masking": {"enabled": True, "method": "synthstrip"},
                    "normalization": {
                        "enabled": True,
                        "template": "/tmp/template.nii.gz",
                        "skull_stripped_outputs": True,
                        "save_transform": False,
                    },
                }
            }
        }
    )

    assert workflow.anat_config.preprocessing.skull_stripped_outputs is False
    assert workflow.anat_config.normalization.skull_stripped_outputs is True
    assert workflow.anat_config.normalization.save_transforms is False


def test_top_level_skull_stripped_output_remains_backward_compatible():
    workflow = _workflow(
        {
            "anat": {
                "preprocessing": {
                    "skull_stripped_outputs": True,
                    "normalization": {
                        "enabled": True,
                        "template": "/tmp/template.nii.gz",
                    },
                }
            }
        }
    )

    assert workflow.anat_config.preprocessing.skull_stripped_outputs is True
    assert workflow.anat_config.normalization.skull_stripped_outputs is True


def test_cached_normalized_brain_does_not_require_native_brain_output(tmp_path: Path):
    workflow = _workflow(
        {
            "anat": {
                "preprocessing": {
                    "normalization": {
                        "enabled": True,
                        "template": "/tmp/template.nii.gz",
                        "space_entity": "InfantTemplate",
                        "skull_stripped_outputs": True,
                    }
                }
            }
        }
    )
    source = ImageFile(
        entities={"sub": "01", "acq": "MPRAGE", "run": "02", "suffix": "T1w"},
        img=tmp_path / "source.nii.gz",
    )

    preproc_entities = dict(source.entities)
    preproc_entities["desc"] = "preproc"
    (tmp_path / build_bids_name(preproc_entities)).touch()

    normalized_brain_entities = dict(preproc_entities)
    normalized_brain_entities["space"] = "InfantTemplate"
    normalized_brain_entities["desc"] = "norm-brain"
    normalized_brain_path = tmp_path / build_bids_name(normalized_brain_entities)
    normalized_brain_path.touch()

    cached = workflow._load_preprocessed_from_output(
        {"t1w_files": [source], "t2w_files": []},
        tmp_path,
    )

    assert cached is not None
    assert cached["preprocessed_t1w_brain"] is None
    assert cached["normalized_t1w_brain"].img == normalized_brain_path
