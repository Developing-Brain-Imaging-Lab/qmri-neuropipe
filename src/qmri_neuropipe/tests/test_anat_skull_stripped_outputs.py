import logging
import shutil
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.core.types import ImageFile
from qmri_neuropipe.io.bids import build_bids_name
from qmri_neuropipe.lib.common.registration import NonlinearRegistrationStep
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


def test_fsl_normalization_writes_skull_stripped_derivative(tmp_path: Path):
    template = tmp_path / "template.nii.gz"
    source_path = tmp_path / "sub-01_T1w.nii.gz"
    mask_path = tmp_path / "sub-01_desc-preproc_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.float32), np.eye(4)), template)
    nib.save(nib.Nifti1Image(np.full((3, 3, 3), 2.0, dtype=np.float32), np.eye(4)), source_path)
    mask_data = np.zeros((3, 3, 3), dtype=np.uint8)
    mask_data[1, 1, 1] = 1
    nib.save(nib.Nifti1Image(mask_data, np.eye(4)), mask_path)

    workflow = _workflow(
        {
            "skip_existing": False,
            "anat": {
                "preprocessing": {
                    "normalization": {
                        "enabled": True,
                        "method": "fsl",
                        "template": str(template),
                        "space_entity": "InfantTemplate",
                        "skull_stripped_outputs": True,
                    }
                }
            },
        }
    )
    source = ImageFile(
        entities={"sub": "01", "suffix": "T1w"},
        img=source_path,
    )
    mask = ImageFile(
        entities={"sub": "01", "desc": "preproc", "suffix": "mask"},
        img=mask_path,
    )
    context = {
        "preprocessed_t1w": source,
        "brain_mask": mask,
        "errors": [],
    }

    def fake_registration_run(step, first_arg, output_dir, **kwargs):
        output_img = output_dir / "sub-01_space-InfantTemplate_desc-norm_T1w.nii.gz"
        transform = output_dir / "sub-01_space-InfantTemplate_desc-norm_transform.mat"
        shutil.copy2(source_path, output_img)
        transform.touch()
        first_arg["current_image"] = ImageFile(
            entities={"sub": "01", "space": "InfantTemplate", "desc": "norm", "suffix": "T1w"},
            img=output_img,
        )
        first_arg["template_transform"] = transform
        first_arg["template_transform_type"] = "fsl"
        return first_arg

    def fake_applywarp(in_file, ref_file, out_file, **kwargs):
        shutil.copy2(in_file, out_file)
        return out_file

    output_dir = tmp_path / "work"
    final_output_dir = tmp_path / "derivatives"
    output_dir.mkdir()
    final_output_dir.mkdir()

    with (
        patch.object(NonlinearRegistrationStep, "run", fake_registration_run),
        patch("qmri_neuropipe.workflows.pipelines.anat.fsl.applywarp", fake_applywarp),
    ):
        result, _ = workflow._run_normalization(
            output_dir,
            context,
            final_output_dir,
            reporter=None,
            figures_dir=output_dir,
            step_metrics=[],
        )

    brain = result["normalized_t1w_brain"]
    assert brain.img.exists()
    assert brain.img.parent == final_output_dir
    assert brain.entities["desc"] == "norm-brain"
    brain_data = nib.load(brain.img).get_fdata()
    assert brain_data[1, 1, 1] == 2.0
    assert np.count_nonzero(brain_data) == 1


def test_normalization_uses_context_returned_by_step(tmp_path: Path):
    template = tmp_path / "template.nii.gz"
    source_path = tmp_path / "sub-01_T1w.nii.gz"
    mask_path = tmp_path / "sub-01_desc-preproc_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.float32), np.eye(4)), template)
    nib.save(nib.Nifti1Image(np.full((3, 3, 3), 2.0, dtype=np.float32), np.eye(4)), source_path)
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.uint8), np.eye(4)), mask_path)

    workflow = _workflow(
        {
            "anat": {
                "preprocessing": {
                    "normalization": {
                        "enabled": True,
                        "method": "fsl",
                        "template": str(template),
                        "space_entity": "InfantTemplate",
                        "skull_stripped_outputs": True,
                    }
                }
            },
        }
    )
    source = ImageFile(entities={"sub": "01", "suffix": "T1w"}, img=source_path)
    mask = ImageFile(
        entities={"sub": "01", "desc": "preproc", "suffix": "mask"},
        img=mask_path,
    )
    original_context = {
        "preprocessed_t1w": source,
        "brain_mask": mask,
        "errors": [],
    }

    def fake_registration_run(step, first_arg, output_dir, **kwargs):
        returned_context = dict(first_arg)
        output_img = output_dir / "sub-01_space-InfantTemplate_desc-norm_T1w.nii.gz"
        transform = output_dir / "sub-01_space-InfantTemplate_desc-norm_transform.mat"
        shutil.copy2(source_path, output_img)
        transform.touch()
        returned_context["current_image"] = ImageFile(
            entities={"sub": "01", "space": "InfantTemplate", "desc": "norm", "suffix": "T1w"},
            img=output_img,
        )
        returned_context["template_transform"] = transform
        returned_context["template_transform_type"] = "fsl"
        return returned_context

    def fake_applywarp(in_file, ref_file, out_file, **kwargs):
        shutil.copy2(in_file, out_file)
        return out_file

    output_dir = tmp_path / "work"
    final_output_dir = tmp_path / "derivatives"
    output_dir.mkdir()
    final_output_dir.mkdir()

    with (
        patch.object(NonlinearRegistrationStep, "run", fake_registration_run),
        patch("qmri_neuropipe.workflows.pipelines.anat.fsl.applywarp", fake_applywarp),
    ):
        result, _ = workflow._run_normalization(
            output_dir,
            original_context,
            final_output_dir,
            reporter=None,
            figures_dir=output_dir,
            step_metrics=[],
        )

    assert "template_transform" not in original_context
    assert result["template_transform"].exists()
    assert result["preprocessed_t1w"].entities["space"] == "InfantTemplate"
    assert result["normalized_t1w_brain"].img.exists()
    assert not any("transform or template is unavailable" in error for error in result["errors"])
