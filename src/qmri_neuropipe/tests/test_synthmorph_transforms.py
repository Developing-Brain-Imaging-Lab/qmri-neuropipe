import logging
from pathlib import Path

import pytest

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.interfaces import freesurfer
from qmri_neuropipe.lib.dmri.normalization import NormalizationStep
from qmri_neuropipe.workflows.pipelines.normalization_workflow import (
    NormalizationWorkflow,
)


@pytest.mark.parametrize("model", [None, "joint", "deform"])
def test_deformable_synthmorph_models_default_to_nifti(model):
    assert freesurfer.synthmorph_transform_extension(model) == ".nii.gz"


@pytest.mark.parametrize("model", ["affine", "rigid"])
def test_linear_synthmorph_models_default_to_lta(model):
    assert freesurfer.synthmorph_transform_extension(model) == ".lta"


def test_synthmorph_rejects_lta_for_deformable_model():
    with pytest.raises(ValueError, match="requires '.nii.gz'"):
        freesurfer.synthmorph_transform_extension("deform", ".lta")


@pytest.mark.parametrize(
    ("model", "filename"),
    [
        ("deform", "deformation.nii.gz"),
        ("affine", "transform.lta"),
    ],
)
def test_synthmorph_register_accepts_model_compatible_transform(
    tmp_path: Path,
    monkeypatch,
    model,
    filename,
):
    commands = []
    monkeypatch.setattr(
        freesurfer,
        "run_cmd",
        lambda command, **kwargs: commands.append(command),
    )

    transform = tmp_path / filename
    freesurfer.mri_synthmorph_register(
        moving=tmp_path / "fa.nii.gz",
        target=tmp_path / "t1w.nii.gz",
        transform_out=transform,
        model=model,
        overwrite=True,
    )

    assert commands
    assert f"-t {transform}" in commands[0]
    assert f"-m {model}" in commands[0]


def test_normalization_workflow_forwards_synthmorph_model():
    config = PipelineConfig(
        config_data={
            "dmri": {
                "normalization": {
                    "enabled": True,
                    "tool": "synthmorph",
                    "synthmorph_model": "deform",
                }
            }
        }
    )
    workflow = NormalizationWorkflow(
        config,
        logging.getLogger("test-synthmorph"),
        None,
    )

    workflow.build_pipeline({})

    step = workflow.steps[0]
    assert isinstance(step, NormalizationStep)
    assert step.kwargs["synthmorph_model"] == "deform"
    assert step.kwargs["synthmorph_transform_ext"] is None
    assert step._synthmorph_transform_extension(
        step.kwargs["synthmorph_model"],
        step.kwargs["synthmorph_transform_ext"],
    ) == ".nii.gz"
