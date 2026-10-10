import logging
from pathlib import Path

import nibabel as nib
import numpy as np

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.interfaces import freesurfer, fsl
from qmri_neuropipe.lib.relax.motion import RelaxometryMotionCorrectionStep


def _step(tmp_path: Path, options: dict) -> RelaxometryMotionCorrectionStep:
    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "out",
    )
    return RelaxometryMotionCorrectionStep(
        config,
        logging.getLogger("test-relaxometry-fsl-motion"),
        {},
        method="fsl",
        options=options,
    )


def _image(path: Path) -> Path:
    nib.save(
        nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.float32), np.eye(4)),
        path,
    )
    return path


def test_fsl_motion_uses_interp_flag_for_sinc(tmp_path, monkeypatch):
    moving = _image(tmp_path / "moving.nii.gz")
    fixed = _image(tmp_path / "fixed.nii.gz")
    output = tmp_path / "registered.nii.gz"
    calls = []

    def fake_flirt(**kwargs):
        calls.append(kwargs)
        return kwargs["out_file"], kwargs.get("omat")

    monkeypatch.setattr(fsl, "flirt", fake_flirt)
    step = _step(
        tmp_path,
        {
            "dof": 6,
            "cost": "normmi",
            "interpolation": "sinc",
            "aff_metric": "mattes",
            "aff_sampling": 32,
        },
    )

    step._register(moving, fixed, output)

    assert calls[0]["extra_args"] == "-interp sinc"
    assert "extra_opts" not in calls[0]


def test_fsl_motion_maps_linear_to_trilinear():
    assert (
        RelaxometryMotionCorrectionStep._normalize_fsl_interpolator("linear")
        == "trilinear"
    )


def test_synthmorph_motion_estimates_rigid_lta_and_applies_it(
    tmp_path, monkeypatch
):
    moving = _image(tmp_path / "moving.nii.gz")
    fixed = _image(tmp_path / "fixed.nii.gz")
    output = tmp_path / "registered.nii.gz"
    calls = {}

    def fake_register(**kwargs):
        calls["register"] = kwargs
        Path(kwargs["transform_out"]).touch()
        return kwargs["transform_out"]

    def fake_apply(**kwargs):
        calls["apply"] = kwargs
        Path(kwargs["out_file"]).touch()
        return kwargs["out_file"]

    monkeypatch.setattr(freesurfer, "mri_synthmorph_register", fake_register)
    monkeypatch.setattr(freesurfer, "mri_synthmorph_apply", fake_apply)
    step = RelaxometryMotionCorrectionStep(
        _step(tmp_path, {}).config,
        logging.getLogger("test-relaxometry-synthmorph-motion"),
        {},
        method="mri_synthmorph",
        options={
            "synthmorph_model": "rigid",
            "synthmorph_register_args": "-j 4",
            "synthmorph_apply_args": "-m linear",
        },
    )

    step._register(moving, fixed, output)

    assert step.method == "synthmorph"
    assert calls["register"]["moving"] == moving
    assert calls["register"]["target"] == fixed
    assert calls["register"]["model"] == "rigid"
    assert calls["register"]["extra_args"] == "-j 4"
    assert calls["apply"]["moving"] == moving
    assert calls["apply"]["target"] == fixed
    assert calls["apply"]["extra_args"] == "-m linear"
    assert output.exists()


def test_synthmorph_motion_rejects_deformable_model(tmp_path):
    step = RelaxometryMotionCorrectionStep(
        _step(tmp_path, {}).config,
        logging.getLogger("test-relaxometry-synthmorph-model"),
        {},
        method="synthmorph",
        options={"synthmorph_model": "deform"},
    )

    import pytest

    with pytest.raises(ValueError, match="linear SynthMorph model"):
        step._synthmorph_model(step.options)
