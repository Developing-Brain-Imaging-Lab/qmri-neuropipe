import logging
from pathlib import Path
from unittest.mock import Mock, patch

import nibabel as nib
import numpy as np

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.core.types import ImageFile
from qmri_neuropipe.interfaces import c3d, fsl
from qmri_neuropipe.lib.common import registration
from qmri_neuropipe.lib.common.registration import CoregistrationStep
from qmri_neuropipe.workflows.pipelines.anat import AnatPreprocessingWorkflow


def _workflow(tmp_path: Path) -> AnatPreprocessingWorkflow:
    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "derivatives",
    )
    return AnatPreprocessingWorkflow(config, logging.getLogger(__name__), None)


def _image(tmp_path: Path, name: str, suffix: str) -> ImageFile:
    path = tmp_path / name
    path.touch()
    return ImageFile(entities={"sub": "01", "suffix": suffix}, img=path)


def _synthetic_outputs(tmp_path: Path, side: str) -> dict[str, Path]:
    output_dir = tmp_path / side
    output_dir.mkdir(parents=True, exist_ok=True)
    t1w = output_dir / "SynthT1.mgz"
    t2w = output_dir / "SynthT2.mgz"
    t1w.touch()
    t2w.touch()
    return {"synth_t1w": t1w, "synth_t2w": t2w}


def test_anatomical_supersynth_coregistration_uses_matched_pair_and_original_grid(
    tmp_path: Path,
):
    workflow = _workflow(tmp_path)
    t1w = _image(tmp_path, "T1w.nii.gz", "T1w")
    t2w = _image(tmp_path, "T2w.nii.gz", "T2w")
    result = _image(tmp_path, "T2w_coreg.nii.gz", "T2w")
    fixed_outputs = _synthetic_outputs(tmp_path, "fixed")
    moving_outputs = _synthetic_outputs(tmp_path, "moving")
    coreg_step = Mock(return_value={"current_image": result})
    coreg_step.method = "fsl"
    context = {"preprocessed_t1w": t1w, "preprocessed_t2w": t2w}

    with patch(
        "qmri_neuropipe.lib.anat.super_synth."
        "ensure_matched_supersynth_registration_inputs",
        return_value=(fixed_outputs, moving_outputs),
    ) as prepare:
        workflow._coregister_to_supersynth(
            "supersynth",
            {"method": "fsl"},
            {
                "skull_strip_registration": True,
                "skull_strip_method": "synthstrip",
            },
            coreg_step,
            False,
            context,
            tmp_path / "work",
            None,
            tmp_path / "figures",
        )

    assert prepare.call_args.args[:2] == (t1w, t2w)
    call = coreg_step.call_args
    assert call.args[0] is t2w
    assert call.kwargs["target"] == fixed_outputs["synth_t1w"]
    options = call.kwargs["options"]
    assert options["registration_fixed"] == fixed_outputs["synth_t1w"]
    assert options["registration_moving"] == moving_outputs["synth_t1w"]
    assert options["application_fixed"] == t1w.img
    assert options["skull_strip_registration"] is True
    assert options["skull_strip_method"] == "synthstrip"
    assert context["preprocessed_t2w_coreg"] is result


def test_anatomical_supersynth_t2w_preference_reverses_original_pair(
    tmp_path: Path,
):
    workflow = _workflow(tmp_path)
    t1w = _image(tmp_path, "T1w.nii.gz", "T1w")
    t2w = _image(tmp_path, "T2w.nii.gz", "T2w")
    result = _image(tmp_path, "T1w_coreg.nii.gz", "T1w")
    fixed_outputs = _synthetic_outputs(tmp_path, "fixed")
    moving_outputs = _synthetic_outputs(tmp_path, "moving")
    coreg_step = Mock(return_value=result)
    coreg_step.method = "ants"
    context = {"preprocessed_t1w": t1w, "preprocessed_t2w": t2w}

    with patch(
        "qmri_neuropipe.lib.anat.super_synth."
        "ensure_matched_supersynth_registration_inputs",
        return_value=(fixed_outputs, moving_outputs),
    ) as prepare:
        workflow._coregister_to_supersynth(
            "supersynth",
            {"method": "ants"},
            {"supersynth_input": "t2w"},
            coreg_step,
            False,
            context,
            tmp_path / "work",
            None,
            tmp_path / "figures",
        )

    assert prepare.call_args.args[:2] == (t2w, t1w)
    assert coreg_step.call_args.args[0] is t1w
    assert coreg_step.call_args.kwargs["options"]["application_fixed"] == t2w.img
    assert context["preprocessed_t1w"] is result


def test_anatomical_multivariate_supersynth_requires_both_contrasts(
    tmp_path: Path,
):
    workflow = _workflow(tmp_path)
    t1w = _image(tmp_path, "T1w.nii.gz", "T1w")
    t2w = _image(tmp_path, "T2w.nii.gz", "T2w")
    result = _image(tmp_path, "T2w_coreg.nii.gz", "T2w")
    fixed_outputs = _synthetic_outputs(tmp_path, "fixed")
    moving_outputs = _synthetic_outputs(tmp_path, "moving")
    coreg_step = Mock(return_value=result)
    coreg_step.method = "ants"
    context = {"preprocessed_t1w": t1w, "preprocessed_t2w": t2w}

    with patch(
        "qmri_neuropipe.lib.anat.super_synth."
        "ensure_matched_supersynth_registration_inputs",
        return_value=(fixed_outputs, moving_outputs),
    ) as prepare, patch.object(
        workflow,
        "_make_coregistration_step",
        return_value=coreg_step,
    ):
        workflow._coregister_to_supersynth(
            "supersynth_multivariate",
            {"method": "ants"},
            {},
            coreg_step,
            False,
            context,
            tmp_path / "work",
            None,
            tmp_path / "figures",
        )

    assert prepare.call_args.kwargs["required_contrasts"] == (
        "synth_t1w",
        "synth_t2w",
    )
    options = coreg_step.call_args.kwargs["options"]
    assert options["registration_fixed_extras"] == [
        fixed_outputs["synth_t2w"]
    ]
    assert options["registration_moving_extras"] == [
        moving_outputs["synth_t2w"]
    ]


def test_fsl_proxy_registration_is_applied_to_original_anatomical_pair(
    tmp_path: Path,
    monkeypatch,
):
    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "derivatives",
    )
    original_moving_path = tmp_path / "T2w.nii.gz"
    original_fixed_path = tmp_path / "T1w.nii.gz"
    proxy_moving = tmp_path / "moving_SynthT1.nii.gz"
    proxy_fixed = tmp_path / "fixed_SynthT1.nii.gz"
    nib.save(
        nib.Nifti1Image(np.ones((4, 4, 4)), np.eye(4)),
        original_moving_path,
    )
    nib.save(
        nib.Nifti1Image(np.ones((5, 5, 5)), np.diag([2, 2, 2, 1])),
        original_fixed_path,
    )
    nib.save(nib.Nifti1Image(np.ones((6, 6, 6)), np.eye(4)), proxy_moving)
    nib.save(nib.Nifti1Image(np.ones((7, 7, 7)), np.eye(4)), proxy_fixed)
    moving = ImageFile(
        entities={"sub": "01", "suffix": "T2w"},
        img=original_moving_path,
    )
    calls = []

    def fake_flirt(**kwargs):
        calls.append(("flirt", kwargs))
        out_file = Path(kwargs["out_file"])
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if "-applyxfm" in kwargs.get("extra_args", ""):
            reference = nib.load(str(kwargs["ref_file"]))
            nib.save(
                nib.Nifti1Image(
                    np.zeros(reference.shape[:3], dtype=np.float32),
                    reference.affine,
                ),
                out_file,
            )
        else:
            out_file.touch()
            np.savetxt(kwargs["omat"], np.eye(4))
        return out_file, kwargs.get("omat")

    def fake_fsl2ants(ref_file, in_file, transform_file, out_file):
        calls.append(
            ("fsl2ants", ref_file, in_file, transform_file, out_file)
        )
        Path(out_file).touch()

    def fake_ants2fsl(ref_file, in_file, transform_file, out_file):
        calls.append(
            ("ants2fsl", ref_file, in_file, transform_file, out_file)
        )
        np.savetxt(out_file, np.eye(4))

    monkeypatch.setattr(fsl, "flirt", fake_flirt)
    monkeypatch.setattr(c3d, "fsl2ants", fake_fsl2ants)
    monkeypatch.setattr(c3d, "ants2fsl", fake_ants2fsl)

    result = CoregistrationStep(
        config,
        logging.getLogger(__name__),
        method="fsl",
    ).run(
        moving,
        tmp_path / "work",
        target=proxy_fixed,
        target_modality="T1w",
        options={
            "registration_moving": proxy_moving,
            "registration_fixed": proxy_fixed,
            "application_fixed": original_fixed_path,
            "output_resolution": "anatomical",
            "transform_type": "Rigid",
        },
        force=True,
    )

    estimation = calls[0][1]
    application = calls[-1][1]
    assert estimation["in_file"] == proxy_moving
    assert estimation["ref_file"] == proxy_fixed
    assert application["in_file"] == original_moving_path
    assert application["ref_file"] == original_fixed_path
    assert nib.load(str(result.img)).shape == (5, 5, 5)


def test_multivariate_skull_stripping_covers_every_synthetic_pair(
    tmp_path: Path,
    monkeypatch,
):
    fixed = [tmp_path / "fixed_SynthT2.mgz", tmp_path / "fixed_SynthFLAIR.mgz"]
    moving = [
        tmp_path / "moving_SynthT2.mgz",
        tmp_path / "moving_SynthFLAIR.mgz",
    ]
    for path in [*fixed, *moving]:
        path.touch()
    calls = []

    def fake_strip(
        config,
        logger,
        image_path,
        output_dir,
        label,
        skull_cfg,
        nthreads,
        force=False,
    ):
        calls.append((image_path, label, skull_cfg, output_dir))
        stripped = output_dir / f"{label}_brain.nii.gz"
        stripped.parent.mkdir(parents=True, exist_ok=True)
        stripped.touch()
        return stripped

    monkeypatch.setattr(registration, "_strip_for_registration", fake_strip)

    prepared_fixed, prepared_moving = (
        registration.prepare_multivariate_registration_images(
            config={},
            logger=logging.getLogger(__name__),
            fixed_images=fixed,
            moving_images=moving,
            output_dir=tmp_path / "work",
            options={
                "skull_strip_registration": True,
                "skull_strip_method": "synthstrip",
            },
            nthreads=4,
        )
    )

    assert [call[0] for call in calls] == [
        moving[0],
        fixed[0],
        moving[1],
        fixed[1],
    ]
    assert all(call[2]["method"] == "synthstrip" for call in calls)
    assert len(prepared_fixed) == len(prepared_moving) == 2
