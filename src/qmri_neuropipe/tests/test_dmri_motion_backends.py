import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.core.types import DWIFile, ImageFile
from qmri_neuropipe.interfaces.tortoise import (
    _stage_tortoise_v4_input,
    _validate_tortoise_v4_gradients,
    build_tortoise_v4_command,
    tortoise_v4_motion_eddy,
)
from qmri_neuropipe.core import ProcessingError
from qmri_neuropipe.lib.dmri.ants_motion import (
    AntsDiffusionMotionCorrectionStep,
    closest_rotation,
    rotate_bvec_table,
)
from qmri_neuropipe.lib.dmri.b0_reference import (
    b0_candidate_indices,
    select_optimal_b0,
)
from qmri_neuropipe.lib.dmri.tortoise_v4 import TortoiseV4CorrectionStep, _image_grid
from qmri_neuropipe.lib.dmri.synb0 import Synb0EstimationStep
from qmri_neuropipe.lib.dmri.topup import TopupStep
from qmri_neuropipe.lib.dmri.apply_topup import ApplyTopupStep
from qmri_neuropipe.workflows.pipelines.integrated_preprocessing_workflow import (
    PreprocessingWorkflow,
)


def _dwi(tmp_path: Path, data: np.ndarray, bvals: np.ndarray) -> DWIFile:
    image = tmp_path / "sub-01_dwi.nii.gz"
    bval = tmp_path / "sub-01_dwi.bval"
    bvec = tmp_path / "sub-01_dwi.bvec"
    sidecar = tmp_path / "sub-01_dwi.json"
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), image)
    np.savetxt(bval, bvals[None, :], fmt="%.1f")
    np.savetxt(bvec, np.zeros((3, data.shape[3])), fmt="%.1f")
    sidecar.write_text("{}\n")
    return DWIFile(
        entities={"sub": "01", "suffix": "dwi"},
        img=image,
        json=sidecar,
        bval=bval,
        bvec=bvec,
    )


def test_tortoise_like_selector_chooses_the_consistent_b0_pair(tmp_path: Path):
    rng = np.random.default_rng(3)
    grid = np.indices((13, 13, 9), dtype=float)
    base = np.exp(-sum((grid[i] - (6, 6, 4)[i]) ** 2 for i in range(3)) / 18.0)
    data = np.stack(
        [
            base,
            base + rng.normal(0, 0.005, base.shape),
            rng.normal(0.2, 0.3, base.shape),
            base * 0.4,
        ],
        axis=3,
    )
    dwi = _dwi(tmp_path, data, np.array([0, 8, 20, 1000]))

    selection = select_optimal_b0(dwi, tmp_path / "reference")

    assert {selection.index, selection.paired_index} == {0, 1}
    assert selection.reference_image.exists()
    assert selection.pair_average_image.exists()
    assert selection.metrics_file.exists()


def test_b0_candidates_fall_back_to_lowest_acquired_shell():
    assert b0_candidate_indices([100, 105, 1000], threshold=50).tolist() == [0, 1]


def test_affine_rotation_is_proper_and_rotates_each_gradient():
    angle = np.pi / 2
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    affine = rotation @ np.diag([1.2, 0.8, 1.1])
    recovered = closest_rotation(affine)
    rotated = rotate_bvec_table(np.array([[1, 0], [0, 0], [0, 0]]), [recovered, recovered])

    assert np.linalg.det(recovered) > 0
    np.testing.assert_allclose(recovered.T @ recovered, np.eye(3), atol=1e-7)
    np.testing.assert_allclose(rotated[:, 0], [0, 1, 0], atol=1e-7)
    np.testing.assert_allclose(rotated[:, 1], [0, 0, 0], atol=1e-7)


def test_tortoise_v4_command_consumes_explicit_pipeline_sidecars(tmp_path: Path):
    dwi = DWIFile(
        entities={"sub": "01"},
        img=tmp_path / "current.nii.gz",
        json=tmp_path / "metadata.json",
        bval=tmp_path / "current.bval",
        bvec=tmp_path / "rotated.bvec",
    )
    command = build_tortoise_v4_command(
        dwi,
        tmp_path / "corrected.nii.gz",
        b0_id=7,
        slice_to_volume=True,
        repol=True,
    )

    assert command[0] == "TORTOISEProcess"
    assert command[command.index("--ub") + 1] == str(dwi.bval)
    assert command[command.index("--uv") + 1] == str(dwi.bvec)
    assert "--up_json" not in command  # v4 discovers basename-matched staged JSON
    assert command[command.index("--b0_id") + 1] == "7"
    assert command[command.index("--s2v") + 1] == "1"
    assert command[command.index("--denoising") + 1] == "off"
    assert command[command.index("--epi") + 1] == "off"


def test_tortoise_v4_writes_nii_then_gzips_requested_output(
    tmp_path: Path, monkeypatch
):
    dwi = _dwi(
        tmp_path,
        np.ones((3, 4, 5, 2), dtype=np.float32),
        np.array([0, 1000]),
    )
    requested = tmp_path / "corrected.nii.gz"
    native = tmp_path / "corrected.nii"
    structural = tmp_path / "T2w.nii.gz"
    reorientation = tmp_path / "T1w.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 4, 5)), np.eye(4)), structural)
    nib.save(nib.Nifti1Image(np.ones((3, 4, 5)), np.eye(4)), reorientation)
    commands = []

    monkeypatch.setattr(
        "qmri_neuropipe.interfaces.tortoise.shutil.which",
        lambda executable: f"/opt/tortoise/{executable}",
    )

    def fake_run_cmd(command, **kwargs):
        commands.append(command)
        nib.save(
            nib.Nifti1Image(np.full((3, 4, 5, 2), 7, dtype=np.float32), np.eye(4)),
            native,
        )
        (tmp_path / "corrected.bvecs").write_text("0 0\n0 0\n0 0\n")
        (tmp_path / "corrected.bvals").write_text("0 1000\n")

    monkeypatch.setattr(
        "qmri_neuropipe.interfaces.tortoise.run_cmd", fake_run_cmd
    )

    result = tortoise_v4_motion_eddy(
        dwi,
        requested,
        structural_file=structural,
        reorientation_file=reorientation,
    )

    assert f"--output {native}" in commands[0]
    assert f"--up_data {tmp_path / 'tortoise_inputs' / 'up.nii'}" in commands[0]
    assert str(tmp_path / "tortoise_inputs" / "structural_0_T2w.nii") in commands[0]
    assert str(tmp_path / "tortoise_inputs" / "reorientation_T1w.nii") in commands[0]
    assert ".nii.gz" not in commands[0]
    assert result.img == requested
    assert requested.exists()
    assert not native.exists()
    np.testing.assert_allclose(np.asanyarray(nib.load(requested).dataobj), 7)


def test_tortoise_stages_compressed_dwi_as_uncompressed_nifti(tmp_path: Path):
    source = _dwi(
        tmp_path,
        np.arange(120, dtype=np.float32).reshape(3, 4, 5, 2),
        np.array([0, 1000]),
    )

    staged = _stage_tortoise_v4_input(source, tmp_path / "staged", "up")

    assert staged.img.name == "up.nii"
    assert not staged.img.is_symlink()
    np.testing.assert_allclose(
        np.asanyarray(nib.load(staged.img).dataobj),
        np.asanyarray(nib.load(source.img).dataobj),
    )


def test_tortoise_v4_command_exposes_full_pipeline_and_reverse_pe(tmp_path: Path):
    up = DWIFile(
        entities={"dir": "AP"}, img=tmp_path / "up.nii.gz", json=tmp_path / "up.json",
        bval=tmp_path / "up.bval", bvec=tmp_path / "up.bvec",
    )
    down = DWIFile(
        entities={"dir": "PA"}, img=tmp_path / "down.nii.gz", json=tmp_path / "down.json",
        bval=tmp_path / "down.bval", bvec=tmp_path / "down.bvec",
    )
    command = build_tortoise_v4_command(
        up,
        tmp_path / "corrected.nii.gz",
        down_file=down,
        structural_file=[tmp_path / "t2w.nii.gz"],
        reorientation_file=tmp_path / "t1w.nii.gz",
        denoising="for_final",
        gibbs=True,
        drift="linear",
        epi="DRBUDDI",
        output_res=[1.5, 1.5, 1.5],
        output_voxels=[128, 140, 96],
        output_data_combination="JacConcat",
        output_signal_redist_method="LSR",
    )

    assert command[command.index("--down_data") + 1] == str(down.img)
    assert command[command.index("--db") + 1] == str(down.bval)
    assert command[command.index("--dv") + 1] == str(down.bvec)
    assert command[command.index("--denoising") + 1] == "for_final"
    assert command[command.index("--gibbs") + 1] == "1"
    assert command[command.index("--drift") + 1] == "linear"
    assert command[command.index("--epi") + 1] == "DRBUDDI"
    assert command[command.index("--output_res") + 1:command.index("--output_res") + 4] == ["1.5"] * 3
    assert command[command.index("--output_data_combination") + 1] == "JacConcat"


def test_tortoise_stages_single_synb0_without_mutating_source(tmp_path: Path):
    image = tmp_path / "synb0.nii.gz"
    sidecar = tmp_path / "metadata.json"
    bval = tmp_path / "synb0.bval"
    bvec = tmp_path / "synb0.bvec"
    nib.save(nib.Nifti1Image(np.ones((5, 6, 4, 1), dtype=np.float32), np.eye(4)), image)
    sidecar.write_text('{"PhaseEncodingDirection": "j-"}\n')
    bval.write_text("0\n")
    bvec.write_text("0\n0\n0\n")
    source = DWIFile(entities={"desc": "synthetic"}, img=image, json=sidecar, bval=bval, bvec=bvec)

    staged = _stage_tortoise_v4_input(source, tmp_path / "staged", "down")

    assert not staged.img.is_symlink()
    assert nib.load(str(source.img)).shape == (5, 6, 4, 1)
    assert nib.load(str(staged.img)).shape == (5, 6, 4, 2)
    assert np.loadtxt(staged.bval).size == 2
    assert np.loadtxt(staged.bvec).shape == (3, 2)
    assert Path(str(staged.img).split(".nii", 1)[0] + ".json").exists()
    _validate_tortoise_v4_gradients(staged, "down")


def test_tortoise_rejects_mismatched_gradient_dimensions_before_execution(tmp_path: Path):
    source = _dwi(
        tmp_path,
        np.ones((5, 6, 4, 3), dtype=np.float32),
        np.array([0, 1000, 1000]),
    )
    source.bvec.write_text("0 0\n0 0\n0 0\n")

    with np.testing.assert_raises_regex(ProcessingError, "3 image volumes"):
        _validate_tortoise_v4_gradients(source, "up")


def test_synb0_phase_encoding_is_opposite_acquired_direction():
    assert Synb0EstimationStep._opposite_phase_encoding("j-") == "j"
    assert Synb0EstimationStep._opposite_phase_encoding("i") == "i-"


def _workflow(method: str) -> PreprocessingWorkflow:
    preprocessing = {
        "motion_correction": {
            "method": method,
            "reference_selection": {"enabled": True},
            "tortoise_v4": {"slice_to_volume": True},
            "ants": {"slice_to_volume": True},
            "native": {"slice_to_volume": True},
        }
    }
    config = PipelineConfig(
        bids_dir=Path("/tmp/bids"),
        output_dir=Path("/tmp/out"),
        config_data={"dmri": {"preprocessing": preprocessing}},
    )
    workflow = PreprocessingWorkflow(config, logging.getLogger(__name__), None)
    workflow.build_pipeline({"dwi_files": [], "topup_groups": []})
    return workflow


def test_workflow_selects_each_new_motion_backend():
    tortoise = _workflow("tortoise_v4").steps
    ants = _workflow("ants").steps
    native = _workflow("ants_native").steps

    assert len(tortoise) == 1 and isinstance(tortoise[0], TortoiseV4CorrectionStep)
    assert len(ants) == 1 and isinstance(ants[0], AntsDiffusionMotionCorrectionStep)
    assert ants[0].mode == "motion" and ants[0].slice_to_volume
    assert len(native) == 1 and isinstance(native[0], AntsDiffusionMotionCorrectionStep)
    assert native[0].mode == "motion_eddy" and native[0].slice_to_volume


def test_top_level_tortoise_stream_is_enabled_by_presence():
    preprocessing = {
        "motion_correction": {"method": "eddy"},
        "tortoise_v4": {"slice_to_volume": True, "denoising": "for_final"},
    }
    config = PipelineConfig(
        bids_dir=Path("/tmp/bids"), output_dir=Path("/tmp/out"),
        config_data={"dmri": {"preprocessing": preprocessing}},
    )
    workflow = PreprocessingWorkflow(config, logging.getLogger(__name__), None)
    workflow.build_pipeline({"dwi_files": [], "topup_groups": []})

    assert len(workflow.steps) == 1
    assert isinstance(workflow.steps[0], TortoiseV4CorrectionStep)


def test_nested_synb0_config_enables_tortoise_drbuddi():
    preprocessing = {
        "tortoise_v4": {
            "synb0": {
                "anatomical_input": "auto",
                "registration_backend": "ants",
            }
        }
    }
    config = PipelineConfig(
        bids_dir=Path("/tmp/bids"), output_dir=Path("/tmp/out"),
        config_data={"dmri": {"preprocessing": preprocessing}},
    )
    workflow = PreprocessingWorkflow(config, logging.getLogger(__name__), None)
    workflow.build_pipeline({
        "dwi_files": [],
        "topup_groups": [],
        "t1w_files": [object()],
    })

    assert [type(step) for step in workflow.steps] == [
        Synb0EstimationStep,
        TortoiseV4CorrectionStep,
    ]
    assert workflow.steps[0].synb0_cfg["registration_backend"] == "ants"
    assert workflow.steps[1].options["epi"] == "DRBUDDI"
    assert workflow.steps[1].options["use_synb0"] is True


def test_full_tortoise_workflow_avoids_duplicate_pipeline_stages():
    preprocessing = {
        "merging": {"enabled": True},
        "denoising": {"enabled": True},
        "degibbs": {"enabled": True},
        "resample": {"enabled": True, "resolution": [2, 2, 2]},
        "distcorr": {"method": "topup"},
        "coregistration": {"enabled": True},
        "motion_correction": {"method": "eddy"},
        "tortoise_v4": {
            "denoising": "for_final",
            "gibbs": True,
            "epi": "DRBUDDI",
            "use_reverse_pe": True,
            "coregistration_to_anatomy": {
                "enabled": True,
                "output_resolution": "anatomical",
            },
        },
    }
    config = PipelineConfig(
        bids_dir=Path("/tmp/bids"), output_dir=Path("/tmp/out"),
        config_data={"dmri": {"preprocessing": preprocessing}},
    )
    workflow = PreprocessingWorkflow(config, logging.getLogger(__name__), None)
    up = DWIFile(entities={"dir": "AP"}, img=Path("/tmp/up.nii.gz"))
    down = DWIFile(entities={"dir": "PA"}, img=Path("/tmp/down.nii.gz"))
    workflow.build_pipeline({"dwi_files": [up, down], "topup_groups": [[up, down]]})

    assert [step.__class__.__name__ for step in workflow.steps] == ["TortoiseV4CorrectionStep"]


def test_post_tortoise_synb0_topup_workflow_order():
    preprocessing = {
        "tortoise_v4": {
            "denoising": "for_final",
            "gibbs": True,
            "epi": "off",
        },
        "distcorr": {
            "method": "synb0",
            "application": "post_tortoise",
            "apply_method": "jac",
            "synb0": {
                "registration": "supersynth",
                "registration_backend": "synthmorph",
            },
            "skull_strip": {
                "enabled": True,
                "method": "synthstrip",
                "strip_moving": True,
                "strip_fixed": False,
            },
        },
    }
    config = PipelineConfig(
        bids_dir=Path("/tmp/bids"), output_dir=Path("/tmp/out"),
        config_data={"dmri": {"preprocessing": preprocessing}},
    )
    workflow = PreprocessingWorkflow(config, logging.getLogger(__name__), None)
    dwi = DWIFile(entities={"suffix": "dwi"}, img=Path("/tmp/dwi.nii.gz"))
    workflow.build_pipeline({
        "dwi_files": [dwi],
        "topup_groups": [],
        "t1w_files": [object()],
    })

    assert [type(step) for step in workflow.steps] == [
        TortoiseV4CorrectionStep,
        Synb0EstimationStep,
        TopupStep,
        ApplyTopupStep,
    ]
    assert workflow.steps[0].options["epi"] == "off"
    assert workflow.steps[1].synb0_cfg["registration_backend"] == "synthmorph"
    assert workflow.steps[1].synb0_cfg["registration"] == "supersynth"
    assert workflow.steps[1].synb0_cfg["skull_strip_registration"] is True
    assert workflow.steps[1].synb0_cfg["skull_strip_method"] == "synthstrip"
    assert workflow.steps[1].synb0_cfg["skull_strip_moving"] is True
    assert workflow.steps[1].synb0_cfg["skull_strip_fixed"] is False
    assert workflow.steps[3].method == "jac"


def test_post_tortoise_accepts_yaml_boolean_off_and_ignores_legacy_synb0():
    preprocessing = {
        "motion_correction": {
            "method": "tortoise_v4",
            "tortoise_v4": {
                "epi": "DRBUDDI",
                "use_synb0": True,
                "synb0": {"enabled": True},
            },
        },
        # An unquoted YAML `off` is loaded by PyYAML as False.  The modern
        # top-level stream must also replace, rather than merge, the legacy
        # TORTOISE options above.
        "tortoise_v4": {"epi": False},
        "distcorr": {
            "method": "synb0",
            "application": "post_tortoise",
        },
    }
    config = PipelineConfig(
        bids_dir=Path("/tmp/bids"), output_dir=Path("/tmp/out"),
        config_data={"dmri": {"preprocessing": preprocessing}},
    )
    workflow = PreprocessingWorkflow(config, logging.getLogger(__name__), None)
    workflow.build_pipeline({
        "dwi_files": [DWIFile(entities={"suffix": "dwi"}, img=Path("/tmp/dwi.nii.gz"))],
        "topup_groups": [],
        "t1w_files": [object()],
    })

    assert [type(step) for step in workflow.steps] == [
        TortoiseV4CorrectionStep,
        Synb0EstimationStep,
        TopupStep,
        ApplyTopupStep,
    ]
    assert workflow.steps[0].options["epi"] == "off"
    assert workflow.steps[0].options.get("use_synb0") is None


def test_post_tortoise_topup_rejects_native_reverse_pe_until_two_stream_support():
    preprocessing = {
        "tortoise_v4": {"epi": "off"},
        "distcorr": {"method": "topup", "application": "post_tortoise"},
    }
    config = PipelineConfig(
        bids_dir=Path("/tmp/bids"), output_dir=Path("/tmp/out"),
        config_data={"dmri": {"preprocessing": preprocessing}},
    )
    workflow = PreprocessingWorkflow(config, logging.getLogger(__name__), None)

    with pytest.raises(ValueError, match="only distcorr.method: synb0"):
        workflow.build_pipeline({
            "dwi_files": [DWIFile(entities={"suffix": "dwi"}, img=Path("/tmp/dwi.nii.gz"))],
            "topup_groups": [],
            "t1w_files": [object()],
        })


def test_tortoise_output_grid_matches_selected_reference(tmp_path: Path):
    native = tmp_path / "dwi.nii.gz"
    anatomical = tmp_path / "T1w.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros((8, 9, 10, 2)), np.diag([2.0, 2.5, 3.0, 1.0])),
        native,
    )
    anatomical_affine = np.diag([-1.0, 1.2, 1.4, 1.0])
    nib.save(nib.Nifti1Image(np.zeros((20, 21, 22)), anatomical_affine), anatomical)
    dwi = DWIFile(entities={}, img=native)

    native_step = TortoiseV4CorrectionStep.__new__(TortoiseV4CorrectionStep)
    native_step.options = {
        "coregistration_to_anatomy": {
            "enabled": True,
            "output_resolution": "native",
        }
    }
    anatomical_step = TortoiseV4CorrectionStep.__new__(TortoiseV4CorrectionStep)
    anatomical_step.options = {
        "coregistration_to_anatomy": {
            "enabled": True,
            "output_resolution": "anatomical",
        }
    }

    assert native_step._resolve_output_grid(dwi, anatomical) == _image_grid(native)
    assert anatomical_step._resolve_output_grid(dwi, anatomical) == _image_grid(anatomical)


def test_tortoise_synthesizes_t2w_when_only_t1w_exists(tmp_path: Path, monkeypatch):
    source = tmp_path / "T1w.nii.gz"
    synth = tmp_path / "SynthT2.mgz"
    nib.save(nib.Nifti1Image(np.ones((4, 5, 6)), np.eye(4)), source)
    nib.save(nib.MGHImage(np.ones((4, 5, 6), dtype=np.float32), np.eye(4)), synth)

    monkeypatch.setattr(
        "qmri_neuropipe.lib.dmri.tortoise_v4.ensure_supersynth_outputs_for_image",
        lambda *args, **kwargs: {"synth_t2w": synth},
    )

    def fake_convert(in_file, out_file):
        nib.save(nib.Nifti1Image(np.ones((4, 5, 6)), np.eye(4)), out_file)

    monkeypatch.setattr(
        "qmri_neuropipe.lib.dmri.tortoise_v4.freesurfer.mri_convert", fake_convert
    )
    config = PipelineConfig(
        bids_dir=tmp_path, output_dir=tmp_path,
        config_data={},
    )
    step = TortoiseV4CorrectionStep(
        config,
        logging.getLogger(__name__),
        None,
        epi="T2Wreg",
        coregistration_to_anatomy={"enabled": True, "reference": "auto"},
    )
    context = {"t1w_files": [source]}

    structurals = step._select_structural(context, tmp_path / "work", False)
    reorientation = step._select_reorientation(
        context, structurals, output_dir=tmp_path / "work", force=False
    )

    assert structurals and structurals[0].name == "desc-supersynth_T2w.nii.gz"
    assert structurals[0].exists()
    assert reorientation == structurals[0]
    assert context["tortoise_t2w_source"] == "mri_super_synth"


@pytest.mark.parametrize(
    ("epi", "extra_options"),
    [
        ("T2Wreg", {}),
        ("DRBUDDI", {"use_for_drbuddi": True}),
    ],
)
def test_tortoise_can_prefer_synthesized_t2w_over_acquired(
    tmp_path: Path, monkeypatch, epi, extra_options
):
    source = tmp_path / "T1w.nii.gz"
    acquired = tmp_path / "acquired_T2w.nii.gz"
    synth = tmp_path / "SynthT2.mgz"
    nib.save(nib.Nifti1Image(np.ones((4, 5, 6)), np.eye(4)), source)
    nib.save(nib.Nifti1Image(np.ones((4, 5, 6)), np.eye(4)), acquired)
    nib.save(nib.MGHImage(np.ones((4, 5, 6), dtype=np.float32), np.eye(4)), synth)

    monkeypatch.setattr(
        "qmri_neuropipe.lib.dmri.tortoise_v4.ensure_supersynth_outputs_for_image",
        lambda *args, **kwargs: {"synth_t2w": synth},
    )

    def fake_convert(in_file, out_file):
        nib.save(nib.Nifti1Image(np.ones((4, 5, 6)), np.eye(4)), out_file)

    monkeypatch.setattr(
        "qmri_neuropipe.lib.dmri.tortoise_v4.freesurfer.mri_convert", fake_convert
    )
    config = PipelineConfig(bids_dir=tmp_path, output_dir=tmp_path, config_data={})
    step = TortoiseV4CorrectionStep(
        config,
        logging.getLogger(__name__),
        None,
        epi=epi,
        t2w_fallback={"source": "synthesized", **extra_options},
    )
    context = {"t1w_files": [source], "t2w_files": [acquired]}

    structurals = step._select_structural(context, tmp_path / "work", False)

    assert structurals and structurals[0].name == "desc-supersynth_T2w.nii.gz"
    assert structurals[0] != acquired
    assert context["tortoise_t2w_source"] == "mri_super_synth"


def test_tortoise_auto_t2w_source_prefers_acquired(tmp_path: Path):
    acquired = tmp_path / "acquired_T2w.nii.gz"
    acquired.touch()
    config = PipelineConfig(bids_dir=tmp_path, output_dir=tmp_path, config_data={})
    step = TortoiseV4CorrectionStep(
        config,
        logging.getLogger(__name__),
        None,
        epi="DRBUDDI",
        t2w_fallback={"source": "auto", "use_for_drbuddi": True},
    )
    context = {"t2w_files": [acquired]}

    structurals = step._select_structural(context, tmp_path / "work", False)

    assert structurals == [acquired]
    assert context["tortoise_t2w_source"] == "acquired"


def test_tortoise_drbuddi_uses_t1w_when_t2w_is_unavailable(tmp_path: Path, caplog):
    t1w = tmp_path / "T1w.nii.gz"
    t1w.touch()
    config = PipelineConfig(bids_dir=tmp_path, output_dir=tmp_path, config_data={})
    step = TortoiseV4CorrectionStep(
        config,
        logging.getLogger(__name__),
        None,
        epi="DRBUDDI",
        t2w_fallback={
            "enabled": False,
            "source": "acquired",
            "use_for_drbuddi": True,
        },
    )
    context = {"t1w_files": [t1w]}

    with caplog.at_level(logging.WARNING):
        structurals = step._select_structural(context, tmp_path / "work", False)

    assert structurals == [t1w]
    assert context["tortoise_structural_source"] == "t1w_fallback"
    assert "using the T1w structural instead" in caplog.text


def test_tortoise_drbuddi_can_run_without_optional_structural(tmp_path: Path, caplog):
    config = PipelineConfig(bids_dir=tmp_path, output_dir=tmp_path, config_data={})
    step = TortoiseV4CorrectionStep(
        config,
        logging.getLogger(__name__),
        None,
        epi="DRBUDDI",
        t2w_fallback={
            "enabled": False,
            "source": "acquired",
            "use_for_drbuddi": True,
        },
    )

    with caplog.at_level(logging.WARNING):
        structurals = step._select_structural({}, tmp_path / "work", False)

    assert structurals is None
    assert "continuing without the optional structural input" in caplog.text


def test_tortoise_coregistration_can_target_synthesized_t2w(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "T1w.nii.gz"
    acquired = tmp_path / "acquired_T2w.nii.gz"
    synth = tmp_path / "SynthT2.mgz"
    nib.save(nib.Nifti1Image(np.ones((4, 5, 6)), np.eye(4)), source)
    nib.save(nib.Nifti1Image(np.ones((4, 5, 6)), np.eye(4)), acquired)
    nib.save(nib.MGHImage(np.ones((7, 8, 9), dtype=np.float32), np.eye(4)), synth)

    monkeypatch.setattr(
        "qmri_neuropipe.lib.dmri.tortoise_v4.ensure_supersynth_outputs_for_image",
        lambda *args, **kwargs: {"synth_t2w": synth},
    )

    def fake_convert(in_file, out_file):
        nib.save(nib.Nifti1Image(np.ones((7, 8, 9)), np.eye(4)), out_file)

    monkeypatch.setattr(
        "qmri_neuropipe.lib.dmri.tortoise_v4.freesurfer.mri_convert", fake_convert
    )
    config = PipelineConfig(bids_dir=tmp_path, output_dir=tmp_path, config_data={})
    step = TortoiseV4CorrectionStep(
        config,
        logging.getLogger(__name__),
        None,
        epi="T2Wreg",
        t2w_fallback={"source": "auto"},
        coregistration_to_anatomy={
            "enabled": True,
            "reference": "synthesized",
            "output_resolution": "anatomical",
        },
    )
    context = {"t1w_files": [source], "t2w_files": [acquired]}
    work_dir = tmp_path / "work"

    structurals = step._select_structural(context, work_dir, False)
    reorientation = step._select_reorientation(
        context, structurals, output_dir=work_dir, force=False
    )

    assert structurals == [acquired]
    assert reorientation and reorientation.name == "desc-supersynth_T2w.nii.gz"
    assert reorientation != acquired
    assert context["tortoise_t2w_source"] == "mri_super_synth"


def test_tortoise_can_skull_strip_structural_and_reorientation_once(
    tmp_path: Path, monkeypatch
):
    structural = tmp_path / "desc-supersynth_T2w.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((5, 6, 7)), np.eye(4)), structural)
    calls = []

    class FakeMaskingStep:
        def __init__(self, *args, **kwargs):
            calls.append(("init", kwargs))

        def __call__(self, image, output_dir, **kwargs):
            calls.append(("run", Path(image.img), Path(output_dir), kwargs))
            output_dir.mkdir(parents=True, exist_ok=True)
            brain = output_dir / "masked.nii.gz"
            mask = output_dir / "mask.nii.gz"
            source = nib.load(image.img)
            nib.save(
                nib.Nifti1Image(np.asanyarray(source.dataobj), source.affine, source.header),
                brain,
            )
            nib.save(
                nib.Nifti1Image(
                    np.ones(source.shape, dtype=np.uint8), source.affine, source.header
                ),
                mask,
            )
            return ImageFile(img=brain, entities={}), ImageFile(img=mask, entities={})

    monkeypatch.setattr(
        "qmri_neuropipe.lib.dmri.tortoise_v4.BrainMaskingStep", FakeMaskingStep
    )
    config = PipelineConfig(bids_dir=tmp_path, output_dir=tmp_path, config_data={})
    step = TortoiseV4CorrectionStep(
        config,
        logging.getLogger(__name__),
        None,
        structural_brain_masking={
            "enabled": True,
            "method": "synthstrip",
            "apply_to": ["structural", "reorientation"],
        },
    )
    context = {}

    masked_structurals, masked_reorientation = step._mask_selected_structurals(
        context,
        [structural],
        structural,
        tmp_path / "work",
        force=False,
        nthreads=4,
    )

    run_calls = [call for call in calls if call[0] == "run"]
    assert len(run_calls) == 1
    assert masked_structurals == [masked_reorientation]
    assert masked_reorientation != structural
    assert {record["role"] for record in context["tortoise_structural_masking"]} == {
        "structural",
        "reorientation",
    }


def test_tortoise_thread_override_precedes_pipeline_cpu_count(tmp_path: Path):
    config = PipelineConfig(
        bids_dir=tmp_path,
        output_dir=tmp_path,
        n_cpus=12,
        config_data={},
    )
    default_step = TortoiseV4CorrectionStep(config, logging.getLogger(__name__), None)
    override_step = TortoiseV4CorrectionStep(
        config, logging.getLogger(__name__), None, nthreads=5
    )

    assert default_step._resolve_nthreads() == 12
    assert default_step._resolve_nthreads(7) == 7
    assert override_step._resolve_nthreads(7) == 5


def test_forced_tortoise_rerun_discards_failed_temp_state(tmp_path: Path):
    temp_folder = tmp_path / "tortoise_work"
    temp_folder.mkdir()
    (temp_folder / "stale.bmtxt").write_text("invalid\n")

    resolved = TortoiseV4CorrectionStep._prepare_temp_folder(tmp_path, force=True)

    assert resolved == temp_folder
    assert not temp_folder.exists()


def test_tortoise_disables_repol_for_synb0_b0_down_by_default(tmp_path: Path):
    config = PipelineConfig(bids_dir=tmp_path, output_dir=tmp_path, config_data={})
    step = TortoiseV4CorrectionStep(
        config,
        logging.getLogger(__name__),
        None,
        repol=True,
        use_synb0=True,
    )

    assert step._resolve_repol() is False


def test_tortoise_keeps_repol_for_native_reverse_pe(tmp_path: Path):
    config = PipelineConfig(bids_dir=tmp_path, output_dir=tmp_path, config_data={})
    step = TortoiseV4CorrectionStep(
        config,
        logging.getLogger(__name__),
        None,
        repol=True,
        use_reverse_pe=True,
    )

    assert step._resolve_repol() is True
