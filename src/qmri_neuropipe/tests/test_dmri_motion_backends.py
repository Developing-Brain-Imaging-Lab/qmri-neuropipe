import logging
from pathlib import Path

import nibabel as nib
import numpy as np

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.core.types import DWIFile
from qmri_neuropipe.interfaces.tortoise import (
    _stage_tortoise_v4_input,
    _validate_tortoise_v4_gradients,
    build_tortoise_v4_command,
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
        config, logging.getLogger(__name__), None, epi="T2Wreg"
    )
    context = {"t1w_files": [source]}

    structurals = step._select_structural(context, tmp_path / "work", False)

    assert structurals and structurals[0].name == "desc-supersynth_T2w.nii.gz"
    assert structurals[0].exists()
    assert context["tortoise_t2w_source"] == "mri_super_synth"


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
