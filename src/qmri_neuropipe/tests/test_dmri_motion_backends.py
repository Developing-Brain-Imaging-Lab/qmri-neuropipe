import logging
from pathlib import Path

import nibabel as nib
import numpy as np

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.core.types import DWIFile
from qmri_neuropipe.interfaces.tortoise import build_tortoise_v4_command
from qmri_neuropipe.lib.dmri.ants_motion import (
    AntsDiffusionMotionCorrectionStep,
    closest_rotation,
    rotate_bvec_table,
)
from qmri_neuropipe.lib.dmri.b0_reference import (
    b0_candidate_indices,
    select_optimal_b0,
)
from qmri_neuropipe.lib.dmri.tortoise_v4 import TortoiseV4CorrectionStep
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
    assert command[command.index("--up_json") + 1] == str(dwi.json)
    assert command[command.index("--b0_id") + 1] == "7"
    assert command[command.index("--s2v") + 1] == "1"
    assert command[command.index("--denoising") + 1] == "off"
    assert command[command.index("--epi") + 1] == "off"


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
