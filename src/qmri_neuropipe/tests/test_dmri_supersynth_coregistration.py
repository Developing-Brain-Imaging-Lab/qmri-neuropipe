import logging
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.core.types import DWIFile, ImageFile
from qmri_neuropipe.lib.anat.super_synth import extract_mean_b0_for_supersynth
from qmri_neuropipe.interfaces.fsl import rotate_bvecs
from qmri_neuropipe.lib.common.registration import (
    CoregistrationStep,
    _coregistration_output_reference,
    _ensure_fsl_registration_nifti,
    _spatial_grids_match,
    _write_header_registered_image,
)
from qmri_neuropipe.utils.execution_engine import ExecutionEngine
from qmri_neuropipe.workflows.pipelines.integrated_preprocessing_workflow import (
    PreprocessingWorkflow,
)


def _config(
    tmp_path: Path,
    reference_image: str,
    method: str = "fsl",
    **coreg_options,
) -> PipelineConfig:
    return PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "derivatives",
        config_data={
            "dmri": {
                "preprocessing": {
                    "coregistration": {
                        "enabled": True,
                        "method": method,
                        "reference_image": reference_image,
                        **coreg_options,
                    }
                }
            }
        },
    )


def _context(tmp_path: Path) -> tuple[dict, DWIFile, ImageFile]:
    dwi_path = tmp_path / "sub-01_dwi.nii.gz"
    bval_path = tmp_path / "sub-01_dwi.bval"
    anat_path = tmp_path / "sub-01_T1w.nii.gz"
    for path in (dwi_path, bval_path, anat_path):
        path.touch()

    dwi = DWIFile(
        img=dwi_path,
        bval=bval_path,
        entities={"sub": "01", "suffix": "dwi"},
    )
    anat = ImageFile(
        img=anat_path,
        entities={"sub": "01", "suffix": "T1w"},
    )
    return {
        "current_image": dwi,
        "dwi_files": [dwi],
        "t1w_files": [anat],
        "t2w_files": [],
        "_execution_output_dir": tmp_path / "work",
    }, dwi, anat


def _fake_supersynth_outputs(input_image, output_dir, *args, **kwargs):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    synth_t1w = output_dir / "SynthT1.mgz"
    synth_t2w = output_dir / "SynthT2.mgz"
    synth_t1w.touch()
    synth_t2w.touch()
    return {"synth_t1w": synth_t1w, "synth_t2w": synth_t2w}


def _fake_mean_b0(input_dwi, output_path, *args, **kwargs):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.touch()
    return output_path


def test_dmri_supersynth_coregistration_prepares_synthetic_pair(tmp_path: Path):
    config = _config(tmp_path, "supersynth")
    context, _, anat = _context(tmp_path)
    engine = ExecutionEngine(config, logging.getLogger(__name__))
    step = CoregistrationStep(config, logging.getLogger(__name__), method="ants")

    with patch(
        "qmri_neuropipe.lib.anat.super_synth.extract_mean_b0_for_supersynth",
        side_effect=_fake_mean_b0,
    ), patch(
        "qmri_neuropipe.lib.anat.super_synth.ensure_supersynth_outputs_for_image",
        side_effect=_fake_supersynth_outputs,
    ):
        kwargs = engine._prepare_step_kwargs(step, context)

    assert kwargs["target"] == anat.img
    assert kwargs["target_modality"] == "T1w"
    assert kwargs["options"]["registration_fixed"].name == "SynthT1.mgz"
    assert kwargs["options"]["registration_moving"].name == "SynthT1.mgz"
    assert kwargs["options"]["application_fixed"] == anat.img
    assert "registration_fixed_extras" not in kwargs["options"]


def test_dmri_supersynth_multivariate_adds_t2w_channel(tmp_path: Path):
    config = _config(tmp_path, "supersynth_multivariate")
    context, _, _ = _context(tmp_path)
    engine = ExecutionEngine(config, logging.getLogger(__name__))
    step = CoregistrationStep(config, logging.getLogger(__name__), method="ants")

    with patch(
        "qmri_neuropipe.lib.anat.super_synth.extract_mean_b0_for_supersynth",
        side_effect=_fake_mean_b0,
    ), patch(
        "qmri_neuropipe.lib.anat.super_synth.ensure_supersynth_outputs_for_image",
        side_effect=_fake_supersynth_outputs,
    ):
        kwargs = engine._prepare_step_kwargs(step, context)

    assert kwargs["options"]["registration_fixed_extras"][0].name == "SynthT2.mgz"
    assert kwargs["options"]["registration_moving_extras"][0].name == "SynthT2.mgz"


def test_dmri_supersynth_coregistration_preserves_fsl_backend(tmp_path: Path):
    config = _config(tmp_path, "supersynth")
    workflow = PreprocessingWorkflow(config, logging.getLogger(__name__), None)

    workflow._add_coregistration_step(
        config.get("dmri.preprocessing"),
        {},
    )

    assert len(workflow.steps) == 1
    assert workflow.steps[0].method == "fsl"


def test_fsl_multivariate_request_falls_back_to_single_synthetic_pair(tmp_path: Path):
    config = _config(tmp_path, "supersynth_multivariate")
    context, _, _ = _context(tmp_path)
    engine = ExecutionEngine(config, logging.getLogger(__name__))
    step = CoregistrationStep(config, logging.getLogger(__name__), method="fsl")

    with patch(
        "qmri_neuropipe.lib.anat.super_synth.extract_mean_b0_for_supersynth",
        side_effect=_fake_mean_b0,
    ), patch(
        "qmri_neuropipe.lib.anat.super_synth.ensure_supersynth_outputs_for_image",
        side_effect=_fake_supersynth_outputs,
    ):
        kwargs = engine._prepare_step_kwargs(step, context)

    assert "registration_fixed_extras" not in kwargs["options"]
    assert "registration_moving_extras" not in kwargs["options"]


def test_dmri_supersynth_coregistration_preserves_freesurfer_backend(tmp_path: Path):
    config = _config(tmp_path, "supersynth", method="freesurfer")
    workflow = PreprocessingWorkflow(config, logging.getLogger(__name__), None)

    workflow._add_coregistration_step(
        config.get("dmri.preprocessing"),
        {},
    )

    assert len(workflow.steps) == 1
    assert workflow.steps[0].method == "freesurfer"


def test_supersynth_mean_b0_uses_all_low_b_volumes(tmp_path: Path):
    dwi_path = tmp_path / "sub-01_dwi.nii.gz"
    bval_path = tmp_path / "sub-01_dwi.bval"
    output_path = tmp_path / "mean_b0.nii.gz"
    data = np.stack([
        np.full((2, 2, 2), 2.0),
        np.full((2, 2, 2), 8.0),
        np.full((2, 2, 2), 100.0),
    ], axis=3)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(dwi_path))
    np.savetxt(bval_path, np.array([[0, 25, 1000]]), fmt="%g")
    dwi = DWIFile(
        img=dwi_path,
        bval=bval_path,
        entities={"sub": "01", "suffix": "dwi"},
    )

    result = extract_mean_b0_for_supersynth(
        dwi,
        output_path,
        logging.getLogger(__name__),
    )

    mean_b0 = np.asarray(nib.load(str(result)).dataobj)
    assert mean_b0.shape == (2, 2, 2)
    assert np.allclose(mean_b0, 5.0)


def test_fsl_registration_converts_mgz_input_to_nifti(tmp_path: Path):
    mgz_path = tmp_path / "SynthT1.mgz"
    data = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    affine = np.array(
        [
            [1.5, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, -5.0],
            [0.0, 0.0, 2.5, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    nib.save(nib.MGHImage(data, affine), mgz_path)

    result = _ensure_fsl_registration_nifti(
        mgz_path,
        tmp_path,
        "moving",
        logging.getLogger(__name__),
    )

    converted = nib.load(result)
    assert result.name == "fsl_moving_SynthT1.nii.gz"
    np.testing.assert_allclose(np.asarray(converted.dataobj), data)
    np.testing.assert_allclose(converted.affine, affine)


def test_native_coregistration_uses_exact_input_dwi_grid(tmp_path: Path):
    dwi = tmp_path / "dwi_128x128x70.nii.gz"
    synthetic_fixed = tmp_path / "SynthT1.mgz"
    anatomical = tmp_path / "T1w_256x256x256.nii.gz"

    assert _coregistration_output_reference(
        dwi,
        synthetic_fixed,
        anatomical,
        "native",
    ) == dwi
    assert _coregistration_output_reference(
        dwi,
        synthetic_fixed,
        anatomical,
        "dwi",
    ) == dwi
    assert _coregistration_output_reference(
        dwi,
        synthetic_fixed,
        anatomical,
        "anatomical",
    ) == anatomical


def test_spatial_grid_check_rejects_same_resolution_with_changed_matrix(tmp_path: Path):
    original = tmp_path / "original.nii.gz"
    changed = tmp_path / "changed.nii.gz"
    same = tmp_path / "same.nii.gz"
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(np.zeros((128, 128, 70)), affine), original)
    nib.save(nib.Nifti1Image(np.zeros((129, 129, 129)), affine), changed)
    nib.save(nib.Nifti1Image(np.zeros((128, 128, 70)), affine), same)

    assert _spatial_grids_match(original, same)
    assert not _spatial_grids_match(original, changed)


def test_header_registration_preserves_voxels_matrix_and_resolution(tmp_path: Path):
    source_path = tmp_path / "source_dwi.nii.gz"
    output_path = tmp_path / "header_coreg_dwi.nii.gz"
    data = np.arange(5 * 6 * 7 * 2, dtype=np.int16).reshape((5, 6, 7, 2))
    source_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(data, source_affine), source_path)
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    transform[:3, 3] = [10.0, -4.0, 3.0]

    _write_header_registered_image(source_path, output_path, transform)

    source = nib.load(str(source_path))
    output = nib.load(str(output_path))
    assert output.shape == source.shape
    np.testing.assert_array_equal(np.asanyarray(output.dataobj), data)
    np.testing.assert_allclose(
        nib.affines.voxel_sizes(output.affine),
        nib.affines.voxel_sizes(source.affine),
    )
    np.testing.assert_allclose(output.affine, transform @ source.affine)
    np.testing.assert_allclose(output.get_qform(), output.get_sform(), atol=1e-6)


def test_bvec_rotation_uses_only_proper_rotation_component(tmp_path: Path):
    bvec_path = tmp_path / "input.bvec"
    matrix_path = tmp_path / "transform.mat"
    output_path = tmp_path / "output.bvec"
    bvec_path.write_text("1 0 0\n0 1 0\n0 0 0\n")
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([2.0, 3.0, 4.0])
    np.savetxt(matrix_path, affine)

    rotate_bvecs(bvec_path, matrix_path, output_path)

    rotated = np.loadtxt(output_path)
    np.testing.assert_allclose(rotated[:, 0], [0.0, 1.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(rotated[:, 1], [-1.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(rotated[:, 2], [0.0, 0.0, 0.0], atol=1e-7)
