import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.core.types import ImageFile
from qmri_neuropipe.lib.relax.motion import RelaxometryMotionCorrectionStep
from qmri_neuropipe.workflows.pipelines.relaxometry import RelaxometryWorkflow
from qmri_neuropipe.workflows.pipelines.relaxometry_config import (
    RelaxometryConfig,
    RelaxometryPreprocConfig,
)


def _workflow(
    tmp_path: Path,
    *,
    spgr_reference=None,
    motion_correction=None,
    skip_existing=False,
):
    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "out",
        skip_existing=skip_existing,
    )
    preprocessing = RelaxometryPreprocConfig(
        spgr_reference=spgr_reference or {"mode": "max_flip"},
        motion_correction=motion_correction
        or {"enabled": False, "method": "fsl"},
    )
    return RelaxometryWorkflow(
        config,
        logging.getLogger("test-spgr-reference"),
        {},
        RelaxometryConfig(preprocessing=preprocessing),
    )


def _spgr(tmp_path: Path, values, flip_angles) -> ImageFile:
    data = np.stack(
        [np.full((3, 3, 3), value, dtype=np.float32) for value in values],
        axis=3,
    )
    image_path = tmp_path / "sub-01_VFA.nii.gz"
    json_path = tmp_path / "sub-01_VFA.json"
    nib.save(nib.Nifti1Image(data, np.eye(4)), image_path)
    json_path.write_text(json.dumps({"FlipAngle": flip_angles}))
    return ImageFile(
        img=image_path,
        json=json_path,
        entities={"sub": "01", "suffix": "VFA"},
    )


def test_max_flip_materializes_the_actual_4d_volume(tmp_path):
    source = _spgr(tmp_path, values=[1, 2, 3], flip_angles=[2, 18, 7])
    workflow = _workflow(tmp_path)

    reference = workflow._build_spgr_reference(
        [source], tmp_path / "anat", {"subject": "01", "session": None}
    )

    result = nib.load(reference.img)
    assert result.shape == (3, 3, 3)
    np.testing.assert_array_equal(result.get_fdata(), 2)
    metadata = json.loads(Path(reference.json).read_text())
    assert metadata["SPGRReferenceGeneration"] == {
        "mode": "max_flip",
        "selected_volume_index": 1,
        "selected_flip_angle": 18.0,
    }


def test_motion_reference_volume_selects_shared_zero_based_volume(tmp_path):
    source = _spgr(tmp_path, values=[10, 20, 30], flip_angles=[2, 8, 18])
    workflow = _workflow(
        tmp_path,
        spgr_reference={"mode": "max_flip"},
        motion_correction={
            "enabled": False,
            "method": "fsl",
            "reference_volume": 0,
        },
    )

    reference = workflow._build_spgr_reference(
        [source], tmp_path / "anat", {"subject": "01", "session": None}
    )

    np.testing.assert_array_equal(nib.load(reference.img).get_fdata(), 10)
    metadata = json.loads(Path(reference.json).read_text())
    assert metadata["SPGRReferenceGeneration"] == {
        "mode": "index",
        "selected_volume_index": 0,
    }


def test_reference_index_out_of_range_is_rejected(tmp_path):
    source = _spgr(tmp_path, values=[1, 2], flip_angles=[2, 18])
    workflow = _workflow(
        tmp_path,
        motion_correction={
            "enabled": True,
            "method": "fsl",
            "reference_volume": 2,
        },
    )

    with pytest.raises(ValueError, match="out of range for 2 logical SPGR"):
        workflow._build_spgr_reference(
            [source], tmp_path / "anat", {"subject": "01", "session": None}
        )


def test_reference_volume_is_not_forwarded_to_registration_backend(tmp_path):
    workflow = _workflow(
        tmp_path,
        motion_correction={
            "enabled": True,
            "method": "fsl",
            "reference_volume": 1,
            "dof": 6,
        },
    )

    motion_step = next(
        step
        for step in workflow.steps
        if step.__class__.__name__ == "RelaxometryMotionCorrectionStep"
    )
    assert motion_step.options == {"dof": 6}


@pytest.mark.parametrize("acquisition", ["SPGR", "SSFP", "IRSPGR"])
def test_recognized_acquisition_uses_nonredundant_preproc_desc(acquisition):
    image = ImageFile(
        img=Path(f"sub-01_acq-{acquisition}_VFA.nii.gz"),
        entities={"sub": "01", "acq": acquisition, "suffix": "VFA"},
    )

    entities = RelaxometryMotionCorrectionStep._preprocessed_entities(
        image, acquisition
    )

    assert entities["desc"] == "preproc"


def test_missing_sequence_acquisition_retains_qualified_desc():
    image = ImageFile(
        img=Path("sub-01_VFA.nii.gz"),
        entities={"sub": "01", "suffix": "VFA"},
    )

    entities = RelaxometryMotionCorrectionStep._preprocessed_entities(
        image, "SSFP"
    )

    assert entities["desc"] == "SSFPpreproc"


def test_existing_series_discovery_accepts_new_and_legacy_names(tmp_path):
    anat_dir = tmp_path / "anat"
    anat_dir.mkdir()
    names = [
        "sub-01_acq-SPGR_desc-preproc_VFA.nii.gz",
        "sub-01_acq-SSFP_desc-preproc_VFA.nii.gz",
        "sub-01_acq-legacy_desc-SPGRpreproc_VFA.nii.gz",
    ]
    for name in names:
        nib.save(
            nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)),
            anat_dir / name,
        )

    workflow = _workflow(tmp_path, skip_existing=True)
    found = workflow._find_existing_preprocessed_series(
        anat_dir,
        modality_label="SPGR",
        context={"subject": "01", "session": None},
    )

    assert {item.img.name for item in found} == {
        "sub-01_acq-SPGR_desc-preproc_VFA.nii.gz",
        "sub-01_acq-legacy_desc-SPGRpreproc_VFA.nii.gz",
    }


def test_motion_and_downstream_receive_the_same_materialized_reference(
    tmp_path, monkeypatch
):
    source = _spgr(tmp_path, values=[1, 7, 3], flip_angles=[2, 18, 7])
    workflow = _workflow(
        tmp_path,
        motion_correction={"enabled": True, "method": "fsl"},
    )
    captured = {}

    def fake_motion(spgr, ssfp, ir, anat_dir, intermediate_dir, reference):
        captured["reference"] = reference
        return spgr, ssfp, ir

    monkeypatch.setattr(workflow, "_run_motion_correction", fake_motion)
    _, _, _, downstream_reference = workflow._prepare_modeling_inputs(
        [source],
        [],
        [],
        tmp_path / "anat",
        tmp_path / "work",
        {"subject": "01", "session": None},
        set(),
        set(),
    )

    assert captured["reference"] is downstream_reference
    np.testing.assert_array_equal(
        nib.load(downstream_reference.img).get_fdata(), 7
    )


def test_reference_builder_supports_separate_3d_spgr_images(tmp_path):
    images = []
    for index, (value, flip_angle) in enumerate(((4, 4), (12, 12))):
        image_path = tmp_path / f"sub-01_flip-{index + 1}_VFA.nii.gz"
        json_path = tmp_path / f"sub-01_flip-{index + 1}_VFA.json"
        nib.save(
            nib.Nifti1Image(np.full((3, 3, 3), value, dtype=np.float32), np.eye(4)),
            image_path,
        )
        json_path.write_text(json.dumps({"FlipAngle": flip_angle}))
        images.append(
            ImageFile(
                img=image_path,
                json=json_path,
                entities={"sub": "01", "flip": str(index + 1), "suffix": "VFA"},
            )
        )

    reference = _workflow(tmp_path)._build_spgr_reference(
        images, tmp_path / "anat", {"subject": "01", "session": None}
    )

    np.testing.assert_array_equal(nib.load(reference.img).get_fdata(), 12)
