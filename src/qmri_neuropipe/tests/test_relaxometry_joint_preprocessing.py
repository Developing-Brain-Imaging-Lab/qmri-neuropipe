import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.core.types import ImageFile
from qmri_neuropipe.lib.common.denoise import DenoisingStep
from qmri_neuropipe.lib.common.gibbs import GibbsUnringingStep
from qmri_neuropipe.workflows.pipelines.relaxometry import RelaxometryWorkflow
from qmri_neuropipe.workflows.pipelines.relaxometry_config import (
    RelaxometryConfig,
    RelaxometryPreprocConfig,
    parse_relaxometry_config,
)


def _series(
    path: Path,
    values: list[float],
    acquisition: str,
    *,
    affine=None,
) -> ImageFile:
    data = np.stack(
        [np.full((3, 3, 3), value, dtype=np.float32) for value in values],
        axis=3,
    )
    nib.save(
        nib.Nifti1Image(data, np.eye(4) if affine is None else affine),
        path,
    )
    json_path = path.with_suffix("").with_suffix(".json")
    json_path.write_text(json.dumps({"FlipAngle": list(range(len(values)))}))
    return ImageFile(
        img=path,
        json=json_path,
        entities={
            "sub": "01",
            "acq": acquisition,
            "suffix": "VFA",
        },
    )


def _workflow(tmp_path: Path) -> RelaxometryWorkflow:
    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "out",
    )
    preprocessing = RelaxometryPreprocConfig(
        joint_spgr_ssfp={"enabled": True},
        denoising={"enabled": True, "method": "mrtrix"},
        degibbs={"enabled": True, "method": "mrtrix"},
    )
    return RelaxometryWorkflow(
        config,
        logging.getLogger("test-joint-relaxometry-preprocessing"),
        {},
        RelaxometryConfig(preprocessing=preprocessing),
    )


def _copy_with_offset(image: ImageFile, output_dir: Path, offset: float) -> ImageFile:
    source = nib.load(str(image.img))
    data = np.asanyarray(source.dataobj) + offset
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"offset-{int(offset)}_{image.img.name}"
    nib.save(nib.Nifti1Image(data, source.affine, source.header), out_path)
    return ImageFile(img=out_path, entities=dict(image.entities), json=image.json)


def test_joint_preprocessing_concatenates_once_and_splits_at_original_boundary(
    tmp_path, monkeypatch
):
    workflow = _workflow(tmp_path)
    spgr = _series(tmp_path / "sub-01_acq-SPGR_VFA.nii.gz", [1, 2], "SPGR")
    ssfp = _series(
        tmp_path / "sub-01_acq-SSFP_VFA.nii.gz", [10, 20, 30], "SSFP"
    )
    calls: list[tuple[str, tuple[int, ...]]] = []

    def denoise(self, image, output_dir, **kwargs):
        calls.append(("denoise", nib.load(str(image.img)).shape))
        return _copy_with_offset(image, Path(output_dir), 100)

    def degibbs(self, image, output_dir, **kwargs):
        calls.append(("gibbs", nib.load(str(image.img)).shape))
        return _copy_with_offset(image, Path(output_dir), 1000)

    monkeypatch.setattr(DenoisingStep, "__call__", denoise)
    monkeypatch.setattr(GibbsUnringingStep, "__call__", degibbs)

    spgr_out, ssfp_out = workflow._preprocess_joint_spgr_ssfp(
        [spgr], [ssfp], tmp_path / "work"
    )

    assert calls == [("denoise", (3, 3, 3, 5)), ("gibbs", (3, 3, 3, 5))]
    spgr_data = nib.load(str(spgr_out[0].img)).get_fdata()
    ssfp_data = nib.load(str(ssfp_out[0].img)).get_fdata()
    assert spgr_data.shape == (3, 3, 3, 2)
    assert ssfp_data.shape == (3, 3, 3, 3)
    np.testing.assert_array_equal(spgr_data[0, 0, 0], [1101, 1102])
    np.testing.assert_array_equal(ssfp_data[0, 0, 0], [1110, 1120, 1130])

    spgr_metadata = json.loads(Path(spgr_out[0].json).read_text())
    ssfp_metadata = json.loads(Path(ssfp_out[0].json).read_text())
    assert spgr_metadata["JointSPGRSSFPPreprocessing"]["volume_range"] == [0, 2]
    assert ssfp_metadata["JointSPGRSSFPPreprocessing"]["volume_range"] == [2, 5]
    assert spgr_metadata["JointSPGRSSFPPreprocessing"]["operations"] == [
        "denoised",
        "Gibbs",
    ]
    assert spgr_metadata["JointSPGRSSFPPreprocessing"]["configuration"] == {
        "denoising": "mrtrix",
        "degibbs": "mrtrix",
    }


def test_joint_preprocessing_rejects_mismatched_native_grids(tmp_path):
    workflow = _workflow(tmp_path)
    spgr = _series(tmp_path / "spgr.nii.gz", [1, 2], "SPGR")
    shifted = np.eye(4)
    shifted[0, 3] = 1.0
    ssfp = _series(
        tmp_path / "ssfp.nii.gz", [10, 20], "SSFP", affine=shifted
    )

    with pytest.raises(ValueError, match="matching native spatial shape and affine"):
        workflow._preprocess_joint_spgr_ssfp(
            [spgr], [ssfp], tmp_path / "work"
        )


def test_joint_preprocessing_requires_one_4d_file_per_modality(tmp_path):
    workflow = _workflow(tmp_path)
    spgr = _series(tmp_path / "spgr.nii.gz", [1, 2], "SPGR")
    ssfp = _series(tmp_path / "ssfp.nii.gz", [10, 20], "SSFP")

    with pytest.raises(ValueError, match="exactly one 4D SPGR"):
        workflow._preprocess_joint_spgr_ssfp(
            [spgr, spgr], [ssfp], tmp_path / "work"
        )


def test_joint_preprocessing_config_defaults_off_and_parses_enabled():
    assert RelaxometryPreprocConfig().joint_spgr_ssfp == {"enabled": False}

    parsed = parse_relaxometry_config(
        {
            "relaxometry": {
                "preprocessing": {"joint_spgr_ssfp": {"enabled": True}}
            }
        }
    )
    assert parsed.preprocessing.joint_spgr_ssfp == {"enabled": True}


def test_joint_cache_requires_matching_operation_configuration(tmp_path):
    image = _series(tmp_path / "cached.nii.gz", [1, 2], "SPGR")
    metadata = json.loads(Path(image.json).read_text())
    metadata["JointSPGRSSFPPreprocessing"] = {
        "enabled": True,
        "configuration": {"denoising": "mrtrix", "degibbs": "mrtrix"},
    }
    Path(image.json).write_text(json.dumps(metadata))

    assert RelaxometryWorkflow._joint_series_was_used(
        [image], {"denoising": "mrtrix", "degibbs": "mrtrix"}
    )
    assert not RelaxometryWorkflow._joint_series_was_used(
        [image], {"denoising": "mppca", "degibbs": "mrtrix"}
    )
