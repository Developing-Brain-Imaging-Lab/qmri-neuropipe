from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from qmri_neuropipe.core.types import DWIFile
from qmri_neuropipe.lib.dmri.outliers import (
    OutlierRemovalStep,
    _resolve_volumes_file,
)


def test_volumes_file_expands_subject_session_aliases_and_glob(tmp_path):
    qc_dir = tmp_path / "sub-01" / "ses-02"
    qc_dir.mkdir(parents=True)
    expected = qc_dir / "sub-01_ses-02_run-1_bad-volumes.txt"
    expected.write_text("12, 47\n103\n")
    pattern = (
        f"{tmp_path}/sub-{{subject}}/ses-{{session}}/"
        "sub-{sub}_ses-{ses}_*_bad-volumes.txt"
    )

    resolved = _resolve_volumes_file(
        pattern,
        {"sub": "01", "ses": "02", "suffix": "dwi"},
    )

    assert resolved == expected


def test_volumes_file_returns_none_when_subject_pattern_has_no_match(tmp_path):
    pattern = str(tmp_path / "sub-{subject}" / "*_bad-volumes.txt")

    assert _resolve_volumes_file(pattern, {"sub": "02"}) is None


def test_volumes_file_rejects_ambiguous_subject_matches(tmp_path):
    first = tmp_path / "sub-01_run-1_bad-volumes.txt"
    second = tmp_path / "sub-01_run-2_bad-volumes.txt"
    first.write_text("1\n")
    second.write_text("2\n")
    pattern = str(tmp_path / "sub-{subject}_*_bad-volumes.txt")

    with pytest.raises(ValueError, match="matched 2 files"):
        _resolve_volumes_file(pattern, {"sub": "01"})


def test_volumes_file_rejects_unavailable_placeholder(tmp_path):
    pattern = str(tmp_path / "sub-{subject}_ses-{session}_bad-volumes.txt")

    with pytest.raises(ValueError, match=r"placeholder \{session\}"):
        _resolve_volumes_file(pattern, {"sub": "01"})


def test_volumes_file_preserves_literal_path_behavior(tmp_path):
    expected = tmp_path / "bad-volumes.txt"
    expected.write_text("3 4\n")

    assert _resolve_volumes_file(str(expected), {}) == Path(expected)


def test_manual_outlier_step_uses_subject_specific_file_and_syncs_gradients(
    tmp_path,
):
    dwi_path = tmp_path / "sub-01_ses-02_dwi.nii.gz"
    data = np.arange(4, dtype=np.float32).reshape(1, 1, 1, 4)
    nib.save(nib.Nifti1Image(data, np.eye(4)), dwi_path)
    bval = tmp_path / "sub-01_ses-02_dwi.bval"
    bvec = tmp_path / "sub-01_ses-02_dwi.bvec"
    bval.write_text("0 1000 2000 3000\n")
    bvec.write_text("0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    dwi = DWIFile(
        entities={"sub": "01", "ses": "02", "suffix": "dwi"},
        img=dwi_path,
        bval=bval,
        bvec=bvec,
    )
    index_file = tmp_path / "sub-01_ses-02_bad-volumes.txt"
    index_file.write_text("1\n")
    step = OutlierRemovalStep(
        config={},
        logger=None,
        provenance=None,
        method="manual",
        volumes_file=str(
            tmp_path / "sub-{subject}_ses-{session}_bad-volumes.txt"
        ),
    )

    result = step.run(
        {"dwi_files": [dwi], "current_image": dwi},
        tmp_path / "out",
    )

    cleaned = result["current_image"]
    assert nib.load(cleaned.img).shape == (1, 1, 1, 3)
    assert np.loadtxt(cleaned.bval) == pytest.approx([0, 2000, 3000])
    np.testing.assert_allclose(
        np.loadtxt(cleaned.bvec),
        [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
