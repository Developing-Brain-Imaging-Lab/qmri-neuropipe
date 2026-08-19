from pathlib import Path

from qmri_neuropipe.io.bids import bids_find


def test_bids_find_ignores_hidden_files_and_directories(tmp_path: Path):
    root = tmp_path / ".mounted-volume" / "bids"
    dwi_dir = root / "sub-01" / "dwi"
    dwi_dir.mkdir(parents=True)

    visible = dwi_dir / "sub-01_dwi.nii.gz"
    visible.touch()
    (dwi_dir / "._sub-01_dwi.nii.gz").touch()

    hidden_dir = root / ".staging" / "sub-02" / "dwi"
    hidden_dir.mkdir(parents=True)
    (hidden_dir / "sub-02_dwi.nii.gz").touch()

    found = bids_find(root, suffix="dwi", extension=".nii.gz")

    assert [entry["path"] for entry in found] == [visible]
