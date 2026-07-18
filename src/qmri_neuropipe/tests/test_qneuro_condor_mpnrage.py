import tarfile
from pathlib import Path

import yaml

from qmri_neuropipe.examples.condor import qneuro_condor


def test_create_bids_archive_overlays_anatomical_derivatives(tmp_path: Path):
    bids = tmp_path / "bids"
    raw_anat = bids / "sub-01" / "ses-01" / "anat"
    raw_dwi = bids / "sub-01" / "ses-01" / "dwi"
    derivative_anat = tmp_path / "mpnrage" / "sub-01" / "ses-01" / "anat"
    archive_dir = tmp_path / "archives"
    raw_anat.mkdir(parents=True)
    raw_dwi.mkdir(parents=True)
    derivative_anat.mkdir(parents=True)
    archive_dir.mkdir()
    (raw_anat / "sub-01_ses-01_T1w.nii.gz").write_text("raw")
    (raw_dwi / "sub-01_ses-01_dwi.nii.gz").write_text("dwi")
    (derivative_anat / "sub-01_ses-01_acq-MPnRAGE_T1w.nii.gz").write_text("mpnrage")
    (derivative_anat / "sub-01_ses-01_acq-MPnRAGE_T1w.json").write_text("{}")

    archive = qneuro_condor.create_bids_archive(
        bids,
        "01",
        "01",
        archive_dir,
        anat_derivatives_dir=tmp_path / "mpnrage",
    )

    with tarfile.open(archive, "r:gz") as tf:
        names = set(tf.getnames())
    assert "sub-01/ses-01/anat/sub-01_ses-01_acq-MPnRAGE_T1w.nii.gz" in names
    assert "sub-01/ses-01/anat/sub-01_ses-01_acq-MPnRAGE_T1w.json" in names
    assert "sub-01/ses-01/dwi/sub-01_ses-01_dwi.nii.gz" in names


def test_apply_t1w_match_acq_preserves_other_anat_config(tmp_path: Path):
    config = tmp_path / "pipeline.yaml"
    config.write_text("anat:\n  preprocessing:\n    denoise: true\ndmri:\n  preprocessing: {}\n")

    qneuro_condor.apply_t1w_match_entities(config, acq="MPnRAGE")

    payload = yaml.safe_load(config.read_text())
    assert payload["anat"]["preprocessing"]["denoise"] is True
    assert payload["anat"]["input"]["t1w_match"] == {
        "entities": {"acq": "MPnRAGE"}
    }
