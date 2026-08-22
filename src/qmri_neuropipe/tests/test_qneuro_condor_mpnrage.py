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


def test_create_bids_archive_packages_preprocessed_dwi_derivatives(tmp_path: Path):
    bids_dwi = tmp_path / "bids" / "sub-01" / "ses-02" / "dwi"
    bids_dwi.mkdir(parents=True)
    (bids_dwi / "sub-01_ses-02_dwi.nii.gz").write_bytes(b"raw\n")

    derivative_dwi = tmp_path / "derivatives" / "sub-01" / "ses-02" / "dwi"
    derivative_dwi.mkdir(parents=True)
    preproc = derivative_dwi / "sub-01_ses-02_desc-preproc_dwi.nii.gz"
    preproc.write_bytes(b"preprocessed\n")
    (derivative_dwi / "sub-01_ses-02_desc-preproc_dwi.bval").write_text("0 1000\n")
    (derivative_dwi / "models" / "DTI").mkdir(parents=True)
    (derivative_dwi / "models" / "DTI" / "FA.nii.gz").write_bytes(b"old model\n")

    archive = qneuro_condor.create_bids_archive(
        tmp_path / "bids",
        "01",
        "02",
        tmp_path,
        preprocessed_dir=tmp_path / "derivatives",
    )

    with tarfile.open(archive, "r:gz") as tf:
        member = tf.extractfile(
            "qneuro_preprocessed/sub-01/ses-02/dwi/"
            "sub-01_ses-02_desc-preproc_dwi.nii.gz"
        )
        assert member is not None
        assert member.read() == b"preprocessed\n"
        assert not any("models/DTI" in name for name in tf.getnames())


def test_modeling_only_config_and_derivatives_are_staged(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("dmri:\n  modeling:\n    dti: true\n")
    qneuro_condor.apply_modeling_only(config, "true")

    payload = yaml.safe_load(config.read_text())
    assert payload["dmri"]["modeling_only"] is True
    assert payload["skip_existing"] is True

    source = tmp_path / "qneuro_preprocessed" / "sub-01" / "ses-02" / "dwi"
    source.mkdir(parents=True)
    (source / "sub-01_ses-02_desc-preproc_dwi.bvec").write_text("0 0 0\n")

    qneuro_condor.stage_preprocessed_tree(tmp_path)

    assert (
        tmp_path
        / "out"
        / "sub-01"
        / "ses-02"
        / "dwi"
        / "sub-01_ses-02_desc-preproc_dwi.bvec"
    ).is_file()


def test_auto_mode_allows_missing_preprocessed_subject(tmp_path: Path):
    bids_dwi = tmp_path / "bids" / "sub-03" / "dwi"
    bids_dwi.mkdir(parents=True)
    (bids_dwi / "sub-03_dwi.nii.gz").write_bytes(b"raw\n")

    archive = qneuro_condor.create_bids_archive(
        tmp_path / "bids",
        "03",
        "none",
        tmp_path,
        preprocessed_dir=tmp_path / "derivatives",
    )

    with tarfile.open(archive, "r:gz") as tf:
        assert "sub-03/dwi/sub-03_dwi.nii.gz" in tf.getnames()
        assert not any(name.startswith("qneuro_preprocessed/") for name in tf.getnames())


def test_apply_modeling_only_supports_auto_mode(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("dmri:\n  modeling: {}\n")

    qneuro_condor.apply_modeling_only(config, "auto")

    payload = yaml.safe_load(config.read_text())
    assert payload["dmri"]["modeling_only"] == "auto"
