import argparse
import tarfile
from pathlib import Path

from qmri_neuropipe.examples.condor import qneuro_condor


def _write_archive(archive: Path, members: dict[str, str]) -> None:
    payload = archive.parent / "payload"
    payload.mkdir()
    with tarfile.open(archive, "w:gz") as tf:
        for arcname, text in members.items():
            source = payload / arcname.replace("/", "_")
            source.write_text(text)
            tf.add(source, arcname=arcname)


def test_collect_outputs_installs_under_bids_derivatives_with_session(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    archive = outputs_dir / "qneuro_outputs_sub-10021_ses-01.tar.gz"
    _write_archive(
        archive,
        {
            "sub-10021/dwi/sub-10021_desc-preproc_dwi.bval": "0 1000\n",
            "logs/pipeline.log": "done\n",
        },
    )
    bids_dir = tmp_path / "rawdata"
    args = argparse.Namespace(
        outputs_dir=str(outputs_dir),
        remote_host="",
        remote_outputs_dir="",
        bids_dir=str(bids_dir),
        derivatives_dir="",
        derivative_name="qneuro",
        subject="10021",
        subjects="",
        subjects_file="",
        session="01",
        sessions="",
        overwrite=False,
        dry_run=False,
    )

    qneuro_condor.cmd_collect_outputs(args)

    derivative_dir = bids_dir / "derivatives" / "qneuro"
    assert (
        derivative_dir
        / "sub-10021"
        / "ses-01"
        / "dwi"
        / "sub-10021_desc-preproc_dwi.bval"
    ).read_text() == "0 1000\n"
    assert (derivative_dir / "logs" / "pipeline.log").read_text() == "done\n"
    assert (derivative_dir / "dataset_description.json").is_file()


def test_collect_outputs_strips_existing_derivatives_prefix(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    archive = outputs_dir / "qneuro_outputs_sub-10021_ses-01.tar.gz"
    _write_archive(
        archive,
        {
            "derivatives/qneuro/sub-10021/ses-01/anat/sub-10021_ses-01_T1w.nii.gz": "nii\n",
        },
    )
    derivatives_dir = tmp_path / "derivatives" / "qneuro"
    args = argparse.Namespace(
        outputs_dir=str(outputs_dir),
        remote_host="",
        remote_outputs_dir="",
        bids_dir="",
        derivatives_dir=str(derivatives_dir),
        derivative_name="qneuro",
        subject="",
        subjects="",
        subjects_file="",
        session="",
        sessions="",
        overwrite=False,
        dry_run=False,
    )

    qneuro_condor.cmd_collect_outputs(args)

    assert (
        derivatives_dir
        / "sub-10021"
        / "ses-01"
        / "anat"
        / "sub-10021_ses-01_T1w.nii.gz"
    ).read_text() == "nii\n"


def test_make_output_tar_does_not_duplicate_directory_subtrees(tmp_path):
    out_dir = tmp_path / "out" / "sub-10021" / "ses-01" / "dwi"
    out_dir.mkdir(parents=True)
    (out_dir / "sub-10021_ses-01_desc-preproc_dwi.bval").write_text("0 1000\n")
    (out_dir / "sub-10021_ses-01_desc-preproc_dwi.bvec").write_text("1 0 0\n")

    qneuro_condor.make_output_tar(tmp_path, "10021", "01")

    archive = tmp_path / "qneuro_outputs_sub-10021_ses-01.tar.gz"
    with tarfile.open(archive, "r:gz") as tf:
        names = tf.getnames()

    assert len(names) == len(set(names))
    assert names.count("sub-10021/ses-01/dwi/sub-10021_ses-01_desc-preproc_dwi.bval") == 1
    assert names.count("sub-10021/ses-01/dwi/sub-10021_ses-01_desc-preproc_dwi.bvec") == 1
