import argparse
import csv

import pytest

from qmri_neuropipe.examples.condor import qneuro_condor


def _submit_staged_args(tmp_path, staging_dir, submit_dir, **overrides):
    config_file = tmp_path / "condor-proc.yaml"
    config_file.write_text("pipeline: dmri\n")
    license_file = tmp_path / "license.txt"
    license_file.write_text("fake-license\n")
    values = {
        "staging_dir": str(staging_dir),
        "submit_dir": str(submit_dir),
        "manifest": "",
        "manifest_name": "qneuro_stage.csv",
        "container_image": "/staging/deaniii/qneuro.sif",
        "config_file": str(config_file),
        "freesurfer_license": str(license_file),
        "gnl_coeff_file": "",
        "transfer_uri": "osdf:///chtc/staging/deaniii/qneuro-staging",
        "subject": "",
        "subjects": "",
        "subjects_file": "",
        "session": "",
        "sessions": "",
        "pipeline": "dmri",
        "cpus": 8,
        "gpus": 0,
        "memory_gb": 32,
        "disk_gb": 40,
        "require_dwi": "true",
        "submit_file_name": "qneuro_generated.sub",
        "queue_file_name": "qneuro_inputs.csv",
        "no_submit": True,
        "requirements": '(OpSys == "LINUX")',
        "getenv": "true",
        "gpu_minimum_capability": "8.0",
        "want_flocking": "true",
        "want_glidein": "true",
        "want_gpu_lab": "false",
        "gpu_job_length": "medium",
        "notification": "",
        "notify_user": "",
        "log_dir": "",
        "output_directory": "",
        "output_destination": "",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_submit_staged_infers_rows_without_manifest(tmp_path):
    staging_dir = tmp_path / "staging"
    submit_dir = tmp_path / "submit"
    staging_dir.mkdir()
    (staging_dir / "sub-10021_ses-01_bids.tar.gz").write_bytes(b"")
    (staging_dir / "sub-10022_ses-02_bids.tar.gz").write_bytes(b"")
    args = _submit_staged_args(tmp_path, staging_dir, submit_dir, subject="10021", session="01")

    qneuro_condor.cmd_submit_staged(args)

    with (submit_dir / "qneuro_inputs.csv").open(newline="") as f:
        rows = list(csv.reader(f))
    assert rows == [
        [
            "10021",
            "01",
            "01",
            "osdf:///chtc/staging/deaniii/qneuro-staging/sub-10021_ses-01_bids.tar.gz",
        ]
    ]
    assert (submit_dir / "qneuro_generated.sub").is_file()


def test_submit_staged_keeps_explicit_manifest_strict(tmp_path):
    staging_dir = tmp_path / "staging"
    submit_dir = tmp_path / "submit"
    staging_dir.mkdir()
    args = _submit_staged_args(
        tmp_path,
        staging_dir,
        submit_dir,
        manifest=str(staging_dir / "missing.csv"),
    )

    with pytest.raises(FileNotFoundError, match="Stage manifest not found"):
        qneuro_condor.cmd_submit_staged(args)
