import json
from pathlib import Path

from typer.testing import CliRunner

from qmri_neuropipe.cli import app
from qmri_neuropipe.io.bids_inventory import inspect_bids_dataset


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "bids"
    _write(
        root / "dataset_description.json",
        json.dumps({"Name": "Inventory fixture", "BIDSVersion": "1.10.0"}),
    )
    _write(root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz")
    _write(root / "sub-01" / "ses-01" / "dwi" / "sub-01_ses-01_dir-AP_dwi.nii.gz")
    _write(root / "sub-01" / "ses-01" / "dwi" / "sub-01_ses-01_dir-AP_dwi.bval")
    _write(root / "sub-02" / "anat" / "sub-02_T2w.nii.gz")

    derivative = root / "derivatives" / "qmri-neuropipe"
    _write(
        derivative / "dataset_description.json",
        json.dumps(
            {
                "Name": "qmri-neuropipe outputs",
                "BIDSVersion": "1.10.0",
                "GeneratedBy": [{"Name": "qmri-neuropipe", "Version": "0.0.1"}],
            }
        ),
    )
    _write(
        derivative
        / "sub-01"
        / "ses-01"
        / "dwi"
        / "sub-01_ses-01_space-T1w_desc-preproc_dwi.nii.gz"
    )
    _write(
        derivative
        / "sub-01"
        / "ses-01"
        / "dwi"
        / "DTI"
        / "sub-01_ses-01_FA.nii.gz"
    )
    _write(
        derivative
        / "sub-01"
        / "ses-01"
        / "dwi"
        / "normalized"
        / "sub-01_ses-01_space-MNI152NLin2009cAsym_model-DTI_FA.nii.gz"
    )
    _write(
        derivative
        / "sub-01"
        / "ses-01"
        / "dwi"
        / "sub-01_ses-01_desc-roi_stats.tsv"
    )
    _write(
        derivative
        / "sub-01"
        / "ses-01"
        / "anat"
        / "sub-01_ses-01_model-DESPOT1_T1.nii.gz"
    )
    return root


def test_inventory_counts_raw_data_and_derivatives(tmp_path: Path):
    inventory = inspect_bids_dataset(_dataset(tmp_path), include_derivatives=True)

    assert inventory.name == "Inventory fixture"
    assert inventory.n_subjects == 2
    assert inventory.n_sessions == 1
    assert inventory.n_observations == 2
    assert inventory.sessionless_subjects == ["02"]
    assert inventory.raw_data.n_files == 3
    assert inventory.raw_data.datatypes == {"anat": 2, "dwi": 1}
    assert inventory.raw_data.suffixes == {"T1w": 1, "T2w": 1, "dwi": 1}
    assert inventory.raw_data.modality_coverage["anat"].n_subjects == 2
    assert inventory.raw_data.modality_coverage["anat"].n_observations == 2
    assert inventory.raw_data.modality_coverage["anat"].suffixes == ["T1w", "T2w"]
    assert inventory.raw_data.modality_coverage["dwi"].n_subjects == 1
    assert inventory.raw_data.modality_coverage["dwi"].n_observations == 1

    assert len(inventory.derivatives) == 1
    derivative = inventory.derivatives[0]
    assert derivative.name == "qmri-neuropipe outputs"
    assert derivative.n_subjects == 1
    assert derivative.data.entities["space"]["T1w"] == 1
    assert derivative.data.entities["space"]["MNI152NLin2009cAsym"] == 1
    assert derivative.data.entities["desc"]["preproc"] == 1
    assert derivative.products["dwi"].models == ["DTI"]
    assert derivative.products["dwi"].roi_stats_observations == 1
    assert derivative.products["dwi"].normalized_observations == 1
    assert derivative.products["dwi"].template_spaces == ["MNI152NLin2009cAsym"]
    assert derivative.products["relaxometry"].models == ["DESPOT1"]


def test_inventory_filters_participant_and_session(tmp_path: Path):
    inventory = inspect_bids_dataset(
        _dataset(tmp_path), participants=["sub-01"], sessions=["ses-01"]
    )

    assert inventory.participants == ["01"]
    assert inventory.n_observations == 1
    assert inventory.raw_data.suffixes == {"T1w": 1, "dwi": 1}


def test_inventory_supports_nested_rawdata_with_sibling_derivatives(tmp_path: Path):
    source = _dataset(tmp_path)
    study_root = tmp_path / "study"
    study_root.mkdir()
    rawdata = study_root / "rawdata"
    source.rename(rawdata)
    (rawdata / "derivatives").rename(study_root / "derivatives")

    inventory = inspect_bids_dataset(
        study_root,
        rawdata_dir="rawdata",
        include_derivatives=True,
    )

    assert inventory.path == str(study_root.resolve())
    assert inventory.rawdata_path == str(rawdata.resolve())
    assert inventory.n_subjects == 2
    assert inventory.raw_data.modality_coverage["dwi"].n_observations == 1
    assert inventory.derivatives[0].products["dwi"].models == ["DTI"]


def test_inspect_cli_accepts_relative_rawdata_dir(tmp_path: Path):
    source = _dataset(tmp_path)
    study_root = tmp_path / "study"
    study_root.mkdir()
    rawdata = study_root / "rawdata"
    source.rename(rawdata)
    (rawdata / "derivatives").rename(study_root / "derivatives")

    result = CliRunner().invoke(
        app,
        [
            "inspect",
            str(study_root),
            "--rawdata-dir",
            "rawdata",
            "--derivatives",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["rawdata_path"] == str(rawdata.resolve())
    assert report["n_subjects"] == 2
    assert report["derivatives"][0]["products"]["dwi"]["models"] == ["DTI"]


def test_processing_gaps_compare_raw_and_derivative_observations(tmp_path: Path):
    root = _dataset(tmp_path)
    _write(root / "sub-02" / "dwi" / "sub-02_dwi.nii.gz")

    inventory = inspect_bids_dataset(root, include_processing_gaps=True)

    dwi = inventory.processing_gaps["dwi"]
    assert dwi.raw_observations == 2
    assert dwi.processed_observations == 1
    assert dwi.missing_observations == 1
    assert dwi.missing_subjects == 1
    assert dwi.missing == ["sub-02"]


def test_processing_gaps_cli_details_lists_missing_observations(tmp_path: Path):
    root = _dataset(tmp_path)
    _write(root / "sub-02" / "dwi" / "sub-02_dwi.nii.gz")

    result = CliRunner().invoke(
        app,
        ["inspect", str(root), "--processing-gaps", "--details"],
    )

    assert result.exit_code == 0, result.output
    assert "Processing gaps" in result.output
    assert "Raw observations" in result.output
    assert "Missing derivative observations" in result.output
    assert "sub-02" in result.output


def test_session_breakdown_reports_raw_derivative_and_gap_counts(tmp_path: Path):
    root = _dataset(tmp_path)
    _write(root / "sub-01" / "ses-02" / "dwi" / "sub-01_ses-02_dwi.nii.gz")
    _write(root / "sub-03" / "ses-02" / "dwi" / "sub-03_ses-02_dwi.nii.gz")

    inventory = inspect_bids_dataset(
        root,
        include_derivatives=True,
        include_processing_gaps=True,
        include_session_breakdown=True,
    )

    assert set(inventory.session_breakdown) == {"ses-01", "ses-02", "sessionless"}
    ses01 = inventory.session_breakdown["ses-01"]
    assert ses01.raw_modalities["dwi"].n_observations == 1
    assert ses01.derivative_products["qmri-neuropipe outputs"]["dwi"].models == ["DTI"]
    assert ses01.processing_gaps["dwi"].missing_observations == 0

    ses02 = inventory.session_breakdown["ses-02"]
    assert ses02.n_subjects == 2
    assert ses02.raw_modalities["dwi"].n_observations == 2
    assert ses02.processing_gaps["dwi"].processed_observations == 0
    assert ses02.processing_gaps["dwi"].missing_observations == 2


def test_by_session_cli_renders_session_tables(tmp_path: Path):
    root = _dataset(tmp_path)
    _write(root / "sub-01" / "ses-02" / "dwi" / "sub-01_ses-02_dwi.nii.gz")

    result = CliRunner().invoke(
        app,
        [
            "inspect",
            str(root),
            "--derivatives",
            "--processing-gaps",
            "--by-session",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Raw coverage by session" in result.output
    assert "Derivative model coverage by session" in result.output
    assert "Processing gaps by session" in result.output
    assert "ses-01" in result.output
    assert "ses-02" in result.output
    assert "sessionless" in result.output


def test_inspect_cli_emits_json(tmp_path: Path):
    result = CliRunner().invoke(app, ["inspect", str(_dataset(tmp_path)), "--format", "json"])

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["n_subjects"] == 2
    assert report["raw_data"]["suffixes"]["dwi"] == 1
    assert report["raw_data"]["modality_coverage"]["anat"]["n_subjects"] == 2


def test_inspect_cli_shows_modality_coverage(tmp_path: Path):
    result = CliRunner().invoke(app, ["inspect", str(_dataset(tmp_path))])

    assert result.exit_code == 0, result.output
    assert "Modality coverage" in result.output
    assert "Subjects" in result.output
    assert "Observations" in result.output
    assert "T1w, T2w" in result.output


def test_inspect_cli_shows_derivative_model_coverage(tmp_path: Path):
    result = CliRunner().invoke(
        app, ["inspect", str(_dataset(tmp_path)), "--derivatives"]
    )

    assert result.exit_code == 0, result.output
    assert "Derivative model coverage" in result.output
    assert "Downstream derivative coverage" in result.output
    assert "qmri-neuropipe" in result.output
    assert "Observations" in result.output
    assert "DTI" in result.output
    assert "DESPOT1" in result.output


def test_inspect_cli_details_shows_per_model_coverage(tmp_path: Path):
    result = CliRunner().invoke(
        app, ["inspect", str(_dataset(tmp_path)), "--details"]
    )

    assert result.exit_code == 0, result.output
    assert "Detailed model coverage" in result.output
    assert "DTI" in result.output
    assert "DESPOT1" in result.output


def test_container_run_forwards_help_to_legacy_runner():
    result = CliRunner().invoke(app, ["container", "run", "--help"])

    assert result.exit_code == 0
    assert "--container-image" in result.output
    assert "--config-file" in result.output
