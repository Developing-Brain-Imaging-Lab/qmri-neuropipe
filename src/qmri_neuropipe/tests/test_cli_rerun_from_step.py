from pathlib import Path

from typer.testing import CliRunner

from qmri_neuropipe.cli import app
from qmri_neuropipe.cli import merge_cli_and_config
from qmri_neuropipe import __version__


def test_cli_reports_package_version():
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == f"qmri-neuropipe {__version__}"


def test_cli_rerun_from_step_overrides_config(tmp_path: Path):
    bids_dir = tmp_path / "bids"
    output_dir = tmp_path / "derivatives"
    bids_dir.mkdir()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"bids_dir: {bids_dir}",
                f"output_dir: {output_dir}",
                "rerun_from_step: brain_masking",
            ]
        ),
        encoding="utf-8",
    )

    config = merge_cli_and_config(
        config_path,
        {"rerun_from_step": "normalization"},
    )

    assert config.get("rerun_from_step") == "normalization"


def test_pipeline_still_runs_without_explicit_run_subcommand(tmp_path: Path):
    bids_dir = tmp_path / "bids"
    output_dir = tmp_path / "derivatives"
    bids_dir.mkdir()

    result = CliRunner().invoke(
        app,
        [
            "--bids-dir",
            str(bids_dir),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Configuration validated successfully" in result.output
    assert "Dry run complete" in result.output


def test_explicit_run_subcommand_remains_available(tmp_path: Path):
    bids_dir = tmp_path / "bids"
    output_dir = tmp_path / "derivatives"
    bids_dir.mkdir()

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--bids-dir",
            str(bids_dir),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Dry run complete" in result.output
