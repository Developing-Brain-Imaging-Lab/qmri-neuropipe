from pathlib import Path

from qmri_neuropipe.cli import merge_cli_and_config


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
