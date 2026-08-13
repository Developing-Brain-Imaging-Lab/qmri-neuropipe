from pathlib import Path

from qmri_neuropipe.interfaces.relaxometry import (
    _append_cli_options,
    _get_qmri_fit_command,
    fit_mcdespot,
)


def _make_executable(path: Path) -> None:
    path.write_text("#!/usr/bin/env sh\nexit 0\n")
    path.chmod(0o755)


def test_get_qmri_fit_command_prefers_hyphenated_cli(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    qmri_fit = bin_dir / "qmri-fit"
    legacy_name = bin_dir / "qmri_fit"
    _make_executable(qmri_fit)
    _make_executable(legacy_name)
    monkeypatch.setenv("PATH", str(bin_dir))

    assert _get_qmri_fit_command("despot1") == [str(qmri_fit), "despot1"]


def test_get_qmri_fit_command_accepts_underscore_cli(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    qmri_fit = bin_dir / "qmri_fit"
    _make_executable(qmri_fit)
    monkeypatch.setenv("PATH", str(bin_dir))

    assert _get_qmri_fit_command("mcdespot") == [str(qmri_fit), "mcdespot"]


def test_value_boolean_options_emit_explicit_false():
    command = ["qmri_fit", "mcdespot"]

    _append_cli_options(
        command,
        {
            "t1f-despot-bounds": False,
            "scale-to-mean": False,
            "progress": False,
            "verbose": True,
        },
    )

    assert command == [
        "qmri_fit",
        "mcdespot",
        "--t1f-despot-bounds=false",
        "--scale-to-mean=false",
        "--verbose",
    ]


def test_unified_mcdespot_command_does_not_emit_retired_t1_option(
    tmp_path, monkeypatch
):
    captured = {}
    monkeypatch.setattr(
        "qmri_neuropipe.interfaces.relaxometry._get_qmri_fit_command",
        lambda *args, **kwargs: ["qmri_fit", "mcdespot"],
    )
    monkeypatch.setattr(
        "qmri_neuropipe.interfaces.relaxometry.run_cmd",
        lambda command, label: captured.update(command=command, label=label),
    )

    fit_mcdespot(
        spgr_file=tmp_path / "spgr.nii.gz",
        ssfp_file=tmp_path / "ssfp.nii.gz",
        t1_file=tmp_path / "t1.nii.gz",
        b1_file=tmp_path / "b1.nii.gz",
        params_file=tmp_path / "params.json",
        out_dir=tmp_path / "out",
    )

    assert captured["command"].startswith("qmri_fit mcdespot ")
    assert "--t1=" not in captured["command"]


def test_legacy_mcdespot_command_keeps_required_t1_option(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "qmri_neuropipe.interfaces.relaxometry._get_qmri_fit_command",
        lambda *args, **kwargs: ["qmri_fit_mcdespot"],
    )
    monkeypatch.setattr(
        "qmri_neuropipe.interfaces.relaxometry.run_cmd",
        lambda command, label: captured.update(command=command, label=label),
    )

    t1_file = tmp_path / "t1.nii.gz"
    fit_mcdespot(
        spgr_file=tmp_path / "spgr.nii.gz",
        ssfp_file=tmp_path / "ssfp.nii.gz",
        t1_file=t1_file,
        b1_file=tmp_path / "b1.nii.gz",
        params_file=tmp_path / "params.json",
        out_dir=tmp_path / "out",
    )

    assert f"--t1={t1_file}" in captured["command"]
