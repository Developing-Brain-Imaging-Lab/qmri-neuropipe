from pathlib import Path

from qmri_neuropipe.interfaces.relaxometry import _get_qmri_fit_command


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
