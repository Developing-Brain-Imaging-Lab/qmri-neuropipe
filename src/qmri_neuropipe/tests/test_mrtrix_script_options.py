from pathlib import Path

from qmri_neuropipe.interfaces import mrtrix


def _capture_commands(monkeypatch):
    commands = []
    monkeypatch.setattr(mrtrix, "_writable_tmpdir", lambda: "/work/.tmp")
    monkeypatch.setattr(
        mrtrix,
        "run_cmd",
        lambda command, **kwargs: commands.append(command),
    )
    return commands


def test_dwibiascorrect_keeps_backend_selector_before_global_options(
    tmp_path: Path, monkeypatch
):
    commands = _capture_commands(monkeypatch)

    mrtrix.dwibiascorrect(
        in_file=tmp_path / "dwi.nii.gz",
        out_file=tmp_path / "corrected.nii.gz",
        method="ants",
        force=True,
    )

    assert commands
    assert commands[0].startswith(
        "dwibiascorrect ants "
        "-config TmpFileDir /work/.tmp "
        "-config ScriptScratchDir /work/.tmp "
        "-scratch /work/.tmp "
    )


def test_dwigradcheck_accepts_global_options_before_image(
    tmp_path: Path, monkeypatch
):
    commands = _capture_commands(monkeypatch)

    mrtrix.dwigradcheck(
        in_file=tmp_path / "dwi.nii.gz",
        nthreads=2,
        force=True,
    )

    assert commands
    assert commands[0].startswith(
        "dwigradcheck "
        "-config TmpFileDir /work/.tmp "
        "-config ScriptScratchDir /work/.tmp "
        "-scratch /work/.tmp "
        f"{tmp_path / 'dwi.nii.gz'} "
    )
